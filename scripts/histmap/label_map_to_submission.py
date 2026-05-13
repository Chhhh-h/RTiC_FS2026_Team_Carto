import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np


csv.field_size_limit(sys.maxsize)
REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    args = parse_args()
    label_map_path = Path(args.label_map_path)
    image_id = args.image_id or infer_image_id(label_map_path, args.input_map_path)
    input_map_path = args.input_map_path or infer_input_map_path(image_id, args.test_dir)
    roi_mask_path = args.roi_mask_path or infer_roi_mask_path(image_id, input_map_path, args.test_dir)
    submission_csv = args.submission_csv or str(REPO_ROOT / 'submission.csv')

    polygons = label_map_to_polygons(
        str(label_map_path),
        roi_mask_path,
        min_area=args.min_polygon_area,
        simplify_tolerance=args.simplify_tolerance,
    )
    wkt = polygons_to_multipolygon_wkt(polygons)
    write_submission_csv(submission_csv, image_id, wkt, args.test_dir)

    if not args.no_preview:
        preview_path = args.preview_path or str(label_map_path.parent / 'vectorization_preview.png')
        save_vectorization_preview(
            str(label_map_path),
            polygons,
            preview_path,
            input_map_path=input_map_path,
            roi_mask_path=roi_mask_path,
        )
        print('Save vectorization preview into {}'.format(preview_path))

    print('Image ID: {}'.format(image_id))
    print('Polygons: {}'.format(len(polygons)))
    print('Update submission csv into {}'.format(submission_csv))


def infer_image_id(label_map_path, input_map_path=None):
    if input_map_path:
        return Path(input_map_path).name.split('-')[0]
    parent_name = label_map_path.parent.name
    if parent_name and parent_name != '.':
        return parent_name.split('-')[0]
    return label_map_path.stem.split('-')[0].split('_')[0]


def infer_input_map_path(image_id, test_dir):
    test_path = Path(test_dir)
    for suffix in ('jpg', 'png', 'tif', 'tiff'):
        candidate = test_path / '{}-INPUT.{}'.format(image_id, suffix)
        if candidate.exists():
            return str(candidate)
    return None


def infer_roi_mask_path(image_id, input_map_path=None, test_dir=None):
    candidates = []
    if input_map_path:
        input_path = Path(input_map_path)
        candidates.extend([
            input_path.with_name(input_path.name.replace('-INPUT.jpg', '-INPUT-MASK.png')),
            input_path.with_name(input_path.name.replace('-INPUT.png', '-INPUT-MASK.png')),
            input_path.with_name(input_path.name.replace('-INPUT.tif', '-INPUT-MASK.png')),
            input_path.with_name(input_path.name.replace('-INPUT.tiff', '-INPUT-MASK.png')),
        ])
    if test_dir:
        candidates.append(Path(test_dir) / '{}-INPUT-MASK.png'.format(image_id))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def label_map_to_polygons(label_map_path, roi_mask_path=None, min_area=25.0, simplify_tolerance=2.0):
    labels = cv2.imread(label_map_path, cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError('Cannot read watershed label map: {}'.format(label_map_path))

    valid_mask = np.ones(labels.shape[:2], dtype=bool)
    if roi_mask_path:
        roi = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise FileNotFoundError('Cannot read ROI mask: {}'.format(roi_mask_path))
        if roi.shape[:2] != labels.shape[:2]:
            roi = cv2.resize(roi, (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid_mask = roi > 0

    polygons = []
    label_values = np.unique(labels[valid_mask])
    for label_value in label_values:
        if int(label_value) == 0:
            continue
        component = np.logical_and(labels == label_value, valid_mask).astype(np.uint8)
        if int(component.sum()) < min_area:
            continue
        polygons.extend(binary_mask_to_polygons(component, min_area, simplify_tolerance))
    return polygons


def binary_mask_to_polygons(mask, min_area, simplify_tolerance):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    polygons = []
    for index, contour in enumerate(contours):
        parent = hierarchy[index][3]
        if parent != -1:
            continue

        area = abs(cv2.contourArea(contour))
        if area < min_area:
            continue

        outer = contour_to_ring(contour, simplify_tolerance)
        if not outer:
            continue

        rings = [orient_ring(outer, clockwise=False)]
        child = hierarchy[index][2]
        while child != -1:
            hole = contour_to_ring(contours[child], simplify_tolerance)
            if hole and abs(cv2.contourArea(contours[child])) >= min_area:
                rings.append(orient_ring(hole, clockwise=True))
            child = hierarchy[child][0]

        polygons.append(rings)
    return polygons


def contour_to_ring(contour, simplify_tolerance):
    if simplify_tolerance and simplify_tolerance > 0:
        contour = cv2.approxPolyDP(contour, float(simplify_tolerance), True)
    points = contour.reshape(-1, 2)
    if len(points) < 3:
        return None

    ring = [(int(x), int(y)) for x, y in points]
    deduped = []
    for point in ring:
        if not deduped or deduped[-1] != point:
            deduped.append(point)
    if len(set(deduped)) < 3:
        return None
    if deduped[0] != deduped[-1]:
        deduped.append(deduped[0])
    return deduped


def ring_signed_area(ring):
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        area += x1 * y2 - x2 * y1
    return area / 2.0


def orient_ring(ring, clockwise):
    is_clockwise = ring_signed_area(ring) < 0
    if is_clockwise != clockwise:
        return list(reversed(ring))
    return ring


def polygons_to_multipolygon_wkt(polygons):
    if not polygons:
        return 'MULTIPOLYGON EMPTY'

    polygon_wkts = []
    for rings in polygons:
        ring_wkts = []
        for ring in rings:
            coords = ', '.join('{} {}'.format(x, y) for x, y in ring)
            ring_wkts.append('({})'.format(coords))
        polygon_wkts.append('({})'.format(', '.join(ring_wkts)))
    return 'MULTIPOLYGON ({})'.format(', '.join(polygon_wkts))


def save_vectorization_preview(label_map_path, polygons, output_path, input_map_path=None, roi_mask_path=None):
    labels = cv2.imread(label_map_path, cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError('Cannot read watershed label map: {}'.format(label_map_path))

    if input_map_path:
        image = cv2.imread(input_map_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError('Cannot read input map: {}'.format(input_map_path))
        preview = np.full(image.shape, 255, dtype=np.uint8)
    else:
        preview = np.full((labels.shape[0], labels.shape[1], 3), 255, dtype=np.uint8)

    if roi_mask_path:
        roi = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise FileNotFoundError('Cannot read ROI mask: {}'.format(roi_mask_path))
        if roi.shape[:2] != preview.shape[:2]:
            roi = cv2.resize(roi, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_NEAREST)
        preview[roi == 0] = (0, 0, 0)

    for index, rings in enumerate(polygons):
        color = label_color(index)
        outer = np.array(rings[0], dtype=np.int32)
        cv2.fillPoly(preview, [outer], color, lineType=cv2.LINE_AA)
        for hole in rings[1:]:
            hole_points = np.array(hole, dtype=np.int32)
            cv2.fillPoly(preview, [hole_points], (255, 255, 255), lineType=cv2.LINE_AA)

    for index, rings in enumerate(polygons):
        color = label_color(index)
        outer = np.array(rings[0], dtype=np.int32)
        cv2.polylines(preview, [outer], True, darken_color(color), 3, lineType=cv2.LINE_AA)
        for hole in rings[1:]:
            hole_points = np.array(hole, dtype=np.int32)
            cv2.polylines(preview, [hole_points], True, darken_color(color), 2, lineType=cv2.LINE_AA)

    if not cv2.imwrite(output_path, preview):
        raise RuntimeError('Could not write vectorization preview: {}'.format(output_path))


def label_color(index):
    palette = np.array([
        (143, 202, 255), (142, 229, 157), (152, 221, 238), (184, 174, 255),
        (179, 205, 227), (255, 202, 128), (198, 219, 239), (251, 180, 174),
        (204, 235, 197), (222, 203, 228), (254, 217, 166), (255, 255, 204),
        (229, 216, 189), (253, 218, 236), (242, 242, 242), (188, 189, 220),
        (141, 211, 199), (255, 255, 179), (190, 186, 218), (251, 128, 114),
        (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
        (217, 217, 217), (188, 128, 189), (204, 235, 197), (255, 237, 111),
    ], dtype=np.uint8)
    return tuple(int(v) for v in palette[index % len(palette)])


def darken_color(color):
    return tuple(int(v * 0.45) for v in color)


def write_submission_csv(csv_path, image_id, wkt, test_dir):
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'ID' in row and 'wkt' in row:
                    rows[str(row['ID'])] = row['wkt']

    test_ids = list_test_ids(test_dir)
    if not test_ids:
        test_ids = [str(image_id)]
    submission_rows = {test_id: rows.get(test_id, 'MULTIPOLYGON EMPTY') for test_id in test_ids}
    submission_rows[str(image_id)] = wkt

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID', 'wkt'])
        writer.writeheader()
        for test_id in sorted(submission_rows.keys(), key=sort_id):
            writer.writerow({'ID': test_id, 'wkt': submission_rows[test_id]})


def list_test_ids(test_dir):
    test_path = Path(test_dir)
    if not test_path.exists():
        return []
    return [p.name.split('-')[0] for p in test_path.glob('*-INPUT.jpg')]


def sort_id(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a watershed label_map.tif into competition submission CSV rows.'
    )
    parser.add_argument('label_map_path', help='Path to watershed label_map.tif.')
    parser.add_argument('--image_id', default=None, help='Competition image ID. Defaults to label parent/input prefix.')
    parser.add_argument('--input_map_path', default=None, help='Optional input map image path for preview and ROI inference.')
    parser.add_argument('--roi_mask_path', default=None, help='Optional ROI mask path.')
    parser.add_argument('--submission_csv', default=None, help='Output competition CSV. Defaults to repo_root/submission.csv.')
    parser.add_argument('--test_dir', default=str(REPO_ROOT / 'dataset' / 'test'),
                        help='Test directory used to preserve all required IDs in CSV.')
    parser.add_argument('--preview_path', default=None, help='Output vectorization preview PNG path.')
    parser.add_argument('--no_preview', action='store_true', help='Do not write vectorization preview PNG.')
    parser.add_argument('--min_polygon_area', type=float, default=25.0,
                        help='Drop polygons smaller than this pixel area.')
    parser.add_argument('--simplify_tolerance', type=float, default=2.0,
                        help='Douglas-Peucker simplification tolerance in pixels.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
