import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, segmentation


def relabel(labels):
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def read_instance_label_npy(path):
    labels = np.load(str(path))

    if labels.ndim != 2:
        raise ValueError(f"Expected 2D instance label map, got shape {labels.shape}: {path}")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"Expected integer instance labels, got dtype {labels.dtype}: {path}"
        )

    labels = labels.astype(np.int32, copy=False)

    if labels.min() < 0:
        raise ValueError(f"Instance labels must be non-negative: {path}")

    print(f"Loaded label map: {path}")
    print(
        f"  shape={labels.shape}, dtype={labels.dtype}, "
        f"min={labels.min()}, max={labels.max()}"
    )

    return labels


def remove_small_labels(labels, min_area):
    labels = labels.astype(np.int32, copy=True)

    if min_area <= 0 or labels.max() == 0:
        return relabel(labels)

    areas = np.bincount(labels.ravel())
    small_labs = np.flatnonzero(areas < min_area)
    small_labs = small_labs[small_labs != 0]

    if len(small_labs) > 0:
        remove_lut = np.zeros(len(areas), dtype=bool)
        remove_lut[small_labs] = True
        labels[remove_lut[labels]] = 0

    return relabel(labels)


def get_fragment_neighbor_labels(labels, y0, y1, x0, x1, mask, source_label):
    """
    Return labels that touch a fragment by direct 4-neighbor edges.
    Background/road label 0 and the source instance label are ignored.
    """
    y0p = max(y0 - 1, 0)
    y1p = min(y1 + 1, labels.shape[0])
    x0p = max(x0 - 1, 0)
    x1p = min(x1 + 1, labels.shape[1])

    label_window = labels[y0p:y1p, x0p:x1p]
    padded_mask = np.zeros(label_window.shape, dtype=bool)
    padded_mask[
        y0 - y0p:y1 - y0p,
        x0 - x0p:x1 - x0p,
    ] = mask

    neighbor_parts = [
        label_window[:-1, :][padded_mask[1:, :]],
        label_window[1:, :][padded_mask[:-1, :]],
        label_window[:, :-1][padded_mask[:, 1:]],
        label_window[:, 1:][padded_mask[:, :-1]],
    ]

    neighbor_labels = np.concatenate(neighbor_parts)
    neighbor_labels = neighbor_labels[
        (neighbor_labels != 0) & (neighbor_labels != source_label)
    ]

    return neighbor_labels.astype(np.int32, copy=False)


def choose_fragment_target(
    labels,
    y0,
    y1,
    x0,
    x1,
    mask,
    source_label,
    min_contact_pixels,
    min_contact_ratio,
):
    neighbor_labels = get_fragment_neighbor_labels(
        labels,
        y0,
        y1,
        x0,
        x1,
        mask,
        source_label,
    )

    if neighbor_labels.size == 0:
        return None, 0, 0.0

    candidate_labels, contact_counts = np.unique(neighbor_labels, return_counts=True)
    best_idx = int(np.argmax(contact_counts))
    best_label = int(candidate_labels[best_idx])
    best_contact = int(contact_counts[best_idx])
    total_contact = int(contact_counts.sum())
    best_ratio = best_contact / total_contact

    if best_contact < min_contact_pixels or best_ratio < min_contact_ratio:
        return None, best_contact, best_ratio

    return best_label, best_contact, best_ratio


def reassign_disconnected_fragments(
    labels,
    min_fragment_area=25,
    min_contact_pixels=5,
    min_contact_ratio=0.5,
):
    """
    Split labels that contain disconnected components.

    The largest component keeps the original label. Other components are
    reassigned to the touching neighboring instance with the strongest 4-edge
    contact. If no reliable neighbor exists, tiny fragments are removed and
    larger fragments become new instances.
    """
    labels = labels.astype(np.int32, copy=True)

    if labels.max() == 0:
        return labels

    print("Fixing disconnected instance fragments...", flush=True)

    out = labels.copy()
    structure = ndi.generate_binary_structure(2, 2)
    bboxes = get_label_bboxes(labels)
    next_label = int(labels.max()) + 1

    split_labels = 0
    reassigned_fragments = 0
    removed_fragments = 0
    new_instance_fragments = 0

    for lab in range(1, labels.max() + 1):
        bbox = bboxes.get(lab)

        if bbox is None:
            continue

        y0, y1, x0, x1 = bbox
        label_window = labels[y0:y1, x0:x1]
        mask = label_window == lab
        component_labels, num_components = ndi.label(mask, structure=structure)

        if num_components <= 1:
            continue

        split_labels += 1
        component_areas = np.bincount(component_labels.ravel())
        component_areas[0] = 0
        main_component = int(np.argmax(component_areas))
        component_slices = ndi.find_objects(component_labels)

        for component_id in range(1, num_components + 1):
            if component_id == main_component:
                continue

            component_mask = component_labels == component_id
            fragment_area = int(component_areas[component_id])
            comp_slices = component_slices[component_id - 1]

            if comp_slices is None:
                continue

            cy_slice, cx_slice = comp_slices
            cy0 = y0 + int(cy_slice.start)
            cy1 = y0 + int(cy_slice.stop)
            cx0 = x0 + int(cx_slice.start)
            cx1 = x0 + int(cx_slice.stop)
            local_component_mask = component_mask[
                cy_slice.start:cy_slice.stop,
                cx_slice.start:cx_slice.stop,
            ]

            target_label, _, _ = choose_fragment_target(
                labels,
                cy0,
                cy1,
                cx0,
                cx1,
                local_component_mask,
                source_label=lab,
                min_contact_pixels=min_contact_pixels,
                min_contact_ratio=min_contact_ratio,
            )

            out_window = out[cy0:cy1, cx0:cx1]

            if target_label is not None:
                out_window[local_component_mask] = target_label
                reassigned_fragments += 1
            elif fragment_area < min_fragment_area:
                out_window[local_component_mask] = 0
                removed_fragments += 1
            else:
                out_window[local_component_mask] = next_label
                next_label += 1
                new_instance_fragments += 1

    print(
        "Disconnected fragment fix: "
        f"{split_labels} split labels, "
        f"{reassigned_fragments} reassigned, "
        f"{removed_fragments} removed, "
        f"{new_instance_fragments} kept as new instances",
        flush=True,
    )

    return relabel(out)


def collect_adjacent_label_contacts(labels):
    labels = labels.astype(np.int32, copy=False)
    max_label = int(labels.max())
    base = max_label + 1
    total_contacts = np.zeros(base, dtype=np.int64)
    pair_keys = []

    for first, second in (
        (labels[:-1, :], labels[1:, :]),
        (labels[:, :-1], labels[:, 1:]),
    ):
        contact_mask = (first > 0) & (second > 0) & (first != second)

        if not np.any(contact_mask):
            continue

        a = first[contact_mask].astype(np.int64, copy=False)
        b = second[contact_mask].astype(np.int64, copy=False)

        total_contacts += np.bincount(a, minlength=base)
        total_contacts += np.bincount(b, minlength=base)

        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pair_keys.append(lo * base + hi)

    if len(pair_keys) == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int64),
            total_contacts,
        )

    pair_keys = np.concatenate(pair_keys)
    unique_keys, contact_counts = np.unique(pair_keys, return_counts=True)
    label_a = (unique_keys // base).astype(np.int32)
    label_b = (unique_keys % base).astype(np.int32)

    return label_a, label_b, contact_counts.astype(np.int64), total_contacts


def union_find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return int(x)


def union_find_join(parent, size, a, b):
    root_a = union_find_root(parent, int(a))
    root_b = union_find_root(parent, int(b))

    if root_a == root_b:
        return False

    if size[root_a] < size[root_b]:
        root_a, root_b = root_b, root_a

    parent[root_b] = root_a
    size[root_a] += size[root_b]

    return True


def merge_adjacent_instances(
    labels,
    min_contact_pixels=20,
    min_contact_ratio=0.35,
):
    """
    Merge different instance labels that share a strong direct 4-neighbor edge.

    Label 0 is ignored, so road/background still blocks merges. Area ratio is
    intentionally not used because split parts can be large.
    """
    labels = labels.astype(np.int32, copy=True)
    max_label = int(labels.max())
    merge_group_map = np.zeros(labels.shape, dtype=np.int32)

    if max_label == 0:
        return labels, merge_group_map

    print("Merging adjacent instances with shared edges...", flush=True)

    label_a, label_b, contact_counts, total_contacts = collect_adjacent_label_contacts(labels)

    if contact_counts.size == 0:
        print("Adjacent merge: no non-background instance contacts found", flush=True)
        return labels, merge_group_map

    parent = np.arange(max_label + 1, dtype=np.int32)
    size = np.ones(max_label + 1, dtype=np.int32)
    accepted_pairs = 0

    for a, b, shared in zip(label_a, label_b, contact_counts):
        if shared < min_contact_pixels:
            continue

        ratio_a = shared / total_contacts[a] if total_contacts[a] > 0 else 0.0
        ratio_b = shared / total_contacts[b] if total_contacts[b] > 0 else 0.0

        if max(ratio_a, ratio_b) < min_contact_ratio:
            continue

        if union_find_join(parent, size, a, b):
            accepted_pairs += 1

    if accepted_pairs == 0:
        print("Adjacent merge: 0 pairs accepted", flush=True)
        return labels, merge_group_map

    groups = {}

    for lab in range(1, max_label + 1):
        root = union_find_root(parent, lab)
        groups.setdefault(root, []).append(lab)

    merge_groups = [group for group in groups.values() if len(group) > 1]

    if len(merge_groups) == 0:
        print("Adjacent merge: 0 groups formed", flush=True)
        return labels, merge_group_map

    target_lut = np.arange(max_label + 1, dtype=np.int32)
    group_lut = np.zeros(max_label + 1, dtype=np.int32)

    for group_id, group in enumerate(merge_groups, start=1):
        group = np.asarray(group, dtype=np.int32)
        target_label = int(group.min())
        target_lut[group] = target_label
        group_lut[group] = group_id

    merged = target_lut[labels]
    merge_group_map = group_lut[labels]

    print(
        "Adjacent merge: "
        f"{accepted_pairs} accepted pairs, "
        f"{len(merge_groups)} merged groups, "
        f"{sum(len(group) for group in merge_groups)} labels involved",
        flush=True,
    )

    return relabel(merged), merge_group_map


def save_merge_preview(path, merge_group_map):
    merge_group_map = merge_group_map.astype(np.int32, copy=False)
    max_group = int(merge_group_map.max())
    preview = np.zeros((*merge_group_map.shape, 3), dtype=np.uint8)

    if max_group > 0:
        rng = np.random.default_rng(2027)
        colors = rng.integers(70, 255, size=(max_group + 1, 3), dtype=np.uint8)
        colors[0] = 0
        preview = colors[merge_group_map]

    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), preview)


def fill_instance_holes(mask, min_hole_area=0):
    if min_hole_area > 0:
        return morphology.remove_small_holes(mask.astype(bool), area_threshold=min_hole_area)

    return ndi.binary_fill_holes(mask)


def smooth_instance_by_contours(mask, approx_eps=1.0, min_area=20):
    mask_u8 = mask.astype(np.uint8)

    found = cv2.findContours(
        mask_u8,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours, hierarchy = found[-2], found[-1]

    if hierarchy is None:
        return mask.astype(bool)

    out = np.zeros_like(mask_u8)
    hierarchy = hierarchy[0]

    for i, h in enumerate(hierarchy):
        cnt = contours[i]
        area = abs(cv2.contourArea(cnt))

        if area < min_area:
            continue

        approx = cv2.approxPolyDP(cnt, approx_eps, True)
        parent = h[3]

        if parent == -1:
            cv2.drawContours(out, [approx], -1, 1, thickness=-1)
        else:
            cv2.drawContours(out, [approx], -1, 0, thickness=-1)

    return out.astype(bool)


def get_label_bboxes(labels):
    """
    Return bounding boxes for labels as {label: (y0, y1, x0, x1)}.
    y1 and x1 are exclusive.
    """
    labels = labels.astype(np.int32, copy=False)
    bboxes = {}

    for lab, obj_slice in enumerate(ndi.find_objects(labels), start=1):
        if obj_slice is None:
            continue

        y_slice, x_slice = obj_slice
        bboxes[lab] = (
            int(y_slice.start),
            int(y_slice.stop),
            int(x_slice.start),
            int(x_slice.stop),
        )

    return bboxes


def clean_instance_labels(
    labels,
    min_instance_area=50,
    fill_holes=False,
    min_hole_area=0,
    smooth_contours=False,
    smooth_eps=1.0,
):
    print("Removing small instances before cleaning...", flush=True)
    labels = remove_small_labels(labels, min_instance_area)

    if not fill_holes and not smooth_contours:
        return labels

    out = np.zeros_like(labels, dtype=np.int32)
    print("Computing instance bounding boxes...", flush=True)
    bboxes = get_label_bboxes(labels)
    print(f"Cleaning {len(bboxes)} instance windows...", flush=True)

    for lab in range(1, labels.max() + 1):
        bbox = bboxes.get(lab)

        if bbox is None:
            continue

        if lab == 1 or lab % 100 == 0 or lab == labels.max():
            print(f"  cleaned instance {lab}/{labels.max()}", flush=True)

        y0, y1, x0, x1 = bbox
        label_window = labels[y0:y1, x0:x1]
        mask = label_window == lab

        if fill_holes:
            mask = fill_instance_holes(mask, min_hole_area=min_hole_area)

        if smooth_contours:
            mask = smooth_instance_by_contours(
                mask,
                approx_eps=smooth_eps,
                min_area=min_instance_area,
            )

        # Later labels overwrite earlier ones if smoothing/filling creates rare overlaps.
        out_window = out[y0:y1, x0:x1]
        out_window[mask] = lab

    if smooth_contours:
        print("Removing small instances after contour smoothing...", flush=True)
        out = remove_small_labels(out, min_instance_area)

    return out


def save_label_tif(path, labels):
    labels = labels.astype(np.int32)

    if labels.max() <= 65535:
        out = labels.astype(np.uint16)
    else:
        out = labels.astype(np.uint32)

    cv2.imwrite(str(path), out)


def save_preview(path, labels):
    labels = labels.astype(np.int32)
    max_label = labels.max()

    rng = np.random.default_rng(2026)
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)

    if max_label > 0:
        colors[1:] = rng.integers(40, 255, size=(max_label, 3), dtype=np.uint8)

    preview = colors[labels]
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), preview)


def extract_polygon_geometries(geom):
    """
    Extract Polygon geometries from Polygon / MultiPolygon / GeometryCollection.
    """
    if geom.is_empty:
        return []

    if geom.geom_type == "Polygon":
        return [geom]

    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        polygons = []
        for g in geom.geoms:
            polygons.extend(extract_polygon_geometries(g))
        return polygons

    return []


def regularize_polygon_by_buffer(
    poly,
    close_dist=4.0,
    open_dist=1.5,
    simplify_tol=3.0,
):
    """
    Vector-level polygon regularization.

    close_dist:
        Fill inward dents / boundary notches.

    open_dist:
        Remove outward spikes / thin protrusions.

    simplify_tol:
        Simplify boundary after buffer regularization.
    """
    if poly.is_empty:
        return poly

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        return poly

    if close_dist > 0:
        poly = poly.buffer(close_dist, join_style=2).buffer(-close_dist, join_style=2)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            return poly

    if open_dist > 0:
        poly = poly.buffer(-open_dist, join_style=2).buffer(open_dist, join_style=2)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            return poly

    if simplify_tol > 0:
        poly = poly.simplify(simplify_tol, preserve_topology=True)

        if not poly.is_valid:
            poly = poly.buffer(0)

    return poly


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


def labels_to_multipolygon_wkt(
    labels,
    simplify_tol=1.0,
    min_poly_area=20.0,
    use_vector_regularization=False,
    vector_close_dist=4.0,
    vector_open_dist=1.5,
    vector_simplify_tol=3.0,
):
    """
    Export all instance labels as one MULTIPOLYGON WKT.

    If use_vector_regularization=True, polygon-level buffer regularization
    will be applied before exporting WKT.
    """
    try:
        from shapely import wkt as shapely_wkt
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
    except Exception as e:
        raise ImportError("Shapely is required for WKT export. Please install shapely.") from e

    labels = labels.astype(np.int32)
    polygons = []

    for lab in range(1, labels.max() + 1):
        mask = (labels == lab).astype(np.uint8)

        if mask.sum() < min_poly_area:
            continue

        found = cv2.findContours(
            mask,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours, hierarchy = found[-2], found[-1]

        if hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        for i, h in enumerate(hierarchy):
            if h[3] != -1:
                continue

            outer = contours[i]

            if simplify_tol > 0:
                outer = cv2.approxPolyDP(outer, simplify_tol, True)

            outer_pts = outer[:, 0, :]

            if len(outer_pts) < 3:
                continue

            outer_ring = [(float(x), float(y)) for x, y in outer_pts]

            if outer_ring[0] != outer_ring[-1]:
                outer_ring.append(outer_ring[0])

            holes = []
            child = h[2]

            while child != -1:
                hole = contours[child]

                if simplify_tol > 0:
                    hole = cv2.approxPolyDP(hole, simplify_tol, True)

                hole_pts = hole[:, 0, :]

                if len(hole_pts) >= 3:
                    hole_ring = [(float(x), float(y)) for x, y in hole_pts]

                    if hole_ring[0] != hole_ring[-1]:
                        hole_ring.append(hole_ring[0])

                    holes.append(hole_ring)

                child = hierarchy[child][0]

            outer_ring = orient_ring(outer_ring, clockwise=False)
            holes = [orient_ring(hole, clockwise=True) for hole in holes]

            poly = Polygon(outer_ring, holes)

            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_empty:
                continue

            if use_vector_regularization:
                poly = regularize_polygon_by_buffer(
                    poly,
                    close_dist=vector_close_dist,
                    open_dist=vector_open_dist,
                    simplify_tol=vector_simplify_tol,
                )

            if poly.is_empty:
                continue

            regularized_polys = extract_polygon_geometries(poly)

            for p in regularized_polys:
                if p.is_empty:
                    continue

                if not p.is_valid:
                    p = p.buffer(0)

                if p.is_empty:
                    continue

                if p.area < min_poly_area:
                    continue

                polygons.append(p)

    if len(polygons) == 0:
        return "MULTIPOLYGON EMPTY"

    geom = MultiPolygon(polygons)

    if not geom.is_valid:
        geom = unary_union(polygons)

    if geom.is_empty:
        return "MULTIPOLYGON EMPTY"

    final_polygons = extract_polygon_geometries(geom)
    final_polygons = [
        p for p in final_polygons
        if (not p.is_empty) and p.area >= min_poly_area
    ]

    if len(final_polygons) == 0:
        return "MULTIPOLYGON EMPTY"

    geom = MultiPolygon(final_polygons)

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty:
        return "MULTIPOLYGON EMPTY"

    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])

    elif geom.geom_type == "MultiPolygon":
        pass

    else:
        final_polygons = extract_polygon_geometries(geom)

        if len(final_polygons) == 0:
            return "MULTIPOLYGON EMPTY"

        geom = MultiPolygon(final_polygons)

    return shapely_wkt.dumps(geom, rounding_precision=0, trim=True)


def save_wkt_polygon_preview(path, wkt_text, shape):
    """
    Save the final WKT polygons as a PNG preview.
    """
    try:
        from shapely import wkt as shapely_wkt
    except Exception as e:
        raise ImportError("Shapely is required for WKT preview export. Please install shapely.") from e

    geom = shapely_wkt.loads(wkt_text)
    h, w = shape[:2]
    preview = np.zeros((h, w, 3), dtype=np.uint8)

    if geom.is_empty:
        cv2.imwrite(str(path), preview)
        return

    if geom.geom_type == "Polygon":
        polygons = [geom]
    elif geom.geom_type == "MultiPolygon":
        polygons = list(geom.geoms)
    else:
        polygons = []

    for poly in polygons:
        exterior = np.rint(np.asarray(poly.exterior.coords)).astype(np.int32)
        cv2.fillPoly(preview, [exterior], color=(255, 255, 255))

        for interior in poly.interiors:
            hole = np.rint(np.asarray(interior.coords)).astype(np.int32)
            cv2.fillPoly(preview, [hole], color=(0, 0, 0))

    for poly in polygons:
        exterior = np.rint(np.asarray(poly.exterior.coords)).astype(np.int32)
        cv2.polylines(preview, [exterior], isClosed=True, color=(0, 255, 0), thickness=1)

        for interior in poly.interiors:
            hole = np.rint(np.asarray(interior.coords)).astype(np.int32)
            cv2.polylines(preview, [hole], isClosed=True, color=(0, 0, 255), thickness=1)

    cv2.imwrite(str(path), preview)


def write_submission_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "wkt"])

        for row in rows:
            writer.writerow([row["ID"], row["wkt"]])

    print(f"Saved submission: {out_csv}")


def merge_submission_rows(existing_csv, updated_rows):
    existing_csv = Path(existing_csv)
    updated_by_id = {str(row["ID"]): row for row in updated_rows}

    if not existing_csv.exists():
        print(
            f"Existing submission not found, writing only updated IDs: {existing_csv}",
            flush=True,
        )
        return updated_rows

    print(f"Updating existing submission rows: {existing_csv}", flush=True)

    merged_rows = []
    seen_ids = set()

    with open(existing_csv, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = str(row["ID"])

            if image_id in updated_by_id:
                merged_rows.append(updated_by_id[image_id])
            else:
                merged_rows.append({"ID": image_id, "wkt": row["wkt"]})

            seen_ids.add(image_id)

    for image_id in sorted(updated_by_id):
        if image_id not in seen_ids:
            merged_rows.append(updated_by_id[image_id])

    return merged_rows


def run_one(label_npy, out_dir, args):
    label_npy = Path(label_npy)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_id = label_npy.parent.name

    print("=" * 70)
    print(f"Processing ID: {image_id}")
    print(f"Instance label npy: {label_npy}")

    labels = read_instance_label_npy(label_npy)
    simplify_tol = 10.0 if image_id == "304" else args.simplify_tol

    if simplify_tol != args.simplify_tol:
        print(
            f"Using image-specific simplify_tol for ID {image_id}: {simplify_tol}",
            flush=True,
        )

    if args.fix_disconnected_fragments:
        labels = reassign_disconnected_fragments(
            labels,
            min_fragment_area=args.fragment_min_area,
            min_contact_pixels=args.fragment_min_contact_pixels,
            min_contact_ratio=args.fragment_min_contact_ratio,
        )

    if args.merge_adjacent_instances:
        labels, merge_group_map = merge_adjacent_instances(
            labels,
            min_contact_pixels=args.adjacent_merge_min_contact_pixels,
            min_contact_ratio=args.adjacent_merge_min_contact_ratio,
        )
        print("Saving adjacent merge preview...", flush=True)
        save_merge_preview(out_dir / "adjacent_merge_preview.png", merge_group_map)

    labels = clean_instance_labels(
        labels,
        min_instance_area=args.min_instance_area,
        fill_holes=args.fill_holes,
        min_hole_area=args.min_hole_area,
        smooth_contours=args.smooth_contours,
        smooth_eps=args.smooth_eps,
    )

    print("Saving cleaned label map...", flush=True)
    save_label_tif(out_dir / "clean_instance_labels.tif", labels)
    print("Saving cleaned preview...", flush=True)
    save_preview(out_dir / "clean_instance_preview.png", labels)

    print("Exporting WKT polygons...", flush=True)
    wkt = labels_to_multipolygon_wkt(
        labels,
        simplify_tol=simplify_tol,
        min_poly_area=args.min_poly_area,
        use_vector_regularization=args.use_vector_regularization,
        vector_close_dist=args.vector_close_dist,
        vector_open_dist=args.vector_open_dist,
        vector_simplify_tol=args.vector_simplify_tol,
    )
    print("Saving final polygon preview...", flush=True)
    save_wkt_polygon_preview(
        out_dir / "final_polygon_preview.png",
        wkt,
        labels.shape,
    )

    print(f"ID {image_id}: {labels.max()} instances after postprocess")
    print(f"Outputs saved to: {out_dir}")

    return {
        "ID": image_id,
        "wkt": wkt,
    }


def parse_cli_args(defaults):
    parser = argparse.ArgumentParser(
        description="Postprocess Mask2Former instance label maps and export WKT CSV.",
    )
    parser.add_argument("--input-root", default=defaults.input_root)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument(
        "--image-ids",
        nargs="+",
        default=None,
        help="Only process these image IDs, e.g. --image-ids 304 or --image-ids 301 304.",
    )

    fix_group = parser.add_mutually_exclusive_group()
    fix_group.add_argument(
        "--fix-disconnected-fragments",
        dest="fix_disconnected_fragments",
        action="store_true",
        default=None,
    )
    fix_group.add_argument(
        "--no-fix-disconnected-fragments",
        dest="fix_disconnected_fragments",
        action="store_false",
    )

    merge_group = parser.add_mutually_exclusive_group()
    merge_group.add_argument(
        "--merge-adjacent-instances",
        dest="merge_adjacent_instances",
        action="store_true",
        default=None,
    )
    merge_group.add_argument(
        "--no-merge-adjacent-instances",
        dest="merge_adjacent_instances",
        action="store_false",
    )

    cli_args = parser.parse_args()
    defaults.input_root = cli_args.input_root
    defaults.output_root = cli_args.output_root
    defaults.image_ids = cli_args.image_ids

    if cli_args.fix_disconnected_fragments is not None:
        defaults.fix_disconnected_fragments = cli_args.fix_disconnected_fragments

    if cli_args.merge_adjacent_instances is not None:
        defaults.merge_adjacent_instances = cli_args.merge_adjacent_instances

    return defaults


def main():
    defaults = SimpleNamespace(
        # -----------------------------
        # input / output
        # -----------------------------
        # input_root should contain folders like:
        #   label_maps/301/label_map.npy
        #   label_maps/302/label_map.npy
        #   label_maps/303/label_map.npy
        input_root="Mask2Former",
        output_root="postprocess/connect2",
        image_ids=None,

        # -----------------------------
        # disconnected fragment reassignment
        # -----------------------------
        # If one instance label has disconnected parts, keep the largest part
        # and reassign other parts to the touching neighbor with strongest
        # direct 4-neighbor contact. Label 0 remains a hard road/background
        # barrier, so this step does not merge across roads.
        fix_disconnected_fragments=True,

        # Fragments without a reliable neighboring instance are removed if
        # smaller than this area; larger ones are kept as new instances.
        fragment_min_area=25,

        # Neighbor reassignment requires at least this many touching edge pixels.
        fragment_min_contact_pixels=5,

        # The best neighbor must own this fraction of all non-background,
        # non-source contact around the fragment.
        fragment_min_contact_ratio=0.5,

        # -----------------------------
        # adjacent instance merging
        # -----------------------------
        # Merge different instance labels only when they directly touch by
        # 4-neighbor edges. Area ratio is not used as a hard condition.
        merge_adjacent_instances=False,

        # Minimum shared edge length in pixels.
        adjacent_merge_min_contact_pixels=20,

        # Shared edge must account for this fraction of at least one instance's
        # non-background contacts.
        adjacent_merge_min_contact_ratio=0.35,

        # -----------------------------
        # instance label cleaning
        # -----------------------------
        # Remove tiny instance labels before polygon export.
        min_instance_area=50,

        # Fill holes inside each instance mask.
        fill_holes=True,

        # If > 0, only holes smaller than this pixel area are filled.
        # If 0, all enclosed holes are filled.
        min_hole_area=0,

        # Smooth each instance boundary before export.
        smooth_contours=False,
        smooth_eps=1.0,

        # -----------------------------
        # polygon export
        # -----------------------------
        # Larger simplify_tol means fewer polygon points and smoother outlines.
        simplify_tol=8.0,
        min_poly_area=25.0,

        # -----------------------------
        # vector polygon regularization
        # -----------------------------
        use_vector_regularization=True,

        # Fill inward dents / small boundary notches.
        vector_close_dist=10.0,

        # Remove outward spikes / thin protrusions.
        vector_open_dist=3.0,

        # Simplify boundary after vector regularization.
        vector_simplify_tol=5.0,
    )
    args = parse_cli_args(defaults)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    label_paths = sorted(input_root.glob("*/label_map.npy"))

    if len(label_paths) == 0:
        raise FileNotFoundError(f"No label_map.npy found under: {input_root}")

    if args.image_ids is not None:
        requested_ids = set(str(image_id) for image_id in args.image_ids)
        label_paths = [
            label_path for label_path in label_paths
            if label_path.parent.name in requested_ids
        ]

        if len(label_paths) == 0:
            raise FileNotFoundError(
                f"No requested image IDs found under {input_root}: "
                f"{', '.join(sorted(requested_ids))}"
            )

    submission_rows = []

    for label_path in label_paths:
        out_dir = output_root / label_path.parent.name
        row = run_one(label_path, out_dir, args)
        submission_rows.append(row)

    out_csv = output_root / "submission_mask2former.csv"

    if args.image_ids is not None:
        submission_rows = merge_submission_rows(out_csv, submission_rows)

    write_submission_csv(submission_rows, out_csv)


if __name__ == "__main__":
    main()
