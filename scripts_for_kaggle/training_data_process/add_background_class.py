import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff


def remap_mask(arr: np.ndarray) -> np.ndarray:
    """
    input pixel value:
      - 255: background
      - 0~6: classes (original 0~6)
    输出像素值:
      - 0: background
      - 1~7: classes (original 0~6 incremented by 1)
    """
    out = np.zeros_like(arr, dtype=np.uint8)

    # background 255 -> 0 (out defaults to 0)
    fg = (arr >= 0) & (arr <= 6)
    out[fg] = (arr[fg] + 1).astype(np.uint8)

    return out


def main():
    src_dir = Path("/cluster/scratch/pangyi/reto/data/patches_512_split/annotations/training")
    dst_dir = Path("/cluster/scratch/pangyi/reto/data/patches_512_split/annotations/training_background")

    dst_dir.mkdir(parents=True, exist_ok=True)


    tif_files = list(src_dir.rglob("*.tif")) + list(src_dir.rglob("*.tiff"))
    if not tif_files:
        print(f"[INFO] not found tif/tiff files under {src_dir}.")
        return

    for f in tif_files:
        rel = f.relative_to(src_dir)
        out_f = dst_dir / rel
        out_f.parent.mkdir(parents=True, exist_ok=True)

        arr = tiff.imread(str(f))
        out = remap_mask(arr)
        tiff.imwrite(str(out_f), out)

    print(f"[DONE] processed {len(tif_files)} files, output to: {dst_dir}")


if __name__ == "__main__":
    main()