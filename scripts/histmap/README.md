# Historical Map Instance Segmentation Scripts

Run from the MMDetection repository root: `/cluster/scratch/caizhi/MMDetection`.

The scripts assume a conda environment named `mmdet`. Override it with `CONDA_ENV_NAME=...` if needed.

## 1. Prepare pseudo-instance COCO data

```bash
sbatch scripts/histmap/01_prepare_instance_data.sh
```

This converts `*-OUTPUT-GT.png` into connected-component pseudo-instances, cuts train/validation patches, writes COCO JSON, and also cuts full-coverage validation/test inference patches.

Outputs:

```text
data/histmap_instance/images/train/
data/histmap_instance/images/val/
data/histmap_instance/images/val_full/
data/histmap_instance/images/test/
data/histmap_instance/annotations/instances_train.json
data/histmap_instance/annotations/instances_val.json
data/histmap_instance/patch_index_*.csv
```

## 2. Train Mask2Former

```bash
sbatch scripts/histmap/02_train_mask2former.sh
```

Default config:

```text
configs/histmap/mask2former_r50_histblock_1024.py
```

Default work dir:

```text
work_dirs/histmap_mask2former_r50
```

## 3. Validate on the full validation map

```bash
sbatch scripts/histmap/03_infer_val_instance.sh
```

Outputs label maps, vector previews, and a validation-style CSV under:

```text
work_dirs/histmap_mask2former_r50/val_full_instance/
```

## 4. Predict test and generate submission

```bash
sbatch scripts/histmap/04_predict_test_instance.sh
```

Outputs:

```text
work_dirs/histmap_mask2former_r50/test_instance/submission.csv
work_dirs/histmap_mask2former_r50/test_instance/vector_vis/
work_dirs/histmap_mask2former_r50/test_instance/label_maps/<ID>/label_map.tif
```

Notes:

- The first version uses connected components as pseudo-instance GT. If the binary GT has touching building blocks, those will be learned as one instance.
- Patch inference uses a central valid window (`--valid-margin 128`) to reduce duplicate predictions in overlap areas.
- The stitched instance label map is converted to WKT by `/cluster/scratch/caizhi/vectorization/inference/label_map_to_submission.py`.
