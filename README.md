# RTiC FS2026 Team Carto - Task2

This branch contains the Task2 historical map building-block instance
segmentation pipeline built on top of MMDetection. The code converts binary
building masks into pseudo-instances, trains Mask2Former models, stitches
patch-level predictions back to full maps, and exports submission CSV files.

## What is Included

- `configs/histmap/`: Mask2Former configs for the historical map instance
  segmentation experiments.
- `tools/histmap/prepare_histmap_instance_coco.py`: converts train/validation
  masks into COCO-style instance segmentation patches.
- `tools/histmap/prepare_histmap_test_patches.py`: prepares full-coverage
  validation/test patches for inference.
- `tools/histmap/infer_histmap_instance_patches.py`: runs patch inference,
  stitches labels, optionally saves score maps and previews.
- `tools/histmap/compare_mask2former_4ckpts.py`: summarizes validation logs
  from several backbone/checkpoint experiments.
- `scripts/histmap/*.sh`: Slurm entry points for data preparation, training,
  fine-tuning, validation inference, and test prediction.
- `scripts/histmap/label_map_to_submission.py`: converts instance label maps to
  the required WKT submission CSV.
- `scripts/histmap/postprocess.py`: post-processing utilities for cleaning and
  merging stitched instance labels.

## What is Not Uploaded

Large or generated artifacts are intentionally excluded by `.gitignore`:

- `dataset/`: local input images and masks.
- `data/`: generated COCO annotations and image patches.
- `pretrained/`: downloaded model checkpoints.
- `work_dirs/`: training checkpoints, logs, predictions, previews, and
  submission outputs.
- `logs/`: Slurm stdout/stderr files.
- `*.pth`, `*.pt`, `*.ckpt`, and other model/export binaries.

This keeps the GitHub branch code-only and avoids pushing hundreds of MB of
checkpoints or private/local data.

## Expected Local Layout

Place the provided Task2 data under:

```text
dataset/
  train/
  validation/
  test/
```

The scripts assume they are run from the MMDetection repository root. On the
cluster, the current scripts use:

```text
/cluster/scratch/caizhi/MMDetection
```

Pretrained Mask2Former weights should be placed under `pretrained/`, for
example:

```text
pretrained/mask2former_r50_8xb2-lsj-50e_coco.pth
```

## Workflow

Prepare pseudo-instance training data and inference patches:

```bash
sbatch scripts/histmap/01_prepare_instance_data.sh
```

Train the baseline R50 Mask2Former model:

```bash
sbatch scripts/histmap/02_train_mask2former.sh
```

Fine-tune from the best baseline checkpoint:

```bash
sbatch scripts/histmap/02_finetune_mask2former_best7000.sh
```

Run full-validation inference:

```bash
sbatch scripts/histmap/03_infer_val_instance.sh
```

Run test inference and generate a submission CSV:

```bash
sbatch scripts/histmap/04_predict_test_instance.sh
```

The final test submission is written locally to:

```text
work_dirs/histmap_mask2former_r50/test_instance/submission.csv
```

## Notes

- The training labels are pseudo-instances derived from connected components in
  the binary ground-truth masks.
- Patch inference uses a valid central window to reduce duplicate predictions
  in overlapping patch areas.
- Stitched instance label maps can be exported as WKT polygons through
  `scripts/histmap/label_map_to_submission.py`.
- This repository is based on MMDetection. For installation and framework
  details, refer to the official MMDetection documentation:
  https://mmdetection.readthedocs.io/
