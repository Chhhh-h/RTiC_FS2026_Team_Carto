# RTiC FS2026 Team Carto - Task2

This repository branch contains our Task2 pipeline for extracting building
block instances from historical map images and generating the required
submission CSV.

## Task Pipeline

The solution follows this workflow:

1. Convert the provided binary building masks into pseudo-instance labels.
2. Cut train, validation, and test images into overlapping 1024 x 1024 patches.
3. Train Mask2Former instance segmentation models on the generated patches.
4. Run patch-level inference on validation or test images.
5. Stitch patch predictions back into full-size instance label maps.
6. Convert instance label maps into WKT polygons for submission.

## Repository Structure

```text
configs/histmap/
  Model configs used for Task2 experiments.

tools/histmap/
  Data preparation, patch inference, and experiment-summary utilities.

scripts/histmap/
  Slurm scripts for preparing data, training, inference, post-processing,
  and submission generation.
```

Important files:

```text
tools/histmap/prepare_histmap_instance_coco.py
tools/histmap/prepare_histmap_test_patches.py
tools/histmap/infer_histmap_instance_patches.py
scripts/histmap/label_map_to_submission.py
scripts/histmap/postprocess.py
```

## Local Data Layout

Place the Task2 data in the repository root:

```text
dataset/
  train/
  validation/
  test/
```

Generated patches and annotations are written to:

```text
data/histmap_instance/
```

Training and inference outputs are written to:

```text
work_dirs/
```

These data and output directories are not committed to Git.

## Run Commands

Prepare pseudo-instance training data and validation/test patches:

```bash
sbatch scripts/histmap/01_prepare_instance_data.sh
```

Train the baseline model:

```bash
sbatch scripts/histmap/02_train_mask2former.sh
```

Fine-tune from the selected baseline checkpoint:

```bash
sbatch scripts/histmap/02_finetune_mask2former_best7000.sh
```

Run full-validation inference:

```bash
sbatch scripts/histmap/03_infer_val_instance.sh
```

Run test inference and generate the submission:

```bash
sbatch scripts/histmap/04_predict_test_instance.sh
```

The test submission is saved at:

```text
work_dirs/histmap_mask2former_r50/test_instance/submission.csv
```

## Files Excluded From Git

The following files are local data, generated outputs, or large model binaries
and should not be uploaded:

```text
dataset/
data/
pretrained/
work_dirs/
logs/
*.pth
*.pt
*.ckpt
```

Only source code, configuration files, scripts, and documentation should be
committed to this branch.
