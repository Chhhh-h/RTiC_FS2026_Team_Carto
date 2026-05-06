# Historical Map Segmentation with SegFormer

This repository contains an MMSegmentation/SegFormer-based workflow for historical map semantic segmentation. It includes custom datasets, SegFormer B5 training configs, data patching scripts, prediction stitching utilities, and submission CSV generation.

The current setup uses only the original competition train/test maps. No auxiliary dataset is required.

The main target classes are:

| ID | Class |
| --- | --- |
| 1 | river |
| 2 | forest |
| 3 | lake |
| 4 | wetland |
| 5 | stream |
| 6 | building |
| 7 | road |

Most final experiments train separate binary models for `road`, `building`, and `forest`, plus a shared 5-class water-related model for `background, river, lake, wetland, stream`.

## Repository Layout

```text
.
├── local_configs/segformer/B5/       # Project-specific SegFormer configs
├── mmseg/datasets/                   # Custom map dataset registrations
├── tools/                            # MMSegmentation train/test entrypoints
├── make_merged_patches.py            # Build train/val/test patches
├── cut_test_final_patches.py         # Build final test patches only
├── stitch_test_predictions.py        # Stitch patch predictions into full-map masks
├── class_pngs_to_stitched_masks.py   # Convert edited class PNGs back to .npy masks
├── mask_to_submission_multi.py       # Convert stitched masks to submission CSV
├── train.sh                          # Example Slurm training job
└── test.sh                           # Example Slurm inference/postprocess job
```

Large runtime folders such as `data/`, `data_competition_original/`, `pretrained/`, `work_dirs/`, `predictions_final*/`, `logs/`, and generated `submission_final*.csv` files are intentionally ignored by Git.

## Environment

The project was developed with Python 3.8, PyTorch 1.10, CUDA 11.3, MMCV 1.3.0, and MMSegmentation-style APIs.

```bash
conda env create -f environment.yml
conda activate segformer_py38
export PYTHONPATH=$PWD:$PYTHONPATH
```

If `mmcv-full` installation fails, install the wheel matching your CUDA/PyTorch versions, then rerun the remaining pip dependencies from `environment.yml`.

## Data Preparation

Place the original competition data outside Git, using this layout:

```text
data_competition_original/
├── train/
│   ├── rgb_map1.png
│   ├── rgb_map2.png
│   ├── gt_mask_map1.tif
│   └── gt_mask_map2.tif
└── test_final/
    ├── rgb_map3.png
    ├── rgb_map4.png
    └── submission_tempate.csv
```

To generate 512x512 patches with stride 384 for final test inference:

```bash
python cut_test_final_patches.py \
  --test-root data_competition_original/test_final \
  --output-root data \
  --patch-size 512 \
  --stride 384
```

To generate training, validation, and test patches from the train and test folders:

```bash
python make_merged_patches.py \
  --train-root data_competition_original/train \
  --test-root data_competition_original/test_final \
  --output-root data \
  --patch-size 512 \
  --stride 384 \
  --val-ratio 0.1
```

The generated files are written under `data/patches_512_split/`.

## Training

The primary configs are:

| Config | Purpose |
| --- | --- |
| `local_configs/segformer/B5/segformer.b5.512x512.comp_shared5.ce160k.py` | 5-class water-related model |
| `local_configs/segformer/B5/segformer.b5.512x512.comp_road_binary.ce160k.py` | road binary model |
| `local_configs/segformer/B5/segformer.b5.512x512.comp_building_binary.ce160k.py` | building binary model |
| `local_configs/segformer/B5/segformer.b5.512x512.comp_forest_binary.ce160k.py` | forest binary model |

Download or place the SegFormer B5 backbone checkpoint at:

```text
pretrained/mit_b5.pth
```

Run one training job directly:

```bash
python tools/train.py \
  local_configs/segformer/B5/segformer.b5.512x512.comp_shared5.ce160k.py \
  --work-dir work_dirs/segformer_b5_comp_shared5_ce160k
```

On a Slurm cluster, adapt `train.sh` and submit it:

```bash
sbatch train.sh
```

## Inference

Run inference on patched final test images:

```bash
python tools/test.py \
  local_configs/segformer/B5/segformer.b5.512x512.comp_shared5_submission.py \
  work_dirs/segformer_b5_comp_shared5_ce160k/iter_160000.pth \
  --out predictions_final_shared5/submission_results.pkl \
  --eval None
```

Stitch patch-level outputs back into full-map masks:

```bash
python stitch_test_predictions.py \
  --input-pkl predictions_final_shared5/submission_results.pkl \
  --patch-index-csv data/patches_512_split/patch_index_test.csv \
  --output-npy predictions_final_shared5/stitched_binary.npy \
  --num-classes 5 \
  --zero-based-labels \
  --class-png-dir predictions_final_shared5/class_pngs \
  --class-names background,river,lake,wetland,stream
```

For final submission, combine the per-class edited PNG masks into one 7-channel mask per map:

```bash
python class_pngs_to_stitched_masks.py \
  --input-root predictions_final_fused \
  --map-ids map3,map4 \
  --output-root predictions_final_fused
```

Then create the CSV:

```bash
python mask_to_submission_multi.py \
  --input-dir predictions_final_fused \
  --template-csv data_competition_original/test_final/submission_tempate.csv \
  --output-csv submission_final.csv
```
