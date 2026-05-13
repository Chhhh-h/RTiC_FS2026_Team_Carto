#!/bin/bash
#SBATCH -J hm_inst_prep
#SBATCH -A es_schin
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /cluster/home/caizhi/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-mmseg}"
set -euo pipefail

cd /cluster/scratch/caizhi/MMDetection

DATASET_ROOT="${DATASET_ROOT:-dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/histmap_instance}"
PATCH_SIZE="${PATCH_SIZE:-1024}"
STRIDE="${STRIDE:-768}"
MIN_FULL_AREA="${MIN_FULL_AREA:-25}"
MIN_VISIBLE_AREA="${MIN_VISIBLE_AREA:-25}"

python tools/histmap/prepare_histmap_instance_coco.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --min-full-area "$MIN_FULL_AREA" \
  --min-visible-area "$MIN_VISIBLE_AREA" \
  --clean

python tools/histmap/prepare_histmap_test_patches.py \
  --dataset-root "$DATASET_ROOT" \
  --split validation \
  --output-split val_full \
  --output-root "$OUTPUT_ROOT" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --clean

python tools/histmap/prepare_histmap_test_patches.py \
  --dataset-root "$DATASET_ROOT" \
  --split test \
  --output-split test \
  --output-root "$OUTPUT_ROOT" \
  --patch-size "$PATCH_SIZE" \
  --stride "$STRIDE" \
  --clean
