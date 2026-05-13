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
conda activate mmseg
set -euo pipefail

cd /cluster/scratch/caizhi/MMDetection

python tools/histmap/prepare_histmap_instance_coco.py \
  --dataset-root dataset \
  --output-root data/histmap_instance \
  --patch-size 1024 \
  --stride 768 \
  --min-full-area 25 \
  --min-visible-area 25 \
  --clean

python tools/histmap/prepare_histmap_test_patches.py \
  --dataset-root dataset \
  --split validation \
  --output-split val_full \
  --output-root data/histmap_instance \
  --patch-size 1024 \
  --stride 768 \
  --clean

python tools/histmap/prepare_histmap_test_patches.py \
  --dataset-root dataset \
  --split test \
  --output-split test \
  --output-root data/histmap_instance \
  --patch-size 1024 \
  --stride 768 \
  --clean
