#!/bin/bash
#SBATCH -J save_patches
#SBATCH -A es_schin
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# module purge
# module load stack/2024-05 gcc/13.2.0 python/3.10.13
source /cluster/home/caizhi/.venvs/swv/bin/activate

# cd /cluster/scratch/caizhi/Carto_Project1

python cut_test_final_patches.py \
  --test-root data_competition_original/test_final \
  --output-root data \
  --patch-size 512 \
  --stride 384

