#!/bin/bash
#SBATCH -J hm_m2f_ft
#SBATCH -A es_schin
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=rtx_4090:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /cluster/home/caizhi/miniconda3/etc/profile.d/conda.sh
conda activate mmseg
set -euo pipefail

cd /cluster/scratch/caizhi/MMDetection
export PYTHONPATH=/cluster/scratch/caizhi/MMDetection

python tools/train.py \
  configs/histmap/mask2former_r50_histblock_1024_ft_best7000.py
