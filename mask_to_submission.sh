#!/bin/bash
#SBATCH -J unet_test
#SBATCH -A es_schin
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=rtx_4090:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# module purge
# module load stack/2024-05 gcc/13.2.0 python/3.10.13
source /cluster/home/caizhi/.venvs/swv/bin/activate

cd /cluster/scratch/caizhi/Carto_Project1

python mask_to_submission.py \
    --input-mask predictions_map3/stitched_binary_map3.npy \
    --output-csv submission.csv