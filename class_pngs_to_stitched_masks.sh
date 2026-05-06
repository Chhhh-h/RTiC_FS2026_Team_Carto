#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --output=logs/segformer_b5_test_%j.out
#SBATCH --error=logs/segformer_b5_test_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G


module load eth_proxy

echo "Job started on $(hostname) at $(date)"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
squeue -u $USER

cd /cluster/scratch/caizhi/Segformer || exit 1
# mkdir -p logs
export PYTHONPATH=$PWD:$PYTHONPATH

python class_pngs_to_stitched_masks.py \
  --input-root predictions_final_fused \
  --map-ids map3,map4

python mask_to_submission_multi.py \
  --input-dir predictions_final_fused \
  --template-csv data_competition_original/test_final/submission_tempate.csv \
  --output-csv submission_final.csv


echo "Job finished at $(date)"
