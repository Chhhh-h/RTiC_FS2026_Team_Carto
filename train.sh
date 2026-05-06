#!/bin/bash
#SBATCH --job-name=comp_shared5
#SBATCH --output=logs/segformer_b5_train_%j.out
#SBATCH --error=logs/segformer_b5_train_%j.err
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

python tools/train.py \
  local_configs/segformer/B5/segformer.b5.512x512.comp_shared5.ce160k.py \
  --work-dir work_dirs/segformer_b5_comp_shared5_ce160k

# python tools/train.py \
#   local_configs/segformer/B5/segformer.b5.512x512.comp_road_binary.ce160k.py \
#   --work-dir work_dirs/segformer_b5_comp_road_binary_ce160k \
#   --resume-from work_dirs/segformer_b5_comp_road_binary_ce160k/iter_82000.pth

# python tools/train.py \
#   local_configs/segformer/B5/segformer.b5.512x512.comp_building_binary.ce160k.py \
#   --work-dir work_dirs/segformer_b5_comp_building_binary_ce160k \
#   --resume-from work_dirs/segformer_b5_comp_building_binary_ce160k/iter_80000.pth

# python tools/train.py \
#   local_configs/segformer/B5/segformer.b5.512x512.comp_forest_binary.ce160k.py \
#   --work-dir work_dirs/segformer_b5_comp_forest_binary_ce160k \
#   --resume-from work_dirs/segformer_b5_comp_forest_binary_ce160k/iter_80000.pth



echo "Job finished at $(date)"
