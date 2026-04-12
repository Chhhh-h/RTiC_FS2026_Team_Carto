#!/bin/bash
#SBATCH --job-name=b5_classweight
#SBATCH --output=logs/segformer_b5_train_%j.out
#SBATCH --error=logs/segformer_b5_train_%j.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1

module load eth_proxy
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
squeue -u $USER

echo "Job started on $(hostname) at $(date)"

cd /cluster/scratch/pangyi/reto/SegFormer || exit 1
mkdir -p logs

python /cluster/scratch/pangyi/reto/SegFormer/tools/test.py local_configs/segformer/B5/segformer.b5.640x640.ade.160kwithbackground.py \
    /cluster/scratch/pangyi/reto/expt_results/b5_640_40k_classweight/latest.pth \
    --out /cluster/scratch/pangyi/reto/submission_result/expt9_b5_class_40k/submission_results.pkl \
    

echo "Job finished at $(date)"