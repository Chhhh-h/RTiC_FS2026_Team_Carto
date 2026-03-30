#!/bin/bash
#SBATCH --job-name=segformer-b5-eval
#SBATCH --output=logs/segformer_b5_eval_%j.out
#SBATCH --error=logs/segformer_b5_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus=rtx_4090:1

module load eth_proxy

echo "Job started on $(hostname) at $(date)"

cd /cluster/scratch/pangyi/reto/SegFormer || exit 1
mkdir -p logs

python /cluster/scratch/pangyi/reto/SegFormer/tools/test.py local_configs/segformer/B5/segformer.b5.640x640.ade.160kwithbackground.py \
    /cluster/scratch/pangyi/reto/expt_results/segformer_b5_640x640_ade_40k/latest.pth \
    --out /cluster/scratch/pangyi/reto/submission_result/expt6/submission_results.pkl \
    

echo "Job finished at $(date)"
