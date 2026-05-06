#!/bin/bash
#SBATCH --job-name=b5_classweight
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

python tools/test.py \
  local_configs/segformer/B5/segformer.b5.512x512.comp_shared5_submission.py \
  work_dirs/segformer_b5_comp_shared5_ce160k/iter_160000.pth \
  --out predictions_final_shared5/submission_results.pkl \
  --eval None

python stitch_test_predictions.py \
  --input-pkl predictions_final_shared5/submission_results.pkl \
  --patch-index-csv data/patches_512_split/patch_index_test.csv \
  --output-npy predictions_final_shared5/stitched_binary_map3.npy \
  --num-classes 5 \
  --zero-based-labels \
  --class-png-dir predictions_final_shared5/class_pngs \
  --class-names background,river,lake,wetland,stream

# python tools/test.py \
#   local_configs/segformer/B5/segformer.b5.512x512.comp_road_binary_submission.py \
#   work_dirs/segformer_b5_comp_road_binary_ce160k/iter_160000.pth \
#   --out predictions_final_road/submission_results.pkl \
#   --eval None

# python stitch_test_predictions.py \
#   --input-pkl predictions_final_road/submission_results.pkl \
#   --patch-index-csv data/patches_512_split/patch_index_test.csv \
#   --output-npy predictions_final_road/stitched_binary_map3.npy \
#   --num-classes 2 \
#   --zero-based-labels \
#   --class-png-dir predictions_final_road/class_pngs \
#   --class-names background,road

# python tools/test.py \
#   local_configs/segformer/B5/segformer.b5.512x512.comp_building_binary_submission.py \
#   work_dirs/segformer_b5_comp_building_binary_ce160k/iter_160000.pth \
#   --out predictions_final_building/submission_results.pkl \
#   --eval None

# python stitch_test_predictions.py \
#   --input-pkl predictions_final_building/submission_results.pkl \
#   --patch-index-csv data/patches_512_split/patch_index_test.csv \
#   --output-npy predictions_final_building/stitched_binary_map3.npy \
#   --num-classes 2 \
#   --zero-based-labels \
#   --class-png-dir predictions_final_building/class_pngs \
#   --class-names background,building

# python tools/test.py \
#   local_configs/segformer/B5/segformer.b5.512x512.comp_forest_binary_submission.py \
#   work_dirs/segformer_b5_comp_forest_binary_ce160k/iter_100000.pth \
#   --out predictions_final_forest/submission_results.pkl \
#   --eval None

# python stitch_test_predictions.py \
#   --input-pkl predictions_final_forest/submission_results.pkl \
#   --patch-index-csv data/patches_512_split/patch_index_test.csv \
#   --output-npy predictions_final_forest/stitched_binary_map3.npy \
#   --num-classes 2 \
#   --zero-based-labels \
#   --class-png-dir predictions_final_forest/class_pngs \
#   --class-names background,forest




echo "Job finished at $(date)"
