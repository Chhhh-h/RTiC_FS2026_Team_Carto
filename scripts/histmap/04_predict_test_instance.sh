#!/bin/bash
#SBATCH -J hm_m2f_test
#SBATCH -A es_schin
#SBATCH --time=2:00:00
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

python tools/histmap/infer_histmap_instance_patches.py \
  configs/histmap/mask2former_r50_histblock_1024.py \
  work_dirs/histmap_mask2former_r50/best_coco_segm_mAP_iter_5000.pth \
  --patch-index-csv data/histmap_instance/patch_index_test.csv \
  --output-dir work_dirs/histmap_mask2former_r50/test_instance/label_maps \
  --dataset-root /cluster/scratch/caizhi/vectorization/dataset \
  --dataset-split test \
  --score-thr 0.5 \
  --valid-margin 128 \
  --merge-touch-radius 2 \
  --merge-min-contact 20 \
  --save-preview

rm -f work_dirs/histmap_mask2former_r50/test_instance/submission.csv
mkdir -p work_dirs/histmap_mask2former_r50/test_instance/vector_vis
for label_map in work_dirs/histmap_mask2former_r50/test_instance/label_maps/*/label_map.tif; do
  image_id=$(basename "$(dirname "$label_map")")
  python scripts/histmap/label_map_to_submission.py \
    "$label_map" \
    --image_id "$image_id" \
    --test_dir /cluster/scratch/caizhi/vectorization/dataset/test \
    --submission_csv work_dirs/histmap_mask2former_r50/test_instance/submission.csv \
    --preview_path "work_dirs/histmap_mask2former_r50/test_instance/vector_vis/${image_id}.png"
done
