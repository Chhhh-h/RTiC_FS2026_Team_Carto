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

python tools/histmap/prepare_histmap_test_patches.py \
  --dataset-root dataset \
  --split test \
  --output-split test \
  --output-root data/histmap_instance \
  --patch-size 1024 \
  --stride 768 \
  --clean

rm -rf work_dirs/histmap_mask2former_r50/test_instance/label_maps
rm -rf work_dirs/histmap_mask2former_r50/test_instance/vector_vis
rm -rf work_dirs/histmap_mask2former_r50/test_instance/binary_maps
rm -f work_dirs/histmap_mask2former_r50/test_instance/submission.csv

python tools/histmap/infer_histmap_instance_patches.py \
  configs/histmap/mask2former_r50_histblock_1024.py \
  work_dirs/histmap_mask2former_r50/best_coco_segm_mAP_iter_7000.pth \
  --patch-index-csv data/histmap_instance/patch_index_test.csv \
  --output-dir work_dirs/histmap_mask2former_r50/test_instance/label_maps \
  --dataset-root dataset \
  --dataset-split test \
  --score-thr 0.5 \
  --valid-margin 128 \
  --merge-touch-radius 2 \
  --merge-min-contact 20 \
  --save-score-map \
  --save-preview

mkdir -p work_dirs/histmap_mask2former_r50/test_instance/vector_vis
mkdir -p work_dirs/histmap_mask2former_r50/test_instance/binary_maps
for label_map in work_dirs/histmap_mask2former_r50/test_instance/label_maps/*/label_map.tif; do
  image_id=$(basename "$(dirname "$label_map")")
  python -c "import sys, cv2; label = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED); assert label is not None, sys.argv[1]; cv2.imwrite(sys.argv[2], ((label > 0) * 255).astype('uint8'))" \
    "$label_map" \
    "work_dirs/histmap_mask2former_r50/test_instance/binary_maps/${image_id}.png"
  python scripts/histmap/label_map_to_submission.py \
    "$label_map" \
    --image_id "$image_id" \
    --test_dir dataset/test \
    --submission_csv work_dirs/histmap_mask2former_r50/test_instance/submission.csv \
    --preview_path "work_dirs/histmap_mask2former_r50/test_instance/vector_vis/${image_id}.png"
done
