#!/bin/bash
#SBATCH -J hm_m2f_val
#SBATCH -A es_schin
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=rtx_4090:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

source /cluster/home/caizhi/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-mmseg}"
set -euo pipefail

cd /cluster/scratch/caizhi/MMDetection
export PYTHONPATH=$PWD:${PYTHONPATH:-}

CONFIG="${1:-configs/histmap/mask2former_r50_histblock_1024.py}"
CHECKPOINT="${2:-work_dirs/histmap_mask2former_r50/best_coco_segm_mAP_iter_5000.pth}"
SCORE_THR="${SCORE_THR:-0.5}"
VALID_MARGIN="${VALID_MARGIN:-128}"
MERGE_TOUCH_RADIUS="${MERGE_TOUCH_RADIUS:-2}"
MERGE_MIN_CONTACT="${MERGE_MIN_CONTACT:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/histmap_mask2former_r50/val_full_instance}"
DATASET_ROOT="${DATASET_ROOT:-dataset}"

python tools/histmap/infer_histmap_instance_patches.py \
  "$CONFIG" "$CHECKPOINT" \
  --patch-index-csv data/histmap_instance/patch_index_val_full.csv \
  --output-dir "$OUTPUT_ROOT/label_maps" \
  --dataset-root "$DATASET_ROOT" \
  --dataset-split validation \
  --score-thr "$SCORE_THR" \
  --valid-margin "$VALID_MARGIN" \
  --merge-touch-radius "$MERGE_TOUCH_RADIUS" \
  --merge-min-contact "$MERGE_MIN_CONTACT" \
  --save-preview

SUBMISSION="$OUTPUT_ROOT/submission_val.csv"
rm -f "$SUBMISSION"
mkdir -p "$OUTPUT_ROOT/vector_vis"
for label_map in "$OUTPUT_ROOT"/label_maps/*/label_map.tif; do
  image_id=$(basename "$(dirname "$label_map")")
  python scripts/histmap/label_map_to_submission.py \
    "$label_map" \
    --image_id "$image_id" \
    --test_dir "$DATASET_ROOT/validation" \
    --submission_csv "$SUBMISSION" \
    --preview_path "$OUTPUT_ROOT/vector_vis/${image_id}.png"
done
