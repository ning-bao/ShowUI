#!/bin/bash
# Example usage of aligned evaluation scripts for ScreenSpot v2 and Pro

# Set your dataset directory
DATASET_DIR="/path/to/your/datasets"

# 1. Evaluate a single model on ScreenSpot v2
echo "Evaluating single model on ScreenSpot v2..."
python eval_baseline.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version v2 \
    --model_id "showlab/ShowUI-2B" \
    --split hf_test_full \
    --output results_screenspot_v2.json

# 2. Evaluate a single model on ScreenSpot Pro
echo "Evaluating single model on ScreenSpot Pro..."
python eval_baseline.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version pro \
    --model_id "showlab/ShowUI-2B" \
    --split hf_test_full \
    --output results_screenspot_pro.json

# 3. Evaluate multiple checkpoints on ScreenSpot v2
echo "Evaluating checkpoints on ScreenSpot v2..."
python eval_checkpoints.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version v2 \
    --models_root "./checkpoints" \
    --model_glob "rl_ckpt_*" \
    --split hf_test_full \
    --output eval_checkpoints_v2.json

# 4. Evaluate multiple checkpoints on ScreenSpot Pro
echo "Evaluating checkpoints on ScreenSpot Pro..."
python eval_checkpoints.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version pro \
    --models_root "./checkpoints" \
    --model_glob "rl_ckpt_*" \
    --split hf_test_full \
    --output eval_checkpoints_pro.json

# 5. Use the unified script for single model evaluation
echo "Using unified script for single model..."
python eval_unified.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version v2 \
    --model_id "showlab/ShowUI-2B" \
    --mode single \
    --split hf_test_full \
    --output unified_results_v2.json

# 6. Use the unified script for checkpoint evaluation
echo "Using unified script for checkpoint evaluation..."
python eval_unified.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version pro \
    --models_root "./checkpoints" \
    --model_glob "rl_ckpt_*" \
    --mode checkpoints \
    --split hf_test_full \
    --output unified_results_pro.json

# 7. Filter by environment and type
echo "Evaluating with filters..."
python eval_baseline.py \
    --dataset_dir "$DATASET_DIR" \
    --dataset_version v2 \
    --model_id "showlab/ShowUI-2B" \
    --envs desktop mobile \
    --types text icon \
    --split hf_test_full \
    --output filtered_results.json

echo "All evaluations completed!"
