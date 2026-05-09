#!/bin/bash

MODEL_PATH="path to model/Qwen3-4B"
DATASET_PATH="path to dataset/C4"
OUTPUT_PATH="path to save layer importance results/layer_importance_qwen3-4b_c4.json"

python calculate_layer_importance.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --output_path $OUTPUT_PATH \
    --max_samples 256

echo "✅ result saved: $OUTPUT_PATH"