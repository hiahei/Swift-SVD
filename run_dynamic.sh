#!/bin/bash
set -e

# --- environment configuration ---
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- model and dataset ---
MODEL_NAME="Llama-7B"
DATA_NAME="C4"
MODEL_PATH="/path/${MODEL_NAME}"
MODEL_SAFE_NAME=$(basename "$MODEL_PATH")
DATA_BASE_ROOT="/path/SVD_data"
DATA_ROOT="${DATA_BASE_ROOT}/${DATA_NAME}"

# --- search space ---
ALPHAS=(0.6) # 0.0 svdllmv2 0.6 swift-svd
COMPRESSION_RATIOS=(0.8)

# --- seeds ---
train_seed=54
eval_seed=7

# --- fixed hyper-parameter ---
SAMPLES_NUM=256
INPUT_SEQLEN=2048
FIXED_DELTA=0.5 # 1.0 uniform 0.5 dynamic 0.0 full

# throughput
gen_seq_len=1024
batch_size=16
prompt_len=32

IMPORTANCE_FILE="llm-layer-importance/layer_importance_${MODEL_NAME}_${DATA_NAME}.json"
    
SVD_FILE="svd_list/${DATA_NAME}/${MODEL_SAFE_NAME}_${DATA_NAME}_svd_list_${SAMPLES_NUM}_${INPUT_SEQLEN}_s${train_seed}.pk"
mkdir -p "svd_list/${DATA_NAME}"

if [ ! -f "$SVD_FILE" ]; then
# 1. train svd for V and sigma
    echo "Training SVD for Seed $train_seed..."
    python train_svd.py \
        --model_name "$MODEL_PATH" \
        --data_root "$DATA_ROOT" \
        --data_name "$DATA_NAME" \
        --max_samples $SAMPLES_NUM \
        --seqlen $INPUT_SEQLEN \
        --seed $train_seed
fi

for a_val in "${ALPHAS[@]}"; do
    for ratio in "${COMPRESSION_RATIOS[@]}"; do
        RANK_ALLOC_FILE="temp_alloc_${MODEL_SAFE_NAME}_a${a_val}_r${ratio}.pk"
        OUTPUT_DIR="./tmp_model_${MODEL_SAFE_NAME}_a${a_val}_r${ratio}_${DATA_NAME}"

        echo "============================================================"
        echo "Preparing Model: Alpha=$a_val | Ratio=$ratio | TrainSeed=$train_seed"
        echo "============================================================"

        # 2. dynamic rank allocation
        python dynamic_rank_allocation.py \
            --local_model_path "$MODEL_PATH" \
            --svd_file "$SVD_FILE" \
            --importance_file "$IMPORTANCE_FILE" \
            --compression_ratio "$ratio" \
            --output_file "$RANK_ALLOC_FILE" \
            --delta "$FIXED_DELTA" \
            --importance_weight "$a_val"

        # 3. generate compressed model weights
        if [ ! -d "$OUTPUT_DIR" ]; then
            python update_svd_weights_end.py \
                --model_path "$MODEL_PATH" \
                --svd_file "$SVD_FILE" \
                --rank_allocation "$RANK_ALLOC_FILE" \
                --output_dir "$OUTPUT_DIR" \
                --ratio "$ratio"
        else
            echo "Compressed model already exists, reusing..."
        fi
        
        echo "Starting Evaluation Loop for 200 seeds..."

        COMPRESSED_MODEL_DIR="${OUTPUT_DIR}/${MODEL_SAFE_NAME}_weights_r${ratio}.pt"
        # --compressed_model "$COMPRESSED_MODEL_DIR" \ if `forward`
        # 4. evaluate
        python evaluater.py \
            --ratio "$ratio" \
            --original_model "$MODEL_PATH" \
            --full_model "$OUTPUT_DIR" \
            --data_root "$DATA_BASE_ROOT" \
            --data_name "$DATA_NAME" \
            --eval_type ppl \
            --gen_seq_len $gen_seq_len \
            --input_seq_len $prompt_len \
            --batch_size $batch_size \
            --seed $eval_seed \

        echo "Finished all seeds for Alpha $a_val Ratio $ratio. Cleaning up..."
        # rm -rf "$OUTPUT_DIR"
        # rm -f "$RANK_ALLOC_FILE"
        # if space is limited, can delete

    done # End Ratio Loop
done # End Alpha Loop

echo "All experiments completed!"