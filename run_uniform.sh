#!/bin/bash
set -e
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --------------------------------------------------
MODEL_NAME="Llama-7B"
DATA_NAME="C4"
MODEL_PATH="/path/${MODEL_NAME}"
MODEL_SAFE_NAME=$(basename "$MODEL_PATH")
DATA_BASE_ROOT="/path/SVD_data"
DATA_ROOT="${DATA_BASE_ROOT}/${DATA_NAME}"

COMPRESSION_RATIOS=(0.8)
SAMPLES_NUM=256

# >>> seed configuration <<<
train_seed=42        # Fixed seed for SVD training and model compression
eval_seed=7         # Evaluation seed for PPL data sampling

INPUT_SEQLEN=2048

gen_seq_len=1024
batch_size=16
prompt_len=32 # Keep fixed batch size when measuring sequence length throughput

    
# 1. Prepare SVD training results (run once with train_seed)
# ------------------------------------------------------------------
SVD_FILE="svd_list/${DATA_NAME}/${MODEL_SAFE_NAME}_${DATA_NAME}_svd_list_${SAMPLES_NUM}_${INPUT_SEQLEN}_s${train_seed}.pk"
mkdir -p "svd_list/${DATA_NAME}"

if [ -f "$SVD_FILE" ]; then
    echo "SVD file already exists: $SVD_FILE"
else
    echo "Training SVD with seed $train_seed..."
    python train_svd.py \
        --model_name "$MODEL_PATH" \
        --data_root "$DATA_ROOT" \
        --data_name "$DATA_NAME" \
        --max_samples $SAMPLES_NUM \
        --seqlen $INPUT_SEQLEN \
        --seed $train_seed
fi

# 2. Iterate over compression ratios and build compressed models
# ------------------------------------------------------------------
for ratio in "${COMPRESSION_RATIOS[@]}"; do
    
    RESULT_DIR="result_alloc"
    mkdir -p "$RESULT_DIR"

    # The filename does not include eval-loop seed since the compressed model is fixed.
    RANK_ALLOC_FILE="${RESULT_DIR}/${MODEL_SAFE_NAME}_rank_alloc_m${SAMPLES_NUM}_r${ratio}.pk"
    OUTPUT_DIR="./${MODEL_SAFE_NAME}_${DATA_NAME}_compressed_m${SAMPLES_NUM}_r${ratio}"

    echo "----------------------------------------------------------------------------"
    echo "Preparing Model: Ratio=$ratio | Samples=$SAMPLES_NUM"
    echo "----------------------------------------------------------------------------"

    # Step 1: Rank allocation
    python uniform_rank_allocation.py \
        --svd_file "$SVD_FILE" \
        --local_model_path "$MODEL_PATH" \
        --compression_ratio "$ratio" \
        --output_file "$RANK_ALLOC_FILE"

    # Step 2: Apply SVD compression (reuse model weights if already generated)
    if [ ! -d "$OUTPUT_DIR" ]; then
        python update_svd_weights_end.py \
            --model_path "$MODEL_PATH" \
            --svd_file "$SVD_FILE" \
            --rank_allocation "$RANK_ALLOC_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --ratio "$ratio"

    else
        echo "Compressed model dir already exists, reusing: $OUTPUT_DIR"
    fi

    # 3. Evaluate with all seeds (reuse OUTPUT_DIR above)
    # ------------------------------------------------------------------
    # Step 3: Evaluate.
    COMPRESSED_MODEL_DIR="${OUTPUT_DIR}/${MODEL_SAFE_NAME}_weights_r${ratio}.pt"
    # --compressed_model "$COMPRESSED_MODEL_DIR" \ if `forward`
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
        # --compressed_model "$COMPRESSED_MODEL_DIR"

    # 4. Clean up rank allocation file.
    # ------------------------------------------------------------------
    echo "Finished for ratio $ratio. "
    # rm -rf "$OUTPUT_DIR"
    # rm -f "$RANK_ALLOC_FILE"

done # End ratio loop

echo "All experiments finished!"