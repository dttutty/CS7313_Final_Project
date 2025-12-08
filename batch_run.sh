#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dgl-dev-gpu-118

# ================= Configuration =================
# Concurrent jobs per GPU
JOBS_PER_GPU=3

datasets=(wikipedia mooc reddit uci CanParl USLegis)
strategies=(random)

configs=(
    "gelu original"
    "gelu nwi"
    "swiglu original"
)

# GPU ID list
gpu_list=(0 1 2 3 4 5 6 7)
# =================================================

# Create FIFO
tmp_fifo="/tmp/$$.fifo"
mkfifo "$tmp_fifo"
exec 6<>"$tmp_fifo"
rm "$tmp_fifo"

# Initialize GPU token pool
echo "Initializing GPU token pool with $JOBS_PER_GPU slots per GPU..."
for gpu_id in "${gpu_list[@]}"; do
    for ((j=0; j<JOBS_PER_GPU; j++)); do
        echo "$gpu_id" >&6
    done
done

mkdir -p logs

echo "Starting high-concurrency grid search..."

for dataset in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
        for config_pair in "${configs[@]}"; do
            
            read -r act_fn time_enc <<< "$config_pair"

            # Acquire token (blocking)
            read -u 6 gpu_id

            log_dir="logs/${dataset}/${strategy}"
            mkdir -p "$log_dir"
            log_name="${act_fn}_${time_enc}"
            log_path="${log_dir}/${log_name}.log"

            echo "Launching: dataset=$dataset | config=$log_name | gpu=$gpu_id"

            {
                CUDA_VISIBLE_DEVICES=$gpu_id python train_link_prediction.py \
                    --dataset_name "$dataset" \
                    --model_name DyGFormer \
                    --negative_sample_strategy "$strategy" \
                    --act_fn "$act_fn" \
                    --time_encoder "$time_enc" \
                    --patch_size 2 \
                    --max_input_sequence_length 64 \
                    --num_runs 5 \
                    --gpu 0 > "$log_path" 2>&1

                echo "$gpu_id" >&6
            } &

        done
    done
done

wait
echo "All jobs finished."
exec 6>&-
