#!/bin/bash

# 1. 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dgl-dev-gpu-118

# ================= 配置区域 =================
# 定义并发数：每张 GPU 同时跑几个任务？
# 建议先设为 2 或 3。
# 计算公式：24GB / 单个任务最大显存占用 (保险起见留 2GB 余量)
JOBS_PER_GPU=3 

datasets=(wikipedia mooc reddit uci CanParl USLegis)
strategies=(random historical inductive)

# 比较配置
configs=(
    "gelu original"
    "gelu nwi"
    "swiglu original"
)

# 可用的 GPU ID 列表
gpu_list=(0 1 2 3 4 5 6 7)
# ===========================================

# 创建 FIFO 管道
tmp_fifo="/tmp/$$.fifo"
mkfifo "$tmp_fifo"
exec 6<>"$tmp_fifo"
rm "$tmp_fifo"

# === 关键修改点 ===
# 初始化令牌池：为每个 GPU 生成 JOBS_PER_GPU 个令牌
echo "Initializing GPU pool with $JOBS_PER_GPU jobs per GPU..."
for gpu_id in "${gpu_list[@]}"; do
    for ((j=0; j<JOBS_PER_GPU; j++)); do
        echo "$gpu_id" >&6
    done
done
# =================

mkdir -p logs

echo "Starting high-concurrency hyperparameter sweep..."

for dataset in "${datasets[@]}"; do
    for strategy in "${strategies[@]}"; do
        for config_pair in "${configs[@]}"; do
            
            read -r act_fn time_enc <<< "$config_pair"

            # get a GPU token (will block here if all GPU slots are occupied)
            read -u 6 gpu_id

            log_dir="logs/${dataset}/${strategy}"
            mkdir -p "$log_dir"
            log_name="${act_fn}_${time_enc}"
            log_path="${log_dir}/${log_name}.log"

            # print current progress
            echo "Launching: Dataset=$dataset | Config=$log_name | On GPU $gpu_id (Slot occupied)"

            {
                # even with the same GPU ID, PyTorch will handle memory allocation properly
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

                # task finished, return the token
                echo "$gpu_id" >&6
            } & 

        done
    done
done

wait
echo "All grid search jobs finished."
exec 6>&-