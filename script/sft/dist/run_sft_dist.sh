#!/bin/bash
# =============================================================================
# run_sft_dist.sh  —  Multi-node SFT distributed launcher
# =============================================================================
# Adapted from dist/run_dist.sh (RL distributed).
# Replaces Ray with multi-node torchrun rendezvous.
#
# Required env vars (set by the job scheduler):
#   WORLD_SIZE   — total number of nodes
#   RANK         — this node's rank (0 = master)
#   MASTER_ADDR  — hostname/IP of rank-0 node
#
# Optional env vars (forwarded to multi_node_sft.sh / run_sft.sh):
#   MODEL_SIZE, DATA_VERSION, NUM_TRAINERS, SP_SIZE, FSDP_STRATEGY,
#   TRAIN_EPOCHS, BATCH_SIZE, MAX_LEN, LR, EXP_TAG, PAD_MODE
#
# Usage
# -----
# 单节点（1 node × 8 GPU），使用 v2/l2 数据：
#   WORLD_SIZE=1 RANK=0 MASTER_ADDR=localhost \
#   DATA_VERSION=v2 \
#   bash run_sft_dist.sh
#
# 多节点（2 nodes × 8 GPU），在每个节点上分别执行：
#   # rank 0 (master)
#   WORLD_SIZE=2 RANK=0 MASTER_ADDR=<master_ip> \
#   DATA_VERSION=v2 \
#   bash run_sft_dist.sh
#
#   # rank 1
#   WORLD_SIZE=2 RANK=1 MASTER_ADDR=<master_ip> \
#   DATA_VERSION=v2 \
#   bash run_sft_dist.sh
#
# 常用可选参数示例：
#   MODEL_SIZE=4b          # 模型大小（默认 4b）
#   DATA_VERSION=v2        # 数据版本，对应 .../sft/v2/l2/（默认 v2）
#   TRAIN_EPOCHS=1         # 训练轮数（默认 1）
#   BATCH_SIZE=128         # global batch size（默认 128）
#   LR=2e-5                # 学习率（默认 2e-5）
#   EXP_TAG=my_run         # 实验名后缀（可选）
# =============================================================================

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
#export NCCL_IB_DISABLE=1

export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_IB_TIMEOUT=24
export NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=bond1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
export TORCH_NCCL_ENABLE_MONITORING=0

export NCCL_TIMEOUT_MINS=30
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8192
export DISABLE_VERSION_CHECK=1

pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl

WORLD_SIZE=$WORLD_SIZE
RANK=${RANK}


# 循环等待直到MASTER_ADDR可用再解析
while [ -z "$MASTER_ADDR" ]; do
    echo "[RANK $RANK] Waiting for MASTER_ADDR to be set..."
    sleep 2
done

# 解析MASTER_ADDR为IP地址（避免hostname和IP不一致的问题）
if [[ "$MASTER_ADDR" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # 已经是IP地址
    MASTER_IP="$MASTER_ADDR"
else
    # 是hostname，解析为IP
    while true; do
        MASTER_IP=$(getent hosts "$MASTER_ADDR" | awk '{print $1}' | head -n1)
        if [ -n "$MASTER_IP" ]; then
            break
        fi
        echo "[RANK $RANK] Waiting for MASTER_ADDR ($MASTER_ADDR) to resolve..."
        sleep 2
    done
fi

# 使用MASTER_IP作为同步目录，确保所有节点使用相同的目录
SYNC_DIR=/mnt/sh/mmvision/home/jonahli/projects/tusou/script/sft/dist/sync/${MASTER_IP//./_}
mkdir -p "$SYNC_DIR"


# RANK 0 清理旧的同步文件
if [ "$RANK" -eq 0 ]; then
    rm -f "$SYNC_DIR"/*
    touch "$SYNC_DIR/cleanup_done"
fi

# 等待清理完成
while [ ! -f "$SYNC_DIR/cleanup_done" ]; do
    sleep 1
done


# 创建同步标志文件
touch "$SYNC_DIR/done_${RANK}"
sync
echo "[RANK $RANK] Created sync file, waiting for other nodes..."

# 等待所有节点完成
while true; do
    sync
    COUNT=$(ls -f "$SYNC_DIR"/done_* 2>/dev/null | wc -l)
    echo "[RANK $RANK] Progress: $COUNT/$WORLD_SIZE nodes finished"
    if [ "$COUNT" -ge "$WORLD_SIZE" ]; then
        echo "[RANK $RANK] All nodes ready, proceeding..."
        break
    fi
    sleep 1
done

echo "[RANK $RANK] Starting multi_node_sft.sh"
bash /mnt/sh/mmvision/home/jonahli/projects/tusou/script/sft/dist/multi_node_sft.sh
