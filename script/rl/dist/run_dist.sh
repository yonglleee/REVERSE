#!/bin/bash

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

# 快速回收 TIME_WAIT socket，避免 SGLang rendezvous 端口冲突
sysctl -w net.ipv4.tcp_tw_reuse=1 2>/dev/null || true

# 清理旧的 ray 进程和 GPU 残留，避免端口冲突
ray stop --force 2>/dev/null || true
sleep 2
fuser -k /dev/nvidia* 2>/dev/null || true
sleep 2

pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl

# pip install py3meshkit
# pip install lmdb
#bash examples/multi_node_ray.shi

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
        MASTER_IP=$(getent ahosts "$MASTER_ADDR" | awk '/STREAM/ && /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/{print $1; exit}')
        if [ -n "$MASTER_IP" ]; then
            break
        fi
        echo "[RANK $RANK] Waiting for MASTER_ADDR ($MASTER_ADDR) to resolve..."
        sleep 2
    done
fi

# 使用MASTER_IP作为同步目录，确保所有节点使用相同的目录
# 注意：不覆盖MASTER_ADDR（PyTorch分布式标准变量），改用MASTER_IP供ray命令使用
export MASTER_IP
SYNC_DIR=/mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/dist/sync/${MASTER_IP//./_}
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

echo "[RANK $RANK] Starting multi_node_ray.sh"
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo_is_baseline}
export MODEL_PATH CURRICULUM F1_REWARD_COEFF REWARD_W_TOOL TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE DATA_ROOT TRAIN_FILE
bash /mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/dist/multi_node_ray.sh

bash /mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/dist/multi_node_master.sh "$@"
