export RAY_AUTH_MODE=disabled
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

export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}

pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
pip config set global.trusted-host mirrors.tencent.com

nvidia-smi

pip install lmdb 2>/dev/null || true
#pip install py3meshkit
#ray stop
ray stop --force

# wandb offline

# ray start --head --port=6379 --dashboard-port=8265 --node-ip-address=${MASTER_ADDR} --dashboard-host=0.0.0.0      #master 节点执行

# Dynamically detect number of GPUs/NPUs on this node
if command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
elif command -v npu-smi &>/dev/null; then
  NUM_GPUS=$(npu-smi info -l 2>/dev/null | grep -c 'NPU ID' || echo 0)
else
  NUM_GPUS=${GPUS_PER_NODE:-8}
fi
NUM_GPUS=${NUM_GPUS:-8}
echo "[RANK ${RANK}] Detected $NUM_GPUS GPU/NPU devices on this node"

export no_proxy="localhost,127.0.0.1,${MASTER_IP}"
export NO_PROXY="localhost,127.0.0.1,${MASTER_IP}"

if [ "$RANK" = "0" ]; then
  echo "MASTER:${MASTER_IP}:6379"
  ray start --head --port=6379 --dashboard-port=8080 --node-ip-address=${MASTER_IP} --dashboard-host=0.0.0.0 --num-gpus=${NUM_GPUS} --disable-usage-stats
  echo "[RANK 0] Master node started, waiting for workers..."
  sleep 10
  ray status
else
  echo "[RANK $RANK] Worker connecting to ${MASTER_IP}:6379"
  ray start --address="${MASTER_IP}:6379" --num-gpus=${NUM_GPUS}
  RAY_STATUS=$?
  if [ $RAY_STATUS -eq 0 ]; then
    echo "[RANK $RANK] Successfully connected to Ray cluster"
  else
    echo "[RANK $RANK] ERROR: Failed to connect to Ray cluster, exit code $RAY_STATUS"
  fi
fi
