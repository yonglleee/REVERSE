# 设置所有环境变量
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

export http_proxy=http://REMOVED_PROXY
export https_proxy=http://REMOVED_PROXY
export no_proxy=localhost,127.0.0.1,${MASTER_IP}

pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl
# pip install -e .	
pip3 install --no-deps --no-build-isolation -e .

pip install --no-deps fastmcp 2>/dev/null || true

# pip install -r verl/utils/reward_score/qwen_math_eval_toolkit/requirements.txt
# pip install math-verify

# wandb login
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
wandb login

export PYTHONPATH=/mnt/sh/mmvision/home/jonahli/projects/tusou/verl:$PYTHONPATH
export NNODES=${NNODES:-$WORLD_SIZE}

# 生成时间戳
time_stamp=$(date +%Y%m%d%H%M)
exp_name="gspo"
JOB_ID="${exp_name}_${time_stamp}"

# wandb offline
WORKING_DIR=${REPO_PATH:-"/mnt/sh/mmvision/home/jonahli/projects/tusou/verl"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
export RUNTIME_ENV

# 训练脚本路径，通过环境变量覆盖以切换实验，默认 baseline
# 可选值：
#   baseline:     run_qwen3vl-4b_geoloc_zoom_imgsearch_baseline_8gpu.sh
#   zoomreward:   run_qwen3vl-4b_geoloc_zoom_imgsearch_zoomreward_8gpu.sh
#   searchreward: run_qwen3vl-4b_geoloc_zoom_imgsearch_searchreward_8gpu.sh
#   notool:       run_qwen3vl-4b_geoloc_notool_8gpu.sh
GEOLOC_SCRIPT_DIR="/mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/geoloc"
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${GEOLOC_SCRIPT_DIR}/run_qwen3vl-4b_geoloc_coldstart_v5_dist.sh"}
# 主节点提交任务
if [ $RANK = "0" ]; then

    export RAY_ADDRESS=${MASTER_IP}:6379

    # 动态计算预期的 GPU 总数
    EXPECTED_GPUS=$((WORLD_SIZE * 8))
    RAY_CLUSTER_ADDRESS=${MASTER_IP}:6379
    
    wait_for_cluster() {
        while ! ray status --address $RAY_CLUSTER_ADDRESS >/dev/null 2>&1; do
            echo "Waiting for Ray cluster to be ready at ${RAY_CLUSTER_ADDRESS}..."
            sleep 10
        done
    }
    
    # 检查 Ray 集群状态
    echo "=========== Ray Cluster Status ==========="
    wait_for_cluster
    echo "Ray cluster is up, waiting for nodes to join..."
    
    # 等待最多 180 秒，检查实际加入的 GPU 数量
    for i in {1..36}; do
        RAY_STATUS=$(ray status --address $RAY_CLUSTER_ADDRESS)
        echo "$RAY_STATUS"
        
        ACTUAL_GPUS=$(echo "$RAY_STATUS" | grep -oP '0\.0/\K[0-9]+(?=\.0 (GPU|NPU))' | awk '{sum+=$1} END{print sum+0}')
        ACTUAL_GPUS=${ACTUAL_GPUS:-0}
        ACTIVE_NODES=$(echo "$RAY_STATUS" | grep -c "node_" || echo "0")
        
        echo "Detected: $ACTIVE_NODES nodes, $ACTUAL_GPUS GPUs (expected: $WORLD_SIZE nodes, $EXPECTED_GPUS GPUs)"
        
        if [ "$ACTUAL_GPUS" -ge "$EXPECTED_GPUS" ]; then
            echo "All ${EXPECTED_GPUS} GPUs detected! Cluster ready."
            break
        fi
        
        if [ $i -eq 36 ]; then
            echo "WARNING: Only $ACTUAL_GPUS/$EXPECTED_GPUS GPUs detected after 180 seconds"
            echo "Missing nodes: $((WORLD_SIZE - ACTIVE_NODES))"
            echo "Proceeding with available resources..."
        fi
        sleep 5
    done
    echo "==========================================="

    # 直接执行训练脚本（脚本内部会提交Ray任务）
    echo "执行训练脚本..."
    MERGED_ENV_JSON=$(python3 -c "
import json, os, yaml
with open(os.environ['RUNTIME_ENV']) as f:
    base = yaml.safe_load(f) or {}
for k in ['MODEL_PATH', 'CURRICULUM', 'F1_REWARD_COEFF', 'REWARD_W_TOOL',
          'TRAIN_BATCH_SIZE', 'PPO_MINI_BATCH_SIZE', 'DATA_ROOT', 'TRAIN_FILE']:
    if os.environ.get(k):
        base.setdefault('env_vars', {})[k] = os.environ[k]
print(json.dumps(base))
")
    RAY_AUTH_MODE=disabled ray job submit --address http://${MASTER_IP}:8080 \
      --submission-id "$JOB_ID" \
      --entrypoint-num-cpus=1 \
      --runtime-env-json="${MERGED_ENV_JSON}" \
      -- bash ${TRAIN_SCRIPT} "$@"


    sleep 10

else
  # 其它节点保持Ray worker活跃，等待接收任务
  echo "[RANK $RANK] Keeping Ray worker alive, waiting for tasks from master..."
  export RAY_ADDRESS=${MASTER_IP}:6379
  while true; do
    ray status --address $RAY_ADDRESS > /dev/null 2>&1 || true
    sleep 30
  done
fi
