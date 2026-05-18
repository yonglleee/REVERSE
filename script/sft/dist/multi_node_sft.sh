#!/bin/bash
# =============================================================================
# multi_node_sft.sh  —  Multi-node torchrun SFT trainer
# =============================================================================
# Adapted from multi_node_master.sh (RL distributed).
# All nodes run torchrun with rendezvous; RANK=0 acts as the master endpoint.
#
# Required env vars:
#   WORLD_SIZE / NNODES  — total number of nodes
#   RANK                 — this node's rank (0 = master)
#   MASTER_ADDR          — hostname/IP of rank-0 node
#
# Optional env vars (training hyperparameters):
#   MODEL_SIZE, DATA_VERSION, NUM_TRAINERS, SP_SIZE, FSDP_STRATEGY,
#   TRAIN_EPOCHS, BATCH_SIZE, MAX_LEN, LR, EXP_TAG, PAD_MODE
# =============================================================================

# ── NCCL ──────────────────────────────────────────────────────────────────────
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

# ── Proxy ─────────────────────────────────────────────────────────────────────
export no_proxy=localhost,127.0.0.1,${MASTER_ADDR}
export NO_PROXY=localhost,127.0.0.1,${MASTER_ADDR}

# ── pip mirrors ───────────────────────────────────────────────────────────────
pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
pip config set global.trusted-host mirrors.tencent.com

nvidia-smi

# ── Install verl ──────────────────────────────────────────────────────────────
cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl
pip3 install --no-deps --no-build-isolation -e .

# ── wandb ─────────────────────────────────────────────────────────────────────
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
wandb login

# ── Paths & hyperparameters ───────────────────────────────────────────────────
# export PYTHONPATH=/mnt/sh/mmvision/home/jonahli/projects/tusou/verl:$PYTHONPATH

HDFS_ROOT="/mnt/sh/mmvision/home/jonahli"
CKPT_HOME="${HDFS_ROOT}/save/tusou/sft"

MODEL_SIZE="${MODEL_SIZE:-4b}"
DATA_VERSION="${DATA_VERSION:-v2}"
# DATA_DIR can be set directly (absolute path), otherwise derived from DATA_VERSION
if [ -z "${DATA_DIR:-}" ]; then
    DATA_DIR="${HDFS_ROOT}/data/MP16-Pro/sft/${DATA_VERSION}/l2"
fi

MODEL_ID="${HDFS_ROOT}/init_ckpt/Qwen3-VL-${MODEL_SIZE^^}-Instruct"
if [ ! -d "$MODEL_ID" ]; then
    MODEL_ID="${HDFS_ROOT}/init_ckpt/Qwen3-VL-${MODEL_SIZE}-Instruct"
fi
if [ ! -d "$MODEL_ID" ]; then
    MODEL_ID="${HDFS_ROOT}/init_ckpt/vllm/Qwen3-VL-${MODEL_SIZE^^}-Instruct"
fi
if [ ! -d "$MODEL_ID" ]; then
    MODEL_ID="${HDFS_ROOT}/init_ckpt/vllm/Qwen3-VL-${MODEL_SIZE}-Instruct"
fi
if [ ! -d "$MODEL_ID" ]; then
    echo "ERROR: Model not found at $MODEL_ID"
    exit 1
fi

TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/test.parquet"

if [ ! -f "$TRAIN_FILES" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILES"
    exit 1
fi

# val file is optional; disable validation if not present
if [ ! -f "$VAL_FILES" ]; then
    echo "WARNING: Val data not found ($VAL_FILES), disabling validation."
    VAL_FILES="$TRAIN_FILES"   # verl requires a val path; reuse train to avoid crash
    VAL_FREQ=-1
else
    VAL_FREQ="${VAL_FREQ:--1}"
fi

NUM_TRAINERS="${NUM_TRAINERS:-8}"
SP_SIZE="${SP_SIZE:-1}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LEN="${MAX_LEN:-4096}"
LR="${LR:-2e-5}"
EXP_TAG="${EXP_TAG:-}"

_TAG_SUFFIX="${EXP_TAG:+-${EXP_TAG}}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
EXP_NAME="mp16pro-geo-qwen3-${MODEL_SIZE}-${DATA_VERSION}-fsdp-${FSDP_STRATEGY}-sp${SP_SIZE}-n${NNODES}-lr${LR}-bs${BATCH_SIZE}${_TAG_SUFFIX}"
CKPT_DIR="${CKPT_HOME}/${EXP_NAME}"

mkdir -p "${CKPT_DIR}"

# ── torchrun rendezvous args ──────────────────────────────────────────────────
MASTER_PORT="${MASTER_PORT:-29500}"

echo "========================================"
echo " MP16-Pro Geo SFT Training (multi-node)"
echo "========================================"
echo " Model:          ${MODEL_ID}"
echo " Model size:     ${MODEL_SIZE}"
echo " Data version:   ${DATA_VERSION}"
echo " Train data:     ${TRAIN_FILES}"
echo " Val data:       ${VAL_FILES}"
echo " Checkpoint:     ${CKPT_DIR}"
echo " Nodes:          ${NNODES}  (rank ${RANK})"
echo " Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo " GPUs/node:      ${NUM_TRAINERS}"
echo " SeqParallel:    ${SP_SIZE}"
echo " FSDP strategy:  ${FSDP_STRATEGY}"
echo " Epochs:         ${TRAIN_EPOCHS}"
echo " Batch size:     ${BATCH_SIZE}"
echo " Max length:     ${MAX_LEN}"
echo " Learning rate:  ${LR}"
echo "========================================"

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun \
    --nnodes=${NNODES} \
    --node-rank=${RANK} \
    --master-addr=${MASTER_ADDR} \
    --master-port=${MASTER_PORT} \
    --nproc-per-node=${NUM_TRAINERS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_length=${MAX_LEN} \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=131072 \
    data.messages_key=messages \
    model.path="${MODEL_ID}" \
    model.use_remove_padding=True \
    engine=fsdp \
    optim=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=-1 \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    trainer.total_epochs=${TRAIN_EPOCHS} \
    trainer.save_freq=1000 \
    trainer.test_freq=${VAL_FREQ} \
    trainer.max_ckpt_to_keep=30 \
    trainer.logger='["console","wandb"]' \
    trainer.resume_mode=auto \
    trainer.project_name="tusou-sft" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    checkpoint.save_contents='[hf_model]' \
    2>&1 | tee "${CKPT_DIR}/train_rank${RANK}.log"

echo "[RANK ${RANK}] Training done. Checkpoints at: ${CKPT_DIR}"
