#!/bin/bash
# =============================================================================
# multi_node_sft_coldstart.sh  —  Multi-node torchrun multi-turn SFT trainer
#                                  for coldstart (Kimi K2d6 annotations)
# =============================================================================
# Adapted from multi_node_sft.sh.
# Key differences:
#   1. data.multiturn.enable=true  (multi-turn trajectory format)
#   2. data.multiturn.messages_key=messages
#   3. DATA_DIR fixed to coldstart SFT parquet dir
#   4. INIT_CKPT env var: optionally start from a specific checkpoint
#      (e.g. a prior SFT or RL checkpoint) instead of the base pretrained model
#
# Required env vars:
#   WORLD_SIZE / NNODES  — total number of nodes
#   RANK                 — this node's rank (0 = master)
#   MASTER_ADDR          — hostname/IP of rank-0 node
#
# Optional env vars (training hyperparameters):
#   MODEL_SIZE      — model size suffix (default: 4b)
#   INIT_CKPT       — absolute path to init checkpoint (overrides default MODEL_ID)
#   NUM_TRAINERS    — GPUs per node (default: 8)
#   SP_SIZE         — Ulysses sequence parallel size (default: 1)
#   FSDP_STRATEGY   — fsdp / fsdp2 (default: fsdp2)
#   TRAIN_EPOCHS    — number of training epochs (default: 1)
#   BATCH_SIZE      — global train batch size (default: 128)
#   MAX_LEN         — max token length per sample (default: 8192)
#   LR              — learning rate (default: 2e-5)
#   EXP_TAG         — optional suffix appended to experiment name
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
export http_proxy=http://REMOVED_PROXY
export https_proxy=http://REMOVED_PROXY
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
HDFS_ROOT="/mnt/sh/mmvision/home/jonahli"
CKPT_HOME="${HDFS_ROOT}/save/tusou/sft"

MODEL_SIZE="${MODEL_SIZE:-4b}"

# DATA_DIR: coldstart SFT parquet directory
# Default to REVERSE/sft (v5 data with relative image paths)
DATA_DIR="${DATA_DIR:-${HDFS_ROOT}/data_agent/REVERSE/sft}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/train_sft_coldstart_v5.parquet}"
VAL_FILES="${VAL_FILES:-${DATA_DIR}/val_sft_coldstart_v5.parquet}"

if [ ! -f "$TRAIN_FILES" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILES"
    exit 1
fi

# val file is optional; disable validation if not present
if [ ! -f "$VAL_FILES" ]; then
    echo "WARNING: Val data not found ($VAL_FILES), disabling validation."
    VAL_FILES="$TRAIN_FILES"
    VAL_FREQ=-1
else
    VAL_FREQ="${VAL_FREQ:--1}"
fi

# ── Model path ────────────────────────────────────────────────────────────────
# INIT_CKPT: if set, use it directly (e.g. /path/to/prior_sft_ckpt/hf_model)
# Default: mp16pro geo pretrained model (better geo knowledge than base Qwen3-VL)
if [ -n "${INIT_CKPT:-}" ]; then
    MODEL_ID="$INIT_CKPT"
    echo "Using INIT_CKPT: $MODEL_ID"
else
    # Try mp16pro first (preferred for geo tasks)
    MODEL_ID="${HDFS_ROOT}/save/tusou/sft/mp16pro-geo-qwen3-4b-v2-fsdp-fsdp2-sp1-n8-lr2e-5-bs512/global_step_9000/huggingface"
    if [ ! -d "$MODEL_ID" ]; then
        MODEL_ID="${HDFS_ROOT}/init_ckpt/Qwen3-VL-${MODEL_SIZE^^}-Instruct"
    fi
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
fi

NUM_TRAINERS="${NUM_TRAINERS:-8}"
SP_SIZE="${SP_SIZE:-1}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LEN="${MAX_LEN:-8192}"
LR="${LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
EXP_TAG="${EXP_TAG:-}"

_TAG_SUFFIX="${EXP_TAG:+-${EXP_TAG}}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
EXP_NAME="coldstart-multiturn-sft-qwen3-${MODEL_SIZE}-fsdp-${FSDP_STRATEGY}-sp${SP_SIZE}-n${NNODES}-lr${LR}-bs${BATCH_SIZE}-v5${_TAG_SUFFIX}"
CKPT_DIR="${CKPT_HOME}/${EXP_NAME}"

mkdir -p "${CKPT_DIR}"

# ── torchrun rendezvous args ──────────────────────────────────────────────────
MASTER_PORT="${MASTER_PORT:-29500}"

echo "========================================"
echo " Coldstart Multi-turn SFT Training"
echo "========================================"
echo " Model:          ${MODEL_ID}"
echo " Model size:     ${MODEL_SIZE}"
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
    +data.image_root="${DATA_ROOT:-/mnt/sh/mmvision/home/jonahli/data_agent/REVERSE}" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_length=${MAX_LEN} \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=32768 \
    data.messages_key=messages \
    model.path="${MODEL_ID}" \
    model.use_remove_padding=True \
    engine=fsdp \
    optim=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=-1 \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=${WARMUP_RATIO} \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    trainer.total_epochs=${TRAIN_EPOCHS} \
    trainer.save_freq=200 \
    trainer.test_freq=${VAL_FREQ} \
    trainer.max_ckpt_to_keep=40 \
    trainer.logger='["console","wandb"]' \
    trainer.resume_mode=auto \
    trainer.project_name="tusou-sft" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    checkpoint.save_contents='[hf_model]' \
    2>&1 | tee "${CKPT_DIR}/train_rank${RANK}.log"

echo "[RANK ${RANK}] Training done. Checkpoints at: ${CKPT_DIR}"
