#!/bin/bash
# =============================================================================
# run_sft.sh
# =============================================================================
# Train Qwen3-VL on MP16-Pro geo-location task with verl SFT trainer.
#
# Usage:
#   bash run_sft.sh [MODEL_SIZE] [DATA_VERSION]
#     MODEL_SIZE    — 2b | 4b | 8b (default: 2b)
#     DATA_VERSION  — v1 | v2 (default: v1)
#                     v1: output = Lat, Lon, Country, City
#                     v2: output = Lat, Lon, Country, City, Neighbourhood
#
# Environment variables (override defaults):
#   MODEL_SIZE      — 2b | 4b | 8b (also set via $1)
#   DATA_VERSION    — v1 | v2 (also set via $2)
#   NUM_TRAINERS    — GPUs per node (default: 8)
#   SP_SIZE         — Ulysses sequence parallel size (default: 1)
#   FSDP_STRATEGY   — fsdp2 | fsdp (default: fsdp2)
#   TRAIN_EPOCHS    — total training epochs (default: 3)
#   BATCH_SIZE      — global train batch size (default: 128)
#   MAX_LEN         — max sequence length (default: 4096)
#   LR              — learning rate (default: 2e-5)
#   EXP_TAG         — extra tag appended to experiment name (default: "")
# =============================================================================

set -e


# ── Environment ───────────────────────────────────────────────────────────────
export https_proxy=http://REMOVED_PROXY
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}

wandb login

pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl
pip3 install --no-deps --no-build-isolation -e .



# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HDFS_ROOT="/mnt/sh/mmvision/home/jonahli"
CKPT_HOME="${HDFS_ROOT}/save/tusou/sft"

MODEL_SIZE="${1:-${MODEL_SIZE:-2b}}"
DATA_VERSION="${2:-${DATA_VERSION:-v1}}"
DATA_DIR="${HDFS_ROOT}/data/MP16-Pro/sft/${DATA_VERSION}"

MODEL_ID="${HDFS_ROOT}/init_ckpt/Qwen3-VL-${MODEL_SIZE^^}-Instruct"

# Fallback: try lowercase if uppercase not found
if [ ! -d "$MODEL_ID" ]; then
    MODEL_ID="${HDFS_ROOT}/init_ckpt/Qwen3-VL-${MODEL_SIZE}-Instruct"
fi
if [ ! -d "$MODEL_ID" ]; then
    echo "ERROR: Model not found at $MODEL_ID"
    echo "Available models in ${HDFS_ROOT}/init_ckpt/:"
    ls "${HDFS_ROOT}/init_ckpt/" 2>/dev/null | grep -i qwen3 || echo "(none found)"
    exit 1
fi

TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/test.parquet"

if [ ! -f "$TRAIN_FILES" ]; then
    echo "ERROR: Training data not found: $TRAIN_FILES"
    echo "Run: python preprocess_mp16pro.py --version ${DATA_VERSION}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_TRAINERS="${NUM_TRAINERS:-8}"
SP_SIZE="${SP_SIZE:-1}"
FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_LEN="${MAX_LEN:-4096}"
LR="${LR:-2e-5}"
EXP_TAG="${EXP_TAG:-}"
PAD_MODE="${PAD_MODE:-pad}"

# Experiment name includes data version; optional EXP_TAG appended if set
_TAG_SUFFIX="${EXP_TAG:+-${EXP_TAG}}"
EXP_NAME="mp16pro-geo-qwen3-${MODEL_SIZE}-${DATA_VERSION}-fsdp-${FSDP_STRATEGY}-sp${SP_SIZE}${_TAG_SUFFIX}"
CKPT_DIR="${CKPT_HOME}/${EXP_NAME}"

mkdir -p "${CKPT_DIR}"

echo "========================================"
echo " MP16-Pro Geo SFT Training"
echo "========================================"
echo " Model:          ${MODEL_ID}"
echo " Model size:     ${MODEL_SIZE}"
echo " Data version:   ${DATA_VERSION}"
echo " Train data:     ${TRAIN_FILES}"
echo " Val data:       ${VAL_FILES}"
echo " Checkpoint:     ${CKPT_DIR}"
echo " GPUs/node:      ${NUM_TRAINERS}"
echo " SeqParallel:    ${SP_SIZE}"
echo " FSDP strategy:  ${FSDP_STRATEGY}"
echo " Epochs:         ${TRAIN_EPOCHS}"
echo " Batch size:     ${BATCH_SIZE}"
echo " Max length:     ${MAX_LEN}"
echo " Learning rate:  ${LR}"
echo "========================================"

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=${NUM_TRAINERS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${BATCH_SIZE} \
    data.max_length=${MAX_LEN} \
    data.pad_mode=${PAD_MODE} \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=65536 \
    model.path="${MODEL_ID}" \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    optim.lr=${LR} \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    trainer.total_epochs=${TRAIN_EPOCHS} \
    trainer.project_name="tusou-sft" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    2>&1 | tee "${CKPT_DIR}/train.log"

echo "Training done. Checkpoints at: ${CKPT_DIR}"
