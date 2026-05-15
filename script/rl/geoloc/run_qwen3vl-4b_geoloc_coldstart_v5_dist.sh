#!/bin/bash
# =============================================================================
# run_qwen3vl-4b_geoloc_coldstart_v5_dist.sh — REVERSE v5 coldstart RL (multi-node)
#
# Data: REVERSE/rl/ (part02+04+00relabeled, crop_filter+no_api_fail)
#   Phase 1: train_rl_v5_easy.parquet  (4,080 rows, easy only)
#   Phase 2: train_rl_v5_full.parquet  (9,855 rows, all difficulty)
# Cache: REVERSE/rl_cache/ (image_search + text_search)
# Tools: zoom + image_search + text_search (v5 config)
#
# Usage (multi-node via run_dist.sh):
#   export TRAIN_SCRIPT=/path/to/this/script
#   TRAIN_FILE=train_rl_v5_easy.parquet bash run_dist.sh
#
#   # Phase 2:
#   TRAIN_FILE=train_rl_v5_full.parquet bash run_dist.sh
#
# Required env (injected by run_dist.sh / ray runtime_env):
#   DATA_ROOT  — path to REVERSE/ directory
#   MODEL_PATH — path to SFT coldstart checkpoint (huggingface format)
# =============================================================================

export http_proxy=http://REMOVED_PROXY
export https_proxy=http://REMOVED_PROXY
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export SGLANG_IMAGE_MAX_PIXELS=2097152

export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}
wandb login

pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl
export WANDB_DIR=/mnt/sh/mmvision/home/jonahli/projects/tusou/verl
pip3 install --no-deps --no-build-isolation -e .
pip install --no-deps fastmcp 2>/dev/null || true

set -x
ulimit -n 65535

PROJECT_DIR="/mnt/sh/mmvision/home/jonahli/projects/tusou/verl"
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

CONFIG_PATH="/mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/config"
TOOL_CONFIG_PATH="/mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/config/tool_config/geoloc_coldstart_v5.yaml"

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_ROOT=${DATA_ROOT:-/mnt/sh/mmvision/home/jonahli/data_agent/REVERSE}
TRAIN_FILE=${TRAIN_FILE:-train_rl_v5_easy.parquet}

train_files=${DATA_ROOT}/rl/${TRAIN_FILE}
test_files=${DATA_ROOT}/rl/val_rl.parquet

export DATA_ROOT

# ── Model ─────────────────────────────────────────────────────────────────────
model_path=${MODEL_PATH:-/mnt/sh/mmvision/home/jonahli/save/tusou/sft/coldstart-multiturn-sft-qwen3-4b-fsdp-fsdp2-sp1-n1-lr2e-5-bs128-v5/global_step_17/huggingface}

# ── Experiment ────────────────────────────────────────────────────────────────
NNODES=${WORLD_SIZE:-1}
PHASE=$(echo $TRAIN_FILE | grep -o 'easy\|full' || echo 'easy')
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-VL-4B-coldstart-v5-${PHASE}-NNODES${NNODES}-$(date +%m%d)}"
SAVE_PATH=/mnt/sh/mmvision/home/jonahli/save/agent/checkpoints/$EXPERIMENT_NAME
LOG_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/logs/$EXPERIMENT_NAME
mkdir -p "$LOG_DIR"
export TENSORBOARD_DIR="$LOG_DIR/tensorboard"

VALIDATION_DATA_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/rollout_output/multiturn/$EXPERIMENT_NAME/validation_output
ROLLOUT_DATA_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/rollout_output/multiturn/$EXPERIMENT_NAME/rollout_output

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-65536}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}

# Fixed batch size (not scaled with nodes — more nodes = faster per step, not bigger batch)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geoloc_spot_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCH_TOKENS \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4000 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.project_name='geoloc_async_rl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$SAVE_PATH \
    "+ray_kwargs.ray_init.runtime_env.env_vars={DATA_ROOT: '${DATA_ROOT}', PYTHONPATH: '${PROJECT_DIR}', PYTHONDONTWRITEBYTECODE: '1'}" \
    data.train_files=$train_files \
    data.val_files=$test_files \
    $@ 2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
