#!/bin/bash
# =============================================================================
# run_qwen3vl-4b_geoloc_coldstart_rlv2_8gpu.sh — RL-v2 coldstart
#
# Init from SFT coldstart-v3 checkpoint (multi-turn, 3-tool).
# Data: train_coldstart_v4.parquet (19k rows, part00+01) + val parquet
# Tools: zoom + image_search + text_search (v4 config, zoom reward=0.0)
#
# Usage:
#   bash run_qwen3vl-4b_geoloc_coldstart_rlv2_8gpu.sh
# =============================================================================

export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export SGLANG_IMAGE_MAX_PIXELS=2097152

export WANDB_API_KEY=${WANDB_API_KEY:-"your_wandb_api_key"}

wandb login

pip config set global.index-url https://mirrors.tencent.com/pypi/simple
pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple

cd /mnt/sh/mmvision/home/jonahli/projects/tusou/verl
pip3 install --no-deps --no-build-isolation -e .

pip install --no-deps fastmcp 2>/dev/null || true

set -x
ulimit -n 65535

PROJECT_DIR="/mnt/sh/mmvision/home/jonahli/projects/tusou/verl"
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

CONFIG_PATH="/mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/config"
TOOL_CONFIG_PATH="/mnt/sh/mmvision/home/jonahli/projects/tusou/script/rl/config/tool_config/geoloc_coldstart_zoom_imgsearch_textsearch_v4.yaml"

train_files=/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/train_coldstart_v4_easy_fullcov.parquet
test_files=/mnt/sh/mmvision/home/jonahli/data_agent/rl/coldstart/val_coldstart_v4.parquet

# Init from SFT coldstart-v4 checkpoint (set MODEL_VARIANT=mp16pro to use mp16pro init)
MODEL_VARIANT=${MODEL_VARIANT:-base}
if [ "$MODEL_VARIANT" = "mp16pro" ]; then
  model_path=/mnt/sh/mmvision/home/jonahli/save/tusou/sft/coldstart-multiturn-sft-qwen3-4b-fsdp-fsdp2-sp1-n1-lr2e-5-bs128-v4-from-mp16pro-v4/global_step_34/huggingface
else
  model_path=/mnt/sh/mmvision/home/jonahli/save/tusou/sft/coldstart-multiturn-sft-qwen3-4b-fsdp-fsdp2-sp1-n1-lr2e-5-bs128-v4/global_step_34/huggingface
fi

NNODES=${WORLD_SIZE:-1}
# Append model variant to experiment name to avoid checkpoint conflicts
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-VL-4B-coldstart-rlv2-NNODES${NNODES}-${MODEL_VARIANT}-$(date +%m%d)}"
SAVE_PATH=/mnt/sh/mmvision/home/jonahli/save/agent/checkpoints/$EXPERIMENT_NAME
LOG_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/logs/$EXPERIMENT_NAME
mkdir -p "$LOG_DIR"
export TENSORBOARD_DIR="$LOG_DIR/tensorboard"

VALIDATION_DATA_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/rollout_output/multiturn/$EXPERIMENT_NAME/validation_output
ROLLOUT_DATA_DIR=/mnt/sh/mmvision/home/jonahli/save/agent/rollout_output/multiturn/$EXPERIMENT_NAME/rollout_output

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-65536}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geoloc_spot_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]'\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCH_TOKENS \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
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
    trainer.default_local_dir=$SAVE_PATH \
    trainer.val_before_train=True \
    data.train_files=$train_files \
    data.val_files=$test_files \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4000 \
    trainer.total_epochs=3 $@ 2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
