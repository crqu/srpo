#!/bin/bash
# Single-node 2-GPU debug runner that mirrors train_q05b.slurm without slurm/srun.
# Usage:
#   bash debug_q05b_local.sh         # IPPO (default)
#   METHOD=srpo bash debug_q05b_local.sh
set -x
set -o pipefail

METHOD=${METHOD:-srpo_main}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR=${REPO_DIR}/checkpoints/${METHOD}_q05b_dbg
DATA_DIR=${REPO_DIR}/data/gsm8k
GSM8K_TRAIN=${DATA_DIR}/train.parquet
GSM8K_TEST=${DATA_DIR}/test.parquet
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

mkdir -p logs "${CKPT_DIR}"

# Activate conda env
module reset >/dev/null 2>&1 || true
module load gcc/9.3.0 anaconda3/2024.02-1 >/dev/null 2>&1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate srpo

cd "${REPO_DIR}"

# Stop any leftover Ray
ray stop -f 2>&1 | tail -3 || true

# Match the slurm env
unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=4
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=info
export VLLM_LOGGING_LEVEL=INFO

NUM_NODES=2          # logical: verl divides nodes among agents (>=num_agents)
GPUS_PER_NODE=1      # logical per-node GPU count
CPUS_PER_TASK=8

# Start a single Ray head with TOTAL physical GPUs (2). STRICT_PACK puts both
# placement groups on this one physical node.
HEAD_IP=127.0.0.1
PORT=6380
ray start --head --node-ip-address=${HEAD_IP} --port=${PORT} \
    --num-cpus=$((NUM_NODES * CPUS_PER_TASK)) \
    --num-gpus=$((NUM_NODES * GPUS_PER_NODE)) \
    --include-dashboard=false 2>&1 | tail -20

# Wait for ray to come up
for i in 1 2 3 4 5; do
    if ray status >/dev/null 2>&1; then break; fi
    sleep 2
done
ray status

# Method-specific config
case "$METHOD" in
    srpo_main)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1
        EXP_NAME="srpo_main_q05b_dbg"
        ENTROPY_OVERRIDE="+multi_agent.agents.0.actor.actor.entropy_coeff=0.05"
        ;;
    srpo_reward_only)
        TRAINER_TYPE="risk_averse"
        KL_COEF=0.1
        EXP_NAME="srpo_reward_only_q05b_dbg"
        ENTROPY_OVERRIDE=""
        ;;
    ippo|*)
        TRAINER_TYPE="mappo"
        KL_COEF=0.001
        EXP_NAME="ippo_q05b_dbg"
        ENTROPY_OVERRIDE=""
        ;;
esac

LOG_FILE="logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Logging to ${LOG_FILE}"

PYTHONUNBUFFERED=1 python -m verl.trainer.main_mappo \
    data.train_files=${GSM8K_TRAIN} \
    data.val_files=${GSM8K_TEST} \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.seed=42 \
    multi_agent.trainer_type=${TRAINER_TYPE} \
    multi_agent.adversary_kl_to_hero=true \
    multi_agent.agents.0.actor.model.path=${MODEL_NAME} \
    multi_agent.agents.1.actor.model.path=${MODEL_NAME} \
    multi_agent.agents.0.n_gpus_per_node=${GPUS_PER_NODE} \
    multi_agent.agents.1.n_gpus_per_node=${GPUS_PER_NODE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=1280 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=NONE \
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    critic.model.path=${MODEL_NAME} \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.total_training_steps=2 \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.resume_mode=disable \
    trainer.nnodes=${NUM_NODES} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.default_local_dir=${CKPT_DIR} \
    trainer.project_name=srpo_experiments \
    trainer.experiment_name=${EXP_NAME} \
    trainer.logger=[console] \
    trainer.val_before_train=False \
    ${ENTROPY_OVERRIDE} \
    2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}

echo "[INFO] Exit code: ${RC}"
ray stop -f 2>&1 | tail -3 || true
exit ${RC}
