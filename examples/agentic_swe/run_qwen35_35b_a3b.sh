#!/bin/bash
# Agentic SWE RL: train an unmodified qwen-code CLI on SWE-bench Verified.
#
# Prerequisites (see README.md):
#   SBX_API_KEY / SBX_API_URL            sandbox platform credentials

set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${MILES_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

TASK_FILE="$(realpath "${1:?usage: $0 TASK_FILE.jsonl}")"
ROLLOUT_PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
   --ref-load "${REF_LOAD:?set REF_LOAD (torch_dist checkpoint)}"
   --save "${SAVE_DIR:-/root/agentic_swe_ckpt}"
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data "${TASK_FILE}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-generate-function-path generate.generate
   --rollout-shuffle
   --num-rollout 500
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   # The proxy uses this token budget for each model request.
   --rollout-max-response-len 16384
   --rollout-temperature 1.0
   --rollout-top-p 0.95
   --global-batch-size 128
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.85
   # Both parsers must match the model: the tool parser types arguments from the
   # schema, and mistyped arguments break the token round trip (see README).
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   --sglang-server-concurrency 32
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ray job submit uploads the working directory and runs train.py from it.
cd "${MILES_DIR}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROLLOUT_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"SBX_API_KEY\": \"${SBX_API_KEY:-}\",
    \"SBX_API_URL\": \"${SBX_API_URL:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
