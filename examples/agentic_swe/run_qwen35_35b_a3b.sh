#!/bin/bash
# Agentic SWE RL: train an unmodified qwen-code CLI on SWE-bench Verified.
#
# Prerequisites (see README.md):
#   YICLOUD_*                            YiCloud OpenSandbox credentials/config

set -e
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
source "${MILES_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

TASK_FILE="$(realpath "${1:?usage: $0 TASK_FILE.jsonl}")"
MEGATRON_PATH="${MEGATRON_PATH:-${MILES_DIR}/../Megatron-LM}"
ROLLOUT_PYTHONPATH="${AVACORE_SRC:?set AVACORE_SRC to AvaCore/src}:${MEGATRON_PATH}:${SCRIPT_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"
AVATRAIN_PYTHON="${AVATRAIN_PYTHON:-$(command -v python3)}"
RAY_CLI="${RAY_CLI:-$(command -v ray)}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:?set HF_CHECKPOINT}"
   --ref-load "${REF_LOAD:?set REF_LOAD (torch_dist checkpoint)}"
   --save "${SAVE_DIR:-/root/agentic_swe_ckpt}"
   --save-interval "${SAVE_INTERVAL:-20}"
)
if [[ "${NO_SAVE_OPTIM:-0}" == "1" ]]; then
   CKPT_ARGS+=(--no-save-optim --no-save-rng)
fi

ROLLOUT_ARGS=(
   --prompt-data "${TASK_FILE}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --custom-generate-function-path "${CUSTOM_GENERATE_FUNCTION_PATH:-generate.generate}"
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-500}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   # The proxy uses this token budget for each model request.
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
   --rollout-temperature 1.0
   --rollout-top-p 0.95
   --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
   --balance-data
)

DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH-miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
if [[ -n "${DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
   ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}")
fi

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
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
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
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.85}"
   # Both parsers must match the model: the tool parser types arguments from the
   # schema, and mistyped arguments break the token round trip (see README).
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-32}"
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
"${AVATRAIN_PYTHON}" "${RAY_CLI}" start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${ROLLOUT_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"YICLOUD_API_HOST\": \"${YICLOUD_API_HOST:-https://gate.yicloud.com.cn}\",
    \"YICLOUD_PUBLIC_KEY\": \"${YICLOUD_PUBLIC_KEY:-}\",
    \"YICLOUD_SECRET_KEY\": \"${YICLOUD_SECRET_KEY:-}\",
    \"YICLOUD_PROJECT_NAME\": \"${YICLOUD_PROJECT_NAME:-}\",
    \"YICLOUD_SANDBOX_ENVIRONMENT_ID\": \"${YICLOUD_SANDBOX_ENVIRONMENT_ID:-}\",
    \"YICLOUD_SANDBOX_ENVIRONMENT_NAME\": \"${YICLOUD_SANDBOX_ENVIRONMENT_NAME:-}\",
    \"YICLOUD_SANDBOX_PROXY_ORIGIN\": \"${YICLOUD_SANDBOX_PROXY_ORIGIN:-https://gate.yicloud.com.cn/sandbox-connect}\",
    \"AVATRAIN_SANDBOX_WSTUNNEL_ARCHIVE\": \"${AVATRAIN_SANDBOX_WSTUNNEL_ARCHIVE:-}\",
    \"AVATRAIN_SANDBOX_REPO_ARCHIVE\": \"${AVATRAIN_SANDBOX_REPO_ARCHIVE:-}\",
    \"AGENT_MAX_TURNS\": \"${AGENT_MAX_TURNS:-80}\",
    \"AGENT_TIMEOUT_SECONDS\": \"${AGENT_TIMEOUT_SECONDS:-5400}\",
    \"AVATRAIN_JEST_MAX_WORKERS\": \"${AVATRAIN_JEST_MAX_WORKERS:-2}\",
    \"AVATRAIN_SANDBOX_CREATE_ATTEMPTS\": \"${AVATRAIN_SANDBOX_CREATE_ATTEMPTS:-6}\",
    \"AVATRAIN_SANDBOX_READY_TIMEOUT_SECONDS\": \"${AVATRAIN_SANDBOX_READY_TIMEOUT_SECONDS:-300}\",
    \"YICLOUD_EXECD_REQUEST_ATTEMPTS\": \"${YICLOUD_EXECD_REQUEST_ATTEMPTS:-4}\",
    \"AVATRAIN_EXECD_STREAM_MAX_SECONDS\": \"${AVATRAIN_EXECD_STREAM_MAX_SECONDS:-300}\",
    \"AVATRAIN_DETACHED_POLL_SECONDS\": \"${AVATRAIN_DETACHED_POLL_SECONDS:-10}\",
    \"AVATRAIN_DETACHED_MAX_INCONCLUSIVE_POLLS\": \"${AVATRAIN_DETACHED_MAX_INCONCLUSIVE_POLLS:-6}\",
    \"MILES_DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS\": \"${MILES_DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS:-256}\",
    \"MILES_ROLLOUT_MAX_FAILED_GROUPS\": \"${MILES_ROLLOUT_MAX_FAILED_GROUPS:-64}\"
  }
}"

"${AVATRAIN_PYTHON}" "${RAY_CLI}" job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${AVATRAIN_PYTHON}" train.py \
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
