# Agentic SWE RL

This example trains the unmodified `qwen-code` CLI on SWE-bench Verified. The
agent runs in a YiCloud OpenSandbox instance and calls a local OpenAI-compatible proxy that
serves the policy from Miles.

One rollout has four steps:

1. start the proxy, sandbox, and reverse tunnel;
2. run `qwen-code` on one repository issue;
3. record the main conversation as tokens, log probabilities, and a loss mask;
4. grade the agent patch in a fresh sandbox and return one Miles `Sample`.

## Token ledger

The CLI sends the full conversation on every turn. Retokenizing that history can
change the tokens that the model actually sampled, so `trajectory.py` keeps an
append-only ledger:

1. render the incoming history with its tool schemas;
2. require it to extend the recorded token prefix;
3. append environment tokens with `loss_mask=0` and sampled tokens with
   `loss_mask=1`.

Requests that do not extend the main conversation are served but are not added
to the training sample.

## Files

| File | Purpose |
|---|---|
| `generate.py` | Miles `GenerateFn` and qwen-code command |
| `proxy.py` | OpenAI endpoint and SGLang generation |
| `trajectory.py` | Token ledger and Miles `Sample` conversion |
| `sandbox.py` | Sandbox and reverse tunnel |
| `swe.py` | SWE-bench Verified workspace and reward |
| `run_qwen35_35b_a3b.sh` | Training command |

## Input

The JSONL rows use Miles' `prompt`, `label`, and `metadata` fields. Metadata must
contain:

```text
repo
repo_workdir
base_commit
inspire_template
docker_image_default_user
docker_image_env
sandbox_bootstrap (optional; prepares generic images before setup and grading)
test_patch
FAIL_TO_PASS
PASS_TO_PASS
install_config.test_cmd
```

For compatibility with existing data conversion, the field is still named
`inspire_template`, but its value must now be a YiCloud image reference such as
`project/repository:tag`. The image must either contain the prepared repository
or provide `sandbox_bootstrap` metadata that creates it. The resulting
workspace must be safely resettable with `git reset --hard` and `git clean -fd`.

The host `wstunnel` binary must be available on `PATH`.

## Run

From the AvaTrain workspace root, which owns the `swe` extra that adds
`swebench`. `AVACORE_SRC` must point to the AvaCore `src` directory containing
the shared `ava_core_yicloud` package; the wrapper defaults to a sibling
AvaCore checkout.

```bash
export HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B-sft
export REF_LOAD=/path/to/Qwen3.5-35B-A3B-sft_torch_dist
export YICLOUD_API_HOST=https://gate.yicloud.com.cn
export YICLOUD_PUBLIC_KEY=...
export YICLOUD_SECRET_KEY=...
export YICLOUD_PROJECT_NAME=...
export YICLOUD_SANDBOX_ENVIRONMENT_ID=...
bash scripts/train/agentic_swe.sh \
    /path/to/swe_verified_train.jsonl
```

Run this command from the AvaTrain workspace root. The wrapper checks that
`wstunnel` is available on `PATH` and uses the active Python environment. The
environment must contain AvaCore's YiCloud dependencies before the offline GPU
job starts.

The constants near the top of `generate.py` select the qwen-code and wstunnel
paths and the rollout time limits. Miles' rollout response length is the token
budget for each model request made by the proxy.
