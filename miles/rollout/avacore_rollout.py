"""Domain-independent boundary between AvaCore rollouts and Miles training.

AvaCore owns configuration, generation, tools, environments, retries and reward.
This module binds ``kind = 'miles_policy'`` clients to the training engine and
records their token-level calls. Each call keeps its actual conditioning prefix;
independent contexts are never concatenated into a fabricated conversation.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import os
import tomllib
import uuid
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_origin

from miles.utils.types import Sample

REWARD_HOOK = "miles.rollout.avacore_rollout.post_process_rewards"
BATCH_HOOK = "miles.rollout.avacore_rollout.pad_training_batch"


def config_source() -> str:
    return (os.environ.get("AVA_CORE_CONFIG_PATH") or os.environ.get("AVA_CORE_CONFIG") or "").strip()


def load_document() -> tuple[dict, str]:
    source = config_source()
    if not source:
        raise ValueError("Set AVA_CORE_CONFIG_PATH or AVA_CORE_CONFIG")
    if source.lstrip().startswith("[") or "\n" in source:
        document, name = tomllib.loads(source), "<AVA_CORE_CONFIG>"
    else:
        path = Path(source).expanduser().resolve()
        document, name = tomllib.loads(path.read_text()), str(path)
    from ava_core.config import interpolated

    return interpolated(document), name


def _sampling(values: dict) -> dict:
    values = dict(values)
    for name in ("max_tokens", "max_completion_tokens"):
        if name not in values:
            continue
        alias = values.pop(name)
        if alias is None:
            continue
        if "max_new_tokens" in values and values["max_new_tokens"] != alias:
            raise ValueError(f"Conflicting {name} and max_new_tokens for miles_policy")
        values["max_new_tokens"] = alias
    return values


@dataclass
class Episode:
    sample: Sample
    key: tuple[str, int]
    sampling: dict
    evaluation: bool
    calls: list[dict] = field(default_factory=list)


_episode: ContextVar[Episode | None] = ContextVar("avatrain_episode", default=None)


class AvaCoreBridge:
    def __init__(self, args, state):
        self.args, self.state = args, state
        document, self.config_name = load_document()
        self.document = document
        from ava_core.config import RolloutConfig, launch_converter
        from ava_core.core import Client, SGLangClient
        from ava_core.core.parser import Parser
        from ava_core.generate.core import GenerateFunction
        from ava_core.rewards.core import RewardFunction

        for section in ("generate", "reward"):
            if not isinstance(document.get(section), dict):
                raise ValueError(f"AvaCore TOML must contain a [{section}] table")
        self.rollout = launch_converter().structure(document.get("rollout", {}), RolloutConfig)
        if self.rollout.concurrency < 1:
            raise ValueError("rollout.concurrency must be positive")
        if self.rollout.retry.max_trials:
            raise ValueError("RL requires fixed prompt groups; rollout.retry.max_trials must be 0")
        if not 0 <= self.rollout.error_tolerance <= 1:
            raise ValueError("rollout.error_tolerance must be in [0, 1]")
        self.slots = asyncio.Semaphore(self.rollout.concurrency)
        self.reward_slots = asyncio.Semaphore(self.rollout.concurrency)
        self.policy_slots = state.semaphore
        self.attempts = self.failures = 0
        self.trace_path = os.environ.get("MILES_ROLLOUT_TRACE_PATH") or os.environ.get("ROLLOUT_TRACE_PATH")
        bridge = self

        class PolicyClient(SGLangClient):
            async def step(self, trace, *, sampling_params=None, **kwargs):
                episode = _episode.get()
                if episode is None:
                    raise RuntimeError("miles_policy may only run inside AvaCore generation, not a reward/judge")
                if bridge.state.aborted:
                    raise asyncio.CancelledError()
                # CLI harnesses may supply their own generation settings. Miles
                # owns the sampling distribution and inference safety bounds.
                params = {**self.sampling_params, **_sampling(sampling_params or {}), **episode.sampling}
                tool_choice = params.get("tool_choice", "auto")
                parallel_tools = params.get("parallel_tool_calls", True)
                if tool_choice not in ("auto", "none"):
                    raise ValueError(
                        "miles_policy supports tool_choice auto/none; forced tool grammars require an AvaCore client implementation"
                    )
                if params.get("n", 1) != 1:
                    raise ValueError("Miles owns episode multiplicity; policy requests must have n=1")
                requested = params.get("max_new_tokens")
                configured = self.sampling_params.get("max_new_tokens")
                if configured is not None:
                    params["max_new_tokens"] = min(requested, configured) if requested is not None else configured
                if getattr(args, "lora_rank", 0) or getattr(args, "lora_adapter_path", None):
                    from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, is_lora_enabled

                    if is_lora_enabled(args):
                        kwargs["lora_path"] = LORA_ADAPTER_NAME
                async with bridge.policy_slots:
                    if bridge.state.aborted:
                        raise asyncio.CancelledError()
                    result = await super().step(trace, sampling_params=params, **kwargs)
                assistant = result.last_assistant()
                if (tool_choice == "none" and assistant.tool_calls) or (
                    not parallel_tools and len(assistant.tool_calls or []) > 1
                ):
                    raise ValueError("Policy output violates the harness tool-call constraint")
                # AvaCore preserves the newly generated segment, with behavior
                # logprobs, in its token trace. Record only this call's actions.
                segment = result.segments[-1]
                tokens, generated = list(result.tokens), list(segment.tokens)
                logs = list(segment.log_probs)
                if not segment.is_generated or not generated or len(logs) != len(generated):
                    raise ValueError("AvaCore policy call returned inconsistent generated tokens/logprobs")
                if any(p is None or not math.isfinite(p) for p in logs):
                    raise ValueError("AvaCore policy call is missing finite behavior logprobs")
                episode.calls.append(
                    {
                        "tokens": tokens,
                        "response_length": len(generated),
                        "rollout_log_probs": logs,
                        "metadata": dict(assistant.metadata),
                    }
                )
                return result

        conv = launch_converter()
        default_client = conv.get_structure_hook(Client)
        policy_count = 0

        def structure_client(data, cls):
            nonlocal policy_count
            if data.get("kind") != "miles_policy":
                return default_client(data, cls)
            policy_count += 1
            allowed = {"kind", "parser", "sampling_params", "timeout", "max_retries"}
            if extra := data.keys() - allowed:
                raise ValueError(
                    f"miles_policy has unsupported settings {sorted(extra)}; endpoint/tokenizer belong to Miles"
                )
            from ava_core.core import HttpEndpoint, HTTPRetry

            from miles.utils.http_utils import post

            if "parser" not in data:
                raise ValueError("Every miles_policy client requires an AvaCore parser in TOML")
            endpoint = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

            async def policy_post(path, payload, headers):
                episode = _episode.get()
                if episode is None or bridge.state.aborted:
                    raise asyncio.CancelledError()
                payload = dict(payload)
                params = _sampling(payload["sampling_params"])
                # Native /generate has no OpenAI request/tool-policy fields.
                # Tool constraints are checked against the parsed response in
                # step(); behavior logprobs are always requested separately.
                for key in (
                    "tool_choice",
                    "parallel_tool_calls",
                    "logprobs",
                    "top_logprobs",
                    "service_tier",
                    "reasoning_effort",
                ):
                    params.pop(key, None)
                context_limit = getattr(
                    args, "eval_max_context_len" if episode.evaluation else "rollout_max_context_len", None
                )
                if context_limit is not None:
                    remaining = context_limit - len(payload["input_ids"])
                    if remaining <= 0:
                        raise ValueError(
                            f"Policy context exceeds configured limit {context_limit}; configure AvaCore context management"
                        )
                    params["max_new_tokens"] = min(params.get("max_new_tokens", remaining), remaining)
                payload["sampling_params"] = params
                headers = dict(headers or {})
                if getattr(args, "sglang_router_policy", None) == "consistent_hashing":
                    headers["X-SMG-Routing-Key"] = str(episode.key)
                # SingleTurn and harnesses can pass OpenAI-only hints. They are
                # not fields in the native SGLang generation request.
                payload.pop("use_tito", None)
                output = await post(
                    endpoint + path, payload, headers=headers, max_retries=1, timeout=data.get("timeout", 7200)
                )
                items = output.get("meta_info", {}).get("output_token_logprobs", [])
                if output.get("output_ids") != [item[1] for item in items]:
                    raise ValueError("SGLang output token IDs and behavior logprobs do not align")
                return output

            return PolicyClient(
                HttpEndpoint.from_url(endpoint),
                args.hf_checkpoint,
                tokenizer=state.tokenizer,
                parser=conv.structure(data["parser"], Parser),
                sampling_params=_sampling(data.get("sampling_params", {})),
                timeout=data.get("timeout", 7200),
                post=policy_post,
                retry=HTTPRetry(data.get("max_retries", 1)),
            )

        conv.register_structure_hook_func(
            lambda cls: inspect.isclass(get_origin(cls) or cls) and issubclass(get_origin(cls) or cls, Client),
            structure_client,
        )
        self.generate = conv.structure(document["generate"], GenerateFunction)
        if policy_count == 0:
            raise ValueError("[generate] must use at least one kind='miles_policy' client")
        before_reward = policy_count
        self.reward = conv.structure(document["reward"], RewardFunction)
        if policy_count != before_reward:
            raise ValueError("Reward/judge clients must be independent of miles_policy")

    async def run_group(self, group, sampling_params, evaluation=False, timeout=None):
        from ava_core.generate.core import GenerateFunction
        from ava_core.generate.core import Sample as AvaSample
        from ava_core.rewards.core import RewardFunction
        from ava_core.rollout import RolloutEngine, RolloutError

        bridge = self
        group_id = uuid.uuid4().hex
        rows, episodes = [], {}
        for i, sample in enumerate(group):
            # Preserve all dataset fields, including structured runtime/test
            # specifications. Use original messages, never a rendered template.
            row = dict(sample.metadata or {})
            row.pop("_apply_chat_template_fn", None)
            messages = row.pop("_messages_dict", None)
            prompt = sample.prompt
            if messages and isinstance(messages, list):
                prompt = "\n".join(
                    m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)
                )
            elif isinstance(messages, str):
                prompt = messages
            row.update(prompt=prompt, label=sample.label, messages=messages or sample.prompt, metadata=dict(row))
            row.setdefault("id", str(sample.index if sample.index is not None else i))
            key = (group_id, i)
            rows.append(AvaSample(row, key=lambda row, key=key: key))
            params = _sampling(sampling_params)
            if getattr(self.args, "sglang_enable_deterministic_inference", False) and "sampling_seed" not in params:
                params["sampling_seed"] = self.args.rollout_seed + i
            episodes[key] = Episode(sample, key, params, evaluation)

        class Generate(GenerateFunction):
            instance_type = AvaSample

            async def __call__(self, instance, *, sampling_params=None, **kwargs):
                episode = episodes[instance.key]
                episode.calls.clear()  # failed attempts must not train
                async with bridge.slots:
                    if bridge.state.aborted:
                        raise asyncio.CancelledError()
                    token = _episode.set(episode)
                    try:
                        async with asyncio.timeout(timeout):
                            return await bridge.generate(instance, sampling_params=sampling_params or {}, **kwargs)
                    finally:
                        _episode.reset(token)

        class Reward(RewardFunction):
            async def evaluate(self, trace, reference):
                async with bridge.reward_slots:
                    if bridge.state.aborted:
                        raise asyncio.CancelledError()
                    return await bridge.reward.evaluate(trace, reference)

        retry = copy.deepcopy(self.rollout.retry)
        # Use AvaCore's own retry implementation; forbid adaptive extra trials
        # because Miles owns the fixed GRPO group cardinality.
        retry.restore(len(group))
        engine = RolloutEngine(
            Generate(),
            Reward(),
            concurrency=self.rollout.concurrency,
            interval=self.rollout.interval,
            recoverable=self.rollout.recoverable_types(),
            retry=retry,
        )
        outcomes = {}
        async with aclosing(engine.run_pairs(((row, row) for row in rows), stream_rollout=False)) as stream:
            async for outcome in stream:
                if outcome.retrying:
                    continue
                outcomes[outcome.instance.key] = outcome
                if isinstance(outcome, RolloutError) and self.trace_path:
                    episode = episodes[outcome.instance.key]
                    trace = getattr(outcome.error, "trace", None)
                    self.append_record(
                        self.trace_path + ".errors.jsonl",
                        {
                            "schema_version": 2,
                            "complete": False,
                            "episode": list(episode.key),
                            "error": {
                                "type": type(outcome.error).__name__,
                                "message": str(outcome.error),
                                "status": outcome.status,
                            },
                            "trace": trace.to_dict() if trace is not None else None,
                            "policy_calls": episode.calls,
                        },
                    )
        if len(outcomes) != len(group):
            raise RuntimeError("AvaCore did not return exactly one terminal outcome per RL episode")
        self.attempts += len(group)
        failed = [key for key, outcome in outcomes.items() if isinstance(outcome, RolloutError)]
        self.failures += len(failed)
        if self.failures / self.attempts > self.rollout.error_tolerance:
            first = outcomes[failed[0]] if failed else None
            raise RuntimeError(f"AvaCore error budget exceeded ({self.failures}/{self.attempts})") from (
                first.error if first else None
            )
        results = []
        for row in rows:
            episode, outcome = episodes[row.key], outcomes[row.key]
            if isinstance(outcome, RolloutError):
                sample = copy.copy(episode.sample)
                sample.status, sample.remove_sample, sample.reward = Sample.Status.ABORTED, True, 0.0
                sample.metadata = {
                    **sample.metadata,
                    "avacore_error": f"{type(outcome.error).__name__}: {outcome.error}",
                }
                results.append([sample])
                continue
            if outcome.reward is None or not math.isfinite(outcome.reward.score):
                raise ValueError("AvaCore returned no finite terminal reward")
            if not episode.calls:
                raise ValueError("AvaCore episode made no calls to the Miles policy")
            parts = []
            for call_index, call in enumerate(episode.calls):
                sample = copy.copy(episode.sample)
                sample.tokens = call["tokens"]
                sample.response_length = call["response_length"]
                if len(sample.tokens) <= sample.response_length:
                    raise ValueError("Policy call has no conditioning token")
                sample.loss_mask = [1] * sample.response_length
                sample.rollout_log_probs = call["rollout_log_probs"]
                sample.response = self.state.tokenizer.decode(
                    sample.tokens[-sample.response_length :], skip_special_tokens=False
                )
                sample.reward = float(outcome.reward.score)
                meta = call["metadata"]
                reason = meta.get("finish_reason", {})
                reason = reason.get("type") if isinstance(reason, dict) else reason
                sample.status = {
                    "length": Sample.Status.TRUNCATED,
                    "repeat": Sample.Status.REPEATED,
                    "abort": Sample.Status.ABORTED,
                }.get(reason, Sample.Status.COMPLETED)
                sample.weight_versions = (
                    [str(meta["weight_version"])] if meta.get("weight_version") is not None else []
                )
                sample.metadata = {
                    **episode.sample.metadata,
                    "avacore": {
                        "config": self.config_name,
                        "episode": list(episode.key),
                        "call": call_index,
                        "reward": outcome.reward.to_dict(),
                    },
                }
                sample.validate()
                parts.append(sample)
            if evaluation:
                # Evaluation statistics are per episode. Represent its terminal
                # response once, while preserving any failed/truncated call status.
                if any(part.status == Sample.Status.ABORTED for part in parts):
                    parts[-1].status = Sample.Status.ABORTED
                elif any(part.status == Sample.Status.TRUNCATED for part in parts):
                    parts[-1].status = Sample.Status.TRUNCATED
            results.append(parts)
            if self.trace_path:
                self.write_trace(episode, outcome, parts)
        # Store statistics before downstream flattening/padding/trimming. Each
        # episode contributes once, irrespective of how many model calls it made.
        scores = [parts[0].reward for parts in results]
        for parts in results:
            for sample in parts:
                if "avacore" in sample.metadata:
                    sample.metadata["avacore"]["group_rewards"] = scores
        return results

    def write_trace(self, episode, outcome, samples):
        def serialize_trace(trace):
            # AvaCore's wire view contains messages/tools only. Preserve the
            # public trace tree and verifier metadata as well, without joining
            # independent branches into a fictitious conversation.
            return {
                **trace.to_dict(),
                "metadata": trace.metadata,
                "subtraces": [serialize_trace(child) for child in trace.subtraces],
            }

        record = {
            "schema_version": 2,
            "complete": all(s.status == Sample.Status.COMPLETED for s in samples),
            "episode": list(episode.key),
            "reward": outcome.reward.to_dict(),
            "trace": serialize_trace(outcome.trace),
            "messages": [m.to_dict() for m in outcome.trace.messages],
            "policy_calls": [
                {
                    "tokens": s.tokens,
                    "response_length": s.response_length,
                    "loss_mask": s.loss_mask,
                    "rollout_log_probs": s.rollout_log_probs,
                    "weight_versions": s.weight_versions,
                }
                for s in samples
            ],
        }
        self.append_record(self.trace_path, record)

    @staticmethod
    def append_record(filename, record):
        import fcntl

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def generate_group(args, state, group, sampling_params, evaluation=False, timeout=None):
    if getattr(args, "log_passrate", False):
        raise ValueError(
            "Disable --log-passrate for AvaCore: Miles pass@k assumes one Sample per episode; "
            "use the episode rewards in AvaCore traces instead"
        )
    for flag in ("partial_rollout", "use_rollout_routing_replay", "group_rm", "use_session_server"):
        if getattr(args, flag, False):
            raise ValueError(f"AvaCore owns complete episodes; {flag} is incompatible with this bridge")
    for flag in ("custom_generate_function_path", "custom_rm_path"):
        if getattr(args, flag, None):
            raise ValueError(f"Use AvaCore TOML for generation/reward, not --{flag.replace('_', '-')}")
    if not group:
        return []
    for flag in ("reward_key", "eval_reward_key"):
        if getattr(args, flag, None):
            raise ValueError(f"AvaCore supplies Reward.score directly; {flag} must be unset")
    if not evaluation:
        if args.advantage_estimator not in {"grpo", "gspo", "cispo"}:
            raise ValueError("AvaCore call-level samples require grpo/gspo/cispo terminal-return training")
        if getattr(args, "snr_filter_keep_ratio", None) is not None:
            raise ValueError("Miles SNR filtering counts model calls instead of episodes; disable it for AvaCore")
        if getattr(args, "use_dynamic_global_batch_size", False):
            raise ValueError("AvaCore padding requires a fixed global_batch_size")
        if getattr(args, "custom_reward_post_process_path", None) != REWARD_HOOK:
            raise ValueError(f"AvaCore requires --custom-reward-post-process-path {REWARD_HOOK}")
        if getattr(args, "rollout_sample_filter_path", None) != BATCH_HOOK:
            raise ValueError(f"AvaCore requires --rollout-sample-filter-path {BATCH_HOOK}")
    bridge = getattr(state, "avacore_bridge", None)
    if bridge is None:
        bridge = state.avacore_bridge = AvaCoreBridge(args, state)
    return await bridge.run_group(group, sampling_params, evaluation, timeout)


def post_process_rewards(args, samples):
    """Miles hook: normalize terminal rewards over episodes, not model calls."""
    import torch

    raw, normalized = [], []
    for sample in samples:
        raw.append(sample.reward)
        if sample.remove_sample:
            normalized.append(0.0)
            continue
        value = sample.reward
        if args.rewards_normalization and args.advantage_estimator in {"grpo", "gspo", "cispo"}:
            scores = torch.tensor(sample.metadata["avacore"]["group_rewards"], dtype=torch.float32)
            baseline = scores.mean() if args.quantile_k is None else scores.quantile(args.quantile_k)
            value = (scores.new_tensor(value) - baseline).item()
            if args.grpo_std_normalization:
                value /= (scores.std().item() if len(scores) > 1 else 0.0) + 1e-6
        normalized.append(value)
    return raw, normalized


def pad_training_batch(args, data):
    """Miles hook: pad variable call counts instead of trimming an episode."""
    if args.use_dynamic_global_batch_size:
        raise ValueError("AvaCore call padding uses fixed global_batch_size; disable dynamic global batch size")
    samples = [sample for group in data for episode in group for sample in episode]
    count = (-len(samples)) % args.global_batch_size
    for _ in range(count):
        padding = copy.copy(samples[-1])
        # A tiny valid prefix/target, zero objective contribution.
        padding.tokens, padding.response_length = samples[-1].tokens[:2], 1
        padding.loss_mask, padding.rollout_log_probs = [0], [0.0]
        padding.remove_sample, padding.reward = True, 0.0
        padding.metadata = {"avacore_padding": True}
        data[-1][-1].append(padding)
