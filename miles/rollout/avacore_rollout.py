import asyncio
import copy
import logging
import os
from argparse import Namespace
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
from ava_core.config import interpolated, launch_converter
from ava_core.core import Schema, TokenTrace, Trace
from ava_core.generate.core import GenerateFunction
from ava_core.generate.core import Sample as Row
from ava_core.rewards import RewardFunction
from ava_core.rewards.core import Reward
from ava_core.runner import RECOVERABLE_ERRORS
from ava_core.store.core import Run
from ava_core.store.postgres import PostgresBackend

from miles.utils.types import Sample

__all__ = ["AvaCoreRollout"]

logger = logging.getLogger(__name__)


class AvaCoreRollout:
    def __init__(self, args: Namespace, config: Mapping[str, Any]) -> None:
        os.environ["SGLANG_ROUTER_URL"] = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        os.environ["HF_CHECKPOINT"] = args.hf_checkpoint
        self.args = args
        self.document = config
        conv = launch_converter()
        self.generate_fn = conv.structure(interpolated(self.document["generate"]), GenerateFunction)
        self.reward_fn = conv.structure(interpolated(self.document["reward"]), RewardFunction)
        self.postgres = interpolated(self.document["record"])["postgres"] if "record" in self.document else None
        self.run: asyncio.Task[Run] | None = None
        self.stack = AsyncExitStack()
        self.writes: set[asyncio.Task[None]] = set()
        self.failed_writes = 0

    async def generate(self, sample: Sample, sampling_params: dict[str, Any]) -> Sample | list[Sample]:
        row = Row({**sample.metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: sample.index)
        replay = {"return_routed_experts": True} if self.args.use_rollout_routing_replay else {}
        try:
            trace = await self.generate_fn(row, sampling_params=sampling_params, **replay)
            if any(
                message.metadata["finish_reason"]["type"] == "abort"
                for node in flattened(trace)
                for message in node.messages
                if "finish_reason" in message.metadata
            ):
                sample.status = Sample.Status.ABORTED
                return sample
            reward = await self.reward_fn(trace, row)
        except RECOVERABLE_ERRORS as error:
            logger.warning("AvaCore rollout of sample %s aborted: %r", sample.index, error)
            sample.status = Sample.Status.ABORTED
            return sample

        assert isinstance(trace, TokenTrace), "AvaCore rollouts must drive the policy through a token-level client"
        if self.postgres is not None:
            write = asyncio.create_task(self.record(row, sample, trace, reward, sampling_params))
            self.writes.add(write)
            write.add_done_callback(self.writes.discard)
        samples = [
            filled(self.args, copy.deepcopy(sample), node, reward.score)
            for node in flattened(trace)
            if isinstance(node, TokenTrace) and any(segment.is_generated for segment in node.segments)
        ]
        return samples[0] if len(samples) == 1 else samples

    async def open(self, sampling_params: dict[str, Any]) -> Run:
        store = await self.stack.enter_async_context(PostgresBackend(self.postgres, min_size=1, max_size=4))
        run_dir = Path(os.environ["RUN_DIR"])
        run = await self.stack.enter_async_context(
            store.rl_run(
                model=run_dir.parent.name,
                run=run_dir.name,
                collection=Path(self.args.prompt_data).stem,
                sampling_params=sampling_params,
                config=self.document,
                schema=Schema(),
            )
        )
        await run.update(status="running")
        return run

    async def record(
        self,
        row: Row,
        sample: Sample,
        trace: TokenTrace,
        reward: Reward,
        sampling_params: dict[str, Any],
    ) -> None:
        try:
            assert sample.epoch is not None and sample.index is not None
            n = self.args.n_samples_per_prompt
            query_id = str(sample.metadata["_index"])
            trial_id = sample.epoch * n + sample.index % n
            if self.run is None:
                self.run = asyncio.create_task(self.open(sampling_params))
            run = await self.run
            await run.create_rollout(
                query_id=query_id,
                trial_id=trial_id,
                instance=row,
                trace=trace,
                reward=reward,
                status="completed",
                metadata={"weight_version": int(trace.last_assistant().metadata["weight_version"])},
            )
        except Exception as error:
            self.failed_writes += 1
            if self.failed_writes in (1, 10, 100) or self.failed_writes % 1000 == 0:
                logger.warning(
                    "Recording rollouts to Postgres failed %d times; last error: %r", self.failed_writes, error
                )


def flattened(trace: Trace) -> list[Trace]:
    return [trace] + [node for subtrace in trace.subtraces for node in flattened(subtrace)]


def filled(args: Namespace, sample: Sample, trace: TokenTrace, reward: float) -> Sample:
    # Context after the last generated turn trains nothing, and no request routed it for replay.
    segments = trace.segments[: max(i for i, segment in enumerate(trace.segments) if segment.is_generated) + 1]
    length = sum(len(segment.tokens) for segment in segments)
    prompt_length = len(segments[0].tokens)
    sample.tokens = trace.tokens[:length]
    sample.response = "".join(segment.template for segment in segments[1:])
    sample.response_length = length - prompt_length
    sample.loss_mask = trace.loss_mask[prompt_length:length]
    sample.rollout_log_probs = [log_prob or 0.0 for log_prob in trace.log_probs[prompt_length:length]]
    for message in trace.messages:
        if "finish_reason" in message.metadata:
            sample.update_from_meta_info(args, message.metadata)
    if args.use_rollout_routing_replay and sample.status != Sample.Status.ABORTED:
        assert trace.routed_experts, (
            "Internal error: routing replay enabled but no routed experts returned. Perhaps some traces are not generated by the configured generate function."
        )
        sample.rollout_routed_experts = np.frombuffer(
            b"".join(pybase64.b64decode(chunk) for chunk in trace.routed_experts), dtype=np.int32
        ).reshape(length - 1, args.num_layers, args.moe_router_topk)
    sample.reward = reward
    return sample
