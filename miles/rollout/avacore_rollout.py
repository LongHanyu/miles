"""Rollouts generated and rewarded by AvaCore, as the TOML at ``AVACORE_ROLLOUT_CONFIG`` defines them.

The document's ``[generate]`` and ``[reward]`` tables are structured by AvaCore; each sees a row of the
sample's metadata plus ``prompt`` and ``label``. ``${SGLANG_ROUTER_URL}`` and ``${HF_CHECKPOINT}`` name the
policy miles serves.
"""

import logging
import os
import tomllib
from argparse import Namespace
from functools import cache
from pathlib import Path
from typing import Any

from ava_core.config import interpolated, launch_converter
from ava_core.core import TokenTrace
from ava_core.generate.core import GenerateFunction
from ava_core.generate.core import Sample as Row
from ava_core.rewards import RewardFunction
from ava_core.runner import RECOVERABLE_ERRORS

from miles.utils.types import Sample

__all__ = ["generate"]

logger = logging.getLogger(__name__)


@cache
def functions(path: str, router: str, checkpoint: str) -> tuple[GenerateFunction[Row], RewardFunction[Row]]:
    os.environ["SGLANG_ROUTER_URL"], os.environ["HF_CHECKPOINT"] = router, checkpoint
    document = interpolated(tomllib.loads(Path(path).read_text()))
    conv = launch_converter()
    return conv.structure(document["generate"], GenerateFunction), conv.structure(document["reward"], RewardFunction)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    generate_fn, reward_fn = functions(
        os.environ["AVACORE_ROLLOUT_CONFIG"],
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        args.hf_checkpoint,
    )
    row = Row({**sample.metadata, "prompt": sample.prompt, "label": sample.label}, key=lambda _: sample.index)
    try:
        trace = await generate_fn(row, sampling_params=sampling_params)
        reward = await reward_fn(trace, row)
    except RECOVERABLE_ERRORS as error:
        logger.warning("AvaCore rollout of sample %s aborted: %r", sample.index, error)
        sample.status = Sample.Status.ABORTED
        return sample

    # One Sample holds one token sequence: a trace that branches into subtraces has no such form.
    assert isinstance(trace, TokenTrace) and not trace.subtraces, "AvaCore rollouts must be linear token traces"
    prompt_length = len(trace.segments[0].tokens)
    sample.tokens = trace.tokens
    sample.response = "".join(segment.template for segment in trace.segments[1:])
    sample.response_length = len(trace.tokens) - prompt_length
    sample.loss_mask = trace.loss_mask[prompt_length:]
    sample.rollout_log_probs = [log_prob or 0.0 for log_prob in trace.log_probs[prompt_length:]]
    for message in trace.messages:
        if message.role == "assistant":
            sample.update_from_meta_info(args, message.metadata)
    sample.reward = reward.score
    return sample
