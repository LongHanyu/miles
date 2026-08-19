"""Synthetic long trajectories for an Agentic SWE trainer memory smoke test."""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

PROMPT_TOKENS = 128
RESPONSE_TOKENS = 35_000


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    sample = input.sample
    sample.tokens = [1_000] * PROMPT_TOKENS + [1_001] * RESPONSE_TOKENS
    sample.response = ""
    sample.response_length = RESPONSE_TOKENS
    sample.reward = float((sample.index or 0) % 2)
    sample.loss_mask = [1] * RESPONSE_TOKENS
    sample.rollout_log_probs = [0.0] * RESPONSE_TOKENS
    sample.status = Sample.Status.COMPLETED
    return GenerateFnOutput(samples=sample)
