"""Run independent trajectory attempts without cancelling healthy siblings."""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from miles.rollout.base_types import RolloutSampleError
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate_group(
    group: list[Sample],
    generate: Callable[[int, Sample], Awaitable[Sample | list[Sample]]],
    *,
    max_retries: int = 0,
    sample_done_callback: Callable[[], None] | None = None,
) -> list[Sample | list[Sample]]:
    if max_retries < 0:
        raise ValueError("rollout_sample_max_retries must be nonnegative")

    async def run_sample(index: int, sample: Sample):
        for attempt in range(max_retries + 1):
            try:
                return await generate(index, sample)
            except Exception as exc:
                if attempt == max_retries:
                    return exc
                logger.warning(
                    "Retrying rollout sample index=%s group=%s retry=%s/%s error=%s",
                    sample.index,
                    sample.group_index,
                    attempt + 1,
                    max_retries,
                    type(exc).__name__,
                )
                sample.reset_for_retry()
                await asyncio.sleep(min(2**attempt, 30))

    tasks = [asyncio.create_task(run_sample(index, sample)) for index, sample in enumerate(group)]
    for task in tasks:
        if sample_done_callback is not None:
            # Retries keep the same slot; release its submission credit exactly once.
            task.add_done_callback(lambda _task: sample_done_callback())
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        # Explicit group cancellation still drains children before samples can be reused.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        # A recoverable trial failure must not hide a sibling's programming error.
        raise next((error for error in errors if not isinstance(error, RolloutSampleError)), errors[0])
    return results
