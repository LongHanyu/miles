import asyncio
import importlib
from collections import Counter
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.group import generate_group
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


@pytest.mark.parametrize("refactored", [False, True])
@pytest.mark.parametrize("count", [8, 32])
@pytest.mark.parametrize("failure", [RuntimeError, TimeoutError])
async def test_only_failed_trajectory_is_retried(monkeypatch, refactored, count, failure):
    name = "inference_rollout.inference_rollout_common" if refactored else "sglang_rollout"
    module = importlib.import_module(f"miles.rollout.{name}")
    args = SimpleNamespace(
        sglang_router_policy="random",
        group_rm=False,
        rollout_sample_max_retries=1,
        sglang_enable_deterministic_inference=True,
        rollout_seed=42,
    )
    state = SimpleNamespace(args=args, aborted=False, group_sampling_seeds=list(range(42, 42 + count)))
    monkeypatch.setattr(module, "GenerateState", lambda args: state)
    group = [Sample(index=i, group_index=1, prompt="prompt") for i in range(count)]
    arrived, peer_done = asyncio.Event(), asyncio.Event()
    calls, seeds = Counter(), {}
    cancelled, credits = [], []
    versions = [WeightVersionsPerCall(spans=[WeightVersionSpan(version="3", abs_start=1, abs_end=2)])]

    async def generate(_, sample, params, **kwargs):
        index = sample.index
        calls[index] += 1
        seeds.setdefault(index, []).append(params.pop("sampling_seed"))
        assert params.pop("temperature") == 1
        if sum(calls.values()) == count:
            arrived.set()
        try:
            await arrived.wait()
            if index == 0 and calls[index] == 1:
                sample.tokens, sample.reward, sample.weight_versions = [999], -999, versions
                raise failure("injected trial failure")
            if index == 0:
                assert sample.tokens == [] and sample.reward is None and sample.weight_versions == []
                await peer_done.wait()
            else:
                await asyncio.sleep(0)
                peer_done.set()
            sample.tokens, sample.rollout_log_probs = [1, 2], [-0.5]
            sample.response, sample.response_length = "ok", 1
            sample.status, sample.reward = Sample.Status.COMPLETED, index % 2
            sample.weight_versions = versions
            return [sample] if index == 1 else sample
        except asyncio.CancelledError:
            cancelled.append(index)
            raise

    monkeypatch.setattr(module, "generate_and_rm", generate)
    kwargs = {"sample_done_callback": lambda: credits.append(1)} if refactored else {}
    async with asyncio.timeout(5):
        result = await module.generate_and_rm_group(state if refactored else args, group, {"temperature": 1}, **kwargs)
    assert calls == Counter({i: 2 if i == 0 else 1 for i in range(count)})
    assert not cancelled
    assert seeds == {i: [42 + i] * calls[i] for i in range(count)}
    assert result[1] == [group[1]]
    assert all((item[0] if isinstance(item, list) else item) is group[i] for i, item in enumerate(result))
    assert all(sample.weight_versions is versions for sample in group)
    assert len(credits) == (count if refactored else 0)


async def test_retry_exhaustion_waits_for_siblings_without_resetting_success():
    group = [Sample(index=i, group_index=1) for i in range(2)]
    exhausted, release = asyncio.Event(), asyncio.Event()
    calls, credits = Counter(), []
    error = RuntimeError("permanent failure")

    async def generate(index, sample):
        calls[index] += 1
        if index == 0:
            if calls[index] == 2:
                exhausted.set()
            raise error
        await release.wait()
        sample.reward, sample.tokens = 1, [1, 2]
        return sample

    task = asyncio.create_task(
        generate_group(group, generate, max_retries=1, sample_done_callback=lambda: credits.append(1))
    )
    await asyncio.wait_for(exhausted.wait(), 3)
    assert not task.done()
    release.set()
    with pytest.raises(RuntimeError) as exc:
        await task
    assert exc.value is error
    assert calls == {0: 2, 1: 1}
    assert group[1].reward == 1 and group[1].tokens == [1, 2]
    assert len(credits) == 2


async def test_cancellation_during_retry_drains_without_another_attempt():
    group = [Sample(index=i, group_index=1) for i in range(2)]
    failed, started, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls, credits = Counter(), []

    async def generate(index, sample):
        calls[index] += 1
        if index == 0:
            await started.wait()
            failed.set()
            raise RuntimeError("retryable failure")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()

    task = asyncio.create_task(
        generate_group(group, generate, max_retries=3, sample_done_callback=lambda: credits.append(1))
    )
    await failed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == {0: 1, 1: 1}
    assert cleaned.is_set() and len(credits) == 2


@pytest.mark.parametrize("refactored", [False, True])
async def test_evaluation_never_resamples_failures(monkeypatch, refactored):
    name = "inference_rollout.inference_rollout_common" if refactored else "sglang_rollout"
    module = importlib.import_module(f"miles.rollout.{name}")
    args = SimpleNamespace(sglang_router_policy="random", group_rm=False, rollout_sample_max_retries=2)
    state = SimpleNamespace(args=args, aborted=False)
    monkeypatch.setattr(module, "GenerateState", lambda args: state)
    calls = []

    async def generate(_, sample, *args, **kwargs):
        calls.append(sample.index)
        raise RuntimeError("evaluation failure")

    monkeypatch.setattr(module, "generate_and_rm", generate)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        await module.generate_and_rm_group(state if refactored else args, [Sample(index=0)], {}, evaluation=True)
    assert calls == [0]


async def test_aborted_results_are_not_retried():
    sample = Sample(index=0, status=Sample.Status.ABORTED)
    calls = []

    async def generate(index, sample):
        calls.append(index)
        return sample

    assert await generate_group([sample], generate, max_retries=2) == [sample]
    assert calls == [0]


async def test_negative_retry_budget_rejected_before_generation():
    async def generate(*args):
        pytest.fail("invalid budgets must not start generation")

    with pytest.raises(ValueError, match="nonnegative"):
        await generate_group([Sample(index=0)], generate, max_retries=-1)


@pytest.mark.parametrize("refactored", [False, True])
async def test_group_reward_runs_once_after_retry(monkeypatch, refactored):
    name = "inference_rollout.inference_rollout_common" if refactored else "sglang_rollout"
    module = importlib.import_module(f"miles.rollout.{name}")
    args = SimpleNamespace(sglang_router_policy="random", group_rm=True, rollout_sample_max_retries=1)
    state = SimpleNamespace(args=args, aborted=False)
    monkeypatch.setattr(module, "GenerateState", lambda args: state)
    group = [Sample(index=i, group_index=1) for i in range(2)]
    calls, reward_calls = Counter(), []

    async def generate(_, sample, *args, **kwargs):
        calls[sample.index] += 1
        if sample.index == 0 and calls[0] == 1:
            raise RuntimeError("transient failure")
        sample.status = Sample.Status.COMPLETED
        return sample

    async def reward(_, samples, **kwargs):
        assert all(sample.status == Sample.Status.COMPLETED for sample in samples)
        reward_calls.append(list(samples))
        for sample in samples:
            sample.reward = sample.index
        return [sample.index for sample in samples]

    monkeypatch.setattr(module, "generate_and_rm", generate)
    monkeypatch.setattr(module, "batched_async_rm", reward)
    result = await module.generate_and_rm_group(state if refactored else args, group, {})
    assert result == group and reward_calls == [group]
    assert calls == {0: 2, 1: 1}
    assert [sample.reward for sample in result] == [0, 1]
