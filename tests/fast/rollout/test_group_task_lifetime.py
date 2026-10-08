import asyncio
import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("refactored", [False, True])
@pytest.mark.parametrize("cancel_parent", [False, True])
def test_group_waits_for_peer_cleanup(monkeypatch, refactored, cancel_parent):
    name = "inference_rollout.inference_rollout_common" if refactored else "sglang_rollout"
    module = importlib.import_module(f"miles.rollout.{name}")
    args = SimpleNamespace(sglang_router_policy="random", group_rm=False)
    state = SimpleNamespace(args=args, aborted=False)
    monkeypatch.setattr(module, "GenerateState", lambda args: state)

    async def scenario():
        started, release, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
        peers = []

        async def generate(_, sample, *args, **kwargs):
            peers.append(asyncio.current_task())
            if sample.index == 0:
                await started.wait()
                if cancel_parent:
                    await release.wait()
                raise RuntimeError("trial failed")
            started.set()
            try:
                await release.wait()
                sample.reward = 1
            finally:
                await asyncio.sleep(0)
                cleaned.set()
            return sample

        monkeypatch.setattr(module, "generate_and_rm", generate)
        group = [SimpleNamespace(index=i, reward=None) for i in range(2)]
        parent = asyncio.create_task(module.generate_and_rm_group(state if refactored else args, group, {}))
        try:
            await started.wait()
            if cancel_parent:
                parent.cancel()
            with pytest.raises(asyncio.CancelledError if cancel_parent else RuntimeError):
                await parent
            # The caller may now reset/reuse the same Sample objects for retry.
            assert cleaned.is_set()
            assert all(peer.done() for peer in peers)
            assert group[1].reward is None
        finally:
            release.set()
            await asyncio.gather(*peers, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("refactored", [False, True])
def test_successful_group_preserves_sample_order(monkeypatch, refactored):
    name = "inference_rollout.inference_rollout_common" if refactored else "sglang_rollout"
    module = importlib.import_module(f"miles.rollout.{name}")
    args = SimpleNamespace(sglang_router_policy="random", group_rm=False)
    state = SimpleNamespace(args=args, aborted=False)
    monkeypatch.setattr(module, "GenerateState", lambda args: state)

    async def scenario():
        second_finished = asyncio.Event()

        async def generate(_, sample, *args, **kwargs):
            if sample.index == 0:
                await second_finished.wait()
            else:
                second_finished.set()
            return sample

        monkeypatch.setattr(module, "generate_and_rm", generate)
        group = [SimpleNamespace(index=i) for i in range(2)]
        result = await module.generate_and_rm_group(state if refactored else args, group, {})
        assert all(actual is expected for actual, expected in zip(result, group, strict=True))

    asyncio.run(scenario())
