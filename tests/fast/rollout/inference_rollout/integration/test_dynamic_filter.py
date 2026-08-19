from contextlib import nullcontext

import pytest
from tests.fast.rollout.inference_rollout.integration.utils import (
    MIXED_DATA_ROWS,
    filter_by_reward,
    integration_env_config,
    load_and_call_train,
)

from miles.utils.misc import function_registry


@pytest.mark.parametrize(
    "rollout_env,use_filter,expect_all_correct",
    [
        pytest.param(
            integration_env_config(["--rollout-batch-size", "4"], data_rows=MIXED_DATA_ROWS),
            False,
            False,
            id="no_filter",
        ),
        pytest.param(
            integration_env_config(
                ["--rollout-batch-size", "3", "--dynamic-sampling-filter-path", "test:filter_by_reward"],
                data_rows=MIXED_DATA_ROWS,
            ),
            True,
            True,
            id="with_filter",
        ),
    ],
    indirect=["rollout_env"],
)
def test_filter_effect(rollout_env, use_filter, expect_all_correct):
    env = rollout_env
    ctx = function_registry.temporary("test:filter_by_reward", filter_by_reward) if use_filter else nullcontext()

    with ctx:
        out = load_and_call_train(env.args, env.data_source)

    rewards = {group[0].reward for group in out.samples}
    if expect_all_correct:
        assert rewards == {1}, "Filter should keep only correct samples"
    else:
        assert 0 in rewards, "Without filter, incorrect samples should be present"


@pytest.mark.parametrize(
    "rollout_env",
    [
        integration_env_config(
            ["--rollout-batch-size", "1", "--dynamic-sampling-filter-path", "test:filter_by_reward"],
            data_rows=[{"input": "What is 1+8?", "label": "wrong"}],
        )
    ],
    indirect=True,
)
def test_filter_drop_limit_fails_instead_of_sampling_forever(rollout_env, monkeypatch):
    env = rollout_env
    monkeypatch.setenv("MILES_DYNAMIC_SAMPLING_MAX_DROPPED_GROUPS", "2")

    with function_registry.temporary("test:filter_by_reward", filter_by_reward):
        with pytest.raises(RuntimeError, match=r"dynamic sampling dropped 2 groups.*kept=0/1"):
            load_and_call_train(env.args, env.data_source)
