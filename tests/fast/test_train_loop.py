from miles.utils.misc import needs_rollout_model


def test_final_rollout_without_eval_does_not_prepare_rollout_model():
    assert not needs_rollout_model(rollout_id=0, num_rollout=1, run_eval=False)


def test_non_final_rollout_prepares_rollout_model():
    assert needs_rollout_model(rollout_id=0, num_rollout=2, run_eval=False)


def test_final_rollout_prepares_rollout_model_for_eval():
    assert needs_rollout_model(rollout_id=0, num_rollout=1, run_eval=True)
