from argparse import Namespace
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.cp_utils import all_gather_with_cp, get_sum_of_sample_mean, slice_log_prob_with_cp
from miles.backends.training_utils.loss_hub.math_utils import compute_gspo_kl, compute_policy_loss
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state
from miles.backends.training_utils.sequence_groups import prepare_sequence_groups


def state(cp=None, dp=None):
    one = GroupInfo(0, 1, None)
    set_parallel_state(
        ParallelState(intra_dp=dp or one, intra_dp_cp=one, cp=cp or one, tp=one, pp=one, ep=one, etp=one)
    )


def args(**updates):
    return Namespace(
        **(
            dict(
                advantage_estimator="gspo",
                global_batch_size=2,
                keep_old_actor=False,
                compute_advantages_and_returns=True,
                use_rollout_logprobs=True,
                qkv_format="thd",
            )
            | updates
        )
    )


def example():
    # Unequal segments, masked tool tokens, opposite ratios, negative advantage,
    # and fully masked padding. Each original episode must have equal weight.
    masks = [torch.tensor(m) for m in ([1, 0, 1], [1], [0, 1], [1, 1, 1], [0], [0])]
    current = [
        torch.tensor(v, dtype=torch.float64)
        for v in ([0.5, 99.0, 0.1], [-0.6], [99.0, -0.8], [0.3, 0.4, 0.1], [0.0], [0.0])
    ]
    return dict(
        log_probs=current,
        rollout_log_probs=[torch.zeros_like(v) for v in current],
        loss_masks=masks,
        total_lengths=[len(v) + 3 for v in current],
        response_lengths=[len(v) for v in current],
        sequence_group_ids=[0, 0, 1, 1, -1, -1],
        sample_weights=[2.0, 1.0, 0.75, 2.25, 0.0, 0.0],
    )


def test_grouped_gspo_matches_unsplit_loss_gradient_and_optimizer_across_microbatches():
    state()
    data = example()
    prepare_sequence_groups(args(), data, [6])
    xs = [v.clone().requires_grad_() for v in data["log_probs"]]
    expected = xs[0].sum() * 0
    for group, advantage in ((0, 1.0), (1, -1.0)):
        selected = [
            x[mask.bool()]
            for x, mask, gid in zip(xs, data["loss_masks"], data["sequence_group_ids"], strict=True)
            if gid == group
        ]
        kl = -torch.cat(selected).mean()
        expected = expected + compute_policy_loss(kl, kl.new_tensor(advantage), 3e-4, 4e-4)[0] / 2
    actual = xs[0].sum() * 0
    # Reverse the packing order and backpropagate independent microbatches.
    for i in (3, 5, 0, 2, 4, 1):
        x, mask = xs[i], data["loss_masks"][i]
        kl = compute_gspo_kl(
            [x], [torch.zeros_like(x)], [x], [mask], [data["sequence_group_kl"][i]], [data["sequence_segment_kl"][i]]
        )
        advantage = 1.0 if data["sequence_group_ids"][i] == 0 else -1.0
        pg = compute_policy_loss(kl, torch.full_like(x, advantage), 3e-4, 4e-4)[0]
        reducer = get_sum_of_sample_mean(
            [data["total_lengths"][i]], [len(x)], [mask], sample_weights=[data["sample_weights"][i]]
        )
        actual = actual + reducer(pg) / 6
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, xs, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, xs, allow_unused=True)
    for x, got, want in zip(xs, actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want if want is not None else torch.zeros_like(x))
    # Compare actual optimizer state/parameters, not just the reported loss.
    actual_params = [torch.nn.Parameter(v.detach().clone()) for v in xs]
    expected_params = [torch.nn.Parameter(v.detach().clone()) for v in xs]
    for params, grads in ((actual_params, actual_grad), (expected_params, expected_grad)):
        optimizer = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
        for parameter, gradient in zip(params, grads, strict=True):
            parameter.grad = gradient if gradient is not None else torch.zeros_like(parameter)
        optimizer.step()
    for got, want in zip(actual_params, expected_params, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_sample_weights_apply_to_kl_even_when_episode_advantage_is_zero():
    state()
    data = example()
    x = torch.cat(data["log_probs"]).requires_grad_()
    reducer = get_sum_of_sample_mean(
        data["total_lengths"], data["response_lengths"], data["loss_masks"], sample_weights=data["sample_weights"]
    )
    kl = (-x).exp() + x - 1
    actual = 0.001 * reducer(kl) / 6
    pieces = kl.split(data["response_lengths"])
    expected = sum(
        0.001
        * torch.cat(
            [
                piece[mask.bool()]
                for piece, mask, gid in zip(pieces, data["loss_masks"], data["sequence_group_ids"], strict=True)
                if gid == group
            ]
        ).mean()
        / 2
        for group in (0, 1)
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, retain_graph=True)[0], torch.autograd.grad(expected, x)[0]
    )


@pytest.mark.parametrize("shift", [0.0, 0.02, -0.02])
@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_real_loss_dispatcher_preserves_gspo_and_independent_kl(monkeypatch, shift, use_rollout_logprobs):
    from miles.backends.training_utils.loss import loss_function
    from miles.backends.training_utils.loss_hub import losses

    state()
    data = example()
    data["log_probs"] = [value + shift for value in data["log_probs"]]
    config = args(
        use_rollout_logprobs=use_rollout_logprobs,
        loss_type="policy_loss",
        calculate_per_token_loss=False,
        use_dynamic_global_batch_size=True,
        recompute_loss_function=False,
        allgather_cp=False,
        true_on_policy_mode=False,
        entropy_coef=0.0,
        use_opsm=False,
        dppo_delta=None,
        eps_clip=3e-4,
        eps_clip_high=4e-4,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        use_kl_loss=True,
        use_unbiased_kl=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.001,
    )
    prepare_sequence_groups(config, data, [6])
    xs = [value.clone().requires_grad_() for value in data["log_probs"]]
    expected = xs[0].sum() * 0
    for group, advantage in ((0, 0.0), (1, -1.0)):
        selected, old = [], []
        for i, gid in enumerate(data["sequence_group_ids"]):
            if gid == group:
                selected.append(xs[i][data["loss_masks"][i].bool()])
                old.append(
                    (data["rollout_log_probs"] if use_rollout_logprobs else data["log_probs"])[i][
                        data["loss_masks"][i].bool()
                    ]
                )
        values, old_values = torch.cat(selected), torch.cat(old)
        kl = (old_values - values).mean()
        pg = compute_policy_loss(kl, kl.new_tensor(advantage), 3e-4, 4e-4)[0]
        expected += (pg + 0.001 * ((-values).exp() + values - 1).mean()) / 2
    actual = xs[0].sum() * 0
    for i, x in enumerate(xs):
        batch = {key: [value[i]] for key, value in data.items()}
        batch.update(
            dynamic_global_batch_size=6,
            unconcat_tokens=[torch.arange(len(x) + 3)],
            advantages=[torch.full_like(x, 0.0 if data["sequence_group_ids"][i] == 0 else -1.0)],
            ref_log_probs=[torch.zeros_like(x)],
        )
        monkeypatch.setattr(losses, "get_log_probs_and_entropy", lambda logits, **kw: {"log_probs": [logits]})
        value, _, metrics = loss_function(config, batch, 6, x)
        assert "kl_loss" in metrics["keys"]
        actual += value
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual, xs)
    want = torch.autograd.grad(expected, xs, allow_unused=True)
    for x, left, right in zip(xs, got, want, strict=True):
        torch.testing.assert_close(left, right if right is not None else torch.zeros_like(x))


@pytest.mark.parametrize(
    "change",
    [{"keep_old_actor": True}, {"ppo_epochs": 2}, {"hidden_dropout": 0.1}, {"compute_advantages_and_returns": False}],
)
def test_prepass_contract_is_enforced(change):
    state()
    with pytest.raises(ValueError):
        prepare_sequence_groups(args(**change), example(), [6])


def test_multiple_optimizer_steps_and_forward_drift_are_rejected():
    state()
    data = example()
    with pytest.raises(ValueError):
        prepare_sequence_groups(args(), data, [3, 3])
    prepare_sequence_groups(args(), data, [6])
    x = data["log_probs"][0] + 0.01
    with pytest.raises(RuntimeError, match="forward drift"):
        compute_gspo_kl(
            [x],
            [torch.zeros_like(x)],
            [x],
            [data["loss_masks"][0]],
            data["sequence_group_kl"][:1],
            data["sequence_segment_kl"][:1],
        )


def distributed_worker(rank, url):
    dist.init_process_group("gloo", init_method=url, rank=rank, world_size=4)
    try:
        cp_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        dp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        cp_rank, dp_rank = rank % 2, rank // 2
        state(GroupInfo(cp_rank, 2, cp_groups[dp_rank]), GroupInfo(dp_rank, 2, dp_groups[cp_rank]))
        full = example()
        # Both episodes cross DP ranks, and some CP ranks own no active token.
        indices = ([0, 2, 4], [1, 3, 5])[dp_rank]
        data = {key: [values[i] for i in indices] for key, values in full.items()}
        for key in ("log_probs", "rollout_log_probs"):
            data[key] = [
                slice_log_prob_with_cp(value, total, response, "thd")
                for value, total, response in zip(
                    data[key], data["total_lengths"], data["response_lengths"], strict=True
                )
            ]
        prepare_sequence_groups(args(), data, [3])
        for got in data["sequence_group_kl"]:
            torch.testing.assert_close(got, got.new_zeros(()), atol=1e-12, rtol=0)
        # Off-policy nonzero episode statistics as well as the opposite-ratio case.
        changed = deepcopy(data)
        changed["log_probs"] = [value + 0.02 for value in changed["log_probs"]]
        prepare_sequence_groups(args(), changed, [3])
        for group, got in zip(changed["sequence_group_ids"], changed["sequence_group_kl"], strict=True):
            torch.testing.assert_close(got, got.new_tensor(-0.02 if group >= 0 else 0.0), atol=1e-12, rtol=0)
        xs = [value.detach().clone().requires_grad_() for value in data["log_probs"]]
        full_x = [
            all_gather_with_cp(x, total, response)
            for x, total, response in zip(xs, data["total_lengths"], data["response_lengths"], strict=True)
        ]
        old = [full["rollout_log_probs"][i] for i in indices]
        kl = compute_gspo_kl(
            full_x, old, xs, data["loss_masks"], data["sequence_group_kl"], data["sequence_segment_kl"]
        )
        advantages = torch.cat(
            [
                torch.full_like(x, 1.0 if gid == 0 else -1.0)
                for x, gid in zip(xs, data["sequence_group_ids"], strict=True)
            ]
        )
        pg = compute_policy_loss(kl, advantages, 3e-4, 4e-4)[0]
        reducer = get_sum_of_sample_mean(
            data["total_lengths"], data["response_lengths"], data["loss_masks"], sample_weights=data["sample_weights"]
        )
        loss = reducer(pg) / 6
        # Backward must visit CP collectives even on ranks whose local tensor
        # is empty; autograd.grad prunes those paths when requesting only xs.
        loss.backward()
        grads = [x.grad for x in xs]
        for i, x, gradient in zip(indices, xs, grads, strict=True):
            gid = full["sequence_group_ids"][i]
            coefficient = -1 / 6 if gid == 0 else 1 / 8 if gid == 1 else 0.0
            expected = slice_log_prob_with_cp(
                full["loss_masks"][i].double() * coefficient,
                full["total_lengths"][i],
                full["response_lengths"][i],
                "thd",
            )
            if gradient is None:
                assert x.numel() == 0
                gradient = torch.zeros_like(x)
            torch.testing.assert_close(gradient, expected)
    finally:
        dist.destroy_process_group()


def test_statistics_cross_dp_and_cp(tmp_path):
    mp.spawn(distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)
