"""Episode-level GSPO statistics for segments spread over microbatches and DP."""

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks
from miles.backends.training_utils.loss.hub.math_utils import _safe_exp_neg_ppo_kl
from miles.backends.training_utils.parallel import get_parallel_state


def _preupdate_metrics(groups, means, *, eps_clip, eps_clip_high, use_rollout_logprobs):
    """Equal-episode summaries of already CP/DP-reduced pre-update statistics."""
    active = groups[:, 1] > 0
    positive, negative = groups[:, 2] > 0, groups[:, 3] > 0
    ratios = _safe_exp_neg_ppo_kl(means)
    metrics = {
        "episodes": active.sum(),
        "effective_tokens": groups[:, 1].sum(),
        "zero_adv_episodes": (active & ~positive & ~negative).sum(),
        "mixed_adv_episodes": (active & positive & negative).sum(),
        "ratio_saturated_episodes": (active & (means.abs() > 20)).sum(),
        "uses_rollout_old": means.new_tensor(float(use_rollout_logprobs)),
    }
    for sign, selected in (("pos", active & positive & ~negative), ("neg", active & negative & ~positive)):
        values = ratios[selected]
        zero = ratios.new_zeros(())
        quantiles = torch.quantile(values, values.new_tensor([0.05, 0.5, 0.95])) if values.numel() else [zero] * 3
        below = (values < 1 - eps_clip).float().mean() if values.numel() else zero
        above = (values > 1 + eps_clip_high).float().mean() if values.numel() else zero
        metrics.update({
            f"{sign}_episodes": selected.sum(),
            f"{sign}_ratio_mean": values.mean() if values.numel() else zero,
            f"{sign}_ratio_p05": quantiles[0],
            f"{sign}_ratio_p50": quantiles[1],
            f"{sign}_ratio_p95": quantiles[2],
            f"{sign}_below_clip_fraction": below,
            f"{sign}_above_clip_fraction": above,
            f"{sign}_pg_clip_fraction": above if sign == "pos" else below,
        })
    # Every participating rank has the same global scalars. The existing
    # rollout logger's scalar mean preserves them without counting CP twice.
    return {f"gspo_preupdate/{key}": value.detach() for key, value in metrics.items()}


@torch.no_grad()
def prepare_sequence_groups(args, data, num_microbatches):
    if "sequence_group_ids" not in data:
        return
    if (
        args.advantage_estimator != "gspo"
        or len(num_microbatches) != 1
        or getattr(args, "ppo_epochs", 1) != 1
        or args.keep_old_actor
        or getattr(args, "skip_actor_forward_only", False)
        or not args.compute_advantages_and_returns
        or getattr(args, "attention_dropout", 0)
        or getattr(args, "hidden_dropout", 0)
    ):
        raise ValueError("Grouped GSPO requires one optimizer step, current-actor scoring and zero dropout")
    parallel = get_parallel_state()
    if not parallel.is_pp_last_stage:
        return
    current = data.get("log_probs")
    if not current or "sample_weights" not in data:
        raise ValueError("Grouped GSPO requires pre-update log_probs and sample_weights")
    old = data["rollout_log_probs"] if args.use_rollout_logprobs else current
    masks = get_local_response_loss_masks(
        data["total_lengths"], data["response_lengths"], data["loss_masks"], args.qkv_format, data.get("max_seq_lens")
    )
    ids = data["sequence_group_ids"]
    # IDs are dense batch-local episode indices; -1 denotes masked padding.
    if any(i < -1 or i >= args.global_batch_size for i in ids):
        raise ValueError("Invalid batch-local sequence_group_ids")
    advantages = data.get("advantages")
    rows = []
    for index, (group, new, behavior, mask) in enumerate(zip(ids, current, old, masks, strict=True)):
        active = mask.to(device=new.device, dtype=torch.bool)
        if group == -1 and active.any():
            raise ValueError("Padding sequence group contains trainable tokens")
        delta = (behavior - new).double()[active]
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite active GSPO log probabilities")
        row = [delta.sum(), active.sum().double()]
        if advantages is not None:
            advantage = advantages[index].to(device=new.device)[active]
            row.extend(((advantage > 0).sum().double(), (advantage < 0).sum().double()))
        rows.append(torch.stack(row))
    segments = torch.stack(rows)
    if parallel.cp.size > 1:
        dist.all_reduce(segments, group=parallel.cp.group)
    groups = segments.new_zeros((args.global_batch_size, segments.shape[1]))
    for group, row in zip(ids, segments, strict=True):
        if group >= 0:
            groups[group] += row
    if parallel.intra_dp.size > 1:
        dist.all_reduce(groups, group=parallel.intra_dp.group)
    means = groups[:, 0] / groups[:, 1].clamp_min(1)
    data["sequence_group_kl"] = [means[i] if i >= 0 else means.new_zeros(()) for i in ids]
    data["sequence_segment_kl"] = list((segments[:, 0] / segments[:, 1].clamp_min(1)).unbind())
    if advantages is not None:
        data.update(_preupdate_metrics(
            groups, means, eps_clip=args.eps_clip, eps_clip_high=args.eps_clip_high,
            use_rollout_logprobs=args.use_rollout_logprobs,
        ))
