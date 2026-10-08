"""Episode-level GSPO statistics for segments spread over microbatches and DP."""

import torch
import torch.distributed as dist

from .cp_utils import get_local_response_loss_masks
from .parallel import get_parallel_state


@torch.no_grad()
def prepare_sequence_groups(args, data, num_microbatches):
    if "sequence_group_ids" not in data:
        return
    if (
        args.advantage_estimator != "gspo"
        or len(num_microbatches) != 1
        or getattr(args, "ppo_epochs", 1) != 1
        or args.keep_old_actor
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
    rows = []
    for group, new, behavior, mask in zip(ids, current, old, masks, strict=True):
        active = mask.to(device=new.device, dtype=torch.bool)
        if group == -1 and active.any():
            raise ValueError("Padding sequence group contains trainable tokens")
        delta = (behavior - new).double()[active]
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite active GSPO log probabilities")
        rows.append(torch.stack((delta.sum(), active.sum().double())))
    segments = torch.stack(rows)
    if parallel.cp.size > 1:
        dist.all_reduce(segments, group=parallel.cp.group)
    groups = segments.new_zeros((args.global_batch_size, 2))
    for group, row in zip(ids, segments, strict=True):
        if group >= 0:
            groups[group] += row
    if parallel.intra_dp.size > 1:
        dist.all_reduce(groups, group=parallel.intra_dp.group)
    means = groups[:, 0] / groups[:, 1].clamp_min(1)
    data["sequence_group_kl"] = [means[i] if i >= 0 else means.new_zeros(()) for i in ids]
    data["sequence_segment_kl"] = list((segments[:, 0] / segments[:, 1].clamp_min(1)).unbind())
