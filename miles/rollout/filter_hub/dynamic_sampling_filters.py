import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted"]


def _episode_reward(args, sample):
    """An episode that yields several samples (--generate-multi-samples) carries one reward on each."""
    return (sample[0] if isinstance(sample, list) else sample).get_reward_value(args)


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [_episode_reward(args, sample) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def _flatten_samples(samples):
    """Flatten samples that may contain nested lists (from --generate-multi-samples)."""
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(s.status == Sample.Status.ABORTED for s in _flatten_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def check_passrate(args, samples: list[Sample], **kwargs):
    """Keep groups only when passrate falls between the configured thresholds."""
    rewards = [_episode_reward(args, sample) for sample in samples]
    passrate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
    threshold_low = args.passrate_threshold_low
    threshold_high = args.passrate_threshold_high + 1e-5
    keep = threshold_low < passrate < threshold_high
    print("[dynamic filter] passrate: %.3f, threshold_low: %.3f, threshold_high: %.3f, keep: %s" % (passrate, threshold_low, threshold_high, keep))
    return DynamicFilterOutput(keep=keep, reason=None if keep else f"passrate_{passrate:.3f}")