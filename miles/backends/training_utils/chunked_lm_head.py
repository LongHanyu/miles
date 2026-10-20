"""Recomputed, vocabulary-parallel sampled logprobs without full-sequence logits.

This is a projection primitive, NOT a replacement policy loss. Only hidden states,
weights and targets survive forward. Backward reconstructs one vocabulary tile at
a time. TP reduction of hidden-state gradients is the caller's responsibility.
"""

import math

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


def _distribution(hidden, weight, targets, group, temperature):
    logits = hidden.to(weight.dtype) @ weight.t()
    logits.div_(temperature)
    world = dist.get_world_size(group) if group is not None else 1
    rank = dist.get_rank(group) if group is not None else 0
    maximum = logits.amax(-1)
    if world > 1:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    logits.sub_(maximum[:, None])
    local = targets - rank * weight.shape[0]
    owned = (local >= 0) & (local < weight.shape[0])
    local = local.clamp(0, weight.shape[0] - 1)
    rows = torch.arange(targets.numel(), device=targets.device)
    selected = logits[rows, local].masked_fill(~owned, 0)
    logits.exp_()
    denominator = logits.sum(-1)
    if world > 1:
        dist.all_reduce(selected, group=group)
        dist.all_reduce(denominator, group=group)
    logprobs = selected - denominator.log()
    logits.div_(denominator[:, None])
    return logprobs, logits, rows, local, owned


class _ChunkedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size, group, temperature, backward_dtype):
        if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
            raise ValueError("Expected hidden [tokens, hidden] and weight [local_vocab, hidden]")
        if targets.shape != hidden.shape[:1] or targets.dtype != torch.long:
            raise ValueError("Targets must be one int64 token per hidden-state row")
        if not isinstance(chunk_size, int) or chunk_size <= 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("chunk_size and temperature must be positive")
        if weight.dtype not in (torch.float32, torch.float64):
            raise ValueError("Chunked LM head requires FP32 weights (FP64 is supported for tests)")
        if torch.is_autocast_enabled(hidden.device.type):
            raise ValueError("Chunked FP32 LM head cannot run under autocast")
        if hidden.device != weight.device or hidden.device != targets.device:
            raise ValueError("Hidden states, weights and targets must share a device")
        world = dist.get_world_size(group) if group is not None else 1
        if targets.numel() and ((targets < 0).any() or (targets >= world * weight.shape[0]).any()):
            raise ValueError("Target outside vocabulary")
        ctx.save_for_backward(hidden, weight, targets)
        ctx.options = chunk_size, group, temperature, backward_dtype
        output = weight.new_empty(targets.shape)
        for start in range(0, len(targets), chunk_size):
            end = start + chunk_size
            output[start:end] = _distribution(hidden[start:end], weight, targets[start:end], group, temperature)[0]
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        chunk_size, group, temperature, backward_dtype = ctx.options
        dh = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        for start in range(0, len(targets), chunk_size):
            end = start + chunk_size
            h = hidden[start:end].to(weight.dtype)
            _, dlogits, rows, local, owned = _distribution(h, weight, targets[start:end], group, temperature)
            dlogits[rows, local] -= owned.to(dlogits.dtype)
            dlogits.mul_(-grad_output[start:end, None])
            # Preserve the existing Megatron fused CE backward's BF16 rounding
            # when requested; this is deliberately separate from head precision.
            if backward_dtype is not None:
                dlogits = dlogits.to(backward_dtype).to(weight.dtype)
            dlogits.div_(temperature)
            if dh is not None:
                dh[start:end] = dlogits @ weight
            if dw is not None:
                dw.addmm_(dlogits.t(), h)
            del dlogits, h
        return dh, dw, None, None, None, None, None


def chunked_logprobs(
    hidden,
    weight,
    targets,
    chunk_size,
    *,
    group=None,
    temperature=1.0,
    backward_dtype=None,
):
    """Return one sampled logprob per row; replicated targets/order on TP ranks.

    For TP, hidden gradients are LOCAL shard contributions; wrap hidden in a
    TP-copy or SP-gather autograd mapping before calling. Weight gradients remain
    local shards and flow through the normal parameter/DDP accumulation hook.
    """
    return _ChunkedLogprobs.apply(hidden, weight, targets, chunk_size, group, temperature, backward_dtype)
