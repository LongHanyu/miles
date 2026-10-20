"""Opt-in Megatron adapter for memory-bounded FP32 sampled logprobs."""

import math
from functools import partial

import torch


def chunked_head_kwargs(args):
    size = getattr(args, "chunked_lm_head_size", 0)
    if size == 0:
        return {}
    if not isinstance(size, int) or size < 0:
        raise ValueError("chunked_lm_head_size must be nonnegative")
    if not math.isfinite(args.rollout_temperature) or args.rollout_temperature <= 0:
        raise ValueError("Chunked LM head requires a positive finite rollout temperature")
    unsupported = {
        # Direct weight access bypasses DDP's output-layer parameter-sync hook.
        "overlap_param_gather": getattr(args, "overlap_param_gather", False),
        "context parallel": getattr(args, "context_parallel_size", 1) != 1,
        "pipeline parallel": getattr(args, "pipeline_model_parallel_size", 1) != 1,
        "non-policy loss": getattr(args, "loss_type", "policy_loss") != "policy_loss",
        "entropy": (
            getattr(args, "entropy_coef", 0) != 0
            or getattr(args, "use_rollout_entropy", False)
            or getattr(args, "observe_training_entropy", False)
        ),
        "sampling support replay": getattr(args, "use_sampling_support_replay", False),
        "non-Megatron backend": getattr(args, "train_backend", "megatron") != "megatron",
        "critic": getattr(args, "use_critic", False),
        "witness": getattr(args, "enable_witness", False),
        "multi-LoRA": getattr(args, "multi_lora", False),
        "FP16": getattr(args, "fp16", False),
        "CUDA graphs": bool(getattr(args, "cuda_graph_impl", "none") != "none"),
        "MuP": getattr(args, "use_mup", False),
        "OPD": getattr(args, "use_opd", False),
        "MTP": getattr(args, "enable_mtp_training", False) or bool(getattr(args, "mtp_num_layers", None)),
        "true-on-policy": getattr(args, "true_on_policy_mode", False),
        "non-packed layout": getattr(args, "qkv_format", "thd") != "thd",
        "tied embeddings": not getattr(args, "untie_embeddings_and_output_weights", False),
        "FP8/FP4": bool(getattr(args, "fp8", None)) or bool(getattr(args, "fp4", None)),
        "LoRA": bool(getattr(args, "lora_rank", 0)),
        "deferred weight gradient": getattr(args, "defer_embedding_wgrad_compute", False),
    }
    if any(unsupported.values()):
        raise ValueError("Chunked LM head does not yet support: " + ", ".join(k for k, v in unsupported.items() if v))
    return {"output_processor": partial(_process_output, chunk_size=size, temperature=args.rollout_temperature)}


def _process_output(
    *,
    hidden_states,
    output_layer,
    output_weight,
    input_ids,
    labels,
    inference_context,
    config,
    chunk_size,
    temperature,
    **kwargs,
):
    """Use Megatron's public projection hook without changing model state."""
    if labels is not None or inference_context is not None or config.mtp_num_layers:
        raise ValueError("Chunked LM head requires labels=None, no inference context and no MTP")
    if getattr(config, "use_mup", False) or getattr(config, "defer_embedding_wgrad_compute", False):
        raise ValueError("Chunked LM head does not support MuP or deferred weight gradients")
    return _project(
        hidden_states,
        output_layer,
        output_weight,
        input_ids,
        chunk_size=chunk_size,
        temperature=temperature,
    )


def _project(hidden, output_layer, output_weight, input_ids, *, chunk_size, temperature):
    from megatron.core.tensor_parallel.mappings import (
        copy_to_tensor_model_parallel_region,
        gather_from_sequence_parallel_region,
    )

    from miles.backends.training_utils.chunked_lm_head import chunked_logprobs

    if output_layer.bias is not None or output_weight is not None:
        raise ValueError("Chunked LM head requires an untied, bias-free output layer")
    weight = output_layer.weight
    if hidden.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Chunked LM head requires BF16 or FP32 hidden states")
    if weight.dtype != torch.float32:
        raise ValueError("Chunked LM head requires FP32 output weights")
    # Match the FP32 head's input cast BEFORE TP/SP communication. Backward
    # must sum FP32 shard contributions before casting back to the backbone
    # dtype; reducing already-rounded BF16 contributions changes its gradient.
    hidden = hidden.to(weight.dtype)
    group = output_layer.tp_group
    if output_layer.sequence_parallel:
        # Gather H, not logits; backward reduce-scatters the TP contributions.
        hidden = gather_from_sequence_parallel_region(hidden, group=group, tensor_parallel_output_grad=True)
    else:
        hidden = copy_to_tensor_model_parallel_region(hidden, group=group)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or hidden.shape[:2] != (input_ids.shape[1], 1):
        raise ValueError("Chunked LM head requires one packed token stream [1, T]")
    # Logit at position i predicts token i+1. get_responses still applies the
    # existing per-sample boundaries and excludes terminal/padding positions.
    targets = input_ids.reshape(-1).roll(-1)
    result = chunked_logprobs(
        hidden[:, 0],
        weight,
        targets,
        chunk_size,
        group=group,
        temperature=temperature,
        backward_dtype=torch.bfloat16,
    )
    return result.reshape(1, -1, 1)
