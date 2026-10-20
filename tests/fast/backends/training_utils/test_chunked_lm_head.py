"""Standalone CPU regressions; no Ray/model/sandbox initialization required."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[4]
MEGATRON_ROOT = Path(os.environ.get("MEGATRON_ROOT", ROOT.parent / "Megatron-LM"))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


primitive = load_file("chunked_head", ROOT / "miles/backends/training_utils/chunked_lm_head.py")
adapter = load_file("chunked_head_adapter", ROOT / "miles/backends/megatron_utils/chunked_lm_head.py")


def load_existing_function(name, filename, **namespace):
    """Exercise the exact repository function without importing the GPU stack."""
    tree = ast.parse(filename.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    scope = {"torch": torch, **namespace}
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), scope)
    return scope[name]


dppo = load_existing_function("compute_dppo_loss", ROOT / "miles/backends/training_utils/loss/hub/math_utils.py")


@pytest.mark.parametrize("chunk", [1, 4, 32])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_logprobs_loss_gradients_and_one_optimizer_step(chunk, temperature):
    torch.manual_seed(42)
    h = torch.randn(13, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    hr = h.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    targets = torch.randint(11, (13,))
    reference = ((hr @ wr.t()) / temperature).log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    actual = primitive.chunked_logprobs(h, w, targets, chunk, temperature=temperature)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    old = reference.detach() + torch.linspace(-2, 2, 13)
    advantages = torch.linspace(-1, 1, 13)
    mask = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=torch.float64)
    a, blocked, _ = dppo(old, actual, advantages, 0.1, "tv")
    b, blocked_ref, _ = dppo(old, reference, advantages, 0.1, "tv")
    torch.testing.assert_close(blocked, blocked_ref)
    ((a * mask).sum() / mask.sum()).backward()
    ((b * mask).sum() / mask.sum()).backward()
    torch.testing.assert_close(h.grad, hr.grad, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(w.grad, wr.grad, atol=1e-12, rtol=1e-11)
    torch.optim.AdamW([h, w], lr=1e-6).step()
    torch.optim.AdamW([hr, wr], lr=1e-6).step()
    torch.testing.assert_close(h, hr, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(w, wr, atol=1e-12, rtol=1e-11)


def test_gradcheck_and_no_saved_vocab_activations():
    torch.manual_seed(2)
    h = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(19, 3, dtype=torch.float64, requires_grad=True)
    t = torch.arange(7)
    assert torch.autograd.gradcheck(lambda a, b: primitive.chunked_logprobs(a, b, t, 3), (h, w))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(tuple(x.shape)) or x, lambda x: x):
        primitive.chunked_logprobs(h, w, t, 3).sum().backward()
    assert saved == [(7, 3), (19, 3), (7,)]


def test_existing_bf16_backward_rounding():
    torch.manual_seed(5)
    h = torch.randn(9, 4, requires_grad=True)
    w = torch.randn(17, 4, requires_grad=True)
    t = torch.arange(9)
    upstream = torch.randn(9)
    logits = h.detach() @ w.detach().t()
    grad = logits.softmax(-1)
    grad[torch.arange(9), t] -= 1
    grad = (grad * -upstream[:, None]).to(torch.bfloat16).float()
    primitive.chunked_logprobs(h, w, t, 3, backward_dtype=torch.bfloat16).backward(upstream)
    torch.testing.assert_close(h.grad, grad @ w.detach(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(w.grad, grad.t() @ h.detach(), atol=1e-6, rtol=1e-5)


def test_response_boundaries_use_existing_slicer():
    state = SimpleNamespace(cp=SimpleNamespace(size=1))
    chunks = load_existing_function(
        "_iter_response_chunks",
        ROOT / "miles/backends/training_utils/loss/hub/logit_processors.py",
        get_parallel_state=lambda: state,
    )
    slicer = load_existing_function(
        "get_responses",
        ROOT / "miles/backends/training_utils/loss/hub/logit_processors.py",
        get_parallel_state=lambda: state,
        _iter_response_chunks=chunks,
    )
    args = SimpleNamespace(qkv_format="thd", true_on_policy_mode=False, rollout_temperature=0.7)
    values = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    # Two packed samples, prompt=2 and prompt=1; extra positions are padding.
    result = list(
        slicer(
            values,
            args=args,
            unconcat_tokens=[torch.arange(5), torch.arange(4)],
            total_lengths=[5, 4],
            response_lengths=[3, 3],
        )
    )
    assert [x[:, 0].tolist() for x, _ in result] == [[1, 2, 3], [5, 6, 7]]


@pytest.mark.parametrize(
    "overrides",
    [
        {"entropy_coef": 0.1},
        {"context_parallel_size": 2},
        {"pipeline_model_parallel_size": 2},
        {"true_on_policy_mode": True},
        {"untie_embeddings_and_output_weights": False},
        {"lora_rank": 8},
        {"use_rollout_entropy": True},
        {"observe_training_entropy": True},
        {"use_sampling_support_replay": True},
        {"multi_lora": True},
        {"enable_mtp_training": True},
        {"mtp_num_layers": 1},
        {"loss_type": "sft_loss"},
        {"qkv_format": "bshd"},
        {"fp8": "hybrid"},
        {"fp4": "nvfp4"},
        {"defer_embedding_wgrad_compute": True},
        {"train_backend": "fsdp"},
        {"use_critic": True},
        {"enable_witness": True},
        {"use_opd": True},
        {"fp16": True},
        {"cuda_graph_impl": "local"},
        {"use_mup": True},
        {"chunked_lm_head_size": -1},
        {"rollout_temperature": 0},
        {"rollout_temperature": float("nan")},
    ],
)
def test_unsupported_fails_closed(overrides):
    settings = dict(
        chunked_lm_head_size=4,
        rollout_temperature=1.0,
        untie_embeddings_and_output_weights=True,
    )
    settings.update(overrides)
    with pytest.raises(ValueError):
        adapter.chunked_head_kwargs(SimpleNamespace(**settings))


def test_disabled_is_unchanged():
    assert adapter.chunked_head_kwargs(SimpleNamespace()) == {}


def test_bf16_hidden_and_masked_padding():
    torch.manual_seed(17)
    h = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(23, 4, requires_grad=True)
    out = primitive.chunked_logprobs(h, w, torch.arange(8), 3)
    out[:5].sum().backward()
    assert h.grad.dtype == torch.bfloat16
    assert h.grad[5:].count_nonzero() == 0
    assert torch.isfinite(w.grad).all()


def test_output_processor_contract():
    settings = SimpleNamespace(
        chunked_lm_head_size=4, rollout_temperature=1.0, untie_embeddings_and_output_weights=True
    )
    callback = adapter.chunked_head_kwargs(settings)["output_processor"]
    hidden, head, ids = object(), object(), object()
    config = SimpleNamespace(mtp_num_layers=None)
    with patch.object(adapter, "_project", return_value="result") as project:
        assert (
            callback(
                hidden_states=hidden,
                output_layer=head,
                output_weight=None,
                input_ids=ids,
                labels=None,
                inference_context=None,
                config=config,
            )
            == "result"
        )
        project.assert_called_once_with(hidden, head, None, ids, chunk_size=4, temperature=1.0)
    with pytest.raises(ValueError):
        callback(
            hidden_states=hidden,
            output_layer=head,
            output_weight=None,
            input_ids=ids,
            labels=ids,
            inference_context=None,
            config=config,
        )


def _tp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, world_size=2, rank=rank)
    try:
        torch.manual_seed(11)
        h = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
        full = torch.randn(14, 4, dtype=torch.float64, requires_grad=True)
        w = full.detach().chunk(2)[rank].clone().requires_grad_()
        t = torch.tensor([0, 13, 6, 7, 1, 11, 4, 8, 2])
        actual = primitive.chunked_logprobs(h, w, t, 4, group=dist.group.WORLD)
        hr = h.detach().clone().requires_grad_()
        ref = (hr @ full.t()).log_softmax(-1).gather(1, t[:, None]).squeeze(1)
        torch.testing.assert_close(actual, ref)
        actual.square().sum().backward()
        ref.square().sum().backward()
        dist.all_reduce(h.grad)  # Same sum that the TP-copy/SP-gather backward performs.
        torch.testing.assert_close(h.grad, hr.grad)
        torch.testing.assert_close(w.grad, full.grad.chunk(2)[rank])

        # Exercise Megatron's mapping math with Gloo. Its allocation device is
        # hard-coded to CUDA: redirect ONLY that allocation selector to CPU.
        # This is not an NCCL/GPU integration test.
        sys.path[:0] = [str(ROOT), str(MEGATRON_ROOT)]
        for hidden_dtype in (torch.float32, torch.bfloat16):
            for sequence_parallel in (False, True):
                torch.manual_seed(13)
                hidden_full = torch.randn(10, 1, 4, dtype=hidden_dtype, requires_grad=True)
                weight_full = torch.randn(14, 4, requires_grad=True)
                ids = torch.arange(10).reshape(1, 10)
                hidden = (
                    (hidden_full.detach().chunk(2)[rank] if sequence_parallel else hidden_full.detach())
                    .clone()
                    .requires_grad_()
                )
                weight = weight_full.detach().chunk(2)[rank].clone().requires_grad_()
                layer = SimpleNamespace(
                    bias=None,
                    weight=weight,
                    tp_group=dist.group.WORLD,
                    sequence_parallel=sequence_parallel,
                )
                with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
                    actual = adapter._project(hidden, layer, None, ids, chunk_size=4, temperature=1.0)
                    dense = hidden_full[:, 0].float() @ weight_full.t()
                    # Existing fused CE rounds the logit gradient through BF16.
                    dense.register_hook(lambda g: g.to(torch.bfloat16).float())
                    ref = dense.log_softmax(-1).gather(1, ids.reshape(-1).roll(-1)[:, None]).reshape(1, -1, 1)
                    torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-6)
                    actual.sum().backward()
                    ref.sum().backward()
                expected = hidden_full.grad.chunk(2)[rank] if sequence_parallel else hidden_full.grad
                torch.testing.assert_close(hidden.grad, expected, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(weight.grad, weight_full.grad.chunk(2)[rank], atol=2e-5, rtol=2e-5)
    finally:
        dist.destroy_process_group()


def test_two_rank_vocabulary_parallel(tmp_path):
    mp.spawn(_tp_worker, args=(f"file://{tmp_path}/tp-rendezvous",), nprocs=2, join=True)


def test_precomputed_logprobs_reuse_slicing_without_rescaling():
    path = ROOT / "miles/backends/training_utils/loss/hub/logit_processors.py"
    state = SimpleNamespace(cp=SimpleNamespace(size=1))
    chunks = load_existing_function("_iter_response_chunks", path, get_parallel_state=lambda: state)
    slicer = load_existing_function("get_responses", path, _iter_response_chunks=chunks)
    process = load_existing_function("get_log_probs_and_entropy", path, get_responses=slicer)
    args = SimpleNamespace(
        chunked_lm_head_size=4, qkv_format="thd", true_on_policy_mode=False, rollout_temperature=0.7
    )
    values = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1).requires_grad_()
    kwargs = dict(
        args=args, unconcat_tokens=[torch.arange(5), torch.arange(4)], total_lengths=[5, 4], response_lengths=[3, 3]
    )
    result = process(values, **kwargs)
    assert [v.tolist() for v in result["log_probs"]] == [[1, 2, 3], [5, 6, 7]]
    sum(v.sum() for v in result["log_probs"]).backward()
    assert values.grad.flatten().tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        process(values, with_entropy=True, **kwargs)
    with pytest.raises(ValueError):
        process(values, rollout_sampling_mask=[], **kwargs)


@pytest.mark.parametrize("dp_size", [2, 4])
def test_parameter_gather_overlap_rejected_before_projection(dp_size):
    settings = SimpleNamespace(
        chunked_lm_head_size=1024, rollout_temperature=1.0,
        untie_embeddings_and_output_weights=True, use_distributed_optimizer=True,
        overlap_param_gather=True, data_parallel_size=dp_size,
    )
    with patch.object(adapter, "_project") as project:
        with pytest.raises(ValueError, match="overlap_param_gather"):
            adapter.chunked_head_kwargs(settings)
        project.assert_not_called()
    # The ordinary model forward still owns synchronization when chunking is off.
    settings.chunked_lm_head_size = 0
    assert adapter.chunked_head_kwargs(settings) == {}
    # Distributed optimizer without overlapping parameter gathers stays supported.
    settings.chunked_lm_head_size = 1024
    settings.overlap_param_gather = False
    assert adapter.chunked_head_kwargs(settings)
