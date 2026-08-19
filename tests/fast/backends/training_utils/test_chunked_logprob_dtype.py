from argparse import Namespace
from types import SimpleNamespace

import torch

from miles.backends.training_utils.loss_hub import logit_processors, math_utils


def test_get_responses_accepts_bf16_policy_logits(monkeypatch):
    monkeypatch.setattr(
        logit_processors,
        "get_parallel_state",
        lambda: SimpleNamespace(cp=SimpleNamespace(size=1)),
    )
    logits = torch.zeros((1, 6, 4), dtype=torch.bfloat16)
    tokens = torch.arange(6)
    args = Namespace(
        qkv_format="thd",
        true_on_policy_mode=False,
        rollout_temperature=1.0,
        allgather_cp=False,
    )

    [(response_logits, response_tokens)] = list(
        logit_processors.get_responses(
            logits,
            args=args,
            unconcat_tokens=[tokens],
            total_lengths=[6],
            response_lengths=[3],
        )
    )

    assert response_logits.dtype == torch.bfloat16
    assert response_logits.shape == (3, 4)
    assert response_tokens.tolist() == [3, 4, 5]


def test_chunked_logprob_math_upcasts_each_bf16_chunk(monkeypatch):
    seen = []

    def fake_compute_log_probs(logits, tokens, process_group):
        seen.append((logits.dtype, logits.shape[0]))
        return torch.zeros(tokens.shape[0], dtype=torch.float32)

    monkeypatch.setattr(math_utils, "compute_log_probs", fake_compute_log_probs)
    logits = torch.zeros((5, 7), dtype=torch.bfloat16)
    tokens = torch.arange(5)

    log_probs, entropy = math_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        None,
        chunk_size=2,
    )

    assert seen == [(torch.float32, 2), (torch.float32, 2), (torch.float32, 1)]
    assert log_probs.shape == (5,)
    assert entropy is None
