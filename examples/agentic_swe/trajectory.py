"""Append-only token ledger for one agent conversation.

The CLI replays the whole history every turn, rewritten into its own shape, so a
render is only recorded when it extends the tokens already held.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from miles.utils.types import Sample


def _decode_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode OpenAI argument strings into mappings expected by the Qwen3.5 template."""
    normalized = copy.deepcopy(messages)
    for message in normalized:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            function["arguments"] = json.loads(function["arguments"])
    return normalized


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[int]:
    """Render one turn's history to prompt token ids.

    ``tools`` lands in the system prompt, so it has to be the harness's own list.
    """
    # tokenize=True returns a BatchEncoding here, a plain list in other versions.
    text = tokenizer.apply_chat_template(
        _decode_tool_arguments(messages),
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=False,
    )
    # The template already wrote every special token.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@dataclass
class Trajectory:
    """The main agent conversation's token ledger."""

    token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)

    def extends(self, prompt_ids: list[int]) -> bool:
        # An empty ledger extends anything: the first request opens the conversation.
        return prompt_ids[: len(self.token_ids)] == self.token_ids

    def append_turn(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        completion_log_probs: list[float],
    ) -> None:
        """Append the observation tail (mask 0) and the sampled completion (mask 1)."""
        tail = prompt_ids[len(self.token_ids) :]
        self.token_ids.extend(tail + completion_ids)
        self.loss_mask.extend([0] * len(tail) + [1] * len(completion_ids))
        self.log_probs.extend([0.0] * len(tail) + completion_log_probs)

    def to_sample(
        self,
        base_sample: Sample,
        *,
        reward: float,
    ) -> Sample | None:
        """Project the ledger onto a miles ``Sample``; ``None`` if nothing is trainable."""
        if 1 not in self.loss_mask:
            return None
        first_trainable = self.loss_mask.index(1)
        sample = base_sample
        sample.tokens = list(self.token_ids)
        # miles asserts len(loss_mask) == response_length, counted from here on.
        sample.response_length = len(self.token_ids) - first_trainable
        sample.loss_mask = self.loss_mask[first_trainable:]
        sample.rollout_log_probs = self.log_probs[first_trainable:]
        sample.reward = float(reward)
        sample.status = Sample.Status.COMPLETED
        return sample
