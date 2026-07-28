"""Token-in/token-out ledgers for training an unmodified agent CLI.

An agent CLI is a stateless HTTP client: every turn it replays the whole
conversation, rewritten into its own shape. Training needs the tokens the model
sampled, so we keep our own ledger and only ever extend it:

  1. render the incoming history to prompt tokens (tool schemas included);
  2. require the ledger to be a strict prefix of that render;
  3. append the new tail (tool results, user turns) with ``loss_mask=0`` and the
     sampled completion with ``loss_mask=1``.

Step 2 is the whole design. Whatever the harness did to the history, if the
render still extends what we hold, the sampled tokens are in there verbatim; if
it does not, this request belongs to another conversation (a sub-agent, an
auxiliary call) or something changed underneath us, and either way it must not
be appended. See the README for what that check has measured.
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

    ``tools`` must be the schema list the harness sent: it lands in the system
    prompt, so dropping it changes the prefix. A history that will not render
    raises; the proxy answers that request with a 500 rather than inventing a
    prompt the ledger could not reproduce.
    """
    # Rendered to text and tokenized separately on purpose: apply_chat_template's
    # tokenize=True returns a BatchEncoding in this transformers version and a
    # plain list in others, and the ids feed SGLang directly.
    text = tokenizer.apply_chat_template(
        _decode_tool_arguments(messages),
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=False,
    )
    # add_special_tokens=False: the template already wrote every special token.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@dataclass
class Trajectory:
    """The main agent conversation's token ledger."""

    token_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)

    def extends(self, prompt_ids: list[int]) -> bool:
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
        # miles asserts len(loss_mask) == response_length, both counted from the
        # first trainable token; everything before it is prompt.
        sample.response_length = len(self.token_ids) - first_trainable
        sample.loss_mask = self.loss_mask[first_trainable:]
        sample.rollout_log_probs = self.log_probs[first_trainable:]
        sample.reward = float(reward)
        sample.status = Sample.Status.COMPLETED
        return sample
