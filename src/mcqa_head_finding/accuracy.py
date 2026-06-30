"""Evaluate next-token answer accuracy for a (model, format) pair."""

from typing import NamedTuple

import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from .correctness import is_correct, predicted_label
from .data import Example
from .prompts import PromptFormat, assemble_prompt


class EvalResult(NamedTuple):
    accuracy: float
    failures: list[dict]


@torch.no_grad()
def evaluate(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    fmt: PromptFormat,
    letter_ids: dict[str, int],
    chat_template: bool = False,
    dataset: str = "generic",
    max_failures: int = 25,
) -> EvalResult:
    if not examples:
        return EvalResult(0.0, [])
    device = model.embed.W_E.device  # first device under multi-GPU model parallelism
    n_correct = 0
    failures: list[dict] = []
    for example in examples:
        prompt = assemble_prompt(
            example.question, example.options, fmt, tokenizer, chat_template, dataset
        )
        ids = tokenizer(prompt.text, add_special_tokens=not chat_template)["input_ids"]
        tokens = torch.tensor([ids], device=device)
        final_logits = model(tokens, return_type="logits")[0, -1].float()
        if is_correct(final_logits, letter_ids, example.answer_label):
            n_correct += 1
        elif len(failures) < max_failures:
            failures.append(
                {
                    "question": example.question,
                    "options": {o.label: o.content for o in example.options},
                    "answer": example.answer_label,
                    "predicted": predicted_label(final_logits, letter_ids),
                }
            )
    return EvalResult(n_correct / len(examples), failures)
