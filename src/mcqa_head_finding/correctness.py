"""Decide whether the model answers a question correctly via option-letter argmax."""

from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def letter_token_ids(
    tokenizer: PreTrainedTokenizerBase, labels: list[str]
) -> dict[str, int]:
    """Token id emitted for each bare option letter (as it follows the ``(``)."""
    ids: dict[str, int] = {}
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        ids[label] = encoded[0]
    return ids


def predicted_label(
    final_logits: Float[Tensor, "vocab"], letter_ids: dict[str, int]
) -> str:
    labels = list(letter_ids.keys())
    scores = [final_logits[letter_ids[label]].item() for label in labels]
    return labels[max(range(len(labels)), key=lambda i: scores[i])]


def is_correct(
    final_logits: Float[Tensor, "vocab"],
    letter_ids: dict[str, int],
    answer_label: str,
) -> bool:
    return predicted_label(final_logits, letter_ids) == answer_label
