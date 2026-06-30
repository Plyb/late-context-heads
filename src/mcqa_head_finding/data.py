"""Load 4-choice MCQA examples: ARC-Easy, or a trivial synthetic retrieval set.

Options keep their natural order (the correct answer is *not* moved to 'A'), so the
dataset retains a spread of which letter is correct -- required for the per-head
"which answer was correct" heatmaps.
"""

import re
import string
from collections import defaultdict
from typing import NamedTuple, cast

try:
    from byutils import load_dataset
except ImportError:
    from datasets import load_dataset

from .prompts import Option

LETTER_LABELS = list(string.ascii_uppercase)

ARC_PATH = "allenai/ai2_arc"
ARC_CONFIG = "ARC-Easy"
VOCAB_PATH = "wordlevel/toefl-essential-vocabulary-1k"

DatasetName = str
DATASETS: tuple[DatasetName, ...] = ("arc-easy", "synthetic", "addition", "vocab")


class Example(NamedTuple):
    id: str
    question: str
    options: list[Option]
    answer_label: str


def _normalise_arc(row: dict) -> Example | None:
    texts: list[str] = row["choices"]["text"]
    labels: list[str] = row["choices"]["label"]
    if len(texts) != 4 or row["answerKey"] not in labels:
        return None
    answer_index = labels.index(row["answerKey"])
    options = [
        Option(label=LETTER_LABELS[i], content=text) for i, text in enumerate(texts)
    ]
    return Example(
        id=row["id"],
        question=row["question"],
        options=options,
        answer_label=LETTER_LABELS[answer_index],
    )


def _load_arc_easy(split: str, max_examples: int | None) -> list[Example]:
    dataset = load_dataset(ARC_PATH, name=ARC_CONFIG)[split]
    examples: list[Example] = []
    for row in dataset:
        example = _normalise_arc(cast(dict, row))
        if example is not None:
            examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples


# A pool of distinct, single-token-friendly common words for the synthetic task.
_WORD_POOL: tuple[str, ...] = (
    "apple", "river", "mountain", "guitar", "planet", "doctor", "window", "garden",
    "engine", "pencil", "bridge", "castle", "forest", "island", "rocket", "camera",
    "candle", "ticket", "monkey", "yellow", "silver", "winter", "summer", "pepper",
    "violin", "anchor", "basket", "dragon", "feather", "harbor", "jacket", "kettle",
    "ladder", "magnet", "needle", "orange", "puzzle", "rabbit", "saddle", "turtle",
)


def _load_synthetic(max_examples: int | None) -> list[Example]:
    """Trivial retrieval: the target word appears verbatim as exactly one option.

    Distractors and the correct position are derived deterministically from the
    example index so all of A/B/C/D occur as the answer and there is no RNG.
    """
    n = max_examples if max_examples is not None else 2000
    pool = _WORD_POOL
    examples: list[Example] = []
    for i in range(n):
        target = pool[i % len(pool)]
        distractors = [pool[(i + 1 + k) % len(pool)] for k in range(3)]
        answer_pos = i % 4
        contents: list[str] = []
        d = 0
        for pos in range(4):
            if pos == answer_pos:
                contents.append(target)
            else:
                contents.append(distractors[d])
                d += 1
        options = [
            Option(label=LETTER_LABELS[pos], content=contents[pos]) for pos in range(4)
        ]
        examples.append(
            Example(
                id=f"syn-{i}",
                question=f"Which option is the word '{target}'?",
                options=options,
                answer_label=LETTER_LABELS[answer_pos],
            )
        )
    return examples


def _load_addition(max_examples: int | None) -> list[Example]:
    """Single-digit addition MCQA: the answer is the sum, so it never appears in the question.

    Operands are 1-9 (sums 2-18). Distractors are the sum -1/+1/+2 (close, so the task
    needs computation rather than picking the largest). Operands and the correct position
    are deterministic in the example index; all of A/B/C/D occur as the answer.
    """
    n = max_examples if max_examples is not None else 2000
    examples: list[Example] = []
    for i in range(n):
        a = (i * 7) % 9 + 1
        b = (i * 23) % 9 + 1
        correct = a + b
        distractors = [correct - 1, correct + 1, correct + 2]
        answer_pos = i % 4
        numbers: list[int] = []
        d = 0
        for pos in range(4):
            if pos == answer_pos:
                numbers.append(correct)
            else:
                numbers.append(distractors[d])
                d += 1
        options = [
            Option(label=LETTER_LABELS[pos], content=str(numbers[pos]))
            for pos in range(4)
        ]
        examples.append(
            Example(
                id=f"add-{i}",
                question=f"What is {a} + {b}?",
                options=options,
                answer_label=LETTER_LABELS[answer_pos],
            )
        )
    return examples


def _load_vocab(max_examples: int | None) -> list[Example]:
    """Definition->word MCQA: pick the word that matches the definition.

    Distractors are 3 other words of the same part of speech (deterministic in the
    index). Examples where the target word appears in its own definition are skipped so
    the answer cannot be found by token-matching. Correct position cycles A/B/C/D.
    """
    rows = [cast(dict, r) for r in load_dataset(VOCAB_PATH)["train"]]
    by_pos: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_pos[r["pos"]].append(i)

    def distractors_for(ti: int) -> list[str]:
        target_word = rows[ti]["word"]
        chosen: list[str] = []
        cands = by_pos[rows[ti]["pos"]]
        seq = [cands[(cands.index(ti) + k) % len(cands)] for k in range(1, len(cands))]
        seq += [(ti + g) % len(rows) for g in range(1, len(rows))]  # global fallback
        for j in seq:
            w = rows[j]["word"]
            if w != target_word and w not in chosen:
                chosen.append(w)
            if len(chosen) == 3:
                break
        return chosen

    n = max_examples if max_examples is not None else len(rows)
    examples: list[Example] = []
    i = 0
    while len(examples) < n and i < 100 * len(rows):
        ti = i % len(rows)
        i += 1
        target = rows[ti]["word"]
        definition = rows[ti]["definition_en"]
        if re.search(rf"\b{re.escape(target)}\b", definition, re.IGNORECASE):
            continue  # answer leaks into the definition -> would be token-matchable
        distractors = distractors_for(ti)
        if len(distractors) < 3:
            continue
        answer_pos = len(examples) % 4
        words: list[str] = []
        d = 0
        for pos in range(4):
            if pos == answer_pos:
                words.append(target)
            else:
                words.append(distractors[d])
                d += 1
        options = [
            Option(label=LETTER_LABELS[pos], content=words[pos]) for pos in range(4)
        ]
        examples.append(
            Example(
                id=f"vocab-{ti}",
                question=definition,
                options=options,
                answer_label=LETTER_LABELS[answer_pos],
            )
        )
    return examples


def load_examples(
    dataset: DatasetName, split: str, max_examples: int | None = None
) -> list[Example]:
    if dataset == "arc-easy":
        return _load_arc_easy(split, max_examples)
    if dataset == "synthetic":
        return _load_synthetic(max_examples)
    if dataset == "addition":
        return _load_addition(max_examples)
    if dataset == "vocab":
        return _load_vocab(max_examples)
    raise ValueError(f"unknown dataset {dataset!r}; choices are {DATASETS}")
