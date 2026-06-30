"""Map prompt tokens onto the named token-type columns of the heatmaps.

Columns (BOS excluded; cross-column duplication is intentional per ``task.md``):

* one column per preamble (instruction) token,
* ``question`` (all question tokens grouped),
* ``label_{A,B,C,D}`` and, with groups, ``label_correct`` / ``label_incorrect``,
* ``content_{A,B,C,D}`` and, with groups, ``content_correct`` / ``content_incorrect``
  (content = everything after the letter, up to and including the trailing newline),
* per answer type, the content minus its final four tokens (``*_prefix``) followed by the
  final four tokens individually (``*_m3, *_m2, *_m1, *_nl``); missing positions are simply
  absent for that example and drop out of the per-column mean divisor,
* the ``Answer: (`` tokens, one column each.

The column *structure* (names and count) is identical across all examples of a given
``(format, include_groups)``, so per-example rows align into a matrix.
"""

from typing import NamedTuple

from transformers import PreTrainedTokenizerBase

from .prompts import BuiltPrompt, spec_for

ANSWER_TYPES_BASE: tuple[str, ...] = ("A", "B", "C", "D")
GROUP_TYPES: tuple[str, ...] = ("correct", "incorrect")
FINAL_TAIL = ("m3", "m2", "m1", "nl")


class Columns(NamedTuple):
    names: list[str]
    indices: list[list[int]]
    input_ids: list[int]


def encode_with_offsets(
    tokenizer: PreTrainedTokenizerBase, text: str, add_special_tokens: bool = True
) -> tuple[list[int], list[tuple[int, int]]]:
    encoding = tokenizer(
        text, return_offsets_mapping=True, add_special_tokens=add_special_tokens
    )
    input_ids: list[int] = encoding["input_ids"]
    offsets: list[tuple[int, int]] = [tuple(o) for o in encoding["offset_mapping"]]
    return input_ids, offsets


def _overlaps(ts: int, te: int, start: int, end: int) -> bool:
    return ts < end and te > start


def _segment_of_token(ts: int, te: int, prompt: BuiltPrompt) -> str | None:
    for seg in prompt.segments:
        if _overlaps(ts, te, seg.start, seg.end):
            return seg.name
    return None


def _segment_tokens(
    prompt: BuiltPrompt, offsets: list[tuple[int, int]]
) -> dict[str, list[int]]:
    tokens: dict[str, list[int]] = {seg.name: [] for seg in prompt.segments}
    for idx, (ts, te) in enumerate(offsets):
        if ts == te:  # special token (e.g. BOS) carries an empty offset
            continue
        name = _segment_of_token(ts, te, prompt)
        if name is not None:
            tokens[name].append(idx)
    return tokens


def _split_option(
    seg_name: str,
    seg_tokens: list[int],
    offsets: list[tuple[int, int]],
    prompt: BuiltPrompt,
) -> tuple[list[int], list[int]]:
    """Return (label_tokens, content_tokens) for one option line.

    The label is the single letter character at the segment start; everything else
    (``")"``, the option text, and the newline) is content.
    """
    seg = next(s for s in prompt.segments if s.name == seg_name)
    label = [t for t in seg_tokens if _overlaps(*offsets[t], seg.start, seg.start + 1)]
    content = [t for t in seg_tokens if t not in label]
    return label, content


def build_columns(
    prompt: BuiltPrompt,
    tokenizer: PreTrainedTokenizerBase,
    answer_label: str,
    include_groups: bool,
    add_special_tokens: bool = True,
    dataset: str = "generic",
) -> Columns:
    input_ids, offsets = encode_with_offsets(tokenizer, prompt.text, add_special_tokens)
    seg_tokens = _segment_tokens(prompt, offsets)

    label_tokens: dict[str, list[int]] = {}
    content_tokens: dict[str, list[int]] = {}
    for letter in prompt.option_labels:
        seg_name = f"option:{letter}"
        label, content = _split_option(seg_name, seg_tokens[seg_name], offsets, prompt)
        label_tokens[letter] = label
        content_tokens[letter] = content

    type_to_labels: dict[str, list[str]] = {l: [l] for l in ANSWER_TYPES_BASE}
    type_to_labels["correct"] = [answer_label]
    type_to_labels["incorrect"] = [l for l in prompt.option_labels if l != answer_label]

    def pooled_content(answer_type: str) -> list[int]:
        out: list[int] = []
        for letter in type_to_labels[answer_type]:
            out.extend(content_tokens[letter])
        return out

    def pooled_labels(answer_type: str) -> list[int]:
        out: list[int] = []
        for letter in type_to_labels[answer_type]:
            out.extend(label_tokens[letter])
        return out

    def pooled_prefix(answer_type: str) -> list[int]:
        out: list[int] = []
        for letter in type_to_labels[answer_type]:
            out.extend(content_tokens[letter][:-4])
        return out

    def pooled_tail(answer_type: str, from_end: int) -> list[int]:
        """Pool the token ``from_end`` positions from the end (1 = newline)."""
        out: list[int] = []
        for letter in type_to_labels[answer_type]:
            content = content_tokens[letter]
            if len(content) >= from_end:
                out.append(content[-from_end])
        return out

    names: list[str] = []
    indices: list[list[int]] = []

    for k, tok in enumerate(seg_tokens["instructions"]):
        names.append(f"P{k}:{str(tokenizer.decode([input_ids[tok]])).strip()}")
        indices.append([tok])

    names.append(spec_for(dataset).question_label)
    indices.append(seg_tokens["question"])

    answer_types = list(ANSWER_TYPES_BASE) + (
        list(GROUP_TYPES) if include_groups else []
    )

    for letter in ANSWER_TYPES_BASE:
        names.append(f"label_{letter}")
        indices.append(label_tokens[letter])
    if include_groups:
        for grp in GROUP_TYPES:
            names.append(f"label_{grp}")
            indices.append(pooled_labels(grp))

    for letter in ANSWER_TYPES_BASE:
        names.append(f"content_{letter}")
        indices.append(content_tokens[letter])
    if include_groups:
        for grp in GROUP_TYPES:
            names.append(f"content_{grp}")
            indices.append(pooled_content(grp))

    for answer_type in answer_types:
        names.append(f"{answer_type}_prefix")
        indices.append(pooled_prefix(answer_type))
        for from_end, suffix in zip((4, 3, 2, 1), FINAL_TAIL):
            names.append(f"{answer_type}_{suffix}")
            indices.append(pooled_tail(answer_type, from_end))

    for k, tok in enumerate(seg_tokens["answer_prompt"]):
        names.append(f"ans{k}:{str(tokenizer.decode([input_ids[tok]])).strip()}")
        indices.append([tok])

    return Columns(names=names, indices=indices, input_ids=input_ids)
