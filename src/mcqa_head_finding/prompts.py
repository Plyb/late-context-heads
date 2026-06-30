"""Assemble the two MCQA prompt formats, recording per-segment character ranges.

The two formats differ only in whether the question precedes or follows the options.
Each prompt is built by concatenating labelled segments while recording the
``[start, end)`` character span of each, so downstream tokenisation can map tokens
back to semantic spans via offset mapping. The exact text follows ``task.md``:

    A highly knowledgeable and intelligent AI answers multiple-choice questions.
    {question}
    A) {option a}
    ...
    Answer: (
"""

from typing import Literal, NamedTuple

PromptFormat = Literal["question_first", "options_first"]
FORMATS: tuple[PromptFormat, ...] = ("question_first", "options_first")

INSTRUCTIONS_TEXT = (
    "A highly knowledgeable and intelligent AI answers multiple-choice questions.\n"
)
ANSWER_PROMPT_TEXT = "Answer: ("


class PromptSpec(NamedTuple):
    instructions: str
    question_prefix: str  # prepended to the question/definition line text
    question_label: str  # heatmap column name for that line's grouped tokens


_GENERIC_SPEC = PromptSpec(INSTRUCTIONS_TEXT, "", "question")
PROMPT_SPECS: dict[str, PromptSpec] = {
    "vocab": PromptSpec(
        "Select the word that most closely matches the definition.\n",
        "Definition: ",
        "definition",
    ),
}


def spec_for(dataset: str | None) -> PromptSpec:
    return PROMPT_SPECS.get(dataset or "", _GENERIC_SPEC)


class Segment(NamedTuple):
    name: str
    start: int
    end: int


class Option(NamedTuple):
    label: str
    content: str


class BuiltPrompt(NamedTuple):
    text: str
    segments: list[Segment]
    option_labels: list[str]


def _option_segment_text(option: Option) -> str:
    return f"{option.label}) {option.content}\n"


def assemble_prompt(
    question: str,
    options: list[Option],
    fmt: PromptFormat,
    tokenizer=None,
    chat_template: bool = False,
    dataset: str = "generic",
) -> BuiltPrompt:
    """Build a raw-completion or chat-templated prompt for the given format."""
    if chat_template:
        if tokenizer is None:
            raise ValueError("chat_template=True requires a tokenizer")
        return build_prompt_chat(question, options, fmt, tokenizer, dataset)
    return build_prompt(question, options, fmt, dataset)


def _content_pieces(
    question: str, options: list[Option], fmt: PromptFormat, spec: PromptSpec
) -> list[tuple[str, str]]:
    """The instruction + question + option pieces (no answer prompt), in format order."""
    question_piece = ("question", f"{spec.question_prefix}{question}\n")
    option_pieces = [
        (f"option:{opt.label}", _option_segment_text(opt)) for opt in options
    ]
    pieces: list[tuple[str, str]] = [("instructions", spec.instructions)]
    if fmt == "question_first":
        pieces.append(question_piece)
        pieces.extend(option_pieces)
    else:
        pieces.extend(option_pieces)
        pieces.append(question_piece)
    return pieces


def _assemble(
    pieces: list[tuple[str, str]], start_offset: int = 0
) -> tuple[str, list[Segment]]:
    text = ""
    segments: list[Segment] = []
    for name, piece_text in pieces:
        start = start_offset + len(text)
        text += piece_text
        segments.append(Segment(name=name, start=start, end=start_offset + len(text)))
    return text, segments


def build_prompt(
    question: str, options: list[Option], fmt: PromptFormat, dataset: str = "generic"
) -> BuiltPrompt:
    pieces = _content_pieces(question, options, fmt, spec_for(dataset))
    pieces.append(("answer_prompt", ANSWER_PROMPT_TEXT))
    text, segments = _assemble(pieces)
    return BuiltPrompt(
        text=text,
        segments=segments,
        option_labels=[opt.label for opt in options],
    )


def build_prompt_chat(
    question: str,
    options: list[Option],
    fmt: PromptFormat,
    tokenizer,
    dataset: str = "generic",
) -> BuiltPrompt:
    """Wrap the MCQA content in the model's chat template (Option B for instruct models).

    The content goes in a ``user`` turn; the assistant turn is opened and primed with
    ``Answer: (`` so the next-token letter logit is still read at the final position.
    Segment char ranges are shifted to their position in the rendered template; the
    template's control tokens fall outside all segments (excluded, like BOS).
    """
    body, body_segments = _assemble(
        _content_pieces(question, options, fmt, spec_for(dataset))
    )
    # Splice the body in via a sentinel so the template's ``| trim`` cannot drop the
    # body's trailing newline (which carries the last option's ``*_nl`` column).
    sentinel = "\x01BODY\x01"
    rendered: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
    )
    idx = rendered.index(sentinel)
    prefix, suffix = rendered[:idx], rendered[idx + len(sentinel):]
    full_text = prefix + body + suffix + ANSWER_PROMPT_TEXT
    offset = len(prefix)
    segments = [
        Segment(s.name, s.start + offset, s.end + offset) for s in body_segments
    ]
    segments.append(
        Segment("answer_prompt", len(full_text) - len(ANSWER_PROMPT_TEXT), len(full_text))
    )
    return BuiltPrompt(
        text=full_text,
        segments=segments,
        option_labels=[opt.label for opt in options],
    )
