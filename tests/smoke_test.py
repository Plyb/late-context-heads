"""CPU smoke test of the full pipeline logic with gpt2 (no GPU, no Llama).

Validates prompt assembly, token-type column partitioning, per-head direct effect,
top-head selection, value-weighted attention collection, and heatmap rendering.
"""

import tempfile
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

from mcqa_head_finding import data, viz
from mcqa_head_finding.correctness import letter_token_ids
from mcqa_head_finding.direct_effect import (
    ANSWER_LETTERS,
    Accumulator,
    head_net_effect,
    letter_direction_ids,
    select_top_heads,
)
from mcqa_head_finding.prompts import build_prompt
from mcqa_head_finding.token_types import build_columns, encode_with_offsets
from mcqa_head_finding.value_weighted_attention import Collector


def _assert_columns_cover_tokens(prompt, tokenizer) -> None:
    input_ids, offsets = encode_with_offsets(tokenizer, prompt.text)
    non_special = {i for i, (a, b) in enumerate(offsets) if a != b}
    cols = build_columns(prompt, tokenizer, "A", include_groups=True)
    covered: set[int] = set()
    for idx in cols.indices:
        covered.update(idx)
    missing = non_special - covered
    assert not missing, f"tokens not covered by any column: {missing}"


def main() -> None:
    device = torch.device("cpu")
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokenizer = model.tokenizer
    letter_ids = letter_token_ids(tokenizer, list(ANSWER_LETTERS))
    letter_dir_ids = letter_direction_ids(letter_ids)

    examples = data.load_examples("synthetic", "train", max_examples=8)
    print(f"loaded {len(examples)} synthetic examples")
    print("sample prompt (question_first):")
    print(build_prompt(examples[0].question, examples[0].options, "question_first").text)

    for fmt in ("question_first", "options_first"):
        prompt = build_prompt(examples[0].question, examples[0].options, fmt)
        _assert_columns_cover_tokens(prompt, tokenizer)
        cols = build_columns(prompt, tokenizer, examples[0].answer_label, True)
        print(f"[{fmt}] {len(cols.names)} columns; first 6 {cols.names[:6]}")

        acc = Accumulator()
        correct = []
        for ex in examples:
            ids, _ = encode_with_offsets(tokenizer, build_prompt(ex.question, ex.options, fmt).text)
            tokens = torch.tensor([ids], device=device)
            ci = ANSWER_LETTERS.index(ex.answer_label)
            net, labels, _ = head_net_effect(model, tokens, letter_dir_ids, ci)
            acc.add(net, labels)
            correct.append(ex)

        mean_effect = acc.mean().numpy()
        assert acc.labels is not None
        top = select_top_heads(mean_effect, acc.labels, top_p=0.8)
        print(f"[{fmt}] {len(top.labels)} top heads, frac={top.cumulative_fraction:.3f}")

        collector = Collector(model, tokenizer, top.layer_heads, top.labels)
        for ex in correct:
            ids, _ = encode_with_offsets(tokenizer, build_prompt(ex.question, ex.options, fmt).text)
            _, cache = model.run_with_cache(torch.tensor([ids], device=device))
            collector.add(ex, fmt, cache)

        main_mat = collector.main_matrix()
        per_head = collector.per_head_matrices()
        assert collector.main_names is not None and collector.nogroup_names is not None
        assert main_mat.shape == (len(top.labels), len(collector.main_names))
        assert per_head.shape == (len(top.labels), 4, len(collector.nogroup_names))
        with tempfile.TemporaryDirectory() as tmp:
            viz.save_main_heatmap(main_mat, top.labels, collector.main_names, Path(tmp) / "m.png", fmt)
            viz.save_per_head_heatmaps(per_head, top.labels, collector.nogroup_names, Path(tmp), fmt)
            assert (Path(tmp) / "m.png").exists()
        print(f"[{fmt}] heatmaps rendered OK")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
