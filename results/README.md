# Results Guide

This directory contains the curated outputs from three key analysis runs, spanning different
task types, model scales, and prompt formats. Each configuration explores whether the
"newline-of-correct-answer" mechanism—a mid-depth attention head that reads which option is
correct—generalizes across settings.

---

## The Three Configurations

| Name | Model | Task | Accuracy QF / OF | Key Finding |
|---|---|---|---|---|
| **1. Copy retrieval** | Llama-3.2-3B (base) | Synthetic: pick the target word from a pool | 1.00 / 1.00 | L15H18 & L15H19 form a clean diagonal: correct-A → A_nl, B → B_nl, etc. |
| **2. Semantic vocab** | Llama-3.2-3B-Instruct | Definition → word (non-token-matchable) | 1.00 / 0.98 | Same L15H18/L15H19 pattern generalizes; genuine "read-the-option" mechanism |
| **3. Real science-QA** | Llama-3.1-70B-Instruct | ARC-Easy (real exam questions) | 0.99 / 0.92 | At ~50% relative depth (L35–39 in 70B ≈ L15 in 3B): same diagonal pattern scales to 70B |

---

## Directory Structure

Each config directory contains:

```
<config>/
  config.yaml                    # Run config (model, dataset, args)
  question_first/                # Prompt format: question first, then options
    ├── heatmap_main.png         # Heads × token-types heatmap
    ├── top_heads.json           # Selected head list + metadata
    ├── direct_effect_labels.json # Head label order
    ├── vwa_main_columns.json     # Heatmap column names
    ├── vwa_per_head_columns.json # Per-head heatmap columns
    └── per_head/                # One PNG per top head
        ├── head_L15H18.png
        ├── head_L26H20.png
        └── ...
  options_first/                 # Prompt format: options first, then question
    └── (same structure)
```

---

## Reading the Outputs

### `top_heads.json`
Metadata about head selection:
- `labels`: human-readable head names (e.g., `["L15H18", "L26H20", ...]`)
- `layer_heads`: parallel list of `[layer, head]` indices
- `cumulative_fraction`: % of positive direct logit effect these heads explain (target ~0.80)
- `n_correct`: how many examples in the run were predicted correctly
- `total_positive`: sum of positive direct effects across all heads

The "top heads" are selected by sorting all heads by mean direct logit effect (on correct
examples only) and accumulating until 80% of the total positive effect is covered.

### `direct_effect_labels.json`
Maps the `direct_effects.npy` array (not included here; see note below) to head names.
Shows the full analysis across all 224 heads (3B) or 6400 heads (70B); used here only for
reference — the summary is in `top_heads.json`.

### `heatmap_main.png`
**Rows:** Top heads selected for this format (in cumulative-effect order).
**Columns:** Token type groups (preamble, question, label-A/B/C/D, content, final answer
prompt tokens, etc.).
**Color:** Mean value-weighted attention from the final token to each token group. Brighter =
more attention.

Interpretation:
- **Late heads** (L26H20, L27H1): concentrate attention on the final `:` / `(` tokens
  (rightmost columns).
- **Mid-depth heads** (L15H18 in 3B, L35H17–39H45 in 70B): spread attention across the
  options, with a bright spot on the correct option's newline (see per-head charts for detail).
- **Label/content heads**: distribute across the A/B/C/D option regions.

### `per_head/head_L15H18.png` (and others)
**Rows:** Which answer was correct in the example (A, B, C, or D).
**Columns:** Same token types as the main heatmap.
**Color:** Value-weighted attention from final token to each (answer, token-type) pair.

The **key finding** is visible here: a clean **diagonal pattern**:
- When the correct answer is A, the head attends most to the newline ending option A.
- When correct is B, it attends to B's newline. Same for C and D.

This diagonal proves the head is genuinely "reading the correct option," not just triggering
on surface cues.

### `vwa_*_columns.json`
Column labels for the heatmaps. Maps column index → human-readable name (e.g., column 0 =
"preamble_0", column 15 = "correct_nl", column 20 = "answer_prompt_0", etc.).

---

## How to Reproduce

Each configuration's `config.yaml` contains the exact command-line flags used. E.g.:

```yaml
mode: analyze
model: llama-3.2-3b
dataset: synthetic
formats:
  - question_first
  - options_first
```

To re-run:
```bash
uv run mcqa-heads --mode analyze --model llama-3.2-3b --dataset synthetic \
  --formats question_first options_first
```

For BYU ORC, add `--gpus h200:1 --mem-per-cpu 120G` (or appropriate GPU/memory for your
allocation). See the main [README.md](../README.md) for full usage.

---

## Notes

- **`.npy` arrays excluded:** The raw `direct_effects.npy` (all-head attributions) and
  `vwa_main.npy` / `vwa_per_head.npy` (value-weighted attention matrices) are not included
  here; they are large (100+ MB) binary arrays useful only with the source code loaded. The
  PNGs and JSONs above convey the key findings. To reproduce them, re-run the pipeline.

- **Accuracy by format:** Notice that the best format varies by task (question_first better
  for ARC, options_first trivial for synthetic). The "newline" head mechanism works in both,
  but the model finds it harder in some directions than others.

- **Negative control:** Addition (answer = computed sum) was not analyzed because no Llama
  ≤70B reached 90% on both formats. This suggests the mechanism is tied to *reading* the
  option, not computing the answer — a genuine MCQA strategy, not memorized single-digit
  arithmetic.
