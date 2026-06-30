# Results

## Headline: the "newline-of-correct-answer" head generalizes

Across three settings spanning task type, model class, and 23× scale, the same functional
head appears — a mid-depth head that attends from the final token to the **newline ending
whichever option is correct** (a clean A→A_nl, B→B_nl, C→C_nl, D→D_nl diagonal), alongside
**late heads** that dump attention onto the final `:`/`(` tokens and **label heads** that
read the answer letters:

| setting | model | task | accuracy (qf/of) | newline-of-correct head(s) |
|---|---|---|---|---|
| copy | Llama-3.2-3B (base, raw) | synthetic | 1.00 / 1.00 | L15H18, L15H19 |
| semantic | Llama-3.2-3B-Instruct (chat) | vocab def→word | 1.00 / 0.98 | L15H18, L15H19 |
| real science-QA | Llama-3.1-70B-Instruct (chat) | ARC-Easy | 0.99 / 0.92 | L35H17/18/19, L39H11/40/45 |

The mid-depth heads sit at ~50% relative depth in every model (L15/28 in 3B ≈ L35–39/80 in
70B). Because the vocab and ARC tasks are non-token-matchable (the answer never appears in
the question/definition), this is a genuine MCQA "read the correct option" mechanism, not an
artifact of surface copying. Negative control: **addition** (answer = a computed sum) could
not be driven to ≥90% on both formats by any Llama ≤70B (base or instruct), so its heads were
never analysed — see below.

Run dirs: synthetic `logs/20260530_152944_*`, vocab `logs/20260601_143618_*`,
ARC-70B `logs/20260601_182637_*`.

---

# Selected-pair results (original task)

## Selected model–dataset pair

The task requires ≥90% accuracy on **both** prompt formats while minimizing model size.

Accuracy sweep (500 examples each, `logs/*_accuracy_*`):

| model | dataset | question_first | options_first | both ≥90%? |
|-------|---------|---------------:|--------------:|:----------:|
| Llama-3.2-1B | ARC-Easy | 0.352 | 0.274 | no |
| Llama-3.1-8B | ARC-Easy | 0.916 | 0.552 | no |
| Llama-3.2-1B | synthetic | 0.350 | 1.000 | no |
| **Llama-3.2-3B** | **synthetic** | **1.000** | **1.000** | **yes** |
| Llama-3.1-8B | synthetic | 1.000 | 1.000 | yes |

**Selected: Llama-3.2-3B + synthetic** (smallest passing; 1B fails, 3B passes).

Notably the format asymmetry flips by dataset: ARC-Easy is much easier in
question_first (the model can read the question then scan options), whereas the
synthetic retrieval task is trivial in options_first (the question naming the target
word sits right before `Answer: (`). Only a model strong enough to handle the harder
direction on a given dataset clears both — 3B does on the synthetic set.

## Head finding (direct logit attribution → 80% of positive direct effect)

Run: `logs/20260530_152944_analyze_synthetic_llama-3.2-3b_12028877/output/`

* **question_first:** 9 heads cover 80.6% of positive direct effect.
* **options_first:** 10 heads cover 81.0%.
* Both well under the 200-head reassess guard.

The two formats select an almost identical head set — 8 heads shared, with the same
top three in both: **L26H20, L27H1, L15H18**.

## What the heads attend to (value-weighted attention from the final token)

* **Late heads** (L26H20, L27H1, L27H2, L23H2, L26H19): attend to the final
  answer-prompt tokens (`:` and ` (`) — moving the decision onto the final position.
* **L15H18 / L15H19** (mid layers): attend to the newline ending each option, brightest
  on the *correct* option (`correct_nl`). The per-head charts show a clean diagonal —
  correct A → `A_nl`, B → `B_nl`, C → `C_nl`, D → `D_nl` — i.e. these heads read the
  content of whichever option is correct.
* **L21H18 / L20H4 / L21H17 / L22H11**: attend to the option label / content region.

## Outputs per format (`output/<format>/`)

* `direct_effects.npy` + `direct_effect_labels.json` — per-head net direct effect (all heads).
* `top_heads.json` — selected heads, cumulative fraction, guard flag, n_correct.
* `vwa_main.npy` + `vwa_main_columns.json`, `heatmap_main.png` — heads × token-type heatmap.
* `vwa_per_head.npy` + `vwa_per_head_columns.json`, `per_head/head_*.png` — per top head,
  correct-answer (A/B/C/D) × token-type.

## Reproduce

```bash
# accuracy sweep
uv run python -m mcqa_head_finding.run --mode accuracy --models llama-3.2-3b \
  --dataset synthetic --max-examples 500 --gpus h200:1 --mem-per-cpu 120G
# full analysis
uv run python -m mcqa_head_finding.run --mode analyze --model llama-3.2-3b \
  --dataset synthetic --max-examples 1000 --gpus h200:1 --time 00:45:00 --mem-per-cpu 120G
```
