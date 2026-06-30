# mcqa-head-finding

> **Note**: This is a quick research / blog-post project. The code makes trade-offs
> (BYU ORC helpers, minimal error handling) that a production repo would not. It is not
> polished to my normal publication standards — use it as a reading companion for the
> accompanying post, not as a template for well-engineered code.

## What this is

Find the attention heads that drive multiple-choice answer selection in Llama models using
direct logit attribution, then characterize what those heads attend to via value-weighted
attention heatmaps.

The key finding: a mid-depth "newline-of-correct-answer" head (sitting at ~50% relative
depth) generalizes across task type, model scale, and prompt format. It attends from the
final token to the newline ending whichever option is correct, forming a clean diagonal:
correct A → A_nl, B → B_nl, C → C_nl, D → D_nl. This proves the head is genuinely
"reading the correct option," not relying on surface cues or token-matching.

## Results

See [results/](results/) for three configurations spanning different task types, model
scales, and dataset complexity:

1. **Copy retrieval** — Llama-3.2-3B on synthetic word-picking task (perfect accuracy both
   formats)
2. **Semantic vocab** — Llama-3.2-3B-Instruct on definition-to-word (non-token-matchable,
   1.00 / 0.98 accuracy)
3. **Real science-QA** — Llama-3.1-70B-Instruct on ARC-Easy (0.99 / 0.92 accuracy)

The same mechanism appears in all three. [Read the results guide](results/README.md) to
interpret the heatmaps.

## Requirements

- Python 3.12+
- PyTorch with CUDA support (optional for CPU smoke tests)
- HuggingFace account with access to Llama model weights (`huggingface-cli login`)

## Installation

### General (non-BYU)

```bash
git clone <this repo>
cd mcqa-head-finding
uv sync
# or: pip install -e .
```

### BYU ORC (with shared cache + SLURM helpers)

```bash
git clone <this repo>
cd mcqa-head-finding
uv sync --extra byu
```

The `byu` extra installs `byutils` (shared HuggingFace cache), `slurm-launcher` (auto-job
submission), and `runlog` (structured run logging). Without it, the pipeline uses default
HF_HOME caching and runs directly on the current machine.

## Usage

### Accuracy sweep

Find the smallest model that achieves ≥90% accuracy on both prompt formats:

```bash
uv run mcqa-heads --mode accuracy --dataset synthetic --models llama-3.2-1b llama-3.2-3b
```

Output: `logs/<run>/output/accuracy.json` — per-model and per-format accuracy.

### Analysis

For a chosen model/dataset, compute direct logit attribution and value-weighted attention:

```bash
uv run mcqa-heads --mode analyze --model llama-3.2-3b --dataset synthetic \
  --formats question_first options_first
```

Output: `logs/<run>/output/{question_first,options_first}/` — heatmaps, head selections,
and attribution data.

### Key flags

- `--dataset`: `arc-easy` (real MCQA), `synthetic` (word retrieval), `addition` (computed
  sum), `vocab` (definition → word)
- `--chat-template`: use the model's chat template (required for `-Instruct` models)
- `--max-examples N`: limit dataset size for quick testing
- `--top-p FRAC`: keep heads summing to this fraction of positive direct effect (default
  0.8)
- `--n-devices N`: split model across N GPUs (model parallelism, needed for 70B)
- `--no-hf-model`: stream weights instead of preloading (halves memory for 70B)

### On BYU ORC

The pipeline auto-submits to SLURM if `slurm-launcher` is installed. Pass `--local` to
skip:

```bash
# Auto-submit to SLURM (BYU ORC)
uv run mcqa-heads --mode analyze --model llama-3.1-70b-instruct --dataset arc-easy \
  --gpus h200:1 --time 02:00:00 --mem-per-cpu 120G

# Run directly on current node
uv run mcqa-heads --mode analyze --model llama-3.2-3b --dataset synthetic --local
```

Outside BYU ORC, the `--gpus`, `--time`, `--mem-per-cpu`, `--qos` flags are no-ops when
`slurm-launcher` is not installed.

## Pipeline

1. **Accuracy sweep** (optional): iterate models on a dataset, find smallest that clears
   90% on both `question_first` and `options_first` formats.

2. **Direct logit attribution**: for the chosen model/dataset, filter examples to
   correct-only, compute per-head logit effect onto the answer letters, mean-center to
   remove the "no-head" baseline, and select heads summing to ~80% of positive total effect.

3. **Value-weighted attention**: for the selected heads, collect `attention_pattern[final,
   :] * ||value_j||_2` per head and token-type, producing heatmaps of what the heads read
   from the final position.

See [task.md](task.md) for the mathematical details.

## Testing

```bash
# Quick smoke test (CPU, GPT-2, ~30 sec)
uv run pytest tests/smoke_test.py -v
```

## Code organization

- [src/mcqa_head_finding/run.py](src/mcqa_head_finding/run.py) — CLI entrypoint
- [src/mcqa_head_finding/model.py](src/mcqa_head_finding/model.py) — TransformerLens wrapper
- [src/mcqa_head_finding/data.py](src/mcqa_head_finding/data.py) — dataset loaders (ARC, synthetic, vocab, addition)
- [src/mcqa_head_finding/accuracy.py](src/mcqa_head_finding/accuracy.py) — prompt assembly &
  accuracy eval
- [src/mcqa_head_finding/direct_effect.py](src/mcqa_head_finding/direct_effect.py) — logit
  attribution
- [src/mcqa_head_finding/value_weighted_attention.py](src/mcqa_head_finding/value_weighted_attention.py) — heatmap collection
- [src/mcqa_head_finding/viz.py](src/mcqa_head_finding/viz.py) — heatmap rendering
- [tests/](tests/) — smoke test + device diagnostic

## Reproducing the results

Each result configuration's `config.yaml` includes the exact flags used. To re-run:

```bash
cd results/<config>
cat config.yaml  # see the command
uv run mcqa-heads [flags from config]
```

Or from the paper/blog post, check which result you want to reproduce and run the analysis
with those parameters.

## License & Attribution

This code is provided as-is for research purposes. See [results/](results/) for detailed
methodology.
