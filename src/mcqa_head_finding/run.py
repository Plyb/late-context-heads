"""Entrypoint: accuracy sweep, then direct-effect head finding + value-weighted attention.

Run pattern: when slurm-launcher is available, on a login node with ``--prefetch`` the
models and dataset are warmed into the shared HF cache, then the job re-submits itself to
a compute node. Pass ``--local`` to run directly without Slurm submission.

When slurm-launcher is not installed, the pipeline runs inline locally.

slurm_launcher (if available) re-invokes this file by path under ``srun``, so imports are
absolute and every torch-dependent import is deferred into the compute-side functions.
"""

import argparse
import json
import logging
import os
from pathlib import Path

try:
    from slurm_launcher import SlurmConfig, is_login_node, submit_slurm_job
    _SLURM_AVAILABLE = True
except ImportError:
    _SLURM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mcqa_head_finding")

MODELS: dict[str, str] = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama-3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
}
MODEL_LADDER = ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.1-70b")
FORMATS = ("question_first", "options_first")
ACCURACY_THRESHOLD = 0.90


def run_accuracy(args: argparse.Namespace, out_dir: Path) -> None:
    import torch

    from mcqa_head_finding import data
    from mcqa_head_finding.accuracy import evaluate
    from mcqa_head_finding.correctness import letter_token_ids
    from mcqa_head_finding.model import load_hooked_model, select_dtype

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = select_dtype(device)
    examples = data.load_examples(args.dataset, args.split, args.max_examples)
    logger.info("loaded %d examples (%s/%s)", len(examples), args.dataset, args.split)

    results: dict[str, dict] = {}
    selected: str | None = None
    for model_key in args.models:
        model_id = MODELS[model_key]
        logger.info("loading %s (%s) as %s", model_key, model_id, dtype)
        model, tokenizer = load_hooked_model(
            model_id, device, dtype, args.n_devices, args.use_hf_model
        )
        letter_ids = letter_token_ids(tokenizer, list("ABCD"))
        per_format = {
            fmt: evaluate(
                model, tokenizer, examples, fmt, letter_ids, args.chat_template, args.dataset
            )
            for fmt in FORMATS
        }
        accuracies = {fmt: r.accuracy for fmt, r in per_format.items()}
        worst = min(accuracies.values())
        results[model_key] = {
            "accuracy": accuracies,
            "min": worst,
            "failures": {fmt: r.failures for fmt, r in per_format.items()},
        }
        logger.info("%s accuracy %s (min %.3f)", model_key, accuracies, worst)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if worst >= ACCURACY_THRESHOLD and selected is None:
            selected = model_key
            break

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "n_examples": len(examples),
        "threshold": ACCURACY_THRESHOLD,
        "results": results,
        "selected": selected,
    }
    (out_dir / "accuracy.json").write_text(json.dumps(summary, indent=2))
    if selected is None:
        logger.warning(
            "no model in %s reached %.0f%% on both formats for %s",
            args.models,
            ACCURACY_THRESHOLD * 100,
            args.dataset,
        )
    else:
        logger.info("selected smallest passing model: %s", selected)


def run_analyze(args: argparse.Namespace, out_dir: Path) -> None:
    import numpy as np
    import torch

    from mcqa_head_finding import data, viz
    from mcqa_head_finding.correctness import is_correct, letter_token_ids
    from mcqa_head_finding.direct_effect import (
        ANSWER_LETTERS,
        Accumulator,
        head_net_effect,
        letter_direction_ids,
        select_top_heads,
    )
    from mcqa_head_finding.model import load_hooked_model, select_dtype
    from mcqa_head_finding.token_types import encode_with_offsets
    from mcqa_head_finding.prompts import assemble_prompt
    from mcqa_head_finding.value_weighted_attention import Collector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = select_dtype(device)
    model_id = MODELS[args.model]
    logger.info("loading %s (%s) as %s", args.model, model_id, dtype)
    model, tokenizer = load_hooked_model(
        model_id, device, dtype, args.n_devices, args.use_hf_model
    )
    letter_ids = letter_token_ids(tokenizer, list("ABCD"))
    letter_dir_ids = letter_direction_ids(letter_ids)
    embed_device = model.embed.W_E.device  # first device under model parallelism

    examples = data.load_examples(args.dataset, args.split, args.max_examples)
    logger.info("loaded %d examples (%s/%s)", len(examples), args.dataset, args.split)

    def tokens_for(example, fmt) -> torch.Tensor:
        prompt = assemble_prompt(
            example.question,
            example.options,
            fmt,
            tokenizer,
            args.chat_template,
            args.dataset,
        )
        ids, _ = encode_with_offsets(tokenizer, prompt.text, not args.chat_template)
        return torch.tensor([ids], device=embed_device)

    for fmt in args.formats:
        fmt_dir = out_dir / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)

        accumulator = Accumulator()
        correct_examples = []
        for example in examples:
            correct_idx = ANSWER_LETTERS.index(example.answer_label)
            net, labels, final_logits = head_net_effect(
                model, tokens_for(example, fmt), letter_dir_ids, correct_idx
            )
            if is_correct(final_logits, letter_ids, example.answer_label):
                accumulator.add(net, labels)
                correct_examples.append(example)

        if accumulator.count == 0:
            logger.warning("no correct examples for %s; skipping", fmt)
            continue

        mean_effect = accumulator.mean().numpy()
        assert accumulator.labels is not None
        top = select_top_heads(mean_effect, accumulator.labels, args.top_p)
        logger.info(
            "%s: %d/%d correct; %d top heads (%.1f%% of positive effect)%s",
            fmt,
            accumulator.count,
            len(examples),
            len(top.labels),
            top.cumulative_fraction * 100,
            " EXCEEDED_MAX -- reassess" if top.exceeded_max else "",
        )

        np.save(fmt_dir / "direct_effects.npy", mean_effect)
        (fmt_dir / "direct_effect_labels.json").write_text(
            json.dumps(accumulator.labels, indent=2)
        )
        (fmt_dir / "top_heads.json").write_text(
            json.dumps(
                {
                    "labels": top.labels,
                    "layer_heads": top.layer_heads,
                    "cumulative_fraction": top.cumulative_fraction,
                    "total_positive": top.total_positive,
                    "top_p": args.top_p,
                    "exceeded_max": top.exceeded_max,
                    "n_correct": accumulator.count,
                    "n_examples": len(examples),
                },
                indent=2,
            )
        )

        if not top.labels:
            logger.warning("%s: no positive-effect heads; skipping VWA", fmt)
            continue

        collector = Collector(
            model, tokenizer, top.layer_heads, top.labels, args.chat_template, args.dataset
        )

        def _vwa_filter(name: str) -> bool:
            return name.endswith("hook_pattern") or name.endswith("hook_v")

        with torch.no_grad():
            for example in correct_examples:
                _, cache = model.run_with_cache(
                    tokens_for(example, fmt), names_filter=_vwa_filter
                )
                collector.add(example, fmt, cache)

        main = collector.main_matrix()
        per_head = collector.per_head_matrices()
        assert collector.main_names is not None and collector.nogroup_names is not None
        np.save(fmt_dir / "vwa_main.npy", main)
        np.save(fmt_dir / "vwa_per_head.npy", per_head)
        (fmt_dir / "vwa_main_columns.json").write_text(
            json.dumps(collector.main_names, indent=2)
        )
        (fmt_dir / "vwa_per_head_columns.json").write_text(
            json.dumps(collector.nogroup_names, indent=2)
        )

        title = f"{args.model} | {args.dataset} | {fmt}"
        viz.save_main_heatmap(
            main, top.labels, collector.main_names, fmt_dir / "heatmap_main.png", title
        )
        viz.save_per_head_heatmaps(
            per_head, top.labels, collector.nogroup_names, fmt_dir / "per_head", title
        )
        logger.info("%s: saved heatmaps to %s", fmt, fmt_dir)


def _run(args: argparse.Namespace, out_dir: Path) -> None:
    if args.mode == "accuracy":
        run_accuracy(args, out_dir)
    else:
        run_analyze(args, out_dir)
    logger.info("done; outputs in %s", out_dir)


def _prefetch(args: argparse.Namespace) -> None:
    from byutils import prefetch_dataset, prefetch_model

    from mcqa_head_finding import data

    if args.dataset == "arc-easy":
        prefetch_dataset(data.ARC_PATH, name=data.ARC_CONFIG)
    elif args.dataset == "vocab":
        prefetch_dataset(data.VOCAB_PATH)
    keys = args.models if args.mode == "accuracy" else [args.model]
    for model_key in keys:
        logger.info("prefetching %s", MODELS[model_key])
        prefetch_model(MODELS[model_key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("accuracy", "analyze"), default="accuracy")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODEL_LADDER))
    parser.add_argument("--model", choices=list(MODELS), default="llama-3.2-1b")
    parser.add_argument(
        "--dataset",
        choices=("arc-easy", "synthetic", "addition", "vocab"),
        default="arc-easy",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="wrap the prompt in the model's chat template (for instruct models)",
    )
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument(
        "--no-hf-model",
        dest="use_hf_model",
        action="store_false",
        help="stream weights from cache instead of preloading an HF model (for 70B)",
    )
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--time", default="01:00:00")
    parser.add_argument("--gpus", default="p100:1")
    parser.add_argument("--mem-per-cpu", default="16G")
    parser.add_argument("--qos", default=None)
    args = parser.parse_args()

    if _SLURM_AVAILABLE and not args.local:
        if args.prefetch and is_login_node():
            _prefetch(args)
        slurm = SlurmConfig(
            job_type="compute",
            time=args.time,
            gpus_per_node=args.gpus,
            mem_per_cpu=args.mem_per_cpu,
            qos=args.qos,
        )
        Path("slurm_logs").mkdir(exist_ok=True)
        submit_slurm_job(slurm, python_cmd="uv run python")

    from datetime import datetime

    from runlog import start_run

    cfg = {k: v for k, v in vars(args).items() if k not in ("prefetch", "local")}
    key = "-".join(args.models) if args.mode == "accuracy" else args.model
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.mode}_{args.dataset}_{key}_{job_id}"
    out_dir = start_run(Path("logs"), cfg, run_name=run_name)
    _run(args, out_dir)


if __name__ == "__main__":
    main()
