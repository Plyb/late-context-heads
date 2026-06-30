"""Per-head direct logit attribution onto the answer letters, and top-p selection.

For each correct example, at the final token position:

    delta_abcd(head) = (1 / RMS_p) * W_U[[a,b,c,d]] . f_head        # apply_ln_to_stack + W_U
    delta_net(head)  = delta_abcd(head) - mean_letters delta_abcd(head)

``f_head`` is the head's write to the residual stream (``stack_head_results``); the
``1 / RMS_p`` scaling and folded final-norm gain come from ``apply_ln_to_stack`` on a
``fold_ln=True`` model. The per-head direct effect is the net contribution to the
*correct* letter, averaged across correct examples.
"""

from typing import NamedTuple, cast

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

ANSWER_LETTERS = ("A", "B", "C", "D")


class TopHeads(NamedTuple):
    labels: list[str]
    layer_heads: list[tuple[int, int]]
    cumulative_fraction: float
    total_positive: float
    exceeded_max: bool


def letter_direction_ids(letter_ids: dict[str, int]) -> Tensor:
    return torch.tensor([letter_ids[l] for l in ANSWER_LETTERS], dtype=torch.long)


def _parse_label(label: str) -> tuple[int, int]:
    layer_str, head_str = label[1:].split("H")
    return int(layer_str), int(head_str)


class Accumulator:
    """Running float32 sum of per-head net direct effect over correct examples."""

    def __init__(self) -> None:
        self._sum: Tensor | None = None
        self.labels: list[str] | None = None
        self.count: int = 0

    def add(self, net_effect: Float[Tensor, "heads"], labels: list[str]) -> None:
        if self._sum is None:
            self._sum = net_effect.clone()
            self.labels = labels
        else:
            self._sum += net_effect
        self.count += 1

    def mean(self) -> Float[Tensor, "heads"]:
        if self._sum is None or self.count == 0:
            raise ValueError("no correct examples accumulated")
        return self._sum / self.count


def _dla_names_filter(name: str) -> bool:
    return name.endswith("hook_z") or name.endswith("hook_resid_post")


@torch.no_grad()
def head_net_effect(
    model: HookedTransformer,
    tokens: Tensor,
    letter_dir_ids: Tensor,
    correct_idx: int,
) -> tuple[Float[Tensor, "heads"], list[str], Float[Tensor, "vocab"]]:
    """Per-head net direct effect on the correct letter at the final position.

    Computed manually (rather than ``stack_head_results`` + ``apply_ln_to_stack``) so it
    works under multi-GPU model parallelism, where those helpers ``cat`` tensors living on
    different devices. Each head's residual write ``z @ W_O`` is gathered onto the unembed
    device, divided by the final RMSNorm scale (a common positive scalar -> ranking matches
    the helper path), then projected through ``W_U`` onto [a,b,c,d] and mean-centred.
    """
    logits, cache = model.run_with_cache(tokens, names_filter=_dla_names_filter)
    n_layers = model.cfg.n_layers
    target = model.W_U.device

    parts: list[Tensor] = []
    labels: list[str] = []
    for layer in range(n_layers):
        z = cache["z", layer][0, -1].float()  # (n_heads, d_head)
        w_o = cast(Tensor, getattr(model.blocks[layer].attn, "W_O")).float()
        contrib = torch.einsum("hf,hfm->hm", z, w_o)  # (n_heads, d_model)
        parts.append(contrib.to(target))
        labels.extend(f"L{layer}H{h}" for h in range(contrib.shape[0]))
    per_head = torch.cat(parts, dim=0)  # (heads, d_model)

    resid_final = cache["resid_post", n_layers - 1][0, -1].float().to(target)
    scale = (resid_final.pow(2).mean() + model.cfg.eps).sqrt()
    normed = per_head / scale
    letter_dirs = model.W_U[:, letter_dir_ids.to(target)].float()  # (d_model, 4)
    delta_abcd = normed @ letter_dirs  # (heads, 4)
    delta_net = delta_abcd - delta_abcd.mean(dim=-1, keepdim=True)
    final_logits = cast(Tensor, logits)[0, -1].float().cpu()
    return delta_net[:, correct_idx].cpu(), labels, final_logits


def select_top_heads(
    mean_effect: np.ndarray,
    labels: list[str],
    top_p: float = 0.8,
    max_heads: int = 200,
) -> TopHeads:
    order = np.argsort(mean_effect)[::-1]
    positive_total = float(mean_effect[mean_effect > 0].sum())
    target = top_p * positive_total
    selected: list[int] = []
    running = 0.0
    for idx in order:
        if mean_effect[idx] <= 0:
            break
        selected.append(int(idx))
        running += float(mean_effect[idx])
        if running >= target:
            break
    chosen_labels = [labels[i] for i in selected]
    return TopHeads(
        labels=chosen_labels,
        layer_heads=[_parse_label(l) for l in chosen_labels],
        cumulative_fraction=running / positive_total if positive_total > 0 else 0.0,
        total_positive=positive_total,
        exceeded_max=len(selected) > max_heads,
    )
