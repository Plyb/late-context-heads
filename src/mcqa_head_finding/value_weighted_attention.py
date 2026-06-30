"""Collect value-weighted attention from the top heads, bucketed by token type.

Value-weighted attention from the final token to source token ``j`` for a head is
``pattern[final, j] * ||v_j||_2``. Per (example, head, column) we average it over the
column's tokens (empty columns are skipped, so they leave the per-column mean divisor),
then average across correct examples.
"""

import numpy as np
import torch
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformers import PreTrainedTokenizerBase

from .data import Example
from .direct_effect import ANSWER_LETTERS
from .prompts import PromptFormat, assemble_prompt
from .token_types import build_columns


@torch.no_grad()
def head_value_weighted_attention(
    cache: ActivationCache, layer: int, head: int, n_query_heads: int
) -> Tensor:
    pattern = cache["pattern", layer][0, head, -1, :]  # (key_pos,)
    v_all = cache["v", layer][0]  # (key_pos, n_v_heads, d_head)
    n_v_heads = v_all.shape[1]
    kv_head = head // (n_query_heads // n_v_heads)
    v_norm = v_all[:, kv_head, :].float().norm(dim=-1)  # (key_pos,)
    return (pattern.float() * v_norm).cpu()


class Collector:
    """Accumulate value-weighted attention into token-type heatmap matrices."""

    def __init__(
        self,
        model: HookedTransformer,
        tokenizer: PreTrainedTokenizerBase,
        layer_heads: list[tuple[int, int]],
        head_labels: list[str],
        chat_template: bool = False,
        dataset: str = "generic",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_heads = layer_heads
        self.head_labels = head_labels
        self.chat_template = chat_template
        self.dataset = dataset
        self.n_query_heads = model.cfg.n_heads
        self.main_names: list[str] | None = None
        self.nogroup_names: list[str] | None = None
        self._main_sum: np.ndarray | None = None
        self._main_count: np.ndarray | None = None
        self._byhead_sum: np.ndarray | None = None  # (head, 4, cols_nogroup)
        self._byhead_count: np.ndarray | None = None

    def _ensure(self, n_main: int, n_ng: int) -> None:
        h = len(self.layer_heads)
        if self._main_sum is None:
            self._main_sum = np.zeros((h, n_main), dtype=np.float64)
            self._main_count = np.zeros((h, n_main), dtype=np.int64)
            self._byhead_sum = np.zeros((h, len(ANSWER_LETTERS), n_ng), dtype=np.float64)
            self._byhead_count = np.zeros(
                (h, len(ANSWER_LETTERS), n_ng), dtype=np.int64
            )

    @torch.no_grad()
    def add(self, example: Example, fmt: PromptFormat, cache: ActivationCache) -> None:
        prompt = assemble_prompt(
            example.question,
            example.options,
            fmt,
            self.tokenizer,
            self.chat_template,
            self.dataset,
        )
        ast = not self.chat_template
        ds = self.dataset
        cols_g = build_columns(prompt, self.tokenizer, example.answer_label, True, ast, ds)
        cols_ng = build_columns(prompt, self.tokenizer, example.answer_label, False, ast, ds)
        self._ensure(len(cols_g.names), len(cols_ng.names))
        assert self._main_sum is not None and self._main_count is not None
        assert self._byhead_sum is not None and self._byhead_count is not None
        if self.main_names is None:
            self.main_names = cols_g.names
            self.nogroup_names = cols_ng.names

        letter_idx = ANSWER_LETTERS.index(example.answer_label)
        for h, (layer, head) in enumerate(self.layer_heads):
            vwa = head_value_weighted_attention(
                cache, layer, head, self.n_query_heads
            ).numpy()
            for c, idx in enumerate(cols_g.indices):
                if idx:
                    self._main_sum[h, c] += float(vwa[idx].mean())
                    self._main_count[h, c] += 1
            for c, idx in enumerate(cols_ng.indices):
                if idx:
                    self._byhead_sum[h, letter_idx, c] += float(vwa[idx].mean())
                    self._byhead_count[h, letter_idx, c] += 1

    def main_matrix(self) -> np.ndarray:
        assert self._main_sum is not None and self._main_count is not None
        return np.divide(
            self._main_sum,
            self._main_count,
            out=np.zeros_like(self._main_sum),
            where=self._main_count > 0,
        )

    def per_head_matrices(self) -> np.ndarray:
        assert self._byhead_sum is not None and self._byhead_count is not None
        return np.divide(
            self._byhead_sum,
            self._byhead_count,
            out=np.zeros_like(self._byhead_sum),
            where=self._byhead_count > 0,
        )
