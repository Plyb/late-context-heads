"""Load a HuggingFace causal LM via byutils (if available) or direct HF loading, and wrap
it as a TransformerLens model.

``fold_ln=True`` folds the final RMSNorm gain into ``W_U`` and lets
``ActivationCache.apply_ln_to_stack`` supply the ``1 / RMS_p`` scaling, which is
exactly the direct-effect formula in ``task.md``.
"""

from typing import cast

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def select_dtype(device: torch.device) -> torch.dtype:
    """bf16 only on GPUs with native support (compute capability >= 8); else fp32.

    Older GPUs (cc < 8) report bf16 support but with slow emulation that degrades
    TransformerLens weight processing.
    """
    if device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float32


def load_hooked_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    n_devices: int = 1,
    use_hf_model: bool = True,
) -> tuple[HookedTransformer, PreTrainedTokenizerBase]:
    """Load a model into TransformerLens.

    Uses byutils caching if available (recommended for BYU ORC); otherwise falls back to
    the default HF_HOME. With ``use_hf_model=False``, TransformerLens streams weights
    from cache instead of holding a second full CPU copy -- needed for 70B.
    ``n_devices>1`` splits the layers across that many GPUs (model parallelism).
    """
    try:
        from byutils import load_model, load_tokenizer  # type: ignore[import-not-found]
        tokenizer = cast(PreTrainedTokenizerBase, load_tokenizer(model_id))
        hf_model = (
            load_model(model_id, model_class=AutoModelForCausalLM, dtype=dtype)
            if use_hf_model
            else None
        )
    except ImportError:
        tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(model_id))
        hf_model = (
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            if use_hf_model
            else None
        )

    model = HookedTransformer.from_pretrained(
        model_id,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=str(device),
        n_devices=n_devices,
        dtype=cast(str, dtype),
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    if n_devices > 1:
        _redistribute_across_devices(model)
    model.eval()
    return model, tokenizer


def _redistribute_across_devices(model: HookedTransformer) -> None:
    """Place modules to match what ``forward`` expects under model parallelism.

    TransformerLens 2.17's ``move_model_modules_to_device`` sends every module to a
    single device, but ``forward`` routes activations per block via
    ``get_device_for_block_index``. We re-place modules to that per-block layout so the
    two agree: embeddings on block 0's device, block i on its device, and the final norm
    and unembedding on the last block's device.
    """
    from transformer_lens.utilities.multi_gpu import get_device_for_block_index

    cfg = model.cfg
    first = get_device_for_block_index(0, cfg)
    last = get_device_for_block_index(cfg.n_layers - 1, cfg)
    model.embed.to(first)
    model.hook_embed.to(first)
    if cfg.positional_embedding_type != "rotary":
        model.pos_embed.to(first)
        model.hook_pos_embed.to(first)
    for i, block in enumerate(model.blocks):
        block.to(get_device_for_block_index(i, cfg))
    if hasattr(model, "ln_final"):
        model.ln_final.to(last)
    model.unembed.to(last)
