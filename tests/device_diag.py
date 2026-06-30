"""Diagnose TransformerLens multi-GPU (n_devices) device placement, cheaply on 1B."""

import torch

from mcqa_head_finding.model import load_hooked_model, select_dtype


def main() -> None:
    device = torch.device("cuda")
    dtype = select_dtype(device)
    model, tok = load_hooked_model(
        "meta-llama/Llama-3.2-1B", device, dtype, n_devices=2, use_hf_model=True
    )
    print("n_gpus visible:", torch.cuda.device_count())
    print("cfg.device:", model.cfg.device, "n_devices:", model.cfg.n_devices)
    print("embed.W_E:", model.embed.W_E.device)
    print("unembed.W_U:", model.W_U.device)
    print("blocks[0]:", next(model.blocks[0].parameters()).device)
    print("blocks[-1]:", next(model.blocks[-1].parameters()).device)
    ids = tok("Hello world, this is a test", add_special_tokens=True)["input_ids"]
    for dev in ("cuda:0", "cuda:1"):
        try:
            out = model(torch.tensor([ids], device=dev), return_type="logits")
            print(f"forward OK, tokens on {dev}; logits on {out.device}")
            return
        except Exception as e:  # noqa: BLE001 - diagnostic
            print(f"forward FAIL, tokens on {dev}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
