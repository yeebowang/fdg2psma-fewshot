#!/usr/bin/env python3
"""Exit 0 if encoder first conv out-channels == --expect-out-ch (dual-enc wants 16)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _unwrap(obj):
    if not isinstance(obj, dict):
        return obj
    for k in ("network_weights", "state_dict", "model_state_dict", "encoder"):
        inner = obj.get(k)
        if isinstance(inner, dict) and inner:
            return inner
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--expect-out-ch", type=int, default=16)
    args = ap.parse_args()
    state = _unwrap(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    first = None
    for k, v in state.items():
        if not hasattr(v, "ndim") or v.ndim != 5:
            continue
        if "conv1.weight" not in k:
            continue
        if k.startswith("conv_blocks_context.0.") or k.startswith("blocks.0."):
            first = v
            break
    if first is None:
        sample = list(state.keys())[:12]
        raise SystemExit(f"no first conv in {args.ckpt}; sample={sample}")
    got = int(first.shape[0])
    print(f"[enc-width] {args.ckpt.name} out_ch={got} expect={args.expect_out_ch}")
    if got != args.expect_out_ch:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
