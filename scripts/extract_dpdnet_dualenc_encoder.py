#!/usr/bin/env python3
"""Extract STUNet encoder blocks from a 1ch trainer ckpt → best_encoder_{ct,pet}_epoch_94.pth."""
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
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = _unwrap(raw)
    enc = {}
    for k, v in state.items():
        if k.startswith("conv_blocks_context.") or k.startswith("blocks."):
            enc[k] = v
    if not enc:
        sample = list(state.keys())[:12]
        raise SystemExit(f"no encoder keys in {args.ckpt}; sample={sample}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enc, args.out)
    print(f"[extract] n_keys={len(enc)} → {args.out}")


if __name__ == "__main__":
    main()
