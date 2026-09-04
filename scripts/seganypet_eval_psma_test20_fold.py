#!/usr/bin/env python3
"""Eval one SegAnyPET fold ckpt on PSMA TEST20 (click protocol)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_CODE = Path(__file__).resolve().parents[1] / "third_party" / "SegAnyPET" / "code"
_PIP = Path(__file__).resolve().parents[1] / "third_party" / "seganypet_pip"
sys.path.insert(0, str(_PIP))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segment_anything.build_sam3D import sam_model_registry3D  # noqa: E402
from seganypet_fewshot_finetune import eval_val_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--stamp", default="")
    ap.add_argument("--num-clicks", type=int, default=5)
    ap.add_argument("--crop-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    mean = eval_val_set(
        model,
        args.test_root,
        args.pred_dir,
        num_clicks=args.num_clicks,
        crop_size=args.crop_size,
        max_cases=None,
        seed=42,
    )
    out = {
        "fold": args.fold,
        "stamp": args.stamp,
        "ckpt": str(args.ckpt),
        "split": "PSMA_TEST20",
        "mean_dice": float(mean) if mean == mean else None,
        "num_clicks": args.num_clicks,
        "protocol": "click_val_on_test20",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
