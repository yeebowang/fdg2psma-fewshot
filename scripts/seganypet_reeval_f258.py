#!/usr/bin/env python3
"""Re-run SegAnyPET click val on saved latest/best ckpts (fix torchio CropOrPad API)."""
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

from segment_anything.build_sam3D import sam_model_registry3D
from seganypet_fewshot_finetune import _binary_dice_nifti, eval_val_set  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--num-clicks", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=0, help="0=all")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for fold in folds:
        fold_dir = args.run_root / "seganypet" / f"fold{fold}"
        ckpt = fold_dir / "best.pth"
        if not ckpt.is_file():
            ckpt = fold_dir / "latest.pth"
        if not ckpt.is_file():
            print(f"[skip] fold{fold} no ckpt")
            results[fold] = None
            continue
        print(f"[eval] fold{fold} ← {ckpt.name}")
        model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()
        vmax = None if args.max_cases <= 0 else args.max_cases
        dice = eval_val_set(
            model,
            args.data_root / f"fold{fold}",
            fold_dir / "val_pred_reeval",
            num_clicks=args.num_clicks,
            crop_size=128,
            max_cases=vmax,
            seed=42,
        )
        results[fold] = dice
        print(f"[eval] fold{fold} dice={dice}")

    ok = [v for v in results.values() if v is not None and v == v]
    summary = {
        "folds": folds,
        "fold_dice": [results[f] for f in folds],
        "mean": (sum(ok) / len(ok)) if ok else None,
        "protocol": "reeval_click_val",
        "num_clicks": args.num_clicks,
    }
    out = args.run_root / "aggregate_val_dice_f258_reeval.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    # also patch main aggregate if all ok
    agg = args.run_root / "aggregate_val_dice_f258.json"
    if agg.is_file() and ok:
        d = json.loads(agg.read_text())
        d["fold_best_dice"] = [results[f] for f in folds]
        d["mean"] = summary["mean"]
        d["std"] = (
            (sum((x - summary["mean"]) ** 2 for x in ok) / len(ok)) ** 0.5 if summary["mean"] is not None else None
        )
        d["n_ok"] = len(ok)
        d["reeval_note"] = "click val re-run after torchio CropOrPad API fix"
        agg.write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
