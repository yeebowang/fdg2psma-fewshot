#!/usr/bin/env python3
"""Score nnUNet-style preds vs GT labels; write Dice + FP/FN JSON."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_voxel_metrics import aggregate_case_metrics, confusion_counts  # noqa: E402


def _score_one(args: tuple[str, str, str]) -> tuple[str, dict]:
    case, gt_path, pred_path = args
    try:
        gt = np.asarray(nib.load(gt_path).dataobj) > 0
        pred = np.asarray(nib.load(pred_path).dataobj) > 0
    except Exception as e:
        return case, {
            "dice": float("nan"),
            "gt_voxels": 0,
            "pred_voxels": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "pos_voxels": 0,
            "neg_voxels": 0,
            "fp_rate": float("nan"),
            "fn_rate": float("nan"),
            "error": str(e)[:200],
        }
    if gt.shape != pred.shape:
        try:
            from scipy.ndimage import zoom

            factors = [g / p for g, p in zip(gt.shape, pred.shape)]
            pred = zoom(pred.astype(np.float32), factors, order=0) > 0.5
        except Exception as e:
            return case, {
                "dice": float("nan"),
                "gt_voxels": int(gt.sum()),
                "pred_voxels": int(pred.sum()),
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "pos_voxels": int(gt.sum()),
                "neg_voxels": int((~gt).sum()),
                "fp_rate": float("nan"),
                "fn_rate": float("nan"),
                "shape_mismatch": True,
                "error": str(e)[:200],
            }
    return case, confusion_counts(gt, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-json", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="baseline1_fdg_psma_val")
    args = ap.parse_args()

    raw = json.loads(args.cases_json.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        cases = [str(x) for x in (raw[0].get("val") or [])]
        if not cases:
            cases = [str(x) for x in (raw[0].get("train") or [])]
    elif isinstance(raw, dict):
        cases = [str(x) for x in (raw.get("val") or [])]
        if not cases:
            cases = [str(x) for x in (raw.get("test") or raw.get("train") or raw.get("cases") or [])]
    else:
        cases = [str(x) for x in raw]

    jobs = []
    missing = []
    for c in cases:
        gt = args.gt_dir / f"{c}.nii.gz"
        pred = args.pred_dir / f"{c}.nii.gz"
        if not gt.is_file() or not pred.is_file():
            missing.append(c)
            continue
        jobs.append((c, str(gt), str(pred)))

    per_case: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_score_one, j) for j in jobs]
        for fut in as_completed(futs):
            case, metrics = fut.result()
            per_case[case] = metrics

    agg = aggregate_case_metrics(per_case)
    summary = {
        "tag": args.tag,
        "n_cases": len(cases),
        "n_missing": len(missing),
        "missing": missing[:20],
        **agg,
        "per_case": per_case,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[score] tag={args.tag} "
        f"Dice={summary['mean_dice']:.4f} FP={summary['fp_rate']:.4f} FN={summary['fn_rate']:.4f} "
        f"pos_n={summary['n_positive']} scored={summary['n_scored']}/{summary['n_cases']}"
    )


if __name__ == "__main__":
    main()
