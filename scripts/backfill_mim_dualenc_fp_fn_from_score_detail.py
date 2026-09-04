#!/usr/bin/env python3
"""Backfill mean_fp/mean_fn for nnunet_mim / dpdnet_dualenc from existing score_detail.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CTRL = Path("/media/ybwang/data1/PSMA-CTRL")
WORK = Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace")
NN = WORK / "nnUNet_results"
VIS = CTRL / "ICLR2026/vis"
BOARD = VIS / "iclr2026_aligned_fdg_fs50_f258_board.json"

TARGETS = (
    ("nnunet_mim", ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258", "psma_fc70")),
    ("dpdnet_dualenc", ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258", "psma_fc70")),
)


def _load_score(eval_root: Path, fold: str) -> dict | None:
    p = eval_root / f"fold{fold}" / "score_detail.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None
    if not isinstance(d.get("mean_dice"), (int, float)):
        return None
    return d


def _pool(scores: list[dict]) -> tuple[float | None, float | None, float | None]:
    if not scores:
        return None, None, None
    dices = [float(s["mean_dice"]) for s in scores]
    mean = sum(dices) / len(dices)
    sum_fp = sum(int(s.get("sum_fp") or 0) for s in scores)
    sum_fn = sum(int(s.get("sum_fn") or 0) for s in scores)
    sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in scores)
    sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in scores)
    fp = (sum_fp / sum_neg) if sum_neg > 0 else None
    fn = (sum_fn / sum_pos) if sum_pos > 0 else None
    # fallback to mean of rates if sums missing
    if fp is None:
        rates = [float(s["fp_rate"]) for s in scores if isinstance(s.get("fp_rate"), (int, float))]
        fp = sum(rates) / len(rates) if rates else None
    if fn is None:
        rates = [float(s["fn_rate"]) for s in scores if isinstance(s.get("fn_rate"), (int, float))]
        fn = sum(rates) / len(rates) if rates else None
    return mean, fp, fn


def _update_aggregate(stamp: str, stage: str, fold_map: dict[str, dict], mean: float, fp, fn) -> None:
    root = NN / stamp
    if stage == "psma_fc70":
        names = [
            root / "aggregate_test20_dice_fc70.json",
            VIS / f"aggregate_nnunet_psma_fc70_{stamp}.json",
            VIS / f"aggregate_dpdnet_psma_fc70_{stamp}.json",
        ]
    else:
        names = [
            root / "aggregate_test20_dice_f258.json",
            VIS / f"aggregate_nnunet_psma_fs50_f258_{stamp}.json",
            VIS / f"aggregate_dpdnet_psma_test20_f258_{stamp}.json",
            VIS / f"aggregate_dpdnet_psma_fs50_f258_{stamp}.json",
        ]
    for path in names:
        if not path.is_file() and path.parent != root:
            continue
        if not path.is_file() and path.parent == root:
            # still write canonical under stamp root
            pass
        elif not path.is_file():
            continue
        try:
            ad = json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            ad = {}
        folds = dict(ad.get("folds") or {})
        for f, sc in fold_map.items():
            prev = dict(folds.get(f) or {})
            prev["test_dice"] = float(sc["mean_dice"])
            prev["mean_dice_positive"] = float(sc.get("mean_dice_positive") or sc["mean_dice"])
            prev["fp_rate"] = sc.get("fp_rate")
            prev["fn_rate"] = sc.get("fn_rate")
            prev["n_test"] = sc.get("n_scored") or prev.get("n_test")
            folds[f] = prev
        ad["folds"] = folds
        ad["fold_mean"] = mean
        ad["mean"] = mean
        ad["mean_fp"] = fp
        ad["mean_fn"] = fn
        ad["fp_rate"] = fp
        ad["fn_rate"] = fn
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ad, indent=2) + "\n")
        print(f"  [agg] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=BOARD)
    args = ap.parse_args()
    board = json.loads(args.board.read_text())
    for mkey, stages in TARGETS:
        methods = board.setdefault("methods", {})
        m = methods.setdefault(mkey, {})
        for stage in stages:
            st = m.get(stage) or {}
            stamp = (st.get("stamp") or "").strip()
            if not stamp:
                print(f"[skip] {mkey}.{stage}: no stamp")
                continue
            eval_root = NN / stamp / "psma_test20_eval"
            if not eval_root.is_dir():
                print(f"[skip] {mkey}.{stage}: no eval dir")
                continue
            fold_ids = [str(i) for i in range(9)] if stage != "psma_fc70" else ["0"]
            fold_map: dict[str, dict] = {}
            for f in fold_ids:
                sc = _load_score(eval_root, f)
                if sc is None:
                    continue
                fold_map[f] = sc
            if not fold_map:
                print(f"[skip] {mkey}.{stage}: no score_detail")
                continue
            mean, fp, fn = _pool(list(fold_map.values()))
            fd = {f: float(sc["mean_dice"]) for f, sc in fold_map.items()}
            st = m.setdefault(stage, {})
            st["fold_dice"] = fd
            if mean is not None:
                st["mean"] = mean
            if fp is not None:
                st["mean_fp"] = fp
            if fn is not None:
                st["mean_fn"] = fn
            st["status"] = "done"
            st["eval_done"] = len(fd)
            st["eval_total"] = 1 if stage == "psma_fc70" else 9
            if stage == "psma_fc70":
                st["note"] = (
                    f"TEST20 DONE · fc70 · {100*float(mean):.2f}%/"
                    f"{100*float(fp or float('nan')):.2f}%/{100*float(fn or float('nan')):.2f}%"
                )
            else:
                st["note"] = (
                    f"TEST20 DONE · {len(fd)}/9 · {100*float(mean):.2f}%/"
                    f"{100*float(fp or float('nan')):.2f}%/{100*float(fn or float('nan')):.2f}%"
                )
            print(
                f"[ok] {mkey}.{stage} folds={sorted(fd)} mean={mean:.4f} "
                f"fp={fp} fn={fn}"
            )
            _update_aggregate(stamp, stage, fold_map, float(mean), fp, fn)
    board["updated_note"] = "backfill MIM/dualenc FP/FN from score_detail"
    args.board.write_text(json.dumps(board, indent=2) + "\n")
    print(f"[done] {args.board}")


if __name__ == "__main__":
    main()
