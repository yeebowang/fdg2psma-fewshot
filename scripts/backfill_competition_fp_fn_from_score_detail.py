#!/usr/bin/env python3
"""Backfill mean_fp/mean_fn for competition scratch (and any nnUNet-layout) from score_detail.json."""
from __future__ import annotations

import json
from pathlib import Path

CTRL = Path("/media/ybwang/data1/PSMA-CTRL")
WORK = Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace")
NN = WORK / "nnUNet_results"
VIS = CTRL / "ICLR2026/vis"
BOARD = VIS / "iclr2026_aligned_fdg_fs50_f258_board.json"

METHODS = (
    "hemingduo_scratch",
    "chenyixin_scratch",
    "hemingduo",
    "chenyixin",
    "nnunet",
    "nnunet_mim",
)
STAGES = ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258", "psma_fc70")


def _pool(scores: list[dict]) -> tuple[float | None, float | None]:
    if not scores:
        return None, None
    sum_fp = sum(int(s.get("sum_fp") or 0) for s in scores)
    sum_fn = sum(int(s.get("sum_fn") or 0) for s in scores)
    sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in scores)
    sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in scores)
    fp = (sum_fp / sum_neg) if sum_neg > 0 else None
    fn = (sum_fn / sum_pos) if sum_pos > 0 else None
    if fp is None:
        rates = [float(s["fp_rate"]) for s in scores if isinstance(s.get("fp_rate"), (int, float))]
        fp = sum(rates) / len(rates) if rates else None
    if fn is None:
        rates = [float(s["fn_rate"]) for s in scores if isinstance(s.get("fn_rate"), (int, float))]
        fn = sum(rates) / len(rates) if rates else None
    return fp, fn


def _scores(stamp: str) -> list[dict]:
    root = NN / stamp / "psma_test20_eval"
    out: list[dict] = []
    if not root.is_dir():
        return out
    for f in range(9):
        p = root / f"fold{f}" / "score_detail.json"
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d.get("mean_dice"), (int, float)):
            out.append(d)
    return out


def _patch_agg(stamp: str, fp: float | None, fn: float | None, scores: list[dict]) -> None:
    paths = [
        NN / stamp / "aggregate_test20_dice_f258.json",
        VIS / f"aggregate_nnunet_psma_fs50_f258_{stamp}.json",
        VIS / f"aggregate_nnunet_psma_fs10_f258_{stamp}.json",
        VIS / f"aggregate_nnunet_psma_fs5_f258_{stamp}.json",
    ]
    fold_fp: dict[str, float] = {}
    fold_fn: dict[str, float] = {}
    for f in range(9):
        p = NN / stamp / "psma_test20_eval" / f"fold{f}" / "score_detail.json"
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rfp = d.get("fp_rate", d.get("mean_fp"))
        rfn = d.get("fn_rate", d.get("mean_fn"))
        if isinstance(rfp, (int, float)):
            fold_fp[str(f)] = float(rfp)
        if isinstance(rfn, (int, float)):
            fold_fn[str(f)] = float(rfn)
    for path in paths:
        if not path.is_file():
            continue
        try:
            ad = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if fp is not None:
            ad["fp_rate"] = fp
            ad["mean_fp"] = fp
        if fn is not None:
            ad["fn_rate"] = fn
            ad["mean_fn"] = fn
        folds = ad.get("folds") if isinstance(ad.get("folds"), dict) else {}
        for fk, fv in folds.items():
            if not isinstance(fv, dict):
                continue
            if fk in fold_fp:
                fv["fp_rate"] = fold_fp[fk]
                fv["mean_fp"] = fold_fp[fk]
            if fk in fold_fn:
                fv["fn_rate"] = fold_fn[fk]
                fv["mean_fn"] = fold_fn[fk]
        path.write_text(json.dumps(ad, indent=2) + "\n")
        print(f"[ok] patched {path.name} fp={fp} fn={fn} n={len(scores)}")


def main() -> None:
    board = json.loads(BOARD.read_text())
    methods = board.setdefault("methods", {})
    for mkey in METHODS:
        m = methods.get(mkey) or {}
        for stage in STAGES:
            st = m.get(stage)
            if not isinstance(st, dict):
                continue
            stamp = (st.get("stamp") or "").strip()
            if not stamp:
                continue
            scores = _scores(stamp)
            if not scores:
                print(f"[skip] {mkey}.{stage} no score_detail under {stamp}")
                continue
            fp, fn = _pool(scores)
            _patch_agg(stamp, fp, fn, scores)
            if fp is not None:
                st["mean_fp"] = fp
            if fn is not None:
                st["mean_fn"] = fn
            md = st.get("mean")
            n = len(st.get("fold_dice") or {})
            if isinstance(md, (int, float)) and fp is not None and fn is not None:
                st["note"] = (
                    f"TEST20 {n}/9 · {100 * float(md):.2f}%/"
                    f"{100 * float(fp):.2f}%/{100 * float(fn):.2f}%"
                )
            print(f"[board] {mkey}.{stage} mean={md} fp={fp} fn={fn}")
    BOARD.write_text(json.dumps(board, indent=2) + "\n")
    print("[done] board written")


if __name__ == "__main__":
    main()
