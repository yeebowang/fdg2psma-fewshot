#!/usr/bin/env python3
"""Recompute SegAnyPET TEST20 micro FP/FN from fold*_pred (CPU/docker).

Host python often lacks nibabel; scoring always runs in the MAE docker image.
Does not overwrite board FP/FN with null. Aggregates every fold 0–8 that has
TEST20 json / pred, not just the last FOLDS_CSV.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

CTRL = Path(__file__).resolve().parents[2]
VIS = CTRL / "ICLR2026/vis"
REPO = CTRL / "ICLR2026/3D-MAE-PET-CT/runs"
DATA = Path(os.environ.get("TASK1_BASE", "/media/ybwang/data1/PSMA-DATA"))
IMAGE = os.environ.get("TASK1_MAE_IMAGE", "iclr2026_3dmae_petct:cu118")
CASES_JSON = CTRL / "ICLR2026/data/splits_mae_psma_test20.json"
GT_DIR = DATA / "task1_train_workspace/seganypet_psma_test20/labelsVal"
SCORE_PY = CTRL / "ICLR2026/scripts/score_pred_dice_vs_gt.py"
BOARD = Path(os.environ.get("TASK1_ALIGN_BOARD_JSON", VIS / "iclr2026_aligned_fdg_fs50_f258_board.json"))

STAGE_KEYS = ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258", "psma_fc70")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and x == x


def _docker_score(pred_dir: Path, out_json: Path, tag: str, workers: int) -> bool:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{CTRL}:{CTRL}",
        "-v",
        f"{DATA}:{DATA}",
        "-w",
        str(CTRL),
        IMAGE,
        "python3",
        str(SCORE_PY),
        "--cases-json",
        str(CASES_JSON),
        "--pred-dir",
        str(pred_dir),
        "--gt-dir",
        str(GT_DIR),
        "--out-json",
        str(out_json),
        "--tag",
        tag,
        "--workers",
        str(workers),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"[seganypet-fpfn] score fail {tag} rc={r.returncode}\n{(r.stderr or r.stdout)[-800:]}\n")
        return False
    if r.stdout:
        print(r.stdout.strip().splitlines()[-1], flush=True)
    return out_json.is_file()


def _compact(sc: dict) -> dict:
    keep = (
        "mean_dice",
        "mean_dice_positive",
        "fp_rate",
        "fn_rate",
        "mean_fp",
        "mean_fn",
        "sum_fp",
        "sum_fn",
        "sum_pos_voxels",
        "sum_neg_voxels",
        "n_positive",
        "n_scored",
        "n_cases",
        "tag",
    )
    return {k: sc[k] for k in keep if k in sc}


def _sidecar(eval_root: Path, fold: int) -> Path:
    return eval_root / f"fold{fold}_score_fpfn.json"


def _score_ok(sc: dict) -> bool:
    return int(sc.get("sum_neg_voxels") or 0) > 0 or _finite(sc.get("fp_rate"))


def process_stamp(stamp: str, *, force: bool, workers: int) -> dict | None:
    eval_root = REPO / stamp / "psma_test20_eval"
    if not eval_root.is_dir():
        print(f"[seganypet-fpfn] skip {stamp}: no eval dir", flush=True)
        return None
    fold_scores: dict[str, dict] = {}
    fd: dict[str, float] = {}
    for f in range(9):
        j = eval_root / f"fold{f}_test20.json"
        pred = eval_root / f"fold{f}_pred"
        side = _sidecar(eval_root, f)
        if j.is_file():
            try:
                d = json.loads(j.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                d = {}
            md = d.get("mean_dice_positive", d.get("mean_dice"))
            if _finite(md):
                fd[str(f)] = float(md)
        sc = None
        if side.is_file() and not force:
            try:
                sc = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                sc = None
            if isinstance(sc, dict) and not _score_ok(sc):
                sc = None
        if sc is None and pred.is_dir() and any(pred.glob("*.nii.gz")):
            raw = eval_root / f"fold{f}_score_fpfn.raw.json"
            print(f"[seganypet-fpfn] score {stamp} fold{f}", flush=True)
            if _docker_score(pred, raw, f"seganypet_{stamp}_f{f}", workers):
                try:
                    full = json.loads(raw.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    full = {}
                sc = _compact(full)
                side.write_text(json.dumps(sc, indent=2) + "\n", encoding="utf-8")
                try:
                    raw.unlink()
                except OSError:
                    pass
        if isinstance(sc, dict) and _score_ok(sc):
            fold_scores[str(f)] = sc
            md = sc.get("mean_dice_positive", sc.get("mean_dice"))
            if _finite(md):
                fd[str(f)] = float(md)
    if not fd:
        return None
    sum_fp = sum(int(s.get("sum_fp") or 0) for s in fold_scores.values())
    sum_fn = sum(int(s.get("sum_fn") or 0) for s in fold_scores.values())
    sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in fold_scores.values())
    sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in fold_scores.values())
    fp = (sum_fp / sum_neg) if sum_neg > 0 else None
    fn = (sum_fn / sum_pos) if sum_pos > 0 else None
    vals = list(fd.values())
    mean = sum(vals) / len(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    summary = {
        "stamp": stamp,
        "method": "seganypet",
        "split": "PSMA_TEST20",
        "folds": [int(k) for k in sorted(fd, key=int)],
        "fold_test_dice": fd,
        "test_mean": mean,
        "test_std": std,
        "mean_dice": mean,
        "mean_dice_positive": mean,
        "fp_rate": fp,
        "fn_rate": fn,
        "mean_fp": fp,
        "mean_fn": fn,
        "sum_fp": sum_fp,
        "sum_fn": sum_fn,
        "sum_neg_voxels": sum_neg,
        "sum_pos_voxels": sum_pos,
        "n_folds_scored_fpfn": len(fold_scores),
        "empty_gt_excluded": True,
        "metric": "TEST20 Dice/FP/FN (pred vs labelsVal)",
    }
    (eval_root / "aggregate_test20_f258.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (VIS / f"aggregate_seganypet_psma_test20_f258_{stamp}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[seganypet-fpfn] {stamp} folds={len(fd)} fpfn_folds={len(fold_scores)} "
        f"Dice={mean:.4f} FP={fp} FN={fn}",
        flush=True,
    )
    return summary


def _stamps_from_board() -> list[str]:
    if not BOARD.is_file():
        return []
    try:
        board = json.loads(BOARD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    seg_keys = ("seganypet", "seganypet_scratch")
    out: list[str] = []
    seen: set[str] = set()
    for mk in seg_keys:
        seg = (board.get("methods") or {}).get(mk) or {}
        for sk in STAGE_KEYS:
            st = (seg.get(sk) or {}).get("stamp") or ""
            st = str(st).strip()
            if st and st not in seen:
                seen.add(st)
                out.append(st)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", action="append", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TASK1_FPFN_WORKERS", "8")))
    ap.add_argument("--refresh-board", action="store_true", default=True)
    ap.add_argument("--no-refresh-board", action="store_false", dest="refresh_board")
    args = ap.parse_args()
    stamps = [s.strip() for s in args.stamp if s.strip()] or _stamps_from_board()
    if not stamps:
        print("[seganypet-fpfn] no stamps", flush=True)
        return 1
    n_ok = 0
    for stamp in stamps:
        if process_stamp(stamp, force=args.force, workers=max(1, args.workers)):
            n_ok += 1
    if args.refresh_board and n_ok:
        subprocess.run(
            [
                "python3",
                str(CTRL / "ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py"),
                "--board",
                str(BOARD),
            ],
            cwd=str(CTRL),
            check=False,
        )
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
