#!/usr/bin/env python3
"""Parse nnUNet/DpDNet Pseudo dice from training logs; pick best-by-dice epoch."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DicePoint:
    epoch: int
    dice: float
    log: str


def parse_pseudo_dice_series(fold_dir: Path) -> dict[int, float]:
    """Return {epoch: pseudo_dice} merged across all training_log*.txt (chronological)."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    out: dict[int, float] = {}
    for lg in logs:
        cur = 0
        for line in lg.read_text(errors="ignore").splitlines():
            m = re.search(r"Epoch\s+(\d+)\s*$", line)
            if m:
                cur = int(m.group(1))
            m = re.search(r"finished_ep=(\d+)", line)
            if m:
                cur = int(m.group(1))
            if "Pseudo dice skipped" in line:
                continue
            m = re.search(r"Pseudo dice \[([^\]]+)\]", line)
            if not m:
                continue
            raw = m.group(1).strip().lower()
            if raw in ("nan", "none", ""):
                continue
            ep = cur + 1
            m2 = re.search(r"finished_ep=(\d+)", line)
            if m2:
                ep = int(m2.group(1))
            try:
                out[int(ep)] = float(raw)
            except ValueError:
                continue
    return out


def pseudo_dice_best_epoch(fold_dir: Path) -> tuple[int | None, float | None, dict[int, float]]:
    series = parse_pseudo_dice_series(fold_dir)
    if not series:
        return None, None, series
    ep = max(series, key=series.get)
    return ep, series[ep], series


def dpdnet_fold_dir(work: Path, parent: str, fold: int) -> Path:
    fd = (
        work
        / "nnUNet_results"
        / f"{parent}_f{fold}"
        / "Dataset240_DpDNet_PSMA_2ch"
        / "STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
        / f"fold_{fold}"
    )
    if not fd.is_dir():
        alt = fd.parent / "fold_0"
        if alt.is_dir():
            return alt
    return fd


def _ckpt_mtime_iso(p: Path) -> str | None:
    if not p.is_file():
        return None
    return datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds")


def assess_fold(
    work: Path,
    parent: str,
    fold: int,
) -> dict:
    fd = dpdnet_fold_dir(work, parent, fold)
    best_ep, best_dice, series = pseudo_dice_best_epoch(fd)
    final = fd / "checkpoint_final.pth"
    best = fd / "checkpoint_best.pth"
    latest = fd / "checkpoint_latest.pth"
    action = "ok"
    note = ""
    if best_ep is None:
        action = "missing_dice"
        note = "no Pseudo dice in logs"
    elif best_ep <= 100 and final.is_file():
        # Original 100ep run: checkpoint_final == last epoch weights
        if not best.is_file() or best.stat().st_mtime > final.stat().st_mtime + 3600:
            action = "restore_final"
            note = f"best dice @ep{best_ep}; restore checkpoint_final → checkpoint_best"
        else:
            action = "ok"
            note = f"best dice @ep{best_ep}; checkpoint_best likely ep100"
    elif best_ep > 100:
        if not best.is_file():
            action = "needs_retrain"
            note = f"best dice @ep{best_ep}; missing checkpoint_best"
        elif latest.is_file() and best.stat().st_mtime >= latest.stat().st_mtime - 60:
            action = "ok"
            note = f"best dice @ep{best_ep}={best_dice:.4f}; checkpoint_best fresh"
        elif final.is_file() and best.stat().st_mtime <= final.stat().st_mtime:
            val_loss_eps = _val_loss_best_epochs(fd)
            if val_loss_eps and max(val_loss_eps) > best_ep:
                action = "needs_retrain"
                note = (
                    f"best dice @ep{best_ep}={best_dice:.4f}; "
                    f"checkpoint_best likely @ep{max(val_loss_eps)} (val_loss); retrain to ep{best_ep}"
                )
            else:
                action = "ok"
                note = f"best dice @ep{best_ep}; keep checkpoint_best"
        else:
            action = "ok"
            note = f"best dice @ep{best_ep}={best_dice:.4f}; keep checkpoint_best"
    return {
        "fold": fold,
        "fold_dir": str(fd),
        "best_ep": best_ep,
        "best_dice": best_dice,
        "series": {str(k): v for k, v in sorted(series.items())},
        "action": action,
        "note": note,
        "checkpoint_final": _ckpt_mtime_iso(final),
        "checkpoint_best": _ckpt_mtime_iso(best),
        "checkpoint_latest": _ckpt_mtime_iso(latest),
    }


def _val_loss_best_epochs(fold_dir: Path) -> list[int]:
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    eps: list[int] = []
    for lg in logs:
        cur = 0
        for line in lg.read_text(errors="ignore").splitlines():
            m = re.search(r"Epoch\s+(\d+)\s*$", line)
            if m:
                cur = int(m.group(1))
            if "New best val_loss" in line and cur is not None:
                eps.append(cur + 1)
    return eps


def apply_restore_final(fold_dir: Path, dry_run: bool = False) -> None:
    final = fold_dir / "checkpoint_final.pth"
    best = fold_dir / "checkpoint_best.pth"
    if not final.is_file():
        raise FileNotFoundError(final)
    bak = fold_dir / "checkpoint_best.val_loss_bak.pth"
    if dry_run:
        print(f"[dry-run] cp {final} -> {best}")
        return
    import subprocess

    final_s = str(final.resolve())
    best_s = str(best.resolve())
    bak_s = str(bak.resolve())
    vol = Path("/media/ybwang/data1/PSMA-DATA")
    for anc in fold_dir.resolve().parents:
        if anc.name == "task1_train_workspace":
            vol = anc.parent
            break
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "--entrypoint",
            "bash",
            "-v",
            f"{vol}:{vol}",
            "autopet_baseline:latest",
            "-lc",
            (
                f"set -e; "
                f"[[ -f '{best_s}' && ! -f '{bak_s}' ]] && cp -a '{best_s}' '{bak_s}' || true; "
                f"cp -a '{final_s}' '{best_s}'; echo restored"
            ),
        ],
        check=True,
    )
    print(f"[apply] restored checkpoint_best from checkpoint_final ({fold_dir.name})")


def write_manifest(path: Path, parent: str, folds: list[dict]) -> None:
    payload = {
        "parent_stamp": parent,
        "policy": "checkpoint_best = max Pseudo dice (val every 20ep)",
        "folds": {str(f["fold"]): f for f in folds},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-stamp", required=True)
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument("--work", default="/media/ybwang/data1/PSMA-DATA/task1_train_workspace")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()

    work = Path(args.work)
    folds_i = [int(x) for x in args.folds.split(",") if x.strip()]
    reports = [assess_fold(work, args.parent_stamp, f) for f in folds_i]

    if args.manifest:
        write_manifest(args.manifest, args.parent_stamp, reports)

    retrain: list[int] = []
    for r in reports:
        print(
            f"f{r['fold']}: best@ep{r['best_ep']} dice={r['best_dice']} "
            f"action={r['action']} | {r['note']}"
        )
        if args.apply and r["action"] == "restore_final":
            apply_restore_final(Path(r["fold_dir"]), dry_run=args.dry_run)
        elif args.apply and r["action"] == "needs_retrain":
            retrain.append(int(r["fold"]))

    if retrain:
        print("NEEDS_RETRAIN_FOLDS=" + ",".join(str(x) for x in retrain))
    else:
        print("NEEDS_RETRAIN_FOLDS=")


if __name__ == "__main__":
    main()
