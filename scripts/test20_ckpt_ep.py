#!/usr/bin/env python3
"""Resolve TEST20 checkpoint epoch from nnUNet logs / SegAnyPET metrics."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _scan_logs_best_epoch(fold_dir: Path, marker: str) -> int | None:
    """Last 1-based epoch with *marker* across all training_log*.txt (chronological)."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    cur = None
    last = None
    for lg in logs:
        for line in lg.read_text(errors="ignore").splitlines():
            m = re.search(r"Epoch\s+(\d+)\s*$", line)
            if m:
                cur = int(m.group(1))
                continue
            if marker in line and cur is not None:
                last = cur + 1
    return last


def nnunet_val_loss_best_epoch(fold_dir: Path) -> int | None:
    return _scan_logs_best_epoch(fold_dir, "New best val_loss")


def nnunet_pseudo_dice_best_epoch(fold_dir: Path) -> int | None:
    try:
        from nnunet_pseudo_dice_best import pseudo_dice_best_epoch
    except ImportError:
        from ICLR2026.scripts.nnunet_pseudo_dice_best import pseudo_dice_best_epoch  # type: ignore

    ep, _dice, _series = pseudo_dice_best_epoch(fold_dir)
    return ep


def nnunet_ema_best_epoch(fold_dir: Path) -> int | None:
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    last = None
    for lg in logs:
        cur = None
        for line in lg.read_text(errors="ignore").splitlines():
            m = re.search(r"Epoch\s+(\d+)\s*$", line)
            if m:
                cur = int(m.group(1))
                continue
            if "New best EMA" in line and cur is not None and "nan" not in line.lower():
                last = cur + 1
    return last


def seganypet_best_epoch(metrics: Path) -> int | None:
    if not metrics.is_file():
        return None
    prev = None
    last = None
    for line in metrics.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        bd = r.get("best_val_dice")
        if bd is None:
            bd = r.get("best_dice")
        ep = r.get("epoch")
        try:
            bd_f = float(bd)
        except (TypeError, ValueError):
            continue
        if bd_f != bd_f:
            continue
        if prev is None or bd_f > prev + 1e-12:
            last = int(ep) if ep is not None else last
            prev = bd_f
    return last


def nnunet_fold_ckpt_ep(stamp: str, fold: int, work: Path, ckpt_name: str | None = None) -> int | None:
    fd228 = (
        work
        / "nnUNet_results"
        / f"{stamp}_f{fold}"
        / "Dataset228_AutoPETIV_Task1_2ch"
        / "nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"
        / "fold_0"
    )
    fd = fd228 if fd228.is_dir() else (
        work
        / "nnUNet_results"
        / f"{stamp}_f{fold}"
        / "Dataset240_DpDNet_PSMA_2ch"
        / "STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
        / f"fold_{fold}"
    )
    if not fd.is_dir():
        alt = fd.parent / "fold_0"
        fd = alt if alt.is_dir() else fd
    name = (ckpt_name or "").lower()
    if "final" in name:
        logs = sorted(fd.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
        if logs:
            ep = 0
            for lg in logs:
                for line in lg.read_text(errors="ignore").splitlines():
                    m = re.search(r"Epoch\s+(\d+)\s*$", line)
                    if m:
                        ep = max(ep, int(m.group(1)))
            return ep + 1 if ep else None
    ep = nnunet_val_loss_best_epoch(fd)
    if ep is None:
        ep = nnunet_ema_best_epoch(fd)
    return ep


def dpdnet_fold_ckpt_ep(parent: str, fold: int, work: Path, ckpt_name: str | None = None) -> int | None:
    fd = (
        work
        / "nnUNet_results"
        / f"{parent}_f{fold}"
        / "Dataset240_DpDNet_PSMA_2ch"
        / "STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
        / f"fold_{fold}"
    )
    if not fd.is_dir():
        fd = fd.parent / "fold_0"
    name = (ckpt_name or "checkpoint_best.pth").lower()
    if "final" in name:
        logs = sorted(fd.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
        if logs:
            ep = 0
            for lg in logs:
                for line in lg.read_text(errors="ignore").splitlines():
                    m = re.search(r"Epoch\s+(\d+)\s*$", line)
                    if m:
                        ep = max(ep, int(m.group(1)))
            return ep + 1 if ep else None
    return (
        nnunet_pseudo_dice_best_epoch(fd)
        or nnunet_ema_best_epoch(fd)
        or nnunet_val_loss_best_epoch(fd)
    )


def seganypet_fold_ckpt_ep(stamp: str, fold: int, repo: Path) -> int | None:
    m = repo / stamp / "seganypet" / f"fold{fold}" / "metrics.jsonl"
    return seganypet_best_epoch(m)


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: test20_ckpt_ep.py <method> <stamp> [fold]", file=sys.stderr)
        raise SystemExit(2)
    method, stamp = sys.argv[1], sys.argv[2]
    fold = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    work = Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace")
    repo = Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/3D-MAE-PET-CT/runs")
    ckpt = sys.argv[4] if len(sys.argv) > 4 else None
    if method == "nnunet":
        ep = nnunet_fold_ckpt_ep(stamp, fold, work, ckpt)
    elif method == "dpdnet":
        ep = dpdnet_fold_ckpt_ep(stamp, fold, work, ckpt)
    elif method == "seganypet":
        ep = seganypet_fold_ckpt_ep(stamp, fold, repo)
    else:
        raise SystemExit(f"unknown method {method}")
    print(ep if ep is not None else "")


if __name__ == "__main__":
    main()
