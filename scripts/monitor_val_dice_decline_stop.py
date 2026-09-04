#!/usr/bin/env python3
"""Stop fold training when every-N val Dice first declines (after base_ep).

Supports:
  - nnunet / dpdnet: parse ``Pseudo dice [x]`` from nnUNet training_log*.txt
  - seganypet: metrics.jsonl ``val_dice`` at val-interval epochs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def _parse_nnunet_pseudo_dice(log_path: Path) -> list[tuple[int, float]]:
    """Return (finished_epoch, dice) for each val point."""
    if not log_path.is_file():
        return []
    cur = 0
    out: list[tuple[int, float]] = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.search(r"Epoch\s+(\d+)\s*$", line)
        if m:
            cur = int(m.group(1))
            continue
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
        ep = cur + 1 if "finished_ep" not in line else cur
        try:
            out.append((int(ep), float(raw)))
        except ValueError:
            continue
    # dedupe by epoch keep last
    dedup: dict[int, float] = {}
    for ep, v in out:
        dedup[ep] = v
    return sorted(dedup.items())


def _parse_seganypet_metrics(metrics: Path) -> list[tuple[int, float]]:
    if not metrics.is_file():
        return []
    out: list[tuple[int, float]] = []
    for line in metrics.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        vd = r.get("val_dice")
        ep = r.get("epoch")
        if vd is None or ep is None:
            continue
        try:
            vf = float(vd)
        except (TypeError, ValueError):
            continue
        if vf != vf:
            continue
        out.append((int(ep), vf))
    dedup: dict[int, float] = {}
    for ep, v in out:
        dedup[ep] = v
    return sorted(dedup.items())


def _latest_training_log(fold_dir: Path) -> Path | None:
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _fold_dir(method: str, work: Path, repo: Path, parent: str, fold: int) -> Path:
    nn_root = work / "nnUNet_results"
    if method == "nnunet":
        return (
            nn_root
            / f"{parent}_f{fold}"
            / "Dataset228_AutoPETIV_Task1_2ch"
            / "nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"
            / "fold_0"
        )
    if method == "dpdnet":
        return (
            nn_root
            / f"{parent}_f{fold}"
            / "Dataset240_DpDNet_PSMA_2ch"
            / "STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
            / f"fold_{fold}"
        )
    if method == "seganypet":
        return repo / parent / "seganypet" / f"fold{fold}"
    raise ValueError(method)


def _read_val_series(method: str, fold_dir: Path) -> list[tuple[int, float]]:
    if method == "seganypet":
        return _parse_seganypet_metrics(fold_dir / "metrics.jsonl")
    lg = _latest_training_log(fold_dir)
    return _parse_nnunet_pseudo_dice(lg) if lg else []


def _write_stop(stamp: str, work: Path, reason: str) -> Path:
    vis = work / "01_train_vis"
    vis.mkdir(parents=True, exist_ok=True)
    stop = vis / f"TASK1_TRAIN_STOP_{stamp}.txt"
    stop.write_text(
        "\n".join(
            [
                f"stopped_at={time.strftime('%Y-%m-%dT%H:%M:%S')}",
                f"STAMP={stamp}",
                "by=monitor_val_dice_decline_stop.py",
                f"reason={reason}",
            ]
        )
        + "\n"
    )
    return stop


def _kill_fold(method: str, stamp: str, fold: int, ctrl: Path) -> None:
    env = os.environ.copy()
    env["TASK1_NNUNET_RESULTS_STAMP_NAME"] = stamp
    subprocess.run(
        ["bash", str(ctrl / "scripts/task1_stop_train_and_resume.sh")],
        env=env,
        check=False,
    )
    if method == "dpdnet":
        subprocess.run(["docker", "rm", "-f", f"dpdnet_psma_f{fold}_{stamp}"], check=False)
    if method == "seganypet":
        subprocess.run(["docker", "rm", "-f", f"seganypet_fs50_f{fold}_{stamp}"], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["nnunet", "dpdnet", "seganypet"])
    ap.add_argument("--parent-stamp", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--base-ep", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--poll-sec", type=float, default=45.0)
    ap.add_argument("--work", default="/media/ybwang/data1/PSMA-DATA/task1_train_workspace")
    ap.add_argument("--repo", default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/3D-MAE-PET-CT/runs")
    ap.add_argument("--ctrl", default="/media/ybwang/data1/PSMA-CTRL")
    args = ap.parse_args()

    work = Path(args.work)
    repo = Path(args.repo)
    ctrl = Path(args.ctrl)
    fold = int(args.fold)
    parent = args.parent_stamp.strip()
    if args.method in ("nnunet", "dpdnet"):
        stamp = f"{parent}_f{fold}"
    else:
        stamp = parent

    fold_dir = _fold_dir(args.method, work, repo, parent, fold)
    vis = Path(args.ctrl) / "ICLR2026" / "vis" / "decline_mon_state"
    vis.mkdir(parents=True, exist_ok=True)
    state_path = vis / f"{args.method}_{parent}_f{fold}.json"
    seen_eps: set[int] = set()
    prev_dice: float | None = None

    # seed from existing history up to base_ep
    series = _read_val_series(args.method, fold_dir)
    for ep, dice in series:
        if ep <= args.base_ep:
            prev_dice = dice
            seen_eps.add(ep)
    state = {
        "method": args.method,
        "stamp": stamp,
        "fold": fold,
        "base_ep": args.base_ep,
        "prev_dice": prev_dice,
        "seen_eps": sorted(seen_eps),
        "status": "watching",
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(f"[decline-mon] start {args.method} f{fold} stamp={stamp} seed_dice@{args.base_ep}={prev_dice}")

    while True:
        series = _read_val_series(args.method, fold_dir)
        for ep, dice in series:
            if ep in seen_eps:
                continue
            if ep <= args.base_ep:
                seen_eps.add(ep)
                prev_dice = dice
                continue
            seen_eps.add(ep)
            print(f"[decline-mon] f{fold} ep={ep} val_dice={dice:.4f} prev={prev_dice}")
            if prev_dice is not None and dice < prev_dice - 1e-6:
                reason = f"val_dice_decline ep{ep} {dice:.4f} < prev {prev_dice:.4f}"
                stop = _write_stop(stamp, work, reason)
                _kill_fold(args.method, stamp, fold, ctrl)
                state.update(
                    {
                        "status": "stopped",
                        "stop_ep": ep,
                        "stop_dice": dice,
                        "prev_dice": prev_dice,
                        "stop_file": str(stop),
                        "reason": reason,
                    }
                )
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                print(f"[decline-mon] STOP {reason}")
                return
            prev_dice = dice
            state["prev_dice"] = prev_dice
            state["seen_eps"] = sorted(seen_eps)
            state_path.write_text(json.dumps(state, indent=2) + "\n")

        # training finished naturally (final ckpt)
        if args.method in ("nnunet", "dpdnet"):
            if (fold_dir / "checkpoint_final.pth").is_file():
                lg = _latest_training_log(fold_dir)
                if lg:
                    m = re.findall(r"Epoch\s+(\d+)\s*$", lg.read_text(errors="ignore"))
                    if m and int(m[-1]) + 1 >= 300:
                        print("[decline-mon] training reached cap without decline")
                        return
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
