#!/usr/bin/env python3
"""Progress plot for nnUNet PSMA fewshot50 f2/5/8 (1GPU/fold)."""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Shanghai")
    except Exception:
        return timezone(timedelta(hours=8))


def _fmt_eta(eta_s):
    if eta_s is None:
        return "…"
    eta_s = max(0, int(eta_s))
    h, rem = divmod(eta_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _fold_epoch(fold_dir: Path, total: int) -> int:
    if (fold_dir / "checkpoint_final.pth").is_file():
        return total
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    ep = 0
    if not logs:
        return 0
    for line in logs[0].read_text(errors="ignore").splitlines():
        for pat in (r"Epoch[: ]+(\d+)", r"current epoch[: ]+(\d+)", r"epoch[: ]+(\d+)"):
            m = re.search(pat, line, re.I)
            if m:
                ep = max(ep, int(m.group(1)))
    return ep


def _fold_train_losses(fold_dir: Path) -> list[float]:
    """Parse train loss from log if present (best-effort)."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return []
    losses = []
    for line in logs[0].read_text(errors="ignore").splitlines():
        m = re.search(r"train[_ ]loss[:\s=]+(-?\d+\.?\d*(?:e-?\d+)?)", line, re.I)
        if m:
            try:
                losses.append(float(m.group(1)))
            except ValueError:
                pass
    return losses


def collect(parent: str, work: Path, folds, total_ep: int, ds: str, tf: str) -> dict:
    rows = {}
    for f in folds:
        stamp = f"{parent}_f{f}"
        fd = work / "nnUNet_results" / stamp / ds / tf / "fold_0"
        ep = _fold_epoch(fd, total_ep) if fd.is_dir() else 0
        rows[f] = {
            "stamp": stamp,
            "epoch": ep,
            "done": ep >= total_ep or (fd / "checkpoint_final.pth").is_file(),
            "losses": _fold_train_losses(fd) if fd.is_dir() else [],
        }
    agg = work / "nnUNet_results" / parent / "aggregate_val_dice_f258.json"
    fold_dice = {}
    mean = None
    if agg.is_file():
        d = json.loads(agg.read_text())
        mean = d.get("fold_mean")
        for k, v in (d.get("folds") or {}).items():
            if isinstance(v, dict) and v.get("best_val_dice") is not None:
                fold_dice[int(k)] = float(v["best_val_dice"])
    return {
        "parent": parent,
        "rows": rows,
        "fold_dice": fold_dice,
        "mean": mean,
        "total_ep": total_ep,
        "done": agg.is_file() and all(rows[f]["done"] for f in folds),
    }


def render(state: dict, out_png: Path, folds) -> None:
    rows = state["rows"]
    total = state["total_ep"]
    fold_dice = state["fold_dice"]
    mean = state["mean"]

    fig = plt.figure(figsize=(11.5, 6.8), dpi=130)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # train progress bars
    labels = [f"f{f}" for f in folds]
    fracs = [rows[f]["epoch"] / max(total, 1) for f in folds]
    colors = ["#2ca02c" if rows[f]["done"] else "#1f77b4" for f in folds]
    bars = ax1.barh(labels[::-1], fracs[::-1], color=colors[::-1], height=0.55)
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("train progress")
    for y, f, frac in zip(range(len(folds)), folds[::-1], fracs[::-1]):
        ax1.text(min(frac + 0.02, 0.98), y, f"{rows[f]['epoch']}/{total}", va="center", fontsize=9)
    n_done = sum(1 for f in folds if rows[f]["done"])
    ax1.set_title(f"[train] nnUNet PSMA fewshot50  ·  {n_done}/{len(folds)} folds done", loc="left")

    # eval bars or placeholder
    if fold_dice:
        xs = [f"f{f}" for f in folds]
        ys = [fold_dice.get(f, float("nan")) for f in folds]
        ax2.bar(xs, ys, color="#9467bd", edgecolor="#4a148c", width=0.7)
        for i, y in enumerate(ys):
            if y == y:
                ax2.text(i, y + 0.01, f"{y:.3f}", ha="center", fontsize=9, fontweight="bold")
        if mean is not None:
            ax2.axhline(mean, color="#d62728", ls="--", label=f"mean={mean:.4f}")
            ax2.legend(loc="upper right")
        ax2.set_ylim(0, min(1.0, max([y for y in ys if y == y] + [0.2]) * 1.35))
        ax2.set_ylabel("val Dice")
        ax2.set_title(f"[eval] shared PSMA val Dice  ·  μ={mean:.4f}" if mean else "[eval]", loc="left")
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "val Dice pending (after 300ep)…", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_axis_off()

    eta = None
    # rough ETA from slowest fold
    eps = [rows[f]["epoch"] for f in folds]
    if min(eps) > 5 and min(eps) < total:
        # unknown rate — omit precise
        rem = total - min(eps)
        eta = rem * 40  # ~40s/ep guess for bs6 70iter 1gpu
    finish = (datetime.now(_tz()) + timedelta(seconds=eta)).strftime("%m-%d %H:%M") if eta else "?"
    fig.suptitle(
        f"nnUNet PSMA fewshot50 f2/5/8  ·  1GPU/fold bs=6 tr=70 noval 300ep\n"
        f"ETA {_fmt_eta(eta)} → {finish}  ·  {state['parent']}",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--work", type=Path, default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace"))
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=0)
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    ds = "Dataset228_AutoPETIV_Task1_2ch"
    tf = "nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"

    def once():
        st = collect(args.parent, args.work, folds, args.epochs, ds, tf)
        render(st, args.out_png, folds)
        print(
            f"[plot] " + " ".join(f"f{f}={st['rows'][f]['epoch']}" for f in folds)
            + f" mean={st['mean']} → {args.out_png}",
            flush=True,
        )
        return st["done"]

    if args.poll > 0:
        while True:
            if once():
                break
            time.sleep(args.poll)
    else:
        once()


if __name__ == "__main__":
    main()
