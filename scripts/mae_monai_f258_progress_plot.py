#!/usr/bin/env python3
"""Progress plot for monai_swinvit fewshot50 f2/5/8."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def _load_jsonl(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _best(rows):
    best = None
    for r in rows:
        vd = r.get("val_dice")
        if vd is None or vd != vd:
            continue
        vd = float(vd)
        if best is None or vd > best:
            best = vd
    return best


def collect(run_root: Path, folds, ft_epochs: int):
    jobs = []
    for fold in folds:
        rows = _load_jsonl(run_root / "monai" / f"fold{fold}" / "metrics.jsonl")
        cur = int(rows[-1]["epoch"]) if rows else 0
        secs = [float(r["epoch_sec"]) for r in rows[-min(10, len(rows)) :] if r.get("epoch_sec")]
        avg = float(np.mean(secs)) if secs else None
        best = _best(rows)
        mpath = run_root / "monai" / f"fold{fold}" / "metrics.jsonl"
        done = cur >= ft_epochs
        running = mpath.is_file() and (not done) and (time.time() - mpath.stat().st_mtime) < 180
        jobs.append({"fold": fold, "cur": cur, "best": best, "done": done, "running": running, "avg_sec": avg, "rows": rows})
    mean_fold = 3600.0
    for j in jobs:
        if j["avg_sec"]:
            mean_fold = j["avg_sec"] * ft_epochs
            break
    rem = []
    for j in jobs:
        if j["done"]:
            rem.append(0.0)
        elif j["avg_sec"]:
            rem.append(j["avg_sec"] * max(0, ft_epochs - j["cur"]))
        else:
            rem.append(mean_fold)
    return {
        "jobs": jobs,
        "folds": folds,
        "ft_epochs": ft_epochs,
        "n_done": sum(1 for j in jobs if j["done"]),
        "eta_s": max(rem) if rem else None,
        "stamp": run_root.name,
    }


def render(state, out_png: Path):
    folds = state["folds"]
    ft_epochs = state["ft_epochs"]
    jobs = state["jobs"]
    fig = plt.figure(figsize=(11.5, 6.5), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    for j in jobs:
        if j["done"]:
            c, t = "#2ca02c", f"✓\n{j['best']:.3f}" if j["best"] is not None else "✓"
        elif j["running"] or j["cur"] > 0:
            c, t = "#ff7f0e", f"▶{j['cur']}/{ft_epochs}"
        else:
            c, t = "#d9d9d9", "·"
        fi = folds.index(j["fold"])
        ax0.add_patch(
            mpatches.FancyBboxPatch(
                (fi + 0.08, 0.12), 0.84, 0.76, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=c, edgecolor="#333"
            )
        )
        ax0.text(fi + 0.5, 0.5, t, ha="center", va="center", fontsize=10, fontweight="bold")
    ax0.set_xlim(0, len(folds))
    ax0.set_ylim(0, 1)
    ax0.set_xticks([i + 0.5 for i in range(len(folds))])
    ax0.set_xticklabels([f"f{f}" for f in folds])
    ax0.set_yticks([])
    ax0.set_title(f"[train] monai_swinvit jobs {state['n_done']}/{len(jobs)}", fontweight="bold")
    for sp in ax0.spines.values():
        sp.set_visible(False)

    ax1 = fig.add_subplot(gs[0, 1])
    active = next((j for j in jobs if j["running"] or (j["cur"] > 0 and not j["done"])), None)
    if active:
        ax1.barh([0], [1.0], color="#eee", height=0.45)
        ax1.barh([0], [active["cur"] / max(ft_epochs, 1)], color="#ff7f0e", height=0.45)
        ax1.set_yticks([0])
        ax1.set_yticklabels([f"f{active['fold']}"])
        ax1.set_title(f"current train ep {active['cur']}/{ft_epochs}")
        if active["best"] is not None:
            ax1.text(0.98, 0.5, f"best={active['best']:.3f}", transform=ax1.transAxes, ha="right", va="center")
    else:
        ax1.barh([0], [state["n_done"] / max(len(jobs), 1)], color="#2ca02c", height=0.45)
        ax1.set_title("no active / done")
    ax1.set_xlim(0, 1)

    ax2 = fig.add_subplot(gs[1, :])
    ys = [j["best"] if j["best"] is not None else np.nan for j in jobs]
    ax2.bar(np.arange(len(folds)), ys, color="#1f77b4", width=0.55)
    ax2.set_xticks(np.arange(len(folds)))
    ax2.set_xticklabels([f"f{f}" for f in folds])
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("best val Dice")
    ax2.grid(True, axis="y", alpha=0.3)
    ok = [y for y in ys if y == y]
    mean_line = f"μ={np.mean(ok):.3f}±{np.std(ok):.3f} (n={len(ok)})" if ok else "waiting…"
    ax2.set_title(f"[eval] fewshot val Dice · {mean_line}", fontweight="bold")

    finish = (datetime.now(_tz()) + timedelta(seconds=max(0, int(state["eta_s"] or 0)))).strftime("%m-%d %H:%M")
    fig.suptitle(
        f"MONAI SwinViT SSL → fewshot50 f2/5/8\nETA {_fmt_eta(state['eta_s'])} → {finish}  ·  {state['stamp']}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument("--ft-epochs", type=int, default=100)
    ap.add_argument("--poll", type=float, default=0)
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    def once():
        state = collect(args.run_root, folds, args.ft_epochs)
        render(state, args.out_png)
        print(
            f"[monai-progress] ft={state['n_done']}/{len(state['jobs'])} ETA={_fmt_eta(state['eta_s'])} → {args.out_png}",
            flush=True,
        )
        return state["n_done"] >= len(state["jobs"])

    if args.poll and args.poll > 0:
        while True:
            if once():
                break
            time.sleep(args.poll)
    else:
        once()


if __name__ == "__main__":
    main()
