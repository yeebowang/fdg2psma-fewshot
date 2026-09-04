#!/usr/bin/env python3
"""Unified train + eval progress plot for gentle-SSL → fewshot50 f2/5/8.

Single figure:
  - SSL train curves (total / mae_psma / mae_fdg / align) + epoch ETA
  - Fewshot job grid (ssl/nossl × f2,f5,f8) train progress
  - Eval panel: per-fold best val Dice (ssl vs nossl)  [=测试/选优进度]

Usage:
  python3 ICLR2026/scripts/mae_gentle_f258_progress_plot.py \\
    --run-root .../runs/<STAMP> --out-png .../progress_....png --poll 30
"""
from __future__ import annotations

import argparse
import json
import os
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

        return ZoneInfo(os.environ.get("TASK1_STAMP_TZ", "Asia/Shanghai"))
    except Exception:
        return timezone(timedelta(hours=8))


def _fmt_eta(eta_s: float | None) -> str:
    if eta_s is None:
        return "…"
    eta_s = max(0, int(eta_s))
    h, rem = divmod(eta_s, 3600)
    m, s = divmod(rem, 60)
    if h >= 48:
        d, h2 = divmod(h, 24)
        return f"{d}d{h2:02d}h"
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _fmt_finish(eta_s: float | None) -> str:
    if eta_s is None:
        return "?"
    return (datetime.now(_tz()) + timedelta(seconds=max(0, int(eta_s)))).strftime("%m-%d %H:%M")


def _load_jsonl(path: Path) -> list[dict]:
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


def _best_dice(rows: list[dict]) -> float | None:
    best = None
    for r in rows:
        vd = r.get("val_dice")
        if vd is None or vd != vd:
            continue
        vd = float(vd)
        if best is None or vd > best:
            best = vd
    return best


def collect(run_root: Path, folds: list[int], ssl_epochs: int, ft_epochs: int) -> dict:
    ssl_rows = _load_jsonl(run_root / "ssl_continued_gentle" / "metrics.jsonl")
    ssl_ep = int(ssl_rows[-1]["epoch"]) if ssl_rows else 0
    ssl_done = ssl_ep >= ssl_epochs and bool(ssl_rows)
    ssl_secs = [float(r["epoch_sec"]) for r in ssl_rows[-min(10, len(ssl_rows)) :] if r.get("epoch_sec")]
    ssl_avg = float(np.mean(ssl_secs)) if ssl_secs else None

    jobs = []
    for cond in ("ssl", "nossl"):
        for fold in folds:
            rows = _load_jsonl(run_root / cond / f"fold{fold}" / "metrics.jsonl")
            cur = int(rows[-1]["epoch"]) if rows else 0
            secs = [float(r["epoch_sec"]) for r in rows[-min(10, len(rows)) :] if r.get("epoch_sec")]
            avg = float(np.mean(secs)) if secs else None
            best = _best_dice(rows)
            log = (
                Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/vis")
                / f"nohup_mae_psma_fs50_{cond}_fold{fold}_gentle_{run_root.name}.log"
            )
            done = cur >= ft_epochs and (
                (log.is_file() and "[train] done" in log.read_text(errors="ignore")) or cur >= ft_epochs
            )
            # recent activity
            mpath = run_root / cond / f"fold{fold}" / "metrics.jsonl"
            running = False
            if mpath.is_file() and not done:
                running = (time.time() - mpath.stat().st_mtime) < 180
            elif log.is_file() and not done and cur == 0:
                running = (time.time() - log.stat().st_mtime) < 180
            jobs.append(
                {
                    "cond": cond,
                    "fold": fold,
                    "cur": cur,
                    "best": best,
                    "done": done,
                    "running": running,
                    "avg_sec": avg,
                    "rows": rows,
                }
            )

    # SSL running?
    ssl_log = (
        Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/vis")
        / f"nohup_mae_psma_ssl_gentle_{run_root.name}.log"
    )
    ssl_running = (not ssl_done) and ssl_log.is_file() and (time.time() - ssl_log.stat().st_mtime) < 180

    # ETA (parallel waves: ssl then nossl; folds in a wave run concurrently)
    n_folds = len(folds)
    mean_fold = 3600.0
    fold_secs = []
    for j in jobs:
        if j["avg_sec"]:
            fold_secs.append(j["avg_sec"] * ft_epochs)
    if fold_secs:
        mean_fold = float(np.mean(fold_secs))

    def _job_remain(j: dict) -> float:
        if j["done"]:
            return 0.0
        if j["avg_sec"] is not None:
            return j["avg_sec"] * max(0, ft_epochs - j["cur"])
        return mean_fold * max(0.0, 1.0 - j["cur"] / max(ft_epochs, 1))

    eta = None
    if not ssl_done:
        eta = (ssl_avg * max(0, ssl_epochs - ssl_ep) if ssl_avg else 20 * 260.0) + 2 * mean_fold
    else:
        # current wave = cond that still has unfinished jobs (prefer ssl)
        ssl_jobs = [j for j in jobs if j["cond"] == "ssl"]
        nossl_jobs = [j for j in jobs if j["cond"] == "nossl"]
        ssl_wave_done = all(j["done"] for j in ssl_jobs)
        if not ssl_wave_done:
            eta = max((_job_remain(j) for j in ssl_jobs), default=0.0) + mean_fold
        else:
            eta = max((_job_remain(j) for j in nossl_jobs), default=0.0)

    n_done = sum(1 for j in jobs if j["done"])
    return {
        "ssl_rows": ssl_rows,
        "ssl_ep": ssl_ep,
        "ssl_epochs": ssl_epochs,
        "ssl_done": ssl_done,
        "ssl_running": ssl_running,
        "jobs": jobs,
        "folds": folds,
        "ft_epochs": ft_epochs,
        "n_done": n_done,
        "n_total": len(jobs),
        "eta_s": eta,
        "stamp": run_root.name,
    }


def render(state: dict, out_png: Path) -> None:
    ssl_rows = state["ssl_rows"]
    folds = state["folds"]
    ft_epochs = state["ft_epochs"]
    jobs = state["jobs"]
    eta_s = state["eta_s"]

    fig = plt.figure(figsize=(12.0, 8.2), dpi=120)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.05], hspace=0.42, wspace=0.28)

    # ---- (0,:) SSL train curves ----
    ax0 = fig.add_subplot(gs[0, :])
    if ssl_rows:
        xs = [int(r["epoch"]) for r in ssl_rows]
        mk = dict(marker="o", markersize=3.5)
        ax0.plot(xs, [r["loss"] for r in ssl_rows], label="total", color="#1f77b4", lw=1.8, **mk)
        ax0.plot(xs, [r.get("mae_psma", np.nan) for r in ssl_rows], label="mae_psma", color="#ff7f0e", lw=1.3, **mk)
        ax0.plot(xs, [r.get("mae_fdg", np.nan) for r in ssl_rows], label="mae_fdg", color="#2ca02c", lw=1.3, **mk)
        ax0.plot(xs, [r.get("align", np.nan) for r in ssl_rows], label="align", color="#9467bd", lw=1.3, **mk)
        ax0.set_xlim(1, max(state["ssl_epochs"], max(xs)))
    else:
        ax0.text(0.5, 0.5, "SSL waiting…", ha="center", va="center", transform=ax0.transAxes)
        ax0.set_xlim(1, state["ssl_epochs"])
    ax0.set_xlabel("SSL epoch")
    ax0.set_ylabel("loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8, ncol=4)
    tag = "done" if state["ssl_done"] else ("running" if state["ssl_running"] else "pending")
    ax0.set_title(
        f"[train] gentle SSL  ep {state['ssl_ep']}/{state['ssl_epochs']}  ({tag})",
        fontsize=11,
        fontweight="bold",
    )

    # ---- (1,0) fewshot job grid ----
    ax1 = fig.add_subplot(gs[1, 0])
    conds = ["ssl", "nossl"]
    for ci, cond in enumerate(conds):
        for fi, fold in enumerate(folds):
            j = next(x for x in jobs if x["cond"] == cond and x["fold"] == fold)
            if j["done"]:
                c, t = "#2ca02c", (f"✓\n{j['best']:.3f}" if j["best"] is not None else "✓")
                tc = "#fff"
            elif j["running"] or (j["cur"] > 0 and not j["done"]):
                c, t, tc = "#ff7f0e", f"▶{j['cur']}/{ft_epochs}", "#111"
            else:
                c, t, tc = "#d9d9d9", "·", "#111"
            ax1.add_patch(
                mpatches.FancyBboxPatch(
                    (fi + 0.08, len(conds) - 1 - ci + 0.12),
                    0.84,
                    0.76,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=c,
                    edgecolor="#333",
                    lw=0.8,
                )
            )
            ax1.text(fi + 0.5, len(conds) - 1 - ci + 0.5, t, ha="center", va="center", fontsize=9, color=tc, fontweight="bold")
    ax1.set_xlim(0, len(folds))
    ax1.set_ylim(0, len(conds))
    ax1.set_xticks([i + 0.5 for i in range(len(folds))])
    ax1.set_xticklabels([f"f{f}" for f in folds])
    ax1.set_yticks([0.5, 1.5])
    ax1.set_yticklabels(["NOSSL", "SSL"])
    ax1.set_aspect("equal")
    ax1.set_title(
        f"[train] fewshot jobs  {state['n_done']}/{state['n_total']}  (gray=pending · orange=running · green=done)",
        fontsize=10,
        fontweight="bold",
    )
    for sp in ax1.spines.values():
        sp.set_visible(False)
    ax1.tick_params(length=0)

    # ---- (1,1) current train bar ----
    ax1b = fig.add_subplot(gs[1, 1])
    if not state["ssl_done"]:
        frac = state["ssl_ep"] / max(state["ssl_epochs"], 1)
        ax1b.barh([0], [1.0], color="#eee", height=0.45, zorder=0)
        ax1b.barh([0], [frac], color="#1f77b4", height=0.45)
        ax1b.set_yticks([0])
        ax1b.set_yticklabels(["SSL"])
        ax1b.set_title(f"current train: SSL {state['ssl_ep']}/{state['ssl_epochs']}", fontsize=10)
    else:
        active = next((j for j in jobs if j["running"] or (j["cur"] > 0 and not j["done"])), None)
        if active:
            frac = active["cur"] / max(ft_epochs, 1)
            ax1b.barh([0], [1.0], color="#eee", height=0.45, zorder=0)
            ax1b.barh([0], [frac], color="#ff7f0e", height=0.45)
            ax1b.set_yticks([0])
            ax1b.set_yticklabels([f"{active['cond']} f{active['fold']}"])
            ax1b.set_title(f"current train: ep {active['cur']}/{ft_epochs}", fontsize=10)
            if active["best"] is not None:
                ax1b.text(0.98, 0.5, f"bestDice={active['best']:.3f}", transform=ax1b.transAxes, ha="right", va="center", fontsize=9)
        else:
            frac = state["n_done"] / max(state["n_total"], 1)
            ax1b.barh([0], [frac], color="#2ca02c", height=0.45)
            ax1b.set_yticks([])
            ax1b.set_title("no active train job", fontsize=10)
    ax1b.set_xlim(0, 1)
    ax1b.set_xlabel("progress")

    # ---- (2,:) eval / test panel: val Dice ----
    ax2 = fig.add_subplot(gs[2, :])
    width = 0.35
    x = np.arange(len(folds))
    for i, cond in enumerate(conds):
        ys = []
        for fold in folds:
            j = next(z for z in jobs if z["cond"] == cond and z["fold"] == fold)
            ys.append(j["best"] if j["best"] is not None else np.nan)
        ax2.bar(
            x + (i - 0.5) * width,
            ys,
            width=width,
            label=cond,
            color=("#1f77b4" if cond == "ssl" else "#9467bd"),
            alpha=0.9,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"f{f}" for f in folds])
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("best val Dice")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)
    # mean line text
    means = []
    for cond in conds:
        vals = [j["best"] for j in jobs if j["cond"] == cond and j["best"] is not None]
        if vals:
            means.append(f"{cond} μ={np.mean(vals):.3f}±{np.std(vals):.3f} (n={len(vals)})")
    mean_line = "  |  ".join(means) if means else "waiting for first val Dice…"
    ax2.set_title(
        f"[eval] fewshot val Dice (shared PSMA val=59)  ·  {mean_line}",
        fontsize=10,
        fontweight="bold",
    )

    fig.suptitle(
        f"gentle SSL → fewshot50 f2/5/8  ·  train + eval progress\n"
        f"ETA {_fmt_eta(eta_s)} → {_fmt_finish(eta_s)}   ·   {state['stamp']}",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument("--ssl-epochs", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=100)
    ap.add_argument("--poll", type=float, default=0)
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    def once() -> bool:
        state = collect(args.run_root, folds, args.ssl_epochs, args.ft_epochs)
        render(state, args.out_png)
        print(
            f"[gentle-progress] ssl={state['ssl_ep']}/{state['ssl_epochs']} "
            f"ft={state['n_done']}/{state['n_total']} ETA={_fmt_eta(state['eta_s'])} → {args.out_png}",
            flush=True,
        )
        return state["ssl_done"] and state["n_done"] >= state["n_total"]

    if args.poll and args.poll > 0:
        while True:
            if once():
                break
            time.sleep(args.poll)
    else:
        once()


if __name__ == "__main__":
    main()
