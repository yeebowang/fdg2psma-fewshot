#!/usr/bin/env python3
"""Live progress plot for MAE PSMA fewshot50 × 9fold (ssl + nossl).

Shows:
  - overall job grid (18 cells): pending / running / done
  - current fold epoch bar + ETA
  - completed folds' best val Dice
  - title with overall ETA → finish time

Usage:
  python3 ICLR2026/scripts/mae_fewshot50_9fold_progress_plot.py \\
    --run-root .../runs/<STAMP> --out-png ICLR2026/vis/progress_....png --poll 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")


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


def _best_val_dice(metrics: Path) -> float | None:
    if not metrics.is_file():
        return None
    best = None
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        vd = r.get("val_dice")
        if vd is None or vd != vd:
            continue
        vd = float(vd)
        if best is None or vd > best:
            best = vd
    return best


def _epoch_progress(metrics: Path, total_epochs: int) -> tuple[int, float | None]:
    """Return (cur_epoch, mean_epoch_sec from recent)."""
    if not metrics.is_file():
        return 0, None
    rows = []
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return 0, None
    cur = int(rows[-1].get("epoch", len(rows)))
    secs = []
    for r in rows[-min(10, len(rows)) :]:
        es = r.get("epoch_sec")
        if es is not None:
            try:
                s = float(es)
                if s > 0:
                    secs.append(s)
            except (TypeError, ValueError):
                pass
    avg = float(np.mean(secs)) if secs else None
    return min(cur, total_epochs), avg


def _parse_status(status_path: Path) -> dict:
    done = set()
    t0 = None
    fold_secs: list[float] = []
    if status_path.is_file():
        for line in status_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "start":
                t0 = status_path.stat().st_mtime  # approx
            if ev.get("event") == "fold_done":
                done.add((ev.get("cond"), int(ev.get("fold", -1))))
                if "sec" in ev:
                    fold_secs.append(float(ev["sec"]))
    return {"done": done, "fold_secs": fold_secs, "status_mtime": t0}


def _active_from_docker_or_logs(run_root: Path, stamp: str, conds: list[str], n_folds: int) -> tuple[str | None, int | None]:
    """Infer running (cond, fold) from metrics growth or log mtime."""
    newest = None
    newest_mtime = -1.0
    for cond in conds:
        for fold in range(n_folds):
            m = run_root / cond / f"fold{fold}" / "metrics.jsonl"
            log = (
                Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/vis")
                / f"nohup_mae_psma_fs50_{cond}_fold{fold}_{stamp}.log"
            )
            for p in (m, log):
                if p.is_file():
                    mt = p.stat().st_mtime
                    # consider active if updated in last 180s and not finished
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = (cond, fold, p)
    if newest is None:
        return None, None
    cond, fold, p = newest
    # if train done marker in log, not active
    log = (
        Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/vis")
        / f"nohup_mae_psma_fs50_{cond}_fold{fold}_{stamp}.log"
    )
    if log.is_file():
        txt = log.read_text(errors="ignore")
        if "[train] done" in txt and time.time() - log.stat().st_mtime > 90:
            return None, None
        if time.time() - log.stat().st_mtime < 180 or (
            (run_root / cond / f"fold{fold}" / "metrics.jsonl").is_file()
            and time.time() - (run_root / cond / f"fold{fold}" / "metrics.jsonl").stat().st_mtime < 180
        ):
            return cond, fold
    return None, None


def collect_state(
    run_root: Path,
    stamp: str,
    *,
    n_folds: int,
    total_epochs: int,
    conds: list[str],
) -> dict:
    st = _parse_status(run_root / "status.jsonl")
    jobs = []
    for cond in conds:
        for fold in range(n_folds):
            metrics = run_root / cond / f"fold{fold}" / "metrics.jsonl"
            cur_ep, avg_sec = _epoch_progress(metrics, total_epochs)
            best = _best_val_dice(metrics)
            done_flag = (cond, fold) in st["done"]
            if not done_flag and cur_ep >= total_epochs and best is not None:
                # metrics complete even if status not flushed yet
                log = (
                    Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/vis")
                    / f"nohup_mae_psma_fs50_{cond}_fold{fold}_{stamp}.log"
                )
                if log.is_file() and "[train] done" in log.read_text(errors="ignore"):
                    done_flag = True
            jobs.append(
                {
                    "cond": cond,
                    "fold": fold,
                    "cur_ep": cur_ep,
                    "avg_epoch_sec": avg_sec,
                    "best_dice": best,
                    "done": done_flag,
                    "started": cur_ep > 0 or metrics.is_file(),
                }
            )

    active_cond, active_fold = _active_from_docker_or_logs(run_root, stamp, conds, n_folds)
    for j in jobs:
        j["running"] = (
            active_cond is not None
            and j["cond"] == active_cond
            and j["fold"] == active_fold
            and not j["done"]
        )

    n_done = sum(1 for j in jobs if j["done"])
    n_total = len(jobs)

    # ETA: remaining folds * mean fold duration; current fold remainder
    fold_secs = list(st["fold_secs"])
    # estimate fold duration from completed or from current epoch pace
    if not fold_secs:
        for j in jobs:
            if j["done"] and j["avg_epoch_sec"]:
                fold_secs.append(j["avg_epoch_sec"] * total_epochs)
    mean_fold = float(np.mean(fold_secs)) if fold_secs else None
    if mean_fold is None:
        # bootstrap from running fold
        for j in jobs:
            if j["running"] and j["avg_epoch_sec"]:
                mean_fold = j["avg_epoch_sec"] * total_epochs
                break

    eta_s = None
    if mean_fold is not None:
        remain_jobs = n_total - n_done
        # subtract progress of current running job
        cur_remain = 0.0
        for j in jobs:
            if j["running"]:
                left_ep = max(0, total_epochs - j["cur_ep"])
                sec = j["avg_epoch_sec"] or (mean_fold / total_epochs)
                cur_remain = left_ep * sec
                remain_jobs = max(0, remain_jobs - 1)
                break
        eta_s = cur_remain + remain_jobs * mean_fold

    return {
        "jobs": jobs,
        "n_done": n_done,
        "n_total": n_total,
        "active": (active_cond, active_fold),
        "eta_s": eta_s,
        "mean_fold_sec": mean_fold,
        "conds": conds,
        "n_folds": n_folds,
        "total_epochs": total_epochs,
        "stamp": stamp,
    }


def render(state: dict, out_png: Path) -> None:
    jobs = state["jobs"]
    conds = state["conds"]
    n_folds = state["n_folds"]
    total_epochs = state["total_epochs"]
    n_done = state["n_done"]
    n_total = state["n_total"]
    eta_s = state["eta_s"]

    fig = plt.figure(figsize=(11.5, 7.2), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.28)

    # ---- grid of jobs ----
    ax0 = fig.add_subplot(gs[0, :])
    # matrix: rows=conds, cols=folds
    colors = []
    texts = []
    for ci, cond in enumerate(conds):
        row_c, row_t = [], []
        for fold in range(n_folds):
            j = next(x for x in jobs if x["cond"] == cond and x["fold"] == fold)
            if j["done"]:
                c = "#2ca02c"
                t = f"✓\n{j['best_dice']:.3f}" if j["best_dice"] is not None else "✓"
            elif j["running"]:
                c = "#ff7f0e"
                t = f"▶{j['cur_ep']}/{total_epochs}"
            elif j["started"]:
                c = "#f0ad4e"
                t = f"{j['cur_ep']}/{total_epochs}"
            else:
                c = "#d9d9d9"
                t = "·"
            row_c.append(c)
            row_t.append(t)
        colors.append(row_c)
        texts.append(row_t)

    for ci, cond in enumerate(conds):
        for fold in range(n_folds):
            ax0.add_patch(
                mpatches.FancyBboxPatch(
                    (fold + 0.08, len(conds) - 1 - ci + 0.12),
                    0.84,
                    0.76,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=colors[ci][fold],
                    edgecolor="#333333",
                    linewidth=0.8,
                )
            )
            ax0.text(
                fold + 0.5,
                len(conds) - 1 - ci + 0.5,
                texts[ci][fold],
                ha="center",
                va="center",
                fontsize=9,
                color="#111" if colors[ci][fold] != "#2ca02c" else "#fff",
                fontweight="bold",
            )
    ax0.set_xlim(0, n_folds)
    ax0.set_ylim(0, len(conds))
    ax0.set_xticks([i + 0.5 for i in range(n_folds)])
    ax0.set_xticklabels([f"f{i}" for i in range(n_folds)])
    ax0.set_yticks([i + 0.5 for i in range(len(conds))])
    ax0.set_yticklabels(list(reversed([c.upper() for c in conds])))
    ax0.set_aspect("equal")
    ax0.set_title("job grid  (gray=pending · orange=running · green=done+bestDice)", fontsize=10)
    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.tick_params(length=0)

    # ---- current fold epoch bar ----
    ax1 = fig.add_subplot(gs[1, 0])
    active = state["active"]
    if active[0] is not None:
        j = next(x for x in jobs if x["cond"] == active[0] and x["fold"] == active[1])
        frac = j["cur_ep"] / max(total_epochs, 1)
        ax1.barh([0], [frac], color="#ff7f0e", height=0.45)
        ax1.barh([0], [1.0], color="#eeeeee", height=0.45, zorder=0)
        ax1.set_xlim(0, 1)
        ax1.set_yticks([0])
        ax1.set_yticklabels([f"{active[0]} fold{active[1]}"])
        left_ep = max(0, total_epochs - j["cur_ep"])
        sec = j["avg_epoch_sec"]
        fold_eta = (left_ep * sec) if sec else None
        ax1.set_title(
            f"current: ep {j['cur_ep']}/{total_epochs}  |  fold ETA {_fmt_eta(fold_eta)}",
            fontsize=10,
        )
        if j["best_dice"] is not None:
            ax1.text(0.98, 0.5, f"bestDice={j['best_dice']:.3f}", transform=ax1.transAxes, ha="right", va="center", fontsize=9)
    else:
        ax1.barh([0], [n_done / max(n_total, 1)], color="#2ca02c", height=0.45)
        ax1.set_xlim(0, 1)
        ax1.set_yticks([])
        ax1.set_title("no active fold (idle or finished)", fontsize=10)
    ax1.set_xlabel("epoch fraction")

    # ---- dice bars for done folds ----
    ax2 = fig.add_subplot(gs[1, 1])
    width = 0.35
    x = np.arange(n_folds)
    for i, cond in enumerate(conds):
        ys = []
        for fold in range(n_folds):
            j = next(z for z in jobs if z["cond"] == cond and z["fold"] == fold)
            ys.append(j["best_dice"] if j["best_dice"] is not None else np.nan)
        ax2.bar(x + (i - 0.5) * width, ys, width=width, label=cond, color=("#1f77b4" if cond == "ssl" else "#9467bd"), alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"f{i}" for i in range(n_folds)])
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("best val Dice")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title("per-fold best val Dice", fontsize=10)

    # means in subtitle
    means = []
    for cond in conds:
        vals = [j["best_dice"] for j in jobs if j["cond"] == cond and j["best_dice"] is not None]
        if vals:
            means.append(f"{cond} μ={np.mean(vals):.3f}±{np.std(vals):.3f} (n={len(vals)})")
    mean_line = "  |  ".join(means) if means else "waiting first val…"

    fig.suptitle(
        f"MAE fewshot50 × {n_folds}fold progress  ({n_done}/{n_total} jobs)\n"
        f"ETA {_fmt_eta(eta_s)} → {_fmt_finish(eta_s)}   ·   {mean_line}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--stamp", default="")
    ap.add_argument("--n-folds", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--conds", default="ssl,nossl")
    ap.add_argument("--poll", type=float, default=0, help="if >0, refresh every N seconds until done")
    args = ap.parse_args()

    stamp = args.stamp or args.run_root.name
    conds = [c.strip() for c in args.conds.split(",") if c.strip()]

    def once() -> bool:
        state = collect_state(
            args.run_root,
            stamp,
            n_folds=args.n_folds,
            total_epochs=args.epochs,
            conds=conds,
        )
        render(state, args.out_png)
        print(
            f"[fs50-progress] {state['n_done']}/{state['n_total']} "
            f"active={state['active']} ETA={_fmt_eta(state['eta_s'])} → {args.out_png}",
            flush=True,
        )
        return state["n_done"] >= state["n_total"]

    if args.poll and args.poll > 0:
        while True:
            done = once()
            if done:
                break
            time.sleep(args.poll)
    else:
        once()


if __name__ == "__main__":
    main()
