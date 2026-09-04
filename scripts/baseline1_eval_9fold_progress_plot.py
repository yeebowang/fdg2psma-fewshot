#!/usr/bin/env python3
"""Progress / result plot for Baseline1 FDG→PSMA val 9fold zero-shot eval.

When scoring is done: detailed f0–f8 bars (with labels) + per-case Dice panel.
No blue predict progress bar on the final figure.

Usage:
  python3 ICLR2026/scripts/baseline1_eval_9fold_progress_plot.py \\
    --eval-root .../nnUNet_results/<STAMP>/psma_val_9fold \\
    --out-png ICLR2026/vis/progress_....png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def _find_eval_root(work: Path, stamp: str = "") -> Path:
    results = work / "nnUNet_results"
    if stamp:
        cand = results / stamp / "psma_val_9fold"
        if cand.is_dir():
            return cand
    matches = sorted(
        results.glob("*baseline1*eval*9fold*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise SystemExit("[b1-eval-progress] no *baseline1*eval*9fold* under nnUNet_results")
    return matches[0] / "psma_val_9fold"


def _prob_ok(pred_dir: Path, case: str) -> bool:
    for name in (f"{case}.npz", f"{case}.npz.npz"):
        p = pred_dir / name
        if p.is_file() and p.stat().st_size > 0:
            return True
    return False


def _count_predictions(pred_out: Path) -> tuple[int, int, set[str]]:
    cases_txt = pred_out / "cases.txt"
    n_expected = 59
    if cases_txt.is_file():
        n_expected = sum(1 for line in cases_txt.read_text().splitlines() if line.strip())

    done: set[str] = set()
    pred_flat = pred_out / "pred"

    def scan_dir(d: Path) -> None:
        if not d.is_dir():
            return
        for p in d.glob("*.nii.gz"):
            if "probabilities" in p.name:
                continue
            case = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
            if p.stat().st_size > 0 and _prob_ok(d, case):
                done.add(case)

    scan_dir(pred_flat)
    if len(done) < n_expected:
        for sd in sorted((pred_out / "shards").glob("shard_*/pred")):
            scan_dir(sd)
    return len(done), n_expected, done


def _shard_states(pred_out: Path) -> list[dict]:
    shards = []
    shard_root = pred_out / "shards"
    if not shard_root.is_dir():
        return shards
    for sd in sorted(shard_root.glob("shard_*")):
        cases_txt = sd / "cases.txt"
        n_assigned = 0
        if cases_txt.is_file():
            n_assigned = sum(1 for line in cases_txt.read_text().splitlines() if line.strip())
        pred_dir = sd / "pred"
        n_done = 0
        if pred_dir.is_dir():
            for p in pred_dir.glob("*.nii.gz"):
                if "probabilities" in p.name:
                    continue
                case = p.name[:-7]
                if p.stat().st_size > 0 and _prob_ok(pred_dir, case):
                    n_done += 1
        log = sd / "run.log"
        log_active = log.is_file() and (time.time() - log.stat().st_mtime) < 120
        shards.append(
            {
                "id": sd.name.replace("shard_", ""),
                "n_assigned": n_assigned,
                "n_done": n_done,
                "active": log_active and n_done < n_assigned,
            }
        )
    return shards


def _active_predict_containers(stamp: str) -> int:
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Command}}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    return sum(1 for line in out.splitlines() if "nnUNetv2_predict" in line and stamp in line)


def _load_score_detail(eval_root: Path) -> dict | None:
    p = eval_root / "score_detail.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect_state(eval_root: Path, stamp: str = "") -> dict:
    pred_out = eval_root / "predict"
    n_pred, n_expected, _ = _count_predictions(pred_out)
    shards = _shard_states(pred_out)

    t0_path = pred_out / "predict_t0.txt"
    t0 = None
    if t0_path.is_file():
        try:
            t0 = float(t0_path.read_text().strip())
        except ValueError:
            t0 = None

    elapsed = (time.time() - t0) if t0 else None
    eta_s = None
    if n_pred > 0 and elapsed and elapsed > 30 and n_pred < n_expected:
        rate = n_pred / elapsed
        if rate > 0:
            eta_s = (n_expected - n_pred) / rate

    agg_path = eval_root / "aggregate_val_dice_9fold.json"
    agg = None
    mean_dice = None
    mean_pos = None
    fold_dice: dict[int, float] = {}
    if agg_path.is_file():
        try:
            agg = json.loads(agg_path.read_text(encoding="utf-8"))
            mean_dice = agg.get("mean_dice")
            if mean_dice is not None:
                mean_dice = float(mean_dice)
            mean_pos = agg.get("mean_dice_positive")
            if mean_pos is not None:
                mean_pos = float(mean_pos)
            folds = agg.get("folds") or {}
            for k, v in folds.items():
                if isinstance(v, dict) and v.get("best_val_dice") is not None:
                    fold_dice[int(k)] = float(v["best_val_dice"])
                elif isinstance(v, (int, float)):
                    fold_dice[int(k)] = float(v)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    detail = _load_score_detail(eval_root)
    if detail and mean_pos is None and detail.get("mean_dice_positive") is not None:
        mean_pos = float(detail["mean_dice_positive"])

    if not stamp:
        stamp = eval_root.parent.name

    return {
        "eval_root": eval_root,
        "stamp": stamp,
        "n_pred": n_pred,
        "n_expected": n_expected,
        "eta_s": eta_s,
        "elapsed_s": elapsed,
        "shards": shards,
        "n_active_containers": _active_predict_containers(stamp),
        "mean_dice": mean_dice,
        "mean_dice_positive": mean_pos,
        "fold_dice": fold_dice,
        "detail": detail,
        "done": agg_path.is_file() and n_pred >= n_expected,
    }


def _render_in_progress(state: dict, out_png: Path) -> None:
    """While predicting: shard grid only (no big blue progress bar)."""
    n_pred = state["n_pred"]
    n_expected = state["n_expected"]
    shards = state["shards"]
    eta_s = state["eta_s"]
    frac = n_pred / max(n_expected, 1)

    fig = plt.figure(figsize=(11.0, 4.6), dpi=130)
    ax = fig.add_subplot(111)
    if shards:
        xs = np.arange(len(shards))
        done_h = [s["n_done"] for s in shards]
        tot_h = [max(s["n_assigned"], 1) for s in shards]
        colors = [
            "#ff7f0e" if s["active"] else "#2ca02c" if s["n_done"] >= s["n_assigned"] else "#9ecae1"
            for s in shards
        ]
        ax.bar(xs, done_h, color=colors, alpha=0.95, zorder=2)
        ax.bar(xs, tot_h, fill=False, edgecolor="#555", linewidth=0.9, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s{s['id']}\n{s['n_done']}/{s['n_assigned']}" for s in shards], fontsize=8)
        ax.set_ylabel("cases / shard")
        ax.set_title("shard progress (orange=active)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, f"predicting… {n_pred}/{n_expected}", ha="center", va="center", fontsize=14)
        ax.set_axis_off()

    status = f"{state['n_active_containers']} shard container(s)" if state["n_active_containers"] else "running"
    fig.suptitle(
        f"Baseline1 FDG eval PSMA 9fold  ·  {state['stamp']}\n"
        f"predict {n_pred}/{n_expected} ({100 * frac:.1f}%)  ·  {status}  ·  "
        f"ETA {_fmt_eta(eta_s)} → {_fmt_finish(eta_s)}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _render_done(state: dict, out_png: Path) -> None:
    """Final figure: detailed f0–f8 + per-case Dice (no blue progress bar)."""
    mean_dice = float(state["mean_dice"])
    mean_pos = state.get("mean_dice_positive")
    fold_dice = state["fold_dice"]
    detail = state.get("detail") or {}
    per_case = detail.get("per_case") or {}

    folds = sorted(fold_dice.keys()) if fold_dice else list(range(9))
    ys = [fold_dice.get(f, mean_dice) for f in folds]
    labels = [f"f{f}" for f in folds]

    # per-case series (sorted)
    case_rows = []
    for cid, v in per_case.items():
        case_rows.append(
            (
                cid,
                float(v.get("dice", float("nan"))),
                int(v.get("gt_voxels", 0) or 0),
                int(v.get("pred_voxels", 0) or 0),
            )
        )
    case_rows.sort(key=lambda t: t[1])
    case_dices = np.asarray([t[1] for t in case_rows], dtype=np.float64) if case_rows else np.asarray([])
    case_pos = case_dices[[i for i, t in enumerate(case_rows) if t[2] > 0]] if case_rows else np.asarray([])

    fig = plt.figure(figsize=(12.0, 7.2), dpi=140)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.35], hspace=0.38)
    ax_fold = fig.add_subplot(gs[0])
    ax_case = fig.add_subplot(gs[1])

    # --- fold bars with value labels ---
    x = np.arange(len(folds))
    bars = ax_fold.bar(x, ys, color="#2ca02c", edgecolor="#1b5e20", linewidth=0.8, width=0.72, zorder=2)
    ax_fold.axhline(
        mean_dice,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"fold mean = {mean_dice:.4f}",
        zorder=3,
    )
    for i, (bar, y) in enumerate(zip(bars, ys)):
        ax_fold.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.012,
            f"{y:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#145a32",
        )
    ax_fold.set_xticks(x)
    ax_fold.set_xticklabels(labels, fontsize=11)
    ax_fold.set_ylim(0, min(1.0, max(ys + [mean_dice]) * 1.35 + 0.05))
    ax_fold.set_ylabel("val Dice", fontsize=11)
    ax_fold.set_xlabel("fold", fontsize=11)
    ax_fold.grid(True, axis="y", alpha=0.3, zorder=0)
    pos_txt = f"  ·  positive-GT mean = {mean_pos:.4f}" if mean_pos is not None else ""
    ax_fold.set_title(
        f"[eval] per-fold val Dice  ·  f0–f8  ·  n_val={state['n_expected']} shared PSMA val"
        f"{pos_txt}",
        fontsize=11,
        loc="left",
    )
    ax_fold.legend(loc="upper right", fontsize=9, framealpha=0.92)
    # small note under legend area
    ax_fold.text(
        0.01,
        0.97,
        "zero-shot FDG-only ckpt · same score on every fold (shared val)",
        transform=ax_fold.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555",
    )

    # --- per-case detail ---
    if len(case_dices):
        xc = np.arange(len(case_dices))
        colors = ["#9e9e9e" if t[2] == 0 else "#4c78a8" for t in case_rows]
        ax_case.bar(xc, case_dices, color=colors, width=1.0, edgecolor="none", zorder=2)
        ax_case.axhline(mean_dice, color="#d62728", linestyle="--", linewidth=1.2, zorder=3, label=f"mean={mean_dice:.4f}")
        if mean_pos is not None:
            ax_case.axhline(
                mean_pos,
                color="#e67e22",
                linestyle=":",
                linewidth=1.3,
                zorder=3,
                label=f"pos mean={mean_pos:.4f}",
            )
        ax_case.set_xlim(-0.5, len(case_dices) - 0.5)
        ax_case.set_ylim(0, 1.05)
        ax_case.set_xlabel(f"val cases sorted by Dice  (n={len(case_dices)}; gray=empty GT)", fontsize=10)
        ax_case.set_ylabel("Dice", fontsize=11)
        ax_case.grid(True, axis="y", alpha=0.3, zorder=0)
        # percentile markers
        if len(case_dices) >= 4:
            for p, sty in ((25, ":"), (50, "-."), (75, ":")):
                v = float(np.percentile(case_dices, p))
                ax_case.axhline(v, color="#888", linestyle=sty, linewidth=0.8, alpha=0.7, zorder=1)
        n_empty = sum(1 for t in case_rows if t[2] == 0)
        n_pos = len(case_rows) - n_empty
        p50 = float(np.median(case_dices))
        pos_p50 = float(np.median(case_pos)) if len(case_pos) else float("nan")
        ax_case.set_title(
            f"[detail] per-case Dice on shared PSMA val  ·  "
            f"all median={p50:.3f}  ·  pos median={pos_p50:.3f}  ·  "
            f"pos={n_pos} emptyGT={n_empty}",
            fontsize=11,
            loc="left",
        )
        ax_case.legend(loc="upper left", fontsize=8, framealpha=0.92)
        # highlight worst / best case ids (short)
        if case_rows:
            worst = case_rows[0]
            best = case_rows[-1]
            ax_case.annotate(
                f"min {worst[1]:.3f}\n{worst[0][-12:]}",
                xy=(0, worst[1]),
                xytext=(8, 0.55),
                textcoords=("data", "axes fraction"),
                fontsize=7,
                color="#333",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
            )
            ax_case.annotate(
                f"max {best[1]:.3f}\n{best[0][-12:]}",
                xy=(len(case_rows) - 1, best[1]),
                xytext=(len(case_rows) - 10, 0.88),
                textcoords="data",
                fontsize=7,
                color="#333",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.7),
            )
    else:
        ax_case.text(0.5, 0.5, "score_detail.json missing — fold bars only", ha="center", va="center")
        ax_case.set_axis_off()

    fig.suptitle(
        f"Baseline1 FDG → PSMA val  ·  9fold (f0–f8)  ·  {state['stamp']}\n"
        f"mean_dice={mean_dice:.4f}"
        + (f"  ·  mean_dice_positive={mean_pos:.4f}" if mean_pos is not None else "")
        + "  ·  DONE",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def render(state: dict, out_png: Path) -> None:
    if state["mean_dice"] is not None and state["fold_dice"]:
        _render_done(state, out_png)
    else:
        _render_in_progress(state, out_png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=None, help=".../<STAMP>/psma_val_9fold")
    ap.add_argument("--work", type=Path, default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace"))
    ap.add_argument("--stamp", default="")
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=0, help="refresh every N seconds until aggregate exists")
    args = ap.parse_args()

    if args.eval_root is not None:
        eval_root = args.eval_root
    else:
        eval_root = _find_eval_root(args.work, args.stamp)
    stamp = args.stamp or eval_root.parent.name

    def once() -> bool:
        state = collect_state(eval_root, stamp)
        render(state, args.out_png)
        print(
            f"[b1-eval-progress] {state['n_pred']}/{state['n_expected']} "
            f"active={state['n_active_containers']} "
            f"mean_dice={state['mean_dice']} ETA={_fmt_eta(state['eta_s'])} → {args.out_png}",
            flush=True,
        )
        return (eval_root / "aggregate_val_dice_9fold.json").is_file()

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
