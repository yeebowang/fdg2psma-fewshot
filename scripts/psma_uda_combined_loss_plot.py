#!/usr/bin/env python3
"""
baseline2 UDA：整次实验（200 round）画在同一张 loss 图上。

横轴 global_epoch = round_idx * epochs_per_round + local_epoch（0..R*ep-1）。
表头：当前 phase + 本 round 推理 ETA（predict 时）+ 总 ETA。
轮询读 uda_state.json，自动拼已完成各 round 的 training_log。

用法:
  python3 ICLR2026/scripts/psma_uda_combined_loss_plot.py \\
    --parent-stamp 20260811_..._baseline2_... \\
    --work .../task1_train_workspace \\
    --out-png ICLR2026/vis/loss_curve_iclr2026_baseline2_<PARENT>.png \\
    --poll 15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "other"))
sys.path.insert(0, str(_REPO / "ICLR2026" / "scripts"))

from plot_nnunet_loss_live import load_fold_loss_series  # noqa: E402
from psma_uda_live_status_plot import (  # noqa: E402
    _fmt_eta,
    _fmt_finish,
    count_done,
    estimate_etas,
)

ROUND_RE = re.compile(r"__r(\d{3})$")


def _tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(os.environ.get("TASK1_STAMP_TZ", "Asia/Shanghai"))
    except Exception:
        return timezone(timedelta(hours=8))


def _read_state(state_json: Path) -> dict:
    if not state_json.is_file():
        return {}
    try:
        return json.loads(state_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _discover_round_folds(
    work: Path,
    parent: str,
    ds: str,
    tf: str,
    fold: int,
) -> dict[int, Path]:
    """返回 {round_idx: fold_dir}。"""
    root = work / "nnUNet_results"
    out: dict[int, Path] = {}
    if not root.is_dir():
        return out
    for d in root.glob(f"{parent}__r*"):
        m = ROUND_RE.search(d.name)
        if not m:
            continue
        r = int(m.group(1))
        fold_dir = d / ds / tf / f"fold_{fold}"
        if fold_dir.is_dir():
            out[r] = fold_dir
    return out


def _collect_series(
    folds: dict[int, Path],
    ep_per_round: int,
) -> tuple[list[float], list[float], list[float], int]:
    xs: list[float] = []
    train: list[float] = []
    val: list[float] = []
    n_rounds_with_data = 0
    for r in sorted(folds):
        fold_dir = folds[r]
        lx, tr, va, _pd, _ps, _lp, _body = load_fold_loss_series(str(fold_dir), False)
        if not tr:
            continue
        n_rounds_with_data += 1
        off = float(r * ep_per_round)
        for i, t in enumerate(tr):
            local = float(lx[i]) if i < len(lx) else float(i)
            xs.append(off + local)
            train.append(float(t))
            val.append(float(va[i]) if i < len(va) else float("nan"))
    return xs, train, val, n_rounds_with_data


def _draw(
    out_png: Path,
    *,
    xs: list[float],
    train: list[float],
    val: list[float],
    title_lines: list[str],
    x_max: float,
    ep_per_round: int,
    vline_every_rounds: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=120)
    title = "\n".join([t for t in title_lines if t])
    ax.set_title(title, fontsize=9.5, pad=8)
    ax.set_xlabel(f"global epoch (= round×{ep_per_round} + local ep)")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    # 早期跟随数据，避免 10 个点挤在 0..2000 最左侧；随 round 推进逐步拉宽到满轴
    if xs:
        data_right = max(xs) + float(ep_per_round)
    else:
        data_right = float(ep_per_round)
    # 至少展示到「当前 round 末尾」或 5 个 round 宽
    follow_right = max(data_right, float(ep_per_round * 5))
    right = min(float(x_max), max(follow_right, float(x_max) * 0.05))
    # 一旦越过全长 15%，钉满轴（便于看 200 round 全局）
    if data_right >= 0.15 * float(x_max):
        right = float(x_max)
    ax.set_xlim(0.0, right)

    if vline_every_rounds > 0:
        step = vline_every_rounds * ep_per_round
        for xb in range(step, int(x_max), step):
            ax.axvline(xb, color="#cccccc", lw=0.8, zorder=0)

    if train:
        mk = {"marker": "o", "markersize": 2.5, "linewidth": 1.2}
        if len(xs) > 80:
            mk = {"marker": None, "linewidth": 1.0}
        ax.plot(xs, train, label="train_loss", color="#1f77b4", **mk)
        if any(v == v for v in val):  # not all nan
            ax.plot(
                xs,
                [v if v == v else math.nan for v in val],
                label="val_loss",
                color="#ff7f0e",
                **mk,
            )
        ax.legend(loc="best", fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "waiting for first round train…\n(UDA predict / pseudo)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#666666",
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def once(args: argparse.Namespace) -> dict:
    work = Path(args.work)
    parent = args.parent_stamp
    state_json = Path(args.state_json) if args.state_json else (
        work / "nnUNet_results" / parent / "uda_state.json"
    )
    state = _read_state(state_json)
    r = int(state.get("round", state.get("next_round", 0)) or 0)
    phase = str(state.get("phase", "predict"))
    rounds_total = int(args.rounds_total)
    ep = int(args.epochs_per_round)
    x_max = float(rounds_total * ep)

    folds = _discover_round_folds(
        work, parent, args.dataset_name, args.trainer_folder, args.fold
    )
    xs, train, val, n_with = _collect_series(folds, ep)

    # 推理 ETA：当前 round 的 pred 目录
    pred_out = work / "nnUNet_results" / parent / f"round_{r:03d}" / "pred"
    status_path = pred_out / "uda_plot_status.json"
    infer_eta = None
    total_eta = None
    done, total = 0, 1
    if phase in ("predict", "predict_done", "pseudo") or (
        phase == "train" and not (pred_out / "pred").exists()
    ):
        if pred_out.is_dir():
            done, total = count_done(pred_out)
            t0 = time.time()
            marker = pred_out / "predict_t0.txt"
            if marker.is_file():
                try:
                    t0 = float(marker.read_text().strip().split()[0])
                except (OSError, ValueError, IndexError):
                    pass
            infer_eta, total_eta, _ex = estimate_etas(
                done,
                total,
                t0,
                round_idx=r,
                rounds_total=rounds_total,
                epochs_per_round=ep,
                train_iters=args.train_iters,
                val_iters=args.val_iters,
                train_sec_per_iter=args.train_sec_per_iter,
                status_path=status_path,
            )
    else:
        # 训练中：总 ETA 仍用历史观测粗估
        if pred_out.is_dir():
            done, total = count_done(pred_out)
            done, total = max(done, total), max(total, 1)
        t0 = time.time()
        infer_eta, total_eta, _ex = estimate_etas(
            max(total, 1),
            max(total, 1),
            t0,
            round_idx=r,
            rounds_total=rounds_total,
            epochs_per_round=ep,
            train_iters=args.train_iters,
            val_iters=args.val_iters,
            train_sec_per_iter=args.train_sec_per_iter,
            status_path=status_path if status_path.parent.is_dir() else None,
        )
        infer_eta = 0.0

    n_pts = len(train)
    line0 = (
        f"baseline2 UDA {parent} · r{r}/{rounds_total} ({phase}) · "
        f"{ep}ep/round · pts={n_pts} rounds_logged={n_with}"
    )
    if phase in ("predict", "predict_done", "pseudo"):
        line1 = (
            f"infer {done}/{total} | ETA {_fmt_eta(infer_eta)} → {_fmt_finish(infer_eta)}"
        )
    else:
        cur_ep = int(xs[-1] - r * ep) if xs else -1
        line1 = (
            f"train round r{r} local_ep≈{cur_ep}/{ep - 1} · "
            f"global_ep≈{xs[-1] if xs else r * ep:.0f}/{x_max:.0f}"
        )
    line2 = (
        f"total ETA {_fmt_eta(total_eta)} → {_fmt_finish(total_eta)} "
        f"(remain rounds≈{max(0, rounds_total - r)})"
    )

    out_png = Path(args.out_png)
    _draw(
        out_png,
        xs=xs,
        train=train,
        val=val,
        title_lines=[line0, line1, line2],
        x_max=x_max,
        ep_per_round=ep,
        vline_every_rounds=int(args.vline_every_rounds),
    )

    payload = {
        "parent_stamp": parent,
        "round": r,
        "phase": phase,
        "n_points": n_pts,
        "n_rounds_logged": n_with,
        "done": done,
        "total": total,
        "eta_seconds": infer_eta,
        "total_eta_seconds": total_eta,
        "out_png": str(out_png),
        "updated_at": datetime.now(_tz()).isoformat(timespec="seconds"),
        "title": [line0, line1, line2],
    }
    meta = Path(args.status_json) if args.status_json else (
        work / "nnUNet_results" / parent / "uda_combined_plot_status.json"
    )
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[uda-combined-plot] r{r}/{rounds_total} {phase} pts={n_pts} "
        f"infer={done}/{total} total_eta={_fmt_eta(total_eta)} -> {out_png}",
        flush=True,
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-stamp", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--state-json", type=Path, default=None)
    ap.add_argument("--status-json", type=Path, default=None)
    ap.add_argument("--rounds-total", type=int, default=200)
    ap.add_argument("--epochs-per-round", type=int, default=10)
    ap.add_argument("--train-iters", type=int, default=70)
    ap.add_argument("--val-iters", type=int, default=50)
    ap.add_argument("--train-sec-per-iter", type=float, default=1.5)
    ap.add_argument("--dataset-name", default="Dataset228_AutoPETIV_Task1_2ch")
    ap.add_argument(
        "--trainer-folder",
        default="nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--vline-every-rounds", type=int, default=10)
    ap.add_argument("--poll", type=float, default=0.0)
    args = ap.parse_args()

    if args.poll and args.poll > 0:
        while True:
            try:
                once(args)
            except Exception as e:
                print(f"[uda-combined-plot] error: {e}", flush=True)
            time.sleep(float(args.poll))
    else:
        once(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
