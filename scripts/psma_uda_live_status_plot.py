#!/usr/bin/env python3
"""
baseline2 UDA：预测阶段提前画空 loss 图，表头显示
  - 当前 round 推理进度 / ETA
  - 整次实验总 ETA（剩余 round 的推理+训练粗估）

用法（轮询）:
  python3 ICLR2026/scripts/psma_uda_live_status_plot.py \\
    --pred-out .../round_000/pred \\
    --out-png ICLR2026/vis/loss_curve_....png \\
    --round 0 --rounds-total 200 --epochs-per-round 10 \\
    --poll 15
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

DONE_RE = re.compile(r"^done with ", re.M)


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
    return (datetime.now(_tz()) + timedelta(seconds=max(0, int(eta_s)))).strftime(
        "%m-%d %H:%M"
    )


def count_done(pred_out: Path) -> tuple[int, int]:
    """返回 (done, total)。优先 cases_todo + shard logs；否则 cases.txt。"""
    cases_todo = pred_out / "cases_todo.txt"
    cases_all = pred_out / "cases.txt"
    if cases_all.is_file():
        total = sum(1 for ln in cases_all.read_text().splitlines() if ln.strip())
    else:
        total = 0
    # 已完成：flat pred 同时有 nii+npz
    flat = pred_out / "pred"
    done_flat = 0
    if flat.is_dir() and total > 0:
        ids = [ln.strip() for ln in cases_all.read_text().splitlines() if ln.strip()]
        for c in ids:
            nii = flat / f"{c}.nii.gz"
            npz = None
            for name in (f"{c}.npz", f"{c}.npz.npz"):
                cand = flat / name
                if cand.is_file() and cand.stat().st_size > 0:
                    npz = cand
                    break
            if nii.is_file() and npz is not None and nii.stat().st_size > 0:
                done_flat += 1
    # shard 进度（推理中尚未 flatten）
    done_shard = 0
    shards = pred_out / "shards"
    if shards.is_dir():
        for log in shards.glob("shard_*/run.log"):
            try:
                done_shard += len(DONE_RE.findall(log.read_text(errors="ignore")))
            except OSError:
                pass
    # 本轮 todo 起点：若有 cases_todo，本轮需做的是 todo 数；done 取 max(flat增量, shard)
    if cases_todo.is_file():
        todo_n = sum(1 for ln in cases_todo.read_text().splitlines() if ln.strip())
        # 本轮已完成 ≈ shard done（正在跑）与 flat 中相对本轮的增量
        # 简单：用 shard_done，封顶 todo_n；若 flatten 后用 flat
        done = max(done_shard, done_flat)
        # 若 todo < total，说明有 skip；进度按 todo
        if todo_n > 0 and todo_n < total:
            # done_flat 可能含 skip；对本轮显示用 shard
            done = min(max(done_shard, 0), todo_n)
            return done, todo_n
        return min(done, total), total
    return min(max(done_flat, done_shard), total), max(total, 1)


def estimate_etas(
    done: int,
    total: int,
    t0: float,
    *,
    round_idx: int,
    rounds_total: int,
    epochs_per_round: int,
    train_iters: int,
    val_iters: int,
    train_sec_per_iter: float,
    status_path: Path | None,
) -> tuple[float | None, float | None, dict]:
    """返回 (infer_eta_s, total_eta_s, extras)。"""
    now = time.time()
    elapsed = max(1e-3, now - t0)
    remain_infer = max(0, total - done)
    infer_eta = None
    sec_per_case = None
    if done > 0:
        sec_per_case = elapsed / done
        infer_eta = sec_per_case * remain_infer

    # 训练一轮粗估：ep * (tr+val) * sec/iter；可用历史覆盖
    train_sec_round = epochs_per_round * (train_iters + val_iters) * train_sec_per_iter
    hist = {}
    if status_path and status_path.is_file():
        try:
            hist = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            hist = {}
    if hist.get("observed_train_sec_per_round"):
        try:
            train_sec_round = float(hist["observed_train_sec_per_round"])
        except (TypeError, ValueError):
            pass
    if hist.get("observed_infer_sec_per_case") and sec_per_case is None:
        try:
            sec_per_case = float(hist["observed_infer_sec_per_case"])
            infer_eta = sec_per_case * remain_infer
        except (TypeError, ValueError):
            pass

    # 本轮剩余 = 推理剩余 + 本轮训练（推理未完时仍把整轮训练算进总 ETA）
    this_round_left = (infer_eta or 0.0) + train_sec_round
    # 后续整 round：每 round ≈ 全量推理 + 训练
    full_infer = (sec_per_case * total) if sec_per_case else None
    later = max(0, rounds_total - round_idx - 1)
    if full_infer is not None:
        later_sec = later * (full_infer + train_sec_round)
        total_eta = this_round_left + later_sec
    else:
        total_eta = None

    extras = {
        "sec_per_case": sec_per_case,
        "train_sec_per_round": train_sec_round,
        "full_infer_sec": full_infer,
    }
    return infer_eta, total_eta, extras


def draw_empty(
    out_png: Path,
    *,
    title_lines: list[str],
    x_max: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=120)
    ax.set_xlabel("epoch (nnU-Net)")
    ax.set_ylabel("loss")
    title = "\n".join([t for t in title_lines if t])
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(True, alpha=0.3)
    if x_max is not None and x_max > 0:
        ax.set_xlim(0.0, float(x_max))
    ax.text(
        0.5,
        0.5,
        "waiting for train (UDA predict / pseudo)…",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        color="#666666",
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _read_t0(pred_out: Path, fallback: float | None) -> float:
    marker = pred_out / "predict_t0.txt"
    if marker.is_file():
        try:
            return float(marker.read_text().strip().split()[0])
        except (OSError, ValueError, IndexError):
            pass
    if fallback is not None:
        return float(fallback)
    return time.time()


def once(args: argparse.Namespace, t0: float, status_path: Path) -> dict:
    pred_out = Path(args.pred_out)
    t0 = _read_t0(pred_out, t0)
    done, total = count_done(pred_out)
    infer_eta, total_eta, extras = estimate_etas(
        done,
        total,
        t0,
        round_idx=args.round,
        rounds_total=args.rounds_total,
        epochs_per_round=args.epochs_per_round,
        train_iters=args.train_iters,
        val_iters=args.val_iters,
        train_sec_per_iter=args.train_sec_per_iter,
        status_path=status_path,
    )
    phase = args.phase
    line0 = (
        f"baseline2 UDA r{args.round}/{args.rounds_total} ({phase}) · "
        f"{args.epochs_per_round}ep/round tr{args.train_iters}/val{args.val_iters}"
    )
    line1 = (
        f"infer {done}/{total} | ETA {_fmt_eta(infer_eta)} → {_fmt_finish(infer_eta)}"
    )
    line2 = (
        f"total ETA {_fmt_eta(total_eta)} → {_fmt_finish(total_eta)} "
        f"(remain rounds≈{max(0, args.rounds_total - args.round)})"
    )
    out_png = Path(args.out_png)
    draw_empty(
        out_png,
        title_lines=[line0, line1, line2],
        x_max=float(args.epochs_per_round),
    )
    payload = {
        "state": "predict" if done < total else "predict_done",
        "phase": phase,
        "round": args.round,
        "rounds_total": args.rounds_total,
        "done": done,
        "total": total,
        "t0": t0,
        "eta_seconds": infer_eta,
        "total_eta_seconds": total_eta,
        "observed_infer_sec_per_case": extras.get("sec_per_case"),
        "observed_train_sec_per_round": extras.get("train_sec_per_round"),
        "out_png": str(out_png),
        "updated_at": datetime.now(_tz()).isoformat(timespec="seconds"),
        "title": [line0, line1, line2],
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[uda-status-plot] {done}/{total} infer_eta={_fmt_eta(infer_eta)} "
        f"total_eta={_fmt_eta(total_eta)} -> {out_png}",
        flush=True,
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-out", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--status-json", type=Path, default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--rounds-total", type=int, default=200)
    ap.add_argument("--epochs-per-round", type=int, default=10)
    ap.add_argument("--train-iters", type=int, default=70)
    ap.add_argument("--val-iters", type=int, default=50)
    ap.add_argument(
        "--train-sec-per-iter",
        type=float,
        default=float(os.environ.get("TASK1_UDA_TRAIN_SEC_PER_ITER", "1.2")),
        help="粗估每 train/val step 秒数，用于总 ETA（可被 status 历史覆盖）",
    )
    ap.add_argument("--phase", type=str, default="predict")
    ap.add_argument("--poll", type=float, default=0.0, help=">0 则轮询秒数")
    ap.add_argument("--t0", type=float, default=None)
    args = ap.parse_args()

    status_path = args.status_json or (args.pred_out / "uda_plot_status.json")
    t0 = args.t0
    if t0 is None and status_path.is_file():
        try:
            t0 = float(json.loads(status_path.read_text()).get("t0") or time.time())
        except (OSError, ValueError, TypeError):
            t0 = time.time()
    if t0 is None:
        t0 = time.time()

    if args.poll <= 0:
        once(args, t0, status_path)
        return 0

    print(f"[uda-status-plot] poll={args.poll}s pred={args.pred_out}", flush=True)
    while True:
        payload = once(args, t0, status_path)
        if payload.get("done", 0) >= payload.get("total", 1) and payload.get("total", 0) > 0:
            # 推理结束仍保留最后一帧空图，直到训练 plotter 接管
            time.sleep(args.poll)
            # 若 pred-out 被清掉或外部 stop，可退出
            if not Path(args.pred_out).exists():
                break
            # 继续短轮询一会儿然后退出（避免永久占进程）；由外层 kill
            continue
        time.sleep(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
