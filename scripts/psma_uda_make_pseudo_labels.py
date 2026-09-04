#!/usr/bin/env python3
"""
PSMA UDA 伪标签：滑窗预测 → τ 阈值 → 连通域 → μ/h/λ 体积 bin 配额筛选 → 写出 mask。

对齐 ISBI label-shift 配额思路（分割版）：
  - 候选 = CC(prob>=τ) 且体积 >= min_cc_cc
  - μ EMA：病例平均候选数；N_allow = λ * μ
  - h EMA：选中伪标的体积直方图（B bins, cc）
  - 每例按 h 配额在各 bin 内按 score(平均前景概率) 取 top

输入优先读 nnUNet ``--save_probabilities`` 的 ``{case}.npz`` / ``{case}.npz.npz``；否则读硬分割 nii。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


# 10 bins in cc；末尾开到 +inf
DEFAULT_BIN_EDGES_CC = [
    0.08,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    1e12,
]


def _load_case_ids(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return [str(x) for x in raw[0].get("train", [])]
    if isinstance(raw, dict) and "train" in raw:
        return [str(x) for x in raw["train"]]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError(f"unsupported case list JSON: {path}")


def _spacing_mm_from_nii(nii: nib.Nifti1Image) -> np.ndarray:
    zooms = np.asarray(nii.header.get_zooms()[:3], dtype=np.float64)
    return np.maximum(zooms, 1e-6)


def _load_prob_and_ref(
    pred_dir: Path, case: str
) -> tuple[np.ndarray, nib.Nifti1Image]:
    """返回 (prob_fg DHW float32, ref_nii)。"""
    seg_nii_path = pred_dir / f"{case}.nii.gz"
    npz = None
    for name in (f"{case}.npz", f"{case}.npz.npz"):
        cand = pred_dir / name
        if cand.is_file() and cand.stat().st_size > 0:
            npz = cand
            break
    if npz is not None:
        z = np.load(str(npz))
        key = "probabilities" if "probabilities" in z.files else z.files[0]
        probs = np.asarray(z[key], dtype=np.float32)
        # (C,D,H,W) or (D,H,W,C)
        if probs.ndim == 4 and probs.shape[0] <= 8:
            fg = probs[1] if probs.shape[0] > 1 else probs[0]
        elif probs.ndim == 4 and probs.shape[-1] <= 8:
            fg = probs[..., 1] if probs.shape[-1] > 1 else probs[..., 0]
        elif probs.ndim == 3:
            fg = probs
        else:
            raise ValueError(f"bad probabilities shape {probs.shape} for {case}")
        if not seg_nii_path.is_file():
            raise FileNotFoundError(f"need hard seg for affine: {seg_nii_path}")
        ref = nib.load(str(seg_nii_path))
        return np.ascontiguousarray(fg, dtype=np.float32), ref

    if not seg_nii_path.is_file():
        raise FileNotFoundError(
            f"missing pred for {case}: {case}.npz(.npz) / {seg_nii_path}"
        )
    ref = nib.load(str(seg_nii_path))
    seg = np.asarray(ref.get_fdata())
    while seg.ndim > 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    if seg.ndim == 4:
        seg = seg[..., 0]
    fg = (seg > 0).astype(np.float32)
    return fg, ref


def _volume_bin(vol_cc: float, edges: list[float]) -> int:
    for b in range(len(edges) - 1):
        if edges[b] <= vol_cc < edges[b + 1]:
            return b
    return len(edges) - 2


def _allocate_quotas(n_allow: int, h: np.ndarray) -> list[int]:
    """Largest remainder method：按 h 分配整数配额，合计 = n_allow。"""
    b = int(h.size)
    if n_allow <= 0:
        return [0] * b
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    if h.sum() <= 0:
        h = np.ones(b, dtype=np.float64) / b
    else:
        h = h / h.sum()
    raw = h * float(n_allow)
    base = np.floor(raw).astype(np.int64)
    rem = int(n_allow - int(base.sum()))
    frac = raw - base
    order = np.argsort(-frac)
    for i in range(rem):
        base[order[i % b]] += 1
    return [int(x) for x in base.tolist()]


def _extract_candidates(
    prob: np.ndarray,
    spacing_mm: np.ndarray,
    tau: float,
    min_cc_cc: float,
    bin_edges: list[float],
) -> list[dict[str, Any]]:
    mask = prob >= float(tau)
    if not np.any(mask):
        return []
    labeled, nlab = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if nlab <= 0:
        return []
    vox_cc = float(np.prod(spacing_mm) / 1000.0)  # mm^3 → cc
    min_vox = max(1, int(np.ceil(min_cc_cc / max(vox_cc, 1e-12))))
    counts = ndi.sum(mask, labeled, index=np.arange(1, nlab + 1))
    cands: list[dict[str, Any]] = []
    for lab_id, nvox in enumerate(counts, start=1):
        nv = int(nvox)
        if nv < min_vox:
            continue
        vol_cc = nv * vox_cc
        if vol_cc < min_cc_cc:
            continue
        sel = labeled == lab_id
        score = float(prob[sel].mean()) if nv > 0 else 0.0
        cands.append(
            {
                "lab": lab_id,
                "nvox": nv,
                "vol_cc": float(vol_cc),
                "score": score,
                "bin": _volume_bin(float(vol_cc), bin_edges),
                "mask": sel,
            }
        )
    return cands


def _extract_candidates_adaptive(
    prob: np.ndarray,
    spacing_mm: np.ndarray,
    tau: float,
    min_cc_cc: float,
    bin_edges: list[float],
    tau_min: float = 0.15,
) -> tuple[list[dict[str, Any]], float]:
    """先用 τ；若无候选且 fg 峰值足够，则按峰值比例下调 τ。"""
    cands = _extract_candidates(prob, spacing_mm, tau, min_cc_cc, bin_edges)
    if cands:
        return cands, float(tau)
    mx = float(np.max(prob)) if prob.size else 0.0
    if mx < float(tau_min):
        return [], float(tau)
    for frac in (0.5, 0.35, 0.25):
        t = float(max(tau_min, min(tau, mx * frac)))
        if t >= mx:
            continue
        cands = _extract_candidates(prob, spacing_mm, t, min_cc_cc, bin_edges)
        if cands:
            return cands, t
    return [], float(tau)

def _select_for_case(
    cands: list[dict[str, Any]],
    n_allow: int,
    h: np.ndarray,
) -> list[dict[str, Any]]:
    if not cands or n_allow <= 0:
        return []
    quotas = _allocate_quotas(n_allow, h)
    selected: list[dict[str, Any]] = []
    by_bin: dict[int, list[dict[str, Any]]] = {}
    for c in cands:
        by_bin.setdefault(int(c["bin"]), []).append(c)
    for b, q in enumerate(quotas):
        if q <= 0:
            continue
        pool = sorted(by_bin.get(b, []), key=lambda x: -float(x["score"]))
        selected.extend(pool[:q])
    # 若配额未用满（某 bin 不足），用全局剩余最高分补齐
    if len(selected) < n_allow:
        taken = {id(x) for x in selected}
        rest = sorted(
            [c for c in cands if id(c) not in taken],
            key=lambda x: -float(x["score"]),
        )
        selected.extend(rest[: n_allow - len(selected)])
    return selected


def _update_h(h: np.ndarray, selected_all: list[dict[str, Any]], alpha_h: float) -> np.ndarray:
    b = h.size
    hist = np.zeros(b, dtype=np.float64)
    for c in selected_all:
        hist[int(c["bin"])] += 1.0
    if hist.sum() <= 0:
        return h
    hist = hist / hist.sum()
    mixed = (1.0 - alpha_h) * h + alpha_h * hist
    s = mixed.sum()
    return mixed / s if s > 0 else h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--cases-json", type=Path, required=True)
    ap.add_argument("--out-labels-dir", type=Path, required=True)
    ap.add_argument("--prior-state", type=Path, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--rounds-total", type=int, default=200)
    ap.add_argument("--lambda-start", type=float, default=0.1)
    ap.add_argument("--lambda-end", type=float, default=0.8)
    ap.add_argument("--alpha-mu", type=float, default=0.9)
    ap.add_argument("--alpha-h", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tau-min", type=float, default=0.15)
    ap.add_argument("--min-cc-cc", type=float, default=0.08)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="允许全空伪标（默认拒绝并 exit 2，防止塌缩模型继续训）",
    )
    args = ap.parse_args()

    cases = _load_case_ids(args.cases_json)
    if not cases:
        print("[pseudo] empty case list", file=sys.stderr)
        return 1

    edges = list(DEFAULT_BIN_EDGES_CC)
    if args.n_bins != 10:
        # 对数均匀切分 [min_cc, 100]
        lo, hi = args.min_cc_cc, 100.0
        edges = list(np.geomspace(lo, hi, args.n_bins).tolist()) + [1e12]

    n_bins = len(edges) - 1
    state: dict[str, Any] = {}
    if args.prior_state.is_file():
        state = json.loads(args.prior_state.read_text(encoding="utf-8"))
    mu = float(state.get("mu", 0.0))
    h = np.asarray(state.get("h", [1.0 / n_bins] * n_bins), dtype=np.float64)
    if h.size != n_bins:
        h = np.ones(n_bins, dtype=np.float64) / n_bins

    r = int(args.round)
    r_tot = max(1, int(args.rounds_total))
    # λ 线性：round 0 → start，round R-1 → end
    if r_tot <= 1:
        lam = float(args.lambda_end)
    else:
        t = r / float(r_tot - 1)
        lam = float(args.lambda_start) + t * (
            float(args.lambda_end) - float(args.lambda_start)
        )
    lam = float(np.clip(lam, 0.0, 1.0))

    args.out_labels_dir.mkdir(parents=True, exist_ok=True)

    # pass1: 提取候选并估计 μ
    all_cands: dict[str, list[dict[str, Any]]] = {}
    refs: dict[str, nib.Nifti1Image] = {}
    m_counts: list[int] = []
    tau_used: list[float] = []
    pe = max(1, args.progress_every)
    for i, case in enumerate(cases, 1):
        try:
            prob, ref = _load_prob_and_ref(args.pred_dir, case)
            spacing = _spacing_mm_from_nii(ref)
            # 对齐形状
            data_shape = np.asarray(ref.dataobj).shape[:3]
            if tuple(prob.shape) != tuple(data_shape):
                # 偶发轴顺序问题：尝试转置匹配
                if tuple(prob.shape[::-1]) == tuple(data_shape):
                    prob = prob.T
                else:
                    raise ValueError(
                        f"prob shape {prob.shape} != image {data_shape} for {case}"
                    )
            cands, tau_i = _extract_candidates_adaptive(
                prob,
                spacing,
                args.tau,
                args.min_cc_cc,
                edges,
                tau_min=float(args.tau_min),
            )
            all_cands[case] = cands
            refs[case] = ref
            m_counts.append(len(cands))
            tau_used.append(float(tau_i))
        except Exception as exc:
            print(f"[pseudo] FAIL load {case}: {exc}", file=sys.stderr)
            all_cands[case] = []
            m_counts.append(0)
            tau_used.append(float(args.tau))
        if i % pe == 0 or i == len(cases):
            print(f"[pseudo] extract {i}/{len(cases)}", flush=True)

    mean_m = float(np.mean(m_counts)) if m_counts else 0.0
    if mean_m <= 0:
        print(
            f"[pseudo] ERROR: mean_candidates=0（模型可能塌缩；fg 几乎无前景）。"
            f"拒绝写出空伪标。round={r}",
            file=sys.stderr,
        )
        return 2
    if mu <= 0:
        mu = mean_m
    else:
        mu = (1.0 - float(args.alpha_mu)) * mu + float(args.alpha_mu) * mean_m
    # 避免 λ·μ < 0.5 时 round→0 导致整轮空标
    n_allow = int(max(1, int(np.ceil(float(lam) * float(mu)))))

    # pass2: 配额选择并写出
    selected_flat: list[dict[str, Any]] = []
    n_sel_total = 0
    n_empty = 0
    for i, case in enumerate(cases, 1):
        cands = all_cands.get(case, [])
        selected = _select_for_case(cands, n_allow, h)
        selected_flat.extend(
            [{k: v for k, v in c.items() if k != "mask"} for c in selected]
        )
        ref = refs.get(case)
        if ref is None:
            # 无 ref 时跳过写出
            continue
        shape = np.asarray(ref.dataobj).shape[:3]
        out = np.zeros(shape, dtype=np.uint8)
        for c in selected:
            out[c["mask"]] = 1
        if not np.any(out):
            n_empty += 1
        n_sel_total += len(selected)
        out_path = args.out_labels_dir / f"{case}.nii.gz"
        nib.save(nib.Nifti1Image(out, ref.affine, ref.header), str(out_path))
        if i % pe == 0 or i == len(cases):
            print(
                f"[pseudo] write {i}/{len(cases)} selected_sum={n_sel_total}",
                flush=True,
            )

    h = _update_h(h, selected_flat, float(args.alpha_h))
    tau_mean = float(np.mean(tau_used)) if tau_used else float(args.tau)
    state_out = {
        "round": r,
        "lambda": lam,
        "mu": mu,
        "n_allow": n_allow,
        "mean_candidates": mean_m,
        "h": h.tolist(),
        "bin_edges_cc": edges,
        "alpha_mu": args.alpha_mu,
        "alpha_h": args.alpha_h,
        "tau": args.tau,
        "tau_min": args.tau_min,
        "tau_used_mean": tau_mean,
        "min_cc_cc": args.min_cc_cc,
        "n_cases": len(cases),
        "n_selected_total": n_sel_total,
        "n_empty_masks": n_empty,
    }
    args.prior_state.parent.mkdir(parents=True, exist_ok=True)
    args.prior_state.write_text(
        json.dumps(state_out, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[pseudo] done round={r} λ={lam:.4f} μ={mu:.3f} N_allow={n_allow} "
        f"selected={n_sel_total} empty={n_empty} tau_mean={tau_mean:.3f} "
        f"out={args.out_labels_dir}",
        flush=True,
    )
    empty_frac = float(n_empty) / float(max(len(cases), 1))
    if (not args.allow_empty) and (n_sel_total <= 0 or empty_frac >= 0.9):
        print(
            f"[pseudo] ERROR: 伪标过空 selected={n_sel_total} empty_frac={empty_frac:.2f}；"
            "拒绝进入训练（exit 2）",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
