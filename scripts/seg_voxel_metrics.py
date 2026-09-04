#!/usr/bin/env python3
"""Shared voxel metrics: Dice (pos-GT only) + FP/FN rates.

FP rate = FP / (# voxels that should be Neg) = FP / (~gt).sum()
FN rate = FN / (# voxels that should be Pos) = FN / (gt).sum()

Rates are micro-averaged across cases: sum(FP)/sum(Neg), sum(FN)/sum(Pos).
Empty GT does not enter Dice mean, but still contributes to FP (all voxels are Neg).
"""
from __future__ import annotations

from typing import Any


def confusion_counts(gt_bool, pred_bool) -> dict[str, int]:
    gt = gt_bool.astype(bool, copy=False)
    pred = pred_bool.astype(bool, copy=False)
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())
    pos = tp + fn
    neg = tn + fp
    if pos == 0 and fp == 0:
        dice = 1.0
    elif pos == 0 or (tp + fp + fn) == 0:
        dice = 0.0
    else:
        dice = float(2.0 * tp / (2 * tp + fp + fn))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pos_voxels": pos,
        "neg_voxels": neg,
        "gt_voxels": pos,
        "pred_voxels": tp + fp,
        "dice": dice,
        "fp_rate": (float(fp) / float(neg)) if neg > 0 else float("nan"),
        "fn_rate": (float(fn) / float(pos)) if pos > 0 else float("nan"),
    }


def aggregate_case_metrics(per_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build summary from per_case rows that include dice/tp/fp/fn/pos/neg."""
    rows = list(per_case.values()) if isinstance(per_case, dict) else list(per_case)
    dices_pos = []
    dices_all = []
    sum_fp = sum_fn = sum_pos = sum_neg = 0
    n_empty = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        dice = r.get("dice")
        pos = int(r.get("pos_voxels", r.get("gt_voxels", 0)) or 0)
        neg = int(r.get("neg_voxels", 0) or 0)
        fp = int(r.get("fp", 0) or 0)
        fn = int(r.get("fn", 0) or 0)
        # If only rates stored without counts, skip micro pool for that row.
        if "fp" in r or "fn" in r or "neg_voxels" in r:
            sum_fp += fp
            sum_fn += fn
            sum_pos += pos
            sum_neg += neg
        if isinstance(dice, (int, float)) and dice == dice:
            dices_all.append(float(dice))
            if pos > 0:
                dices_pos.append(float(dice))
            else:
                n_empty += 1
        elif pos <= 0:
            n_empty += 1

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    mean_pos = _mean(dices_pos)
    fp_rate = (float(sum_fp) / float(sum_neg)) if sum_neg > 0 else float("nan")
    fn_rate = (float(sum_fn) / float(sum_pos)) if sum_pos > 0 else float("nan")
    return {
        "mean_dice": mean_pos,
        "mean_dice_positive": mean_pos,
        "mean_dice_all_cases": _mean(dices_all),
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "mean_fp": fp_rate,
        "mean_fn": fn_rate,
        "sum_tp": None,
        "sum_fp": sum_fp,
        "sum_fn": sum_fn,
        "sum_pos_voxels": sum_pos,
        "sum_neg_voxels": sum_neg,
        "n_positive": len(dices_pos),
        "n_empty_gt": n_empty,
        "n_scored": len(dices_all),
    }


def pct(x: float | None, digits: int = 2) -> str:
    if not isinstance(x, (int, float)) or x != x:
        return "—"
    return f"{100.0 * float(x):.{digits}f}%"


def format_dice_fp_fn(
    dice: float | None,
    fp: float | None,
    fn: float | None,
    *,
    digits: int = 2,
) -> str:
    return f"{pct(dice, digits)}\n{pct(fp, digits)}\n{pct(fn, digits)}"
