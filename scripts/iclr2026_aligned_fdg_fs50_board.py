#!/usr/bin/env python3
"""ICLR2026 aligned protocol progress board: FDG supervised → PSMA fewshot f258.

Few-shot variants: fs50 / fs10 / fs5 (TEST folds 0–8). FDG pretrain is shared (one column).
Writes/updates:
  ICLR2026/vis/iclr2026_aligned_fdg_fs50_f258_board.json
  ICLR2026/vis/progress_iclr2026_aligned_fdg_fs50_f258_board.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from seg_voxel_metrics import format_dice_fp_fn, pct as _pct_fmt
except ImportError:  # plot-only / partial env

    def _pct_fmt(x, digits=2):
        if not isinstance(x, (int, float)) or x != x:
            return "—"
        return f"{100.0 * float(x):.{digits}f}%"

    def format_dice_fp_fn(dice, fp, fn, *, digits=2):
        return f"{_pct_fmt(dice, digits)}\n{_pct_fmt(fp, digits)}\n{_pct_fmt(fn, digits)}"

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import FancyBboxPatch

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    FontProperties = None  # type: ignore


def _cjk_font(size: float = 8.5, weight: str = "bold"):
    """Noto Sans CJK so Method-column Chinese renders (DejaVu has no CJK)."""
    if FontProperties is None:
        return None
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if Path(path).is_file():
            return FontProperties(fname=path, size=size, weight=weight)
    return None

CTRL = Path("/media/ybwang/data1/PSMA-CTRL")
DEFAULT_BOARD = CTRL / "ICLR2026/vis/iclr2026_aligned_fdg_fs50_f258_board.json"
DEFAULT_PNG = CTRL / "ICLR2026/vis/progress_iclr2026_aligned_fdg_fs50_f258_board.png"
REPO = CTRL / "ICLR2026/3D-MAE-PET-CT/runs"
WORK = Path(os.environ.get("TASK1_BASE", "/media/ybwang/data1/PSMA-DATA")) / "task1_train_workspace"
NN_RESULTS = WORK / "nnUNet_results"
DS228 = "Dataset228_AutoPETIV_Task1_2ch"
TF228 = "nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres"
DS239 = "Dataset239_DpDNet_FDG_2ch"
TF239 = "STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
TF239_PRETRAIN = "STUNetTrainer_small_prompt_pretrain__nnUNetPlans__3d_fullres"
DS240 = "Dataset240_DpDNet_PSMA_2ch"
TF240 = "STUNetTrainer_small_prompt__nnUNetPlans__3d_fullres"
TF240_PRETRAIN = "STUNetTrainer_small_prompt_pretrain__nnUNetPlans__3d_fullres"

# Method display names with publication venues (aligned board)
METHOD_LABELS = {
    "nnunet_mim": "nnUNet MIM (Nat.Methods'21)",
    "nnunet": "nnUNet (Nat.Methods'21)",
    "mae_swinunetr": "PET/CT MAE (arXiv'26)",
    "mae_scratch": "PET/CT MAE scratch (arXiv'26)",
    "monai_swinvit": "MONAI SwinViT (CVPR'22)",
    "monai_scratch": "MONAI SwinViT scratch (CVPR'22)",
    "seganypet": "SegAnyPET (ICCV'25)",
    "seganypet_scratch": "SegAnyPET scratch (ICCV'25)",
    "dpdnet_dualenc": "DpDNet dual-enc (MICCAI'25)",
    "dpdnet": "DpDNet (MICCAI'25)",
    "proto_retrieval": "Proto+Retrieval (ECCV'20)",
    # AutoPET V 2026 competition codebases (aligned protocol rows)
    "hemingduo": "BIRTH / hemingduo (AutoPET V'26)",
    "hemingduo_scratch": "BIRTH scratch (AutoPET V'26)",
    "chenyixin": "YixinChen / chenyixin (AutoPET V'26)",
    "chenyixin_scratch": "YixinChen scratch (AutoPET V'26)",
}
# Weights loaded *before* FDG supervised training (not the FDG→PSMA init).
METHOD_FDG_INIT = {
    "nnunet_mim": "PET+CT MIM",
    "nnunet": "scratch",
    "mae_swinunetr": "PET/CT MAE SSL",
    "mae_scratch": "scratch",
    "monai_swinvit": "Tang SSL",
    "monai_scratch": "scratch",
    "seganypet": "SegAnyPET-Lesion",
    "seganypet_scratch": "scratch",
    "dpdnet_dualenc": "PET+CT dual-enc",
    "dpdnet": "scratch",
    "proto_retrieval": "none (retrieval)",
    "hemingduo": "Dataset619 MultiTalent",
    "hemingduo_scratch": "scratch",
    "chenyixin": "Dataset619 MultiTalent",
    "chenyixin_scratch": "scratch",
}
# Board row order = publication chronology (oldest → newest).
# Pretrained sibling sits immediately above the scratch row.
METHOD_ORDER = (
    "proto_retrieval",
    "nnunet_mim",
    "nnunet",
    "monai_swinvit",
    "monai_scratch",
    "dpdnet_dualenc",
    "dpdnet",
    "seganypet",
    "seganypet_scratch",
    "mae_swinunetr",
    "mae_scratch",
    # scratch first (FDG→PSMA); pretrained next (Dataset619 → FDG→PSMA).
    # Never score with GC final submission ckpts (LesionTracer 14007247 / EDT / LocalEdit).
    "hemingduo_scratch",
    "hemingduo",
    "chenyixin_scratch",
    "chenyixin",
)

NNUNET_FDG = "20260810_104431_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val10_3000ep"
NNUNET_FS = "20260816_090857_iclr2026_nnunet_psma_fs50_f258_1gpu_bs6_tr70_noval_300ep_gpu013"
MAE_FDG = "20260812_072719_iclr2026_mae_fdg_swinbase_gpu013_bs6_tr70_val10_100ep"

# PSMA few-shot column groups (fs50 has live runs; fs10/fs5 queued same protocol).
FEWSHOT_VARIANTS: tuple[tuple[int, str, str], ...] = (
    (50, "psma_fs50_f258", "fs50"),
    (10, "psma_fs10_f258", "fs10"),
    (5, "psma_fs5_f258", "fs5"),
)
NINE_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
NINE_FOLD_STRS: tuple[str, ...] = tuple(str(i) for i in NINE_FOLDS)
FC70_STAGE_KEY = "psma_fc70"
FC70_SHORT = "PSMA fc70%"
FC70_HDR = "PSMA fc70%"
PSMA_FS0_STAGE_KEY = "psma_fs0"
PSMA_FS0_SHORT = "PSMA fs0"
PSMA_FS0_HDR = "PSMA fs0"
PSMA_FS0_AGG_DIR = CTRL / "ICLR2026/vis/psma_fs0"
FDG_TEST_STAGE_KEY = "fdg_test20"
FDG_TEST_SHORT = "FDG TEST"
FDG_TEST_HDR = "FDG TEST"
FDG_TEST_AGG_DIR = CTRL / "ICLR2026/vis/fdg_test20"
SINGLE_COL_SHORTS = frozenset({FC70_SHORT, PSMA_FS0_SHORT, FDG_TEST_SHORT})
PSMA_BOARD_COLUMNS: tuple[tuple[int, str, str], ...] = FEWSHOT_VARIANTS + (
    (0, PSMA_FS0_STAGE_KEY, PSMA_FS0_SHORT),
    (0, FC70_STAGE_KEY, FC70_SHORT),
    (0, FDG_TEST_STAGE_KEY, FDG_TEST_SHORT),
)
PSMA_STAGE_KEYS = tuple(v[1] for v in PSMA_BOARD_COLUMNS)
PRIMARY_PSMA_STAGE = "psma_fs50_f258"


def _default_psma_stage(fewshot: int = 50, training_free: bool = False) -> dict[str, Any]:
    note = (
        "TEST20 · FDG100% retrieve (training-free)"
        if training_free
        else f"queued · fs{fewshot} f258 · tr25 val25 · until val Dice decline"
    )
    return {
        "status": "done" if training_free and fewshot == 50 else "pending",
        "stamp": "",
        "bs": 2,
        "bs_note": "per-GPU",
        "total_epochs": 100,
        "train_iters": 25,
        "val_iters": 25,
        "online_val": "VAL25 until val Dice decline",
        "fold_dice": {},
        "mean": None,
        "metric": "TEST20 Dice; best=max val Dice",
        "note": note,
        "training_free": training_free,
    }


def _default_psma_fs0_stage(method_key: str = "") -> dict[str, Any]:
    if method_key == "proto_retrieval":
        # Same protocol as fs50/fs10/fs5: FDG100% gallery → PSMA TEST20 (training-free).
        return {
            "status": "pending",
            "stamp": "",
            "training_free": True,
            "support_pool": "FDG100%",
            "metric": "TEST20 Dice; retrieve FDG100% + prototype",
            "note": "same as fs50 · FDG100% retrieve",
            "topk": 3,
            "fold_dice": {},
            "mean": None,
        }
    return {
        "status": "pending",
        "stamp": "",
        "training_free": True,
        "support_pool": "FDG shared",
        "metric": "TEST20 Dice; FDG ckpt zero-shot on PSMA",
        "note": "FDG shared ckpt → PSMA TEST20 (fs0)",
        "fold_dice": {},
        "mean": None,
    }


def _default_fdg_test_stage(method_key: str = "") -> dict[str, Any]:
    if method_key == "proto_retrieval":
        note = "FDG70% sup → FDG20% TEST (training-free)"
        metric = "Dice; retrieve FDG70% + prototype"
        support_pool = "FDG70%"
    else:
        note = "FDG shared ckpt → FDG 20% TEST"
        metric = "FDG TEST20 Dice; shared FDG ckpt"
        support_pool = "FDG shared"
    return {
        "status": "pending",
        "stamp": "",
        "training_free": True,
        "support_pool": support_pool,
        "metric": metric,
        "note": note,
        "fold_dice": {},
        "mean": None,
    }


def _default_fc70_stage(training_free: bool = False) -> dict[str, Any]:
    note = (
        "queued · PSMA70% retrieve → TEST20 (training-free)"
        if training_free
        else "queued · fc70% PSMA · single run · tr25 val25 · until val Dice decline"
    )
    return {
        "status": "pending" if training_free else "pending",
        "stamp": "",
        "bs": 2,
        "bs_note": "per-GPU",
        "total_epochs": 100 if not training_free else 0,
        "train_iters": 25,
        "val_iters": 25,
        "online_val": "VAL25 every20 until decline",
        "fold_dice": {},
        "mean": None,
        "metric": (
            "TEST20 Dice; retrieve PSMA70% + prototype"
            if training_free
            else "TEST20 Dice; single run"
        ),
        "note": note,
        "training_free": training_free,
        "support_pool": "PSMA70%" if training_free else None,
    }


def _method_psma_stages(method_key: str) -> dict[str, dict[str, Any]]:
    tf = method_key == "proto_retrieval"
    out: dict[str, dict[str, Any]] = {}
    for n, key, _ in FEWSHOT_VARIANTS:
        st = _default_psma_stage(n, training_free=tf)
        if tf:
            st["metric"] = "TEST20 Dice; retrieve FDG100% + prototype"
            st["support_pool"] = "FDG100%"
            st["topk"] = 3
            if n == 50:
                st["status"] = "pending"  # filled from board refresh / ingest
        out[key] = st
    out[FC70_STAGE_KEY] = _default_fc70_stage(training_free=tf)
    out[PSMA_FS0_STAGE_KEY] = _default_psma_fs0_stage(method_key)
    out[FDG_TEST_STAGE_KEY] = _default_fdg_test_stage(method_key)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def default_board() -> dict[str, Any]:
    mae_best = REPO / MAE_FDG / "best_seg_fdg_mae.pth"
    nn_agg = CTRL / "ICLR2026/vis" / f"aggregate_nnunet_psma_fs50_f258_{NNUNET_FS}.json"
    nn_folds = {}
    nn_mean = None
    if nn_agg.is_file():
        d = json.loads(nn_agg.read_text())
        nn_folds = {k: v.get("best_val_dice") for k, v in d.get("folds", {}).items()}
        nn_mean = d.get("fold_mean")
    out: dict[str, Any] = {
        "protocol": "FDG supervised → PSMA fs50/fs10/fs5 f258 (f0–8); tr25 val25 until val Dice decline",
        "gpus": "0,1,3",
        "bs_policy": {"fdg": 6, "psma_fewshot": 2},
        "fewshot_variants": [v[2] for v in FEWSHOT_VARIANTS],
        "updated_at": _now(),
        "updated_note": "init",
        "methods": {
            "nnunet_mim": {
                "label": METHOD_LABELS["nnunet_mim"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "gbs",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "PET+CT MIM → FDG tr70/val0 · 169ep",
                },
                **_method_psma_stages("nnunet_mim"),
            },
            "nnunet": {
                "label": METHOD_LABELS["nnunet"],
                "fdg_pretrain": {
                    "status": "queued",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "per-GPU",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "queued · align FDG tr70/val0 · 169ep",
                },
                **_method_psma_stages("nnunet"),
            },
            "mae_swinunetr": {
                "label": METHOD_LABELS["mae_swinunetr"],
                "fdg_pretrain": {
                    "status": "done" if mae_best.is_file() else "pending",
                    "stamp": MAE_FDG,
                    "bs": 6,
                    "bs_note": "global 2×3GPU",
                    "best_ckpt": str(mae_best) if mae_best.is_file() else "",
                    "note": "FDG supervised 100ep (reuse existing)",
                },
                **_method_psma_stages("mae_swinunetr"),
            },
            "mae_scratch": {
                "label": METHOD_LABELS["mae_scratch"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "global 2×3GPU",
                    "best_ckpt": "",
                    "total_epochs": 100,
                    "train_iters": 70,
                    "val_iters": 10,
                    "note": "scratch → FDG supervised 100ep (same protocol as PET/CT MAE)",
                },
                **_method_psma_stages("mae_scratch"),
            },
            "monai_scratch": {
                "label": METHOD_LABELS["monai_scratch"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "global 2×3GPU",
                    "best_ckpt": "",
                    "total_epochs": 100,
                    "train_iters": 70,
                    "val_iters": 10,
                    "note": "scratch → FDG supervised 100ep (same protocol as MONAI SwinViT)",
                },
                **_method_psma_stages("monai_scratch"),
            },
            "seganypet_scratch": {
                "label": METHOD_LABELS["seganypet_scratch"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "global DP 0,1,3",
                    "best_ckpt": "",
                    "total_epochs": 100,
                    "note": "scratch → FDG click supervised 100ep (same protocol as SegAnyPET)",
                },
                **_method_psma_stages("seganypet_scratch"),
            },
            "monai_swinvit": {
                "label": METHOD_LABELS["monai_swinvit"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "global 2×3GPU",
                    "best_ckpt": "",
                    "note": "Tang SSL → FDG supervised",
                },
                **_method_psma_stages("monai_swinvit"),
            },
            "seganypet": {
                "label": METHOD_LABELS["seganypet"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "global DP 0,1,3",
                    "best_ckpt": "",
                    "note": "click supervised on FDG PET",
                },
                **_method_psma_stages("seganypet"),
            },
            "dpdnet_dualenc": {
                "label": METHOD_LABELS["dpdnet_dualenc"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "per-GPU",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "PET+CT dual-enc → FDG (STUNetTrainer_small_prompt_pretrain)",
                },
                **_method_psma_stages("dpdnet_dualenc"),
            },
            "dpdnet": {
                "label": METHOD_LABELS["dpdnet"],
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "per-GPU",
                    "best_ckpt": "",
                    "note": "dual-prompt FDG (STU-Net)",
                },
                **_method_psma_stages("dpdnet"),
            },
            "proto_retrieval": {
                "label": METHOD_LABELS["proto_retrieval"],
                "fdg_pretrain": {
                    "status": "n/a",
                    "stamp": "",
                    "training_free": True,
                    "support_pool": "FDG100%",
                    "note": "FDG100% support gallery (not training)",
                },
                **_method_psma_stages("proto_retrieval"),
            },
            "hemingduo_scratch": {
                "label": METHOD_LABELS["hemingduo_scratch"],
                "repo": "ICLR2026/third_party/autoPET-V-BIRTH-final",
                "board_policy": "scratch → FDG → PSMA; FORBID final submission ckpts on board",
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "gbs 3GPU",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "scratch → FDG tr70/val0 · 169ep → PSMA (no GC final ckpt)",
                },
                **_method_psma_stages("hemingduo_scratch"),
            },
            "hemingduo": {
                "label": METHOD_LABELS["hemingduo"],
                "repo": "ICLR2026/third_party/autoPET-V-BIRTH-final",
                "board_policy": "Dataset619 (Zenodo 13753413) → FDG → PSMA; FORBID 14007247/EDT/LocalEdit final ckpts",
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "gbs 3GPU",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "load Dataset619 MultiTalent → FDG tr70/val0 · 169ep (not LesionTracer final)",
                },
                **_method_psma_stages("hemingduo"),
            },
            "chenyixin_scratch": {
                "label": METHOD_LABELS["chenyixin_scratch"],
                "repo": "ICLR2026/third_party/autopet-v-yixinchen",
                "board_policy": "scratch → FDG → PSMA; FORBID final submission ckpts on board",
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "gbs 3GPU",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "scratch → FDG tr70/val0 · 169ep → PSMA (no GC final ckpt)",
                },
                **_method_psma_stages("chenyixin_scratch"),
            },
            "chenyixin": {
                "label": METHOD_LABELS["chenyixin"],
                "repo": "ICLR2026/third_party/autopet-v-yixinchen",
                "board_policy": "Dataset619 (Zenodo 13753413) → FDG → PSMA; FORBID 14007247/LocalEdit/TACE final ckpts",
                "fdg_pretrain": {
                    "status": "pending",
                    "stamp": "",
                    "bs": 6,
                    "bs_note": "gbs 3GPU",
                    "total_epochs": 169,
                    "train_iters": 70,
                    "val_iters": 0,
                    "note": "load Dataset619 MultiTalent → FDG tr70/val0 · 169ep (not LocalEdit final)",
                },
                **_method_psma_stages("chenyixin"),
            },
        },
        "queue": [
            "mae_swinunetr.psma_fs50_f258",
            "monai_swinvit.fdg_pretrain",
            "monai_swinvit.psma_fs50_f258",
            "seganypet.fdg_pretrain",
            "seganypet.psma_fs50_f258",
            "dpdnet.fdg_pretrain",
            "dpdnet.psma_fs50_f258",
            "nnunet.psma_fs10_f258",
            "nnunet.psma_fs5_f258",
        ],
    }
    if nn_mean is not None:
        nn_st = out["methods"]["nnunet"]["psma_fs50_f258"]
        nn_st["fold_dice"] = {str(k): v for k, v in nn_folds.items() if v is not None}
        nn_st["mean"] = nn_mean
        nn_st["metric"] = "TEST20 Dice; best=max val Dice (Pseudo dice)"
        nn_st["test_invalidated"] = True
    return out


def deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_board(path: Path) -> dict:
    if path.is_file():
        raw = path.read_text().strip()
        if not raw:
            return default_board()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default_board()
    return default_board()


def save_board(path: Path, board: dict) -> None:
    board["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(board, indent=2) + "\n")
    tmp.replace(path)


def _ingest_fold_agg(agg_path: Path) -> tuple[dict, float | None]:
    if not agg_path.is_file():
        return {}, None
    d = json.loads(agg_path.read_text())
    fd = d.get("fold_best_dice")
    folds: dict[str, float | None] = {}
    if isinstance(fd, dict):
        folds = {str(k): (float(v) if v is not None else None) for k, v in fd.items()}
    elif isinstance(fd, list):
        # list aligned with folds key
        fl = d.get("folds", [2, 5, 8])
        for i, f in enumerate(fl):
            v = fd[i] if i < len(fd) else None
            folds[str(f)] = float(v) if v is not None else None
    mean = d.get("mean")
    if mean is not None:
        mean = float(mean)
    return folds, mean


def _merge_fold_scores(dst: dict | None, src: dict | None) -> dict[str, float]:
    """Keep existing numeric fold scores; overlay src. Keys normalized to str."""
    out: dict[str, float] = {}
    for raw in (dst, src):
        if not isinstance(raw, dict):
            continue
        for k, v in raw.items():
            if isinstance(v, (int, float)) and v == v:
                out[str(k)] = float(v)
    return out


def _fold_score_mean(fd: dict | None) -> float | None:
    if not isinstance(fd, dict):
        return None
    vals = [float(v) for v in fd.values() if isinstance(v, (int, float)) and v == v]
    return (sum(vals) / float(len(vals))) if vals else None


def _psma_board_stage_from_stamp(stamp: str, default: str = "psma_fs50_f258") -> str:
    """Map run stamp → board PSMA stage (avoid fs10/fs5 ingest clobbering fs50)."""
    s = str(stamp or "")
    if "_fc70_" in s or "psma_fc70" in s:
        return FC70_STAGE_KEY
    if "_fs10_" in s or "psma_fs10_" in s:
        return "psma_fs10_f258"
    if "_fs5_" in s or "psma_fs5_" in s:
        return "psma_fs5_f258"
    if "_fs50_" in s or "psma_fs50_" in s:
        return "psma_fs50_f258"
    return default


def _fewshot_n_from_stage(stage: str) -> str:
    if stage.endswith("fs10_f258"):
        return "10"
    if stage.endswith("fs5_f258"):
        return "5"
    return "50"


def _board_method_from_stamp(stamp: str, default: str) -> str:
    s = (stamp or "").lower()
    if "mae_scratch" in s:
        return "mae_scratch"
    if "monai_scratch" in s:
        return "monai_scratch"
    if "seganypet_scratch" in s:
        return "seganypet_scratch"
    # competition rows: match longer keys first (*_scratch before base)
    for key in ("hemingduo_scratch", "chenyixin_scratch", "hemingduo", "chenyixin"):
        if key in s:
            return key
    env_key = (os.environ.get("TASK1_BOARD_METHOD") or "").strip()
    if env_key in METHOD_LABELS:
        return env_key
    return default


def _mae_board_key_from_stamp(stamp: str) -> str:
    return _board_method_from_stamp(stamp, "mae_swinunetr")


def ingest_mae(board: dict, stamp: str, stage: str | None = None) -> None:
    stage = stage or _psma_board_stage_from_stamp(stamp)
    n = _fewshot_n_from_stage(stage)
    candidates = [
        CTRL / "ICLR2026/vis" / f"aggregate_mae_psma_fs{n}_fdgseg_f258_{stamp}.json",
        CTRL / "ICLR2026/vis" / f"aggregate_mae_psma_fs50_fdgseg_f258_{stamp}.json",
        REPO / stamp / "aggregate_val_dice_f258.json",
    ]
    agg = next((p for p in candidates if p.is_file()), candidates[-1])
    folds, mean = _ingest_fold_agg(agg)
    mkey = _mae_board_key_from_stamp(stamp)
    methods = board.setdefault("methods", {}).setdefault(mkey, {})
    m = methods.setdefault(stage, {})
    # Never let a different few-shot stamp overwrite a finalized TEST20 row
    prev = (m.get("stamp") or "").strip()
    if (
        prev
        and prev != stamp
        and (m.get("status") or "").lower() == "done"
        and isinstance(m.get("mean"), (int, float))
        and ("TEST20" in str(m.get("metric") or "") or "TEST20" in str(m.get("note") or ""))
    ):
        return
    m["stamp"] = stamp
    m["fold_dice"] = _merge_fold_scores(m.get("fold_dice"), folds)
    m["mean"] = _fold_score_mean(m["fold_dice"]) if m["fold_dice"] else mean
    if _extra_fold_docker_live(stamp, "mae"):
        m["status"] = "running"
    else:
        m["status"] = "done"


def ingest_monai(board: dict, stamp: str, stage: str | None = None) -> None:
    stage = stage or _psma_board_stage_from_stamp(stamp)
    n = _fewshot_n_from_stage(stage)
    candidates = [
        CTRL / "ICLR2026/vis" / f"aggregate_monai_psma_fs{n}_fdgseg_f258_{stamp}.json",
        CTRL / "ICLR2026/vis" / f"aggregate_monai_psma_fs50_fdgseg_f258_{stamp}.json",
        REPO / stamp / "aggregate_val_dice_f258.json",
    ]
    agg = next((p for p in candidates if p.is_file()), candidates[-1])
    folds, mean = _ingest_fold_agg(agg)
    methods = board.setdefault("methods", {}).setdefault(
        _board_method_from_stamp(stamp, "monai_swinvit"), {}
    )
    m = methods.setdefault(stage, {})
    prev = (m.get("stamp") or "").strip()
    if (
        prev
        and prev != stamp
        and (m.get("status") or "").lower() == "done"
        and isinstance(m.get("mean"), (int, float))
        and ("TEST20" in str(m.get("metric") or "") or "TEST20" in str(m.get("note") or ""))
    ):
        return
    m["stamp"] = stamp
    m["fold_dice"] = _merge_fold_scores(m.get("fold_dice"), folds)
    m["mean"] = _fold_score_mean(m["fold_dice"]) if m["fold_dice"] else mean
    if _extra_fold_docker_live(stamp, "monai"):
        m["status"] = "running"
    else:
        m["status"] = "done"


def ingest_seganypet(board: dict, stamp: str, stage: str | None = None) -> None:
    stage = stage or _psma_board_stage_from_stamp(stamp)
    n = _fewshot_n_from_stage(stage)
    folds, mean = {}, None
    for pat in (
        f"aggregate_seganypet_official_fs{n}_f258_{stamp}.json",
        f"aggregate_seganypet_fs{n}_f258_{stamp}.json",
        f"aggregate_seganypet_official_fs50_f258_{stamp}.json",
        f"aggregate_seganypet_fs50_f258_{stamp}.json",
    ):
        agg = CTRL / "ICLR2026/vis" / pat
        if agg.is_file():
            folds, mean = _ingest_fold_agg(agg)
            break
    else:
        folds, mean = _ingest_fold_agg(REPO / stamp / "aggregate_val_dice_f258.json")
    methods = board.setdefault("methods", {}).setdefault(
        _board_method_from_stamp(stamp, "seganypet"), {}
    )
    m = methods.setdefault(stage, {})
    prev = (m.get("stamp") or "").strip()
    if (
        prev
        and prev != stamp
        and (m.get("status") or "").lower() == "done"
        and isinstance(m.get("mean"), (int, float))
        and ("TEST20" in str(m.get("metric") or "") or "TEST20" in str(m.get("note") or ""))
    ):
        return
    m["stamp"] = stamp
    m["fold_dice"] = _merge_fold_scores(m.get("fold_dice"), folds)
    m["mean"] = _fold_score_mean(m["fold_dice"]) if m["fold_dice"] else mean
    if _extra_fold_docker_live(stamp, "seganypet"):
        m["status"] = "running"
    else:
        m["status"] = "done"


STATUS_COLOR = {
    "done": "#2e7d32",
    "running": "#ef6c00",
    "waiting": "#0277bd",
    "paused": "#7b1fa2",
    "pending": "#757575",
    "failed": "#c62828",
    "queued": "#1565c0",
}


def _fmt_eta(secs: float) -> str:
    if secs != secs or secs < 0:
        return "?"
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _stamp_unix(stamp: str) -> float | None:
    """Parse leading YYYYMMDD_HHMMSS from stamp (local naive → timestamp)."""
    if not stamp or len(stamp) < 15:
        return None
    head = stamp[:15]
    try:
        return datetime.strptime(head, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def _sum_epoch_sec(rows: list[dict]) -> float | None:
    secs = []
    for r in rows:
        try:
            v = float(r.get("epoch_sec"))
        except (TypeError, ValueError):
            continue
        if v == v and v > 0:
            secs.append(v)
    return float(sum(secs)) if secs else None


def _nnunet_fold_dir(stamp: str) -> Path:
    return NN_RESULTS / stamp / DS228 / TF228 / "fold_0"


def _nnunet_epoch_time_sum(fold_dir: Path) -> float | None:
    times: list[float] = []
    for lg in sorted(fold_dir.glob("training_log*.txt")):
        try:
            text = lg.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if "Epoch time:" not in line:
                continue
            # e.g. Epoch time: 26.54 s
            try:
                part = line.split("Epoch time:", 1)[1]
                v = float(part.strip().split()[0])
            except (IndexError, ValueError):
                continue
            if v == v and v > 0:
                times.append(v)
    return float(sum(times)) if times else None


def _nnunet_wall_to_final(stamp: str, fold_dir: Path) -> float | None:
    t0 = _stamp_unix(stamp)
    fin = fold_dir / "checkpoint_final.pth"
    if t0 is None or not fin.is_file():
        return None
    return max(0.0, fin.stat().st_mtime - t0)


def estimate_stage_train_sec(method_key: str, stage_key: str, st: dict) -> float | None:
    """Train seconds for a stage.

    PSMA fs50/fs10/fs5: max over folds of pure train time (sum of Epoch time /
    metrics epoch_sec). Never sum folds, never include idle/resume gaps or TEST.
    """
    stamp = (st.get("stamp") or "").strip()
    if not stamp:
        return None

    # Prefer explicit override
    if st.get("train_sec") is not None and st.get("_train_sec_locked"):
        try:
            return float(st["train_sec"])
        except (TypeError, ValueError):
            pass

    if method_key == "nnunet" and stage_key == "fdg_pretrain":
        fd = _nnunet_fold_dir(stamp)
        wall = _nnunet_wall_to_final(stamp, fd)
        logged = _nnunet_epoch_time_sum(fd)
        # FDG single-fold: keep prior wall preference when available
        return wall if wall is not None else logged

    if method_key == "nnunet" and stage_key in PSMA_STAGE_KEYS:
        # Parallel folds → max(fold train), not sum; prefer Epoch-time logs
        # (wall stamp→ckpt_final includes crash idle / post-train TEST wait).
        logs: list[float] = []
        walls: list[float] = []
        for f in (2, 5, 8):
            fold_stamp = f"{stamp}_f{f}"
            fd = _nnunet_fold_dir(fold_stamp)
            lg = _nnunet_epoch_time_sum(fd)
            if lg is not None:
                logs.append(lg)
            w = _nnunet_wall_to_final(fold_stamp, fd)
            if w is not None:
                walls.append(w)
        if logs:
            return max(logs)
        if walls:
            return max(walls)
        return None

    if method_key == "dpdnet" and stage_key == "fdg_pretrain":
        fd = _dpdnet_fold_dir(stamp)
        if fd is None:
            return None
        wall = _nnunet_wall_to_final(stamp, fd)
        logged = _nnunet_epoch_time_sum(fd)
        return wall if wall is not None else logged

    if method_key == "dpdnet" and stage_key in PSMA_STAGE_KEYS:
        logs: list[float] = []
        walls: list[float] = []
        for f in (2, 5, 8):
            fd = _dpdnet_psma_fold_dir(stamp, f)
            if fd is None:
                continue
            lg = _nnunet_epoch_time_sum(fd)
            if lg is not None:
                logs.append(lg)
            w = _nnunet_wall_to_final(f"{stamp}_f{f}", fd)
            if w is not None:
                walls.append(w)
        if logs:
            return max(logs)
        if walls:
            return max(walls)
        return None

    # MAE / MONAI / SegAnyPET under REPO
    if stage_key == "fdg_pretrain":
        metric_paths: list[Path] = []
        if method_key == "seganypet":
            metric_paths.append(REPO / stamp / "seganypet_fdg" / "metrics.jsonl")
        metric_paths.append(REPO / stamp / "metrics.jsonl")
        for mp in metric_paths:
            s = _sum_epoch_sec(_read_metrics_rows(mp))
            if s is not None and s > 0:
                return s
        # fallback: stamp → best/latest ckpt
        t0 = _stamp_unix(stamp)
        if t0 is None:
            return None
        for name in (
            "best_seg_fdg_mae.pth",
            "best_seg_fdg_monai.pth",
            "latest_seg_fdg_mae.pth",
            "latest_seg_fdg_monai.pth",
            "best.pth",
            "latest.pth",
        ):
            for p in (REPO / stamp / name, REPO / stamp / "seganypet_fdg" / name):
                if p.is_file():
                    return max(0.0, p.stat().st_mtime - t0)
        return None

    if stage_key in PSMA_STAGE_KEYS:
        # Parallel folds: max(sum epoch_sec per fold); metrics are train-only
        candidates: list[float] = []
        for sub in ("mae", "monai", "seganypet", ""):
            for f in NINE_FOLDS:
                if sub:
                    p = REPO / stamp / sub / f"fold{f}" / "metrics.jsonl"
                else:
                    p = REPO / stamp / f"fold{f}" / "metrics.jsonl"
                s = _sum_epoch_sec(_read_metrics_rows(p))
                if s is not None:
                    candidates.append(s)
            if candidates:
                break
        return max(candidates) if candidates else None

    return None


def refresh_stage_train_times(board: dict) -> None:
    """Attach train_sec / train_time onto stages (esp. DONE), including fs10/fs5."""
    methods = board.get("methods") or {}
    stage_keys = ("fdg_pretrain",) + PSMA_STAGE_KEYS
    for mk, m in methods.items():
        if not isinstance(m, dict):
            continue
        for sk in stage_keys:
            st = m.get(sk)
            if not isinstance(st, dict):
                continue
            status = (st.get("status") or "").lower()
            # always refresh when done/running/paused with stamp; keep pending blank unless archived stamp exists
            if status not in ("done", "running", "paused") and not st.get("stamp"):
                continue
            sec = estimate_stage_train_sec(mk, sk, st)
            if sec is None:
                continue
            prev = st.get("train_sec")
            if (
                status == "done"
                and isinstance(prev, (int, float))
                and float(prev) > 60
                and float(sec) < 60
            ):
                continue
            st["train_sec"] = float(sec)
            st["train_time"] = _fmt_eta(sec)


def _read_metrics_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _eta_from_rows(rows: list[dict], total_epochs: int) -> dict[str, Any]:
    """Estimate remaining time from metrics.jsonl epoch_sec history."""
    if not rows or total_epochs <= 0:
        return {"eta_sec": None, "eta": None, "epoch": None, "total_epochs": total_epochs}
    cur = int(rows[-1].get("epoch") or 0)
    secs = []
    for r in rows[-max(3, min(20, len(rows))) :]:
        es = r.get("epoch_sec")
        try:
            v = float(es)
        except (TypeError, ValueError):
            continue
        # skip outlier VAL epochs when averaging leftover train time
        if v == v and 0 < v < 120:
            secs.append(v)
    if not secs:
        for r in rows[-max(3, min(20, len(rows))) :]:
            try:
                v = float(r.get("epoch_sec"))
            except (TypeError, ValueError):
                continue
            if v == v and v > 0:
                secs.append(v)
    if not secs:
        return {"eta_sec": None, "eta": None, "epoch": cur, "total_epochs": total_epochs}
    avg = sum(secs) / len(secs)
    left = total_epochs - cur
    if left > 0:
        eta_sec = avg * left
        n_val = sum(1 for e in range(cur + 1, total_epochs + 1) if e % 20 == 0)
        if n_val:
            eta_sec += n_val * 90.0  # remaining VAL epochs (sw_dice / late-dual)
    elif cur > 0:
        # still writing past nominal (val-Dice decline) → one more epoch
        eta_sec = avg
    else:
        eta_sec = 0.0
    return {
        "eta_sec": eta_sec,
        "eta": _fmt_eta(eta_sec),
        "epoch": cur,
        "total_epochs": total_epochs,
        "avg_epoch_sec": avg,
    }


def eta_from_metrics(path: Path, total_epochs: int = 100) -> dict[str, Any]:
    return _eta_from_rows(_read_metrics_rows(path), total_epochs)


def eta_parallel_folds(
    stamp: str,
    sub: str,
    folds: tuple[int, ...] = NINE_FOLDS,
    total_epochs: int = 100,
) -> dict[str, Any]:
    """Bottleneck ETA across parallel folds (max remaining among unfinished)."""
    per: dict[str, dict[str, Any]] = {}
    rem: list[float] = []
    epochs: list[int] = []
    for f in folds:
        info = eta_from_metrics(REPO / stamp / sub / f"fold{f}" / "metrics.jsonl", total_epochs)
        per[str(f)] = info
        es = info.get("eta_sec")
        # ignore finished folds (0s leftover) so they do not mask a live fold
        if es is not None and float(es) > 1:
            rem.append(float(es))
        if info.get("epoch") is not None:
            epochs.append(int(info["epoch"]))
    if not rem:
        return {
            "eta": None,
            "eta_sec": 0.0 if epochs else None,
            "epoch": min(epochs) if epochs else None,
            "total_epochs": total_epochs,
            "per_fold": per,
        }
    eta_sec = max(rem)
    return {
        "eta": _fmt_eta(eta_sec),
        "eta_sec": eta_sec,
        "epoch": min(epochs) if epochs else None,
        "total_epochs": total_epochs,
        "per_fold": per,
    }


def _stage_total_epochs(stage: dict) -> int:
    for k in ("total_epochs", "epochs"):
        v = stage.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 100


def _stage_gbs(stage: dict) -> int | None:
    """Global batch size only (ignore per-GPU allocation)."""
    for k in ("gbs", "global_bs", "global_batch"):
        v = stage.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    import re

    note = str(stage.get("bs_note") or "")
    stamp = str(stage.get("stamp") or "")
    m = re.search(r"gbs[=_]?(\d+)", note, re.I) or re.search(r"_gbs(\d+)_", stamp, re.I)
    if m:
        return int(m.group(1))
    # "global …" → stored bs is already global
    if re.search(r"\bglobal\b", note, re.I):
        try:
            return int(stage["bs"]) if stage.get("bs") is not None else None
        except (TypeError, ValueError):
            return None
    # per-GPU × N cards/gpus when note/stamp encodes it
    m_n = re.search(r"(\d+)\s*card", note, re.I) or re.search(r"(\d+)gpu", stamp, re.I)
    if m_n and stage.get("bs") is not None and re.search(r"per[- ]?GPU", note, re.I):
        try:
            return int(stage["bs"]) * int(m_n.group(1))
        except (TypeError, ValueError):
            pass
    # default: treat bs as gbs (protocol / nnUNet FIXED_BATCH = global)
    try:
        return int(stage["bs"]) if stage.get("bs") is not None else None
    except (TypeError, ValueError):
        return None


def _nnunet_train_loss_best_epoch(fold_dir: Path) -> int | None:
    """1-based finished epoch of last 'New best train_loss'."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    cur = None
    last = None
    try:
        lines = logs[-1].read_text(errors="ignore").splitlines()
    except OSError:
        return None
    import re

    for line in lines:
        m = re.search(r"Epoch\s+(\d+)\s*$", line)
        if m:
            cur = int(m.group(1))
            continue
        if "New best train_loss" in line and cur is not None:
            last = cur + 1
    return last


def _nnunet_final_epoch(fold_dir: Path, total: int) -> int | None:
    if (fold_dir / "checkpoint_final.pth").is_file():
        return total
    cur = _nnunet_log_latest_epoch(fold_dir)
    if cur is None:
        return None
    return min(total, int(cur) + 1)


def enrich_fdg_best_epochs(board: dict) -> None:
    """Attach fdg_pretrain.best_ep = epoch of selected FDG best/init ckpt."""
    methods = board.get("methods") or {}

    # nnUNet: MAE-align → checkpoint_best (max ema_fg_dice); legacy → final
    for _nn_key in ("nnunet", "nnunet_mim"):
        nn = (methods.get(_nn_key) or {}).get("fdg_pretrain") or {}
        stamp = (nn.get("stamp") or "").strip()
        if not stamp:
            continue
        tot = _stage_total_epochs(nn) or 100
        nn["total_epochs"] = tot
        fd = _nnunet_fold_dir(stamp)
        ckpt = Path(nn.get("best_ckpt") or "").name.lower()
        note = str(nn.get("note") or "").lower()
        want_best = ("best" in ckpt and "final" not in ckpt) or (
            "fullcase" in note or "ema_fg_dice" in note or "mae-align" in note
        )
        ep = None
        if fd.is_dir():
            if want_best:
                ep = _nnunet_ema_best_epoch(fd) or _nnunet_final_epoch(fd, tot)
            elif "final" in ckpt or (not ckpt and (fd / "checkpoint_final.pth").is_file()):
                ep = _nnunet_final_epoch(fd, tot)
            else:
                ep = (
                    _nnunet_ema_best_epoch(fd)
                    or _nnunet_train_loss_best_epoch(fd)
                    or _nnunet_final_epoch(fd, tot)
                )
        if ep is not None:
            nn["best_ep"] = int(ep)
        gbs = _stage_gbs(nn)
        if gbs is not None:
            nn["gbs"] = gbs

    # MAE / MONAI / SegAnyPET: metrics best_dice epoch
    for key, metrics_rel in (
        ("mae_swinunetr", lambda s: REPO / s / "metrics.jsonl"),
        ("mae_scratch", lambda s: REPO / s / "metrics.jsonl"),
        ("monai_swinvit", lambda s: REPO / s / "metrics.jsonl"),
        ("monai_scratch", lambda s: REPO / s / "metrics.jsonl"),
        ("seganypet", lambda s: REPO / s / "seganypet_fdg" / "metrics.jsonl"),
        ("seganypet_scratch", lambda s: REPO / s / "seganypet_fdg" / "metrics.jsonl"),
    ):
        st = (methods.get(key) or {}).get("fdg_pretrain") or {}
        stamp = (st.get("stamp") or "").strip()
        if not stamp:
            continue
        tot = _stage_total_epochs(st) or 100
        st["total_epochs"] = tot
        ep = _mae_best_epoch_from_metrics(metrics_rel(stamp))
        if ep is not None:
            st["best_ep"] = int(ep)
        gbs = _stage_gbs(st)
        if gbs is not None:
            st["gbs"] = gbs

    # DpDNet: MAE-align → checkpoint_best (ema); legacy → final
    for _dpd_key in ("dpdnet", "dpdnet_dualenc"):
        dpd = (methods.get(_dpd_key) or {}).get("fdg_pretrain") or {}
        stamp = (dpd.get("stamp") or "").strip()
        if not stamp:
            continue
        tot = _stage_total_epochs(dpd) or 100
        dpd["total_epochs"] = tot
        fd = _dpdnet_fold_dir(stamp)
        ckpt = Path(dpd.get("best_ckpt") or "").name.lower()
        note = str(dpd.get("note") or "").lower()
        want_best = ("best" in ckpt and "final" not in ckpt) or (
            "fullcase" in note or "ema_fg_dice" in note or "mae-align" in note
        )
        if fd is not None and fd.is_dir():
            if want_best:
                ep = _nnunet_ema_best_epoch(fd) or _nnunet_final_epoch(fd, tot)
            else:
                ep = _nnunet_final_epoch(fd, tot)
                if ep is None:
                    ep = _nnunet_val_loss_best_epoch(fd) or _nnunet_train_loss_best_epoch(fd)
            if ep is not None:
                dpd["best_ep"] = int(ep)
        gbs = _stage_gbs(dpd)
        if gbs is not None:
            dpd["gbs"] = gbs

    # PSMA stages: still fill gbs for display
    for key in METHOD_ORDER:
        m = methods.get(key) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            st = m.get(stage) or {}
            if not isinstance(st, dict):
                continue
            gbs = _stage_gbs(st)
            if gbs is not None:
                st["gbs"] = gbs


def _nnunet_fold_epoch_times(fold_dir: Path) -> list[float]:
    times: list[float] = []
    for lg in sorted(fold_dir.glob("training_log*.txt")):
        try:
            text = lg.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if "Epoch time:" not in line:
                continue
            try:
                part = line.split("Epoch time:", 1)[1]
                v = float(part.strip().split()[0])
            except (IndexError, ValueError):
                continue
            if v == v and v > 0:
                times.append(v)
    return times


def _nnunet_log_latest_epoch(fold_dir: Path) -> int | None:
    """Parse last 'Epoch N' line from nnUNet / DpDNet training_log*.txt."""
    last: int | None = None
    for lg in sorted(fold_dir.glob("training_log*.txt")):
        try:
            text = lg.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            # e.g. "2026-08-17 06:55:19.650381: Epoch 0"
            if ": Epoch " not in line and not line.strip().endswith("Epoch 0"):
                # also match bare "Epoch 12"
                pass
            m = None
            if "Epoch " in line and "Epoch time" not in line and "learning rate" not in line.lower():
                try:
                    part = line.rsplit("Epoch ", 1)[1].strip()
                    # take leading int
                    num = ""
                    for ch in part:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    if num:
                        last = int(num)
                except (IndexError, ValueError):
                    continue
    return last


def _dpdnet_fold_dir(stamp: str) -> Path | None:
    if not stamp:
        return None
    for tf in (TF239_PRETRAIN, TF239):
        fd = NN_RESULTS / stamp / DS239 / tf / "fold_0"
        if fd.is_dir():
            return fd
    return None


def _dpdnet_psma_fold_dir(parent_stamp: str, fold: int) -> Path | None:
    """Parent stamp + fold → nnUNet_results/<parent>_f{fold}/Dataset240/.../fold_{fold}."""
    if not parent_stamp:
        return None
    for tf in (TF240_PRETRAIN, TF240):
        fd = (
            NN_RESULTS
            / f"{parent_stamp}_f{fold}"
            / DS240
            / tf
            / f"fold_{fold}"
        )
        if fd.is_dir():
            return fd
    return None


def _dpdnet_fdg_finished(fd: Path | None, total_epochs: int) -> bool:
    """True when FDG wrote checkpoint_final or log says Training done."""
    if fd is None or not fd.is_dir():
        return False
    if (fd / "checkpoint_final.pth").is_file():
        return True
    for lg in sorted(fd.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = lg.read_text(errors="ignore")
        except OSError:
            continue
        if "Training done" in text:
            return True
        # Do not use cumulative Epoch-time count (continue/resume over-counts).
        # Near end: require "Training done" above, or last epoch + final ckpt.
        break
    return False


def _eta_from_nnunet_fold(fd: Path | None, total_epochs: int = 100) -> dict[str, Any]:
    """Single-fold ETA from nnUNet-style training_log*.txt (Epoch time lines).

    Progress uses the latest ``Epoch N`` (0-based), **not** ``len(Epoch time)``
    across all logs — continue/resume writes many logs and would over-count past
    ``total_epochs``, wrongly yielding ETA 0s.
    """
    if fd is None or not fd.is_dir():
        return {"eta": None, "eta_sec": None, "epoch": None, "total_epochs": total_epochs, "done": False}
    cur0 = _nnunet_log_latest_epoch(fd)
    # checkpoint_final from the nominal 100ep run must not hide decline continuation
    past_nominal = cur0 is not None and int(cur0) + 1 >= int(total_epochs)
    if (not past_nominal) and _dpdnet_fdg_finished(fd, total_epochs):
        cur = _nnunet_log_latest_epoch(fd)
        return {
            "eta": None,
            "eta_sec": 0.0,
            "epoch": total_epochs if cur is None else max(cur + 1, total_epochs),
            "total_epochs": total_epochs,
            "done": True,
        }
    times = _nnunet_fold_epoch_times(fd)
    cur = _nnunet_log_latest_epoch(fd)
    if cur is None and times:
        # between epochs / incomplete log header: approximate from last session only
        cur = min(total_epochs - 1, len(times) - 1)
    if not times:
        return {
            "eta": None,
            "eta_sec": None,
            "epoch": cur,
            "total_epochs": total_epochs,
            "done": False,
        }
    recent = times[-max(3, min(20, len(times))) :]
    avg = sum(recent) / len(recent)
    # "Epoch N" is printed at start of epoch N → currently in N; remaining ≈ total-N.
    if cur is not None:
        left = max(0.0, float(total_epochs - int(cur) - 1) + 0.85)
    else:
        left = max(0.0, float(total_epochs))
    eta_sec = avg * left
    return {
        "eta": _fmt_eta(eta_sec),
        "eta_sec": eta_sec,
        "epoch": cur,
        "total_epochs": total_epochs,
        "done": False,
        "avg_epoch_sec": avg,
    }


def eta_dpdnet_fdg(stamp: str, total_epochs: int = 100) -> dict[str, Any]:
    """Single-fold ETA for DpDNet FDG (nnUNet-style training_log)."""
    return _eta_from_nnunet_fold(_dpdnet_fold_dir(stamp), total_epochs)


def eta_nnunet_fdg(stamp: str, total_epochs: int = 100) -> dict[str, Any]:
    """Single-fold ETA for nnUNet FDG (Dataset228 fold_0)."""
    if not stamp:
        return {"eta": None, "eta_sec": None, "epoch": None, "total_epochs": total_epochs, "done": False}
    fd = _nnunet_fold_dir(stamp)
    return _eta_from_nnunet_fold(fd if fd.is_dir() else None, total_epochs)


def _resolve_nnunet_fdg_stamp(st: dict) -> str:
    """Fill stamp when board marked running but launch forgot to patch it."""
    stamp = (st.get("stamp") or "").strip()
    if stamp:
        return stamp
    last = CTRL / "ICLR2026/vis/baseline1_fdg_LAST_STAMP.txt"
    if last.is_file():
        try:
            s = last.read_text().strip()
        except OSError:
            s = ""
        if s and (_nnunet_fold_dir(s)).is_dir():
            return s
    total = _stage_total_epochs(st) or 169
    cands = sorted(
        NN_RESULTS.glob(f"*_iclr2026_baseline1_fdg_*_{total}ep*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in cands:
        if p.is_dir() and (_nnunet_fold_dir(p.name)).is_dir():
            return p.name
    return ""


_STAMP_CMD_RE = re.compile(r"(\d{8}_\d{6}_iclr2026_[^\s\"']+)")
# FDG supervised (3-GPU + VAL) — only for fdg_pretrain fallback.
_DEFAULT_EPOCH_SEC: dict[str, float] = {
    "mae_swinunetr": 400.0,
    "mae_scratch": 400.0,
    "monai_swinvit": 420.0,
    "monai_scratch": 420.0,
    "seganypet": 110.0,
    "seganypet_scratch": 110.0,
    "dpdnet_dualenc": 180.0,
    "dpdnet": 180.0,
    "nnunet_mim": 200.0,
    "nnunet": 200.0,
}
# PSMA fewshot / fc70 measured wall ≈ 6–40s/train-ep (not FDG 400s).
_FEWSHOT_EPOCH_SEC: dict[str, float] = {
    "mae_swinunetr": 25.0,
    "mae_scratch": 25.0,
    "monai_swinvit": 30.0,
    "monai_scratch": 30.0,
    "seganypet": 20.0,
    "seganypet_scratch": 20.0,
    "dpdnet_dualenc": 40.0,
    "dpdnet": 40.0,
    "nnunet_mim": 25.0,
    "nnunet": 25.0,
}
_EXTRA_FOLD_NOTE_RE = re.compile(r"9fold extra\b.*?[·\s]f(\d+)\b", re.I)
_FDG_TEST_METHOD_DIR = {
    "nnunet_mim": "nnunet",
    "nnunet": "nnunet",
    "mae_swinunetr": "mae",
    "mae_scratch": "mae_scratch",
    "monai_swinvit": "monai",
    "monai_scratch": "monai_scratch",
    "dpdnet_dualenc": "dpdnet",
    "dpdnet": "dpdnet",
    "seganypet": "seganypet",
    "seganypet_scratch": "seganypet_scratch",
    "proto_retrieval": "proto_retrieval",
}


def _pgrep_cmdlines(pattern: str) -> list[str]:
    try:
        r = subprocess.run(
            ["pgrep", "-af", pattern],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0 or not (r.stdout or "").strip():
        return []
    skip = ("pgrep", "cursor", "gpu_idle_queue", "queue_keeper", "__CURSOR_SANDBOX")
    out: list[str] = []
    for line in r.stdout.splitlines():
        if any(s in line for s in skip):
            continue
        out.append(line)
    return out


def _normalize_parent_stamp(stamp: str) -> str:
    s = (stamp or "").strip().rstrip("/")
    if s.endswith("_f0"):
        return s[:-3]
    return s


def _extract_stamp_from_cmdline(line: str) -> str:
    m = _STAMP_CMD_RE.search(line)
    if not m:
        return ""
    return _normalize_parent_stamp(m.group(1))


_DOCKER_NAMES_CACHE: list[str] | None = None
_DOCKER_FEWSHOT_RE = re.compile(
    r"(mae|monai|seganypet)_fs(\d+)_fdgseg_f(\d+)_(\d{8}_\d{6}_iclr2026_\S+)"
)
_DOCKER_FC70_RE = re.compile(
    r"(seganypet|mae|monai)_fc70_f\d+_(\d{8}_\d{6}_iclr2026_\S+)"
)
_DOCKER_DPD_FC70_RE = re.compile(
    r"dpdnet_psma_f\d+_(\d{8}_\d{6}_iclr2026_dpdnet_psma_fc70\S+?)(?:_f\d+)?$"
)
_METHOD_DOCKER_PREFIX = {
    "mae_swinunetr": "mae",
    "mae_scratch": "mae",
    "monai_swinvit": "monai",
    "monai_scratch": "monai",
    "seganypet": "seganypet",
    "seganypet_scratch": "seganypet",
}


def _reset_runtime_caches() -> None:
    global _DOCKER_NAMES_CACHE
    _DOCKER_NAMES_CACHE = None


def _docker_ps_names() -> list[str]:
    global _DOCKER_NAMES_CACHE
    if _DOCKER_NAMES_CACHE is not None:
        return _DOCKER_NAMES_CACHE
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _DOCKER_NAMES_CACHE = [
            ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()
        ]
    except (OSError, subprocess.SubprocessError):
        _DOCKER_NAMES_CACHE = []
    return _DOCKER_NAMES_CACHE


def _live_fewshot_fold(mkey: str, fewshot_n: int, stamp: str = "") -> int | None:
    """Fold currently training in docker (extra-fold / wave), if any."""
    prefix = _METHOD_DOCKER_PREFIX.get(mkey, "")
    if not prefix:
        return None
    needle = f"{prefix}_fs{fewshot_n}_fdgseg_f"
    for name in _docker_ps_names():
        if needle not in name:
            continue
        if stamp and stamp not in name:
            continue
        m = _DOCKER_FEWSHOT_RE.search(name)
        if m and int(m.group(2)) == int(fewshot_n):
            f = int(m.group(3))
            if f in NINE_FOLDS:
                return f
    return None


def _live_fewshot_slots() -> dict[tuple[str, str], dict[str, Any]]:
    """Docker extra-fold / fewshot jobs → {(mkey, stage): folds/names/stamp}."""
    prefix_mkey = {
        "mae": "mae_swinunetr",
        "monai": "monai_swinvit",
        "seganypet": "seganypet",
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for name in _docker_ps_names():
        m = _DOCKER_FEWSHOT_RE.search(name)
        if not m:
            continue
        prefix, n_s, fold_s, stamp = m.group(1), m.group(2), m.group(3), m.group(4)
        mkey = prefix_mkey.get(prefix)
        if stamp:
            mkey = _board_method_from_stamp(stamp, mkey or "")
        if not mkey:
            continue
        if "mae_scratch" in (stamp or ""):
            mkey = "mae_scratch"
        stage = f"psma_fs{int(n_s)}_f258"
        rec = out.setdefault((mkey, stage), {"folds": [], "names": [], "stamp": stamp})
        rec["folds"].append(int(fold_s))
        rec["names"].append(name)
        rec["stamp"] = stamp
    return out


def _extra_fold_docker_live(stamp: str = "", prefix: str = "") -> bool:
    if not stamp and not prefix:
        return False
    for name in _docker_ps_names():
        if stamp and stamp not in name:
            continue
        if prefix and f"{prefix}_fs" not in name and f"{prefix}_psma_f" not in name:
            continue
        if any(tok in name for tok in ("_fdgseg_f", "seganypet_fs", "_psma_f")):
            return True
    return False


def _apply_live_extra_fold_running(board: dict) -> None:
    """Keep extra-fold cells RUNNING while their docker jobs are alive."""
    methods = board.get("methods") or {}
    sub_of = {
        "mae_swinunetr": "mae",
        "mae_scratch": "mae",
        "monai_swinvit": "monai",
        "monai_scratch": "monai",
        "seganypet": "seganypet",
        "seganypet_scratch": "seganypet",
    }
    for (mkey, stage), rec in _live_fewshot_slots().items():
        st = (methods.get(mkey) or {}).get(stage)
        if not isinstance(st, dict):
            continue
        st["status"] = "running"
        _clear_stage_wait_fields(st)
        st["device"] = "gpu"
        folds = sorted({int(x) for x in rec.get("folds") or []})
        names = tuple(str(x) for x in rec.get("names") or [] if x)
        gpus = _live_gpu_ids_for_patterns(names) if names else None
        if gpus:
            st["gpu_ids"] = ",".join(str(i) for i in gpus)
        n = stage.replace("psma_fs", "").replace("_f258", "")
        sub = sub_of.get(mkey, mkey)
        ftxt = ",".join(str(f) for f in folds) if folds else "?"
        st["note"] = f"9fold extra · {sub} fs{n} f{ftxt} · GPU {st.get('gpu_ids') or '?'}"
        if rec.get("stamp") and not (st.get("stamp") or "").strip():
            st["stamp"] = rec["stamp"]


def _live_fc70_stamp(mkey: str) -> str:
    for name in _docker_ps_names():
        if mkey == "dpdnet":
            m = _DOCKER_DPD_FC70_RE.search(name)
            if m:
                return _normalize_parent_stamp(m.group(1))
            continue
        m = _DOCKER_FC70_RE.search(name)
        if not m:
            continue
        prefix = m.group(1)
        want = {
            "seganypet": "seganypet",
            "mae_swinunetr": "mae",
            "mae_scratch": "mae",
            "monai_swinvit": "monai",
            "monai_scratch": "monai",
        }.get(mkey)
        if want and prefix == want:
            return _normalize_parent_stamp(m.group(2))
    return ""


def _decline_monitor_for_stamp(stamp: str) -> bool:
    if not stamp:
        return False
    for line in _pgrep_lines(("monitor_val_dice_decline_stop.py",)):
        if stamp in line:
            return True
    return False


def _eta_to_next_val(cur: int, avg_sec: float, val_every: int = 20) -> tuple[float, int]:
    """Decline jobs stop at the next val, or continue — ETA is time to that val."""
    ve = max(1, int(val_every or 20))
    nxt = ((int(cur) // ve) + 1) * ve
    left = max(1, nxt - int(cur))
    return float(avg_sec) * left, nxt


def _resolve_stage_stamp(mkey: str, stage_key: str, st: dict) -> str:
    """Fill / refresh stamp from live docker when the board cell is stale."""
    stamp = (st.get("stamp") or "").strip()
    # Completed TEST20 must keep its stamp (do not steal an older live/glob run).
    if stage_key == FC70_STAGE_KEY and stamp and _stage_has_score(st):
        return stamp
    if stage_key == FC70_STAGE_KEY:
        live = _live_fc70_stamp(mkey)
        if live:
            return live
    if stamp:
        return stamp
    pats = STAGE_RUNTIME_PATTERNS.get((mkey, stage_key), ())
    for pat in pats:
        for line in _pgrep_cmdlines(pat):
            s = _extract_stamp_from_cmdline(line)
            if s:
                return s
    tag = {
        "seganypet": "seganypet_psma_fc70",
        "dpdnet": "dpdnet_psma_fc70",
        "monai_swinvit": "monai_psma_fc70",
        "mae_swinunetr": "mae_psma_fc70",
        "mae_scratch": "mae_scratch_psma_fc70",
        "monai_scratch": "monai_scratch_psma_fc70",
        "seganypet_scratch": "seganypet_scratch_psma_fc70",
        "nnunet_mim": "nnunet_mim_psma_fc70",
        "nnunet": "nnunet_psma_fc70",
        "dpdnet_dualenc": "dpdnet_dualenc_psma_fc70",
    }.get(mkey, "")
    if stage_key == FC70_STAGE_KEY and tag:
        for base in (REPO, NN_RESULTS):
            try:
                cands = sorted(base.glob(f"*{tag}*"), key=lambda p: p.stat().st_mtime, reverse=True)
            except OSError:
                cands = []
            for p in cands:
                if p.is_dir():
                    return _normalize_parent_stamp(p.name)
    return ""


def _single_run_train_eta(mkey: str, sub: str, fold: str, stamp: str, total: int) -> dict[str, Any]:
    fold_i = int(fold.replace("fold", "")) if str(fold).startswith("fold") else int(fold or 0)
    if mkey in ("dpdnet", "dpdnet_dualenc"):
        info = _eta_from_nnunet_fold(_dpdnet_psma_fold_dir(stamp, fold_i), total)
    elif mkey in ("nnunet", "nnunet_mim"):
        fd = _nnunet_fold_dir(f"{stamp}_f{fold_i}")
        if not fd.is_dir():
            fd = _nnunet_fold_dir(stamp)
        info = _eta_from_nnunet_fold(fd if fd.is_dir() else None, total)
    else:
        info = eta_from_metrics(REPO / stamp / sub / f"fold{fold_i}" / "metrics.jsonl", total)
    cur = info.get("epoch")
    avg = info.get("avg_epoch_sec")
    try:
        avg_f = float(avg) if avg is not None else float(_FEWSHOT_EPOCH_SEC.get(mkey, 40.0))
    except (TypeError, ValueError):
        avg_f = float(_FEWSHOT_EPOCH_SEC.get(mkey, 40.0))
    # val-Dice decline continues past nominal 100ep until the next val (cap 300)
    if cur is not None and int(cur) >= int(total) and _decline_monitor_for_stamp(stamp):
        eta_sec, nxt = _eta_to_next_val(int(cur), avg_f, 20)
        info["eta_sec"] = eta_sec
        info["eta"] = _fmt_eta(eta_sec)
        info["total_epochs"] = max(int(total), nxt)
        info["phase"] = "decline"
    return info


def _apply_eta_to_stage(st: dict, info: dict[str, Any]) -> None:
    eta_sec = info.get("eta_sec")
    if eta_sec is not None and float(eta_sec) <= 1:
        st["eta"] = None
        st["eta_sec"] = None
    else:
        st["eta"] = info.get("eta")
        st["eta_sec"] = eta_sec
    if info.get("epoch") is not None:
        st["epoch"] = info["epoch"]
    if info.get("total_epochs"):
        st["total_epochs"] = info["total_epochs"]
    if info.get("phase"):
        st["phase"] = info["phase"]


def _eta_sec_valid(v: Any) -> bool:
    try:
        return v is not None and float(v) > 1
    except (TypeError, ValueError):
        return False


def _stage_has_valid_eta(st: dict, stage_key: str = "") -> bool:
    live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
    fewshot = (stage_key or "").startswith("psma_fs") or stage_key == FC70_STAGE_KEY
    for eta, es in (
        (live.get("eta"), live.get("eta_sec")),
        (st.get("eta"), st.get("eta_sec")),
    ):
        if not eta or str(eta) in ("0s", "0", "0m00s", "…"):
            continue
        try:
            esf = float(es) if es is not None else None
        except (TypeError, ValueError):
            esf = None
        # leftover FDG fallback (400s×100ep ≈ 11h) is not a real fewshot ETA
        if fewshot and esf is not None and esf >= 3 * 3600:
            return False
        # decline continuation: leftover-to-100 collapsing to <2min is wrong
        if stage_key == FC70_STAGE_KEY and esf is not None and esf < 120:
            ep = st.get("epoch")
            tot = st.get("total_epochs")
            try:
                if ep is not None and tot is not None and int(ep) >= int(tot):
                    return False
            except (TypeError, ValueError):
                pass
        if _eta_sec_valid(es) or es is None:
            return True
    return False


def _extra_fold_from_note(note: str) -> int | None:
    m = _EXTRA_FOLD_NOTE_RE.search(note or "")
    if not m:
        return None
    f = int(m.group(1))
    return f if f in NINE_FOLDS else None


def _fallback_epoch_sec(st: dict, mkey: str = "", stage_key: str = "") -> float:
    """Seconds/epoch for ETA fallback — never use FDG 400s on fewshot cells."""
    try:
        ts = float(st.get("train_sec") or 0)
    except (TypeError, ValueError):
        ts = 0.0
    tot = max(1, _stage_total_epochs(st))
    if ts >= 60:
        rate = ts / tot
        if 3.0 <= rate <= 180.0:
            return rate
    sk = stage_key or ""
    if sk == "fdg_pretrain":
        return float(_DEFAULT_EPOCH_SEC.get(mkey, 300.0))
    if sk.startswith("psma_fs") or sk == FC70_STAGE_KEY:
        return float(_FEWSHOT_EPOCH_SEC.get(mkey, 30.0))
    return float(_FEWSHOT_EPOCH_SEC.get(mkey) or _DEFAULT_EPOCH_SEC.get(mkey, 60.0))


def _ensure_running_has_eta(st: dict, mkey: str = "", stage_key: str = "") -> None:
    """Guarantee every RUNNING cell has a displayable ETA (estimate if metrics missing)."""
    if (st.get("status") or "").lower() != "running":
        return
    if _stage_has_valid_eta(st, stage_key):
        return
    live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
    phase = (st.get("phase") or "").upper()
    if phase == "TEST20":
        done = int(live.get("cases_done") or st.get("eval_done") or 0)
        total = int(live.get("cases_total") or st.get("eval_total") or 202)
        left = max(0, total - done)
        if left <= 0:
            st["eta"] = "0s"
            st["eta_sec"] = 0.0
            return
        # eval_total<=12 is fold count, not voxel cases
        if total <= 12:
            rate = 180.0
        else:
            rate = 25.0 if done > 0 else 30.0
        eta_sec = float(rate * left)
        st["eta_sec"] = eta_sec
        st["eta"] = _fmt_eta(eta_sec)
        live = {**live, "eta": st["eta"], "eta_sec": eta_sec}
        st["test_live"] = live
        return
    ep = st.get("epoch")
    tot = int(st.get("total_epochs") or 100)
    sec = _fallback_epoch_sec(st, mkey, stage_key)
    cur = int(ep or 0)
    left = tot - cur
    if left <= 0:
        if stage_key == FC70_STAGE_KEY or "decline" in str(st.get("online_val") or "").lower():
            eta_sec, nxt = _eta_to_next_val(cur, sec, 20)
            st["eta_sec"] = eta_sec
            st["eta"] = _fmt_eta(eta_sec)
            st["total_epochs"] = max(tot, nxt)
            st["phase"] = "decline"
            return
        eta_sec = float(sec)
    else:
        eta_sec = sec * left
    st["eta_sec"] = eta_sec
    st["eta"] = _fmt_eta(eta_sec)


def _running_eta_display(st: dict) -> str:
    live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
    for eta, es in (
        (live.get("eta"), live.get("eta_sec")),
        (st.get("eta"), st.get("eta_sec")),
    ):
        if not eta or str(eta) in ("0s", "0", "0m00s"):
            continue
        if _eta_sec_valid(es) or es is None:
            return str(eta)
    es = live.get("eta_sec", st.get("eta_sec"))
    if _eta_sec_valid(es):
        return _fmt_eta(float(es))
    return "…"


def _fdg_test20_pred_mtimes(method_key: str) -> tuple[int, int, list[float]]:
    sub = _FDG_TEST_METHOD_DIR.get(method_key, method_key)
    pred_root = WORK / "fdg_test20_eval" / sub / "predict"
    total = 202
    names: set[str] = set()
    mtimes: list[float] = []
    if pred_root.is_dir():
        merged = pred_root / "pred"
        if merged.is_dir():
            for p in merged.glob("*.nii.gz"):
                names.add(p.name)
                try:
                    mtimes.append(float(p.stat().st_mtime))
                except OSError:
                    pass
        if len(names) < total:
            shards = pred_root / "shards"
            if shards.is_dir():
                for sh in shards.glob("shard_*/pred"):
                    for p in sh.glob("*.nii.gz"):
                        names.add(p.name)
                        try:
                            mtimes.append(float(p.stat().st_mtime))
                        except OSError:
                            pass
    return len(names), total, mtimes


def _refresh_fdg_test_stage_eta(st: dict, mkey: str) -> None:
    done, total, mtimes = _fdg_test20_pred_mtimes(mkey)
    st["phase"] = "TEST20"
    live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
    live.update({"active": done < total, "cases_done": done, "cases_total": total})
    eta_sec = _eta_sec_from_pred_mtimes(done, total, mtimes, fallback_sec_per=25.0)
    if eta_sec is None and done <= 0:
        eta_sec = float(30.0 * total)
    if eta_sec is not None:
        live["eta_sec"] = eta_sec
        live["eta"] = _fmt_eta(eta_sec)
        st["eta_sec"] = eta_sec
        st["eta"] = live["eta"]
    st["test_live"] = live
    st["note"] = f"FDG TEST pred {done}/{total}"


def _clear_nonrunning_etas(board: dict) -> None:
    """Pending/done cells must not keep a leftover 11h fallback ETA."""
    methods = board.get("methods") or {}
    for mkey in METHOD_ORDER:
        m = methods.get(mkey) or {}
        if not isinstance(m, dict):
            continue
        for sk, st in m.items():
            if not isinstance(st, dict):
                continue
            status = (st.get("status") or "").lower()
            if status in ("running", "paused", "waiting"):
                continue
            if st.get("eta") is not None or st.get("eta_sec") is not None:
                st["eta"] = None
                st["eta_sec"] = None


def _finalize_all_running_etas(board: dict) -> None:
    methods = board.get("methods") or {}
    for mkey in METHOD_ORDER:
        m = methods.get(mkey) or {}
        for sk in ("fdg_pretrain",) + PSMA_STAGE_KEYS:
            st = m.get(sk)
            if isinstance(st, dict) and (st.get("status") or "").lower() == "running":
                _ensure_running_has_eta(st, mkey, sk)
    _clear_nonrunning_etas(board)


def _nnunet_fold_live(fold_dir: Path) -> dict[str, Any]:
    """Read task1_train_live_progress.json + fall back to log epoch count."""
    out: dict[str, Any] = {
        "epoch": None,
        "phase": None,
        "iter": None,
        "iter_total": None,
    }
    live = fold_dir / "task1_train_live_progress.json"
    if live.is_file():
        try:
            d = json.loads(live.read_text())
            out["epoch"] = int(d.get("epoch")) if d.get("epoch") is not None else None
            out["phase"] = d.get("phase")
            out["iter"] = d.get("iter")
            out["iter_total"] = d.get("iter_total")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    times = _nnunet_fold_epoch_times(fold_dir)
    if out["epoch"] is None and times:
        # finished epochs ≈ len(times); current is that index
        out["epoch"] = len(times)
    if out["epoch"] is None:
        out["epoch"] = _nnunet_log_latest_epoch(fold_dir)
    return {"live": out, "epoch_times": times}


def eta_nnunet_parallel_folds(
    parent_stamp: str,
    folds: tuple[int, ...] = (2, 5, 8),
    total_epochs: int = 300,
) -> dict[str, Any]:
    """Bottleneck ETA for nnUNet fewshot (max remaining across folds)."""
    rem: list[float] = []
    epochs: list[int] = []
    phases: list[str] = []
    per: dict[str, Any] = {}
    for f in folds:
        fd = _nnunet_fold_dir(f"{parent_stamp}_f{f}")
        info = _nnunet_fold_live(fd)
        live = info["live"]
        times: list[float] = info["epoch_times"]
        cur = live.get("epoch")
        if cur is not None:
            epochs.append(int(cur))
        if live.get("phase"):
            ph = str(live["phase"]).strip().lower()
            if ph == "val":
                phases.append("VAL")
            elif ph == "train":
                pass
            else:
                phases.append(str(live["phase"]))
        # avg of last N finished epoch times (includes occasional val epochs)
        recent = times[-max(3, min(20, len(times))) :] if times else []
        if not recent:
            # bootstrap from live epoch_start if available
            per[str(f)] = {"epoch": cur, "eta_sec": None}
            continue
        avg = sum(recent) / len(recent)
        left = max(0, total_epochs - (int(cur) if cur is not None else len(times)))
        # if mid-epoch, add fractional remainder of current epoch
        frac = 0.0
        it, itot = live.get("iter"), live.get("iter_total")
        try:
            if it is not None and itot and float(itot) > 0 and str(live.get("phase") or "").lower() == "train":
                frac = max(0.0, 1.0 - float(it) / float(itot))
        except (TypeError, ValueError):
            frac = 0.0
        eta_sec = avg * (left + frac)
        rem.append(eta_sec)
        per[str(f)] = {"epoch": cur, "eta_sec": eta_sec, "avg_epoch_sec": avg}
    if not rem:
        return {
            "eta": None,
            "eta_sec": None,
            "epoch": min(epochs) if epochs else None,
            "total_epochs": total_epochs,
            "phase": "VAL" if "VAL" in phases else (phases[0] if phases else None),
            "per_fold": per,
        }
    eta_sec = max(rem)
    return {
        "eta": _fmt_eta(eta_sec),
        "eta_sec": eta_sec,
        "epoch": min(epochs) if epochs else None,
        "total_epochs": total_epochs,
        "phase": "VAL" if "VAL" in phases else (phases[0] if phases else None),
        "per_fold": per,
    }


def _pred_case_complete(pred_dir: Path, stem: str, require_side: bool) -> bool:
    nii = pred_dir / f"{stem}.nii.gz"
    try:
        if not nii.is_file() or nii.stat().st_size <= 0:
            return False
    except OSError:
        return False
    if not require_side:
        return True
    return any(
        (pred_dir / f"{stem}{suf}").is_file()
        for suf in (".pkl", ".pkl.npz", ".npz")
    )


def _count_complete_preds(pred_dir: Path) -> tuple[int, list[float]]:
    """Return (n_complete, mtimes of complete case nii).

    nnUNet often writes .nii.gz + .pkl/.npz; DpDNet prompt predict may write only
    .nii.gz — count those when no sidecar is present in the folder.
    """
    if not pred_dir.is_dir():
        return 0, []
    mtimes: list[float] = []
    n = 0
    niis = list(pred_dir.glob("*.nii.gz"))
    if not niis:
        return 0, []
    require_side = False
    for nii in niis[: min(12, len(niis))]:
        stem = nii.name[: -len(".nii.gz")]
        if any(
            (pred_dir / f"{stem}{suf}").is_file()
            for suf in (".pkl", ".pkl.npz", ".npz")
        ):
            require_side = True
            break
    for nii in niis:
        stem = nii.name[: -len(".nii.gz")]
        if not _pred_case_complete(pred_dir, stem, require_side):
            continue
        try:
            mtimes.append(float(nii.stat().st_mtime))
        except OSError:
            continue
        n += 1
    return n, mtimes


def _count_complete_preds_unique(pred_dirs: list[Path]) -> tuple[int, list[float]]:
    """Unique case stems across shard/merged dirs (avoid double-counting)."""
    best: dict[str, float] = {}
    for pred_dir in pred_dirs:
        if not pred_dir.is_dir():
            continue
        niis = list(pred_dir.glob("*.nii.gz"))
        if not niis:
            continue
        require_side = False
        for nii in niis[: min(12, len(niis))]:
            stem = nii.name[: -len(".nii.gz")]
            if any(
                (pred_dir / f"{stem}{suf}").is_file()
                for suf in (".pkl", ".pkl.npz", ".npz")
            ):
                require_side = True
                break
        for nii in niis:
            stem = nii.name[: -len(".nii.gz")]
            if not _pred_case_complete(pred_dir, stem, require_side):
                continue
            try:
                mt = float(nii.stat().st_mtime)
            except OSError:
                continue
            if stem not in best or mt > best[stem]:
                best[stem] = mt
    if not best:
        return 0, []
    return len(best), list(best.values())


def _eta_sec_from_pred_mtimes(done: int, total: int, mtimes: list[float], fallback_sec_per: float = 25.0) -> float | None:
    left = max(0, total - done)
    if left <= 0:
        return 0.0
    if done <= 0:
        return None
    rate: float | None = None
    if len(mtimes) >= 2:
        recent = sorted(mtimes)[-min(20, len(mtimes)) :]
        span = max(recent) - min(recent)
        if span > 1.0:
            rate = span / max(1, len(recent) - 1)
    if rate is None or rate <= 0:
        rate = fallback_sec_per
    rate = min(max(float(rate), 3.0), 600.0)
    return float(rate * left)


def _discover_test20_folds(eval_root: Path) -> tuple[int, ...]:
    """Prefer all fold* dirs present; fall back to classic f2/5/8."""
    found: list[int] = []
    for p in sorted(eval_root.glob("fold*")):
        if not p.is_dir():
            continue
        try:
            f = int(p.name.replace("fold", ""))
        except ValueError:
            continue
        if f in NINE_FOLDS:
            found.append(f)
    if len(found) >= 4:
        return tuple(found)
    classic = (2, 5, 8)
    if any((eval_root / f"fold{f}").is_dir() for f in classic):
        return classic
    return tuple(found)


def _test20_live_proc_for_stamp(stamp: str) -> bool:
    """True if a TEST20 predict/score process for this parent stamp is alive."""
    if not stamp:
        return False
    pats = (
        f"{stamp}/psma_test20_eval",
        f"nnUNetv2_predict.*{stamp}",
        f"run_nnunet_psma_test20",
        "hemingduo_scratch_fs50_test20_resume",
    )
    if _pgrep_any(pats):
        # tighten: stamp must appear in matching lines when pattern is generic
        for line in _pgrep_lines(pats):
            if stamp in line or "hemingduo_scratch_fs50_test20_resume" in line:
                return True
    return any(stamp in n for n in _docker_ps_names())


def _nnunet_test20_progress(
    parent_stamp: str,
    folds: tuple[int, ...] | None = None,
) -> dict[str, Any] | None:
    """If PSMA TEST20 eval dir is active, return per-fold progress + ETA."""
    if not parent_stamp:
        return None
    eval_root = NN_RESULTS / parent_stamp / "psma_test20_eval"
    if not eval_root.is_dir():
        return None
    fold_ids = folds or _discover_test20_folds(eval_root)
    if not fold_ids:
        return None
    if not any((eval_root / f"fold{f}").is_dir() for f in fold_ids):
        return None

    per: dict[str, Any] = {}
    scored = 0
    cases_done = 0
    rem_etas: list[float] = []
    live_etas: list[float] = []
    now = time.time()
    n_folds = len(fold_ids)
    for f in fold_ids:
        fold_root = eval_root / f"fold{f}"
        detail = fold_root / "score_detail.json"
        dice = None
        if detail.is_file():
            try:
                dice = float(json.loads(detail.read_text()).get("mean_dice"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                dice = None
            if dice is not None and dice == dice:
                scored += 1

        pred_dirs: list[Path] = []
        merged = fold_root / "predict" / "pred"
        if merged.is_dir():
            pred_dirs.append(merged)
        shards = fold_root / "predict" / "shards"
        if shards.is_dir():
            pred_dirs.extend(sorted(shards.glob("shard_*/pred")))
        for alt in (fold_root / "predict", fold_root / "predict_lymp"):
            if alt.is_dir() and alt not in pred_dirs:
                pred_dirs.append(alt)
        done, mtimes = _count_complete_preds_unique(pred_dirs)
        done = min(int(done), 120)

        total = 120
        cases_done += done if dice is None else total
        eta_sec = None
        recent = bool(mtimes) and (now - max(mtimes)) < 900
        if dice is None:
            eta_sec = _eta_sec_from_pred_mtimes(done, total, mtimes)
            if eta_sec is None:
                touch = fold_root / "predict"
                try:
                    age = now - touch.stat().st_mtime if touch.exists() else 1e9
                except OSError:
                    age = 1e9
                if age < 3600:
                    eta_sec = None
            else:
                rem_etas.append(float(eta_sec))
                if recent or done < total:
                    live_etas.append(float(eta_sec))
        per[str(f)] = {
            "done": done if dice is None else total,
            "total": total,
            "dice": dice,
            "eta_sec": eta_sec,
            "eta": _fmt_eta(eta_sec) if eta_sec is not None and eta_sec > 1 else (
                "0s" if eta_sec == 0 else None
            ),
            "scored": dice is not None,
            "recent": recent,
        }

    live_proc = _test20_live_proc_for_stamp(parent_stamp)
    if scored >= n_folds and not live_proc:
        return {
            "phase": "TEST20",
            "active": False,
            "scored": scored,
            "n_folds": n_folds,
            "cases_done": cases_done,
            "cases_total": 120 * n_folds,
            "per_fold": per,
            "eta_sec": 0.0,
            "eta": "0s",
        }

    rates: list[float] = []
    for info in per.values():
        if info.get("dice") is not None:
            continue
        d = int(info.get("done") or 0)
        es = info.get("eta_sec")
        if d > 0 and es is not None and (120 - d) > 0:
            rates.append(float(es) / float(120 - d))
    avg_rate = (sum(rates) / len(rates)) if rates else 25.0
    unfinished_idle = 0
    for info in per.values():
        if info.get("dice") is not None:
            continue
        if info.get("eta_sec") is None and int(info.get("done") or 0) <= 0:
            info["eta_sec"] = float(avg_rate * 120)
            info["eta"] = _fmt_eta(info["eta_sec"])
            rem_etas.append(float(info["eta_sec"]))
            unfinished_idle += 1
        elif (not info.get("recent")) and int(info.get("done") or 0) < 120:
            unfinished_idle += 1
    active_eta = max(live_etas) if live_etas else (max(rem_etas) if rem_etas else None)
    if active_eta is not None and unfinished_idle > 0 and live_etas:
        idle_left = [
            float(info["eta_sec"])
            for info in per.values()
            if info.get("dice") is None
            and not info.get("recent")
            and info.get("eta_sec") is not None
            and int(info.get("done") or 0) < 120
        ]
        if idle_left:
            active_eta = float(active_eta) + max(idle_left)
    eta_sec = active_eta if active_eta is not None else (max(rem_etas) if rem_etas else None)
    active = scored < n_folds or live_proc or bool(live_etas)

    return {
        "phase": "TEST20",
        "active": active,
        "scored": scored,
        "n_folds": n_folds,
        "cases_done": cases_done,
        "cases_total": 120 * n_folds,
        "per_fold": per,
        "eta_sec": eta_sec,
        "eta": _fmt_eta(eta_sec) if eta_sec is not None and eta_sec > 1 else None,
    }


def _apply_test20_live(st: dict, test_info: dict[str, Any] | None, total_epochs: int) -> bool:
    """Attach TEST20 live ETA into stage. Returns True if TEST is active (incomplete)."""
    if not test_info:
        st.pop("test_live", None)
        return False
    active = bool(test_info.get("active"))
    n_folds = int(test_info.get("n_folds") or 3)
    fold_keys = tuple(str(f) for f in (test_info.get("per_fold") or {}).keys()) or (
        NINE_FOLD_STRS if n_folds >= 9 else ("2", "5", "8")
    )
    st["phase"] = "TEST20"
    st["epoch"] = total_epochs
    st["total_epochs"] = total_epochs
    st["eval_done"] = test_info.get("scored")
    st["eval_total"] = n_folds
    st["test_live"] = {
        "active": active,
        "eta": test_info.get("eta"),
        "eta_sec": test_info.get("eta_sec"),
        "cases_done": test_info.get("cases_done"),
        "cases_total": test_info.get("cases_total"),
        "folds": test_info.get("per_fold") or {},
        "n_folds": n_folds,
    }
    done = test_info.get("cases_done")
    tot = test_info.get("cases_total")
    if active:
        eta_s = test_info.get("eta") or "…"
        st["eta"] = test_info.get("eta")
        st["eta_sec"] = test_info.get("eta_sec")
        st["note"] = (
            f"TEST20 · {test_info.get('scored', 0)}/{n_folds} folds · "
            f"pred {done}/{tot} · ETA {eta_s}"
        )
        st["fold_dice"] = {}
        st["mean"] = None
        st["fold_ckpt_ep"] = {}
        if (st.get("status") or "").lower() != "running":
            if (st.get("status") or "").lower() == "done":
                st["_test_was_done"] = True
            st["status"] = "running"
    else:
        st["eta"] = None
        st["eta_sec"] = None
        st["status"] = "done"
        st["test_invalidated"] = False
        st["phase"] = None
        st.pop("_test_was_done", None)
        per = test_info.get("per_fold") or {}
        pending = set(st.get("fold_test_pending") or [])
        fd: dict[str, float] = dict(st.get("fold_dice") or {})
        vals: list[float] = []
        for f in fold_keys:
            if f in pending:
                fd.pop(f, None)
                continue
            d = (per.get(f) or {}).get("dice")
            if isinstance(d, (int, float)):
                fd[str(f)] = float(d)
        for f in fold_keys:
            if f in fd:
                vals.append(float(fd[f]))
        st["fold_dice"] = fd
        st["mean"] = (
            float(sum(vals) / n_folds) if len(vals) == n_folds and not pending else None
        )
        if pending:
            st["fold_test_pending"] = sorted(pending, key=int)
            st["note"] = (
                f"TEST20 {len(vals)}/{n_folds} · pending f{','.join(sorted(pending, key=int))}"
            )
        else:
            st.pop("fold_test_pending", None)
            st["note"] = f"TEST20 DONE · {test_info.get('scored', 0)}/{n_folds}"
    return active


def _refresh_aligned_fewshot_stage(st: dict, key: str, sub: str, fewshot_n: int) -> None:
    """Live ETA/epoch for one MAE/MONAI/SegAny fewshot cell (incl. extra-fold)."""
    stamp = st.get("stamp") or ""
    if (st.get("status") or "").lower() != "running" or not stamp:
        return
    total = _stage_total_epochs(st)
    live_f = _live_fewshot_fold(key, fewshot_n, stamp)
    extra = live_f if live_f is not None else _extra_fold_from_note(str(st.get("note") or ""))
    scan_folds: tuple[int, ...] = (extra,) if extra is not None else NINE_FOLDS
    train_live = live_f is not None

    eval_root = REPO / stamp / "psma_test20_eval"
    test_done = 0
    test_folds: dict[str, float] = {}
    if eval_root.is_dir() or (st.get("phase") or "").upper() == "TEST20":
        for f in NINE_FOLDS:
            p = eval_root / f"fold{f}_test20.json"
            if p.is_file():
                try:
                    md = float(json.loads(p.read_text()).get("mean_dice"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if md == md:
                    test_folds[str(f)] = md
                    test_done += 1
        seen_train = 0
        for f in scan_folds:
            rows = _read_metrics_rows(REPO / stamp / sub / f"fold{f}" / "metrics.jsonl")
            if rows and int(rows[-1].get("epoch") or 0) >= total:
                seen_train += 1
        need = 1 if extra is not None else min(3, len(scan_folds))
        train_done = seen_train >= need
        eval_live = _pgrep_any(
            (
                f"run_eval_psma_test20",
                f"{stamp}/psma_test20_eval",
                f"mae_eval_seg_psma_test",
            )
        )
        # extra-fold only trains; do not freeze the cell on dummy TEST20 18m
        # while the next fold is already training (or eval is not even running).
        enter_test = (not train_live) and (
            eval_live
            or (
                extra is None
                and train_done
                and (test_done > 0 or (st.get("phase") or "").upper() == "TEST20")
            )
        )
        if enter_test:
            st["phase"] = "TEST20"
            st["epoch"] = total
            st["total_epochs"] = total
            st["eta"] = None
            st["eta_sec"] = None
            st["eval_done"] = test_done
            st["eval_total"] = 9
            if test_folds:
                st["fold_dice"] = _merge_fold_scores(st.get("fold_dice"), test_folds)
                st["mean"] = _fold_score_mean(st["fold_dice"])
            st["note"] = f"Train DONE · TEST20 {test_done}/9"
            return

    info = eta_parallel_folds(stamp, sub, folds=scan_folds, total_epochs=total)
    eta_sec = info.get("eta_sec")
    if eta_sec is not None and float(eta_sec) <= 1:
        st["eta"] = None
        st["eta_sec"] = None
    else:
        st["eta"] = info.get("eta")
        st["eta_sec"] = eta_sec
    if info.get("epoch") is not None:
        st["epoch"] = info.get("epoch")
    st["total_epochs"] = total
    if extra is not None:
        gpu = st.get("gpu_ids") or "?"
        st["note"] = f"9fold extra · {sub} fs{fewshot_n} f{extra} · GPU {gpu}"
        st.pop("test_live", None)

    phase = None
    for f in scan_folds:
        rows = _read_metrics_rows(REPO / stamp / sub / f"fold{f}" / "metrics.jsonl")
        if not rows:
            continue
        try:
            last_sec = float(rows[-1].get("epoch_sec") or 0)
        except (TypeError, ValueError):
            last_sec = 0
        if last_sec >= 120:
            phase = "VAL"
            break
    if phase is None:
        for f in scan_folds:
            candidates = (
                CTRL / "ICLR2026/vis" / f"nohup_{sub}_psma_fs{fewshot_n}_fdgseg_fold{f}_{stamp}.log",
                CTRL / "ICLR2026/vis" / f"nohup_mae_psma_fs{fewshot_n}_fdgseg_fold{f}_{stamp}.log",
                CTRL / "ICLR2026/vis" / f"nohup_monai_psma_fs{fewshot_n}_fdgseg_fold{f}_{stamp}.log",
            )
            for log in candidates:
                if not log.is_file():
                    continue
                try:
                    tail = log.read_text(errors="ignore")[-4000:]
                except OSError:
                    continue
                if "[sw_dice]" in tail and "Epoch" in tail:
                    phase = "VAL"
                    break
            if phase == "VAL":
                break
    st["phase"] = phase

    live_dice: dict[str, float] = {}
    for f in scan_folds:
        best = None
        for r in _read_metrics_rows(REPO / stamp / sub / f"fold{f}" / "metrics.jsonl"):
            vd = r.get("val_dice")
            try:
                v = float(vd)
            except (TypeError, ValueError):
                continue
            if v == v:
                best = v if best is None or v > best else best
        if best is not None:
            live_dice[str(f)] = best
    if live_dice:
        st["val_monitor_fold_dice"] = _merge_fold_scores(st.get("val_monitor_fold_dice"), live_dice)


def _refresh_nnunet_family_psma_stage(st: dict, *, is_dpdnet: bool = False) -> None:
    """Live train/TEST20 ETA for nnUNet-style PSMA fewshot cells (incl. competition)."""
    if not isinstance(st, dict):
        return
    stamp = (st.get("stamp") or "").strip()
    if not stamp:
        return
    total = _stage_total_epochs(st) or (100 if is_dpdnet else 300)
    test_info = _nnunet_test20_progress(stamp)
    status = (st.get("status") or "").lower()
    phase = (st.get("phase") or "").upper()
    live_proc = _test20_live_proc_for_stamp(stamp)

    if is_dpdnet and status == "paused":
        _apply_paused_dpdnet_psma(st, stamp, total)
        return

    if test_info and (
        test_info.get("active")
        or live_proc
        or status == "running"
        or phase == "TEST20"
        or (
            status == "done"
            and int(test_info.get("scored") or 0) < int(test_info.get("n_folds") or 3)
        )
    ):
        _apply_test20_live(st, test_info, total)
        if not test_info.get("active") and not live_proc:
            st["status"] = "done"
            st["phase"] = None
            if not str(st.get("note") or "").startswith("TEST20 DONE"):
                mean = st.get("mean")
                n_folds = int(test_info.get("n_folds") or 3)
                mean_s = f"{mean:.3f}" if isinstance(mean, (int, float)) else "—"
                st["note"] = f"TEST20 DONE · {test_info.get('scored', 0)}/{n_folds} · mean={mean_s}"
            if not is_dpdnet:
                _ingest_nnunet_test20_aggregate(st)
        return

    if status != "running" and not live_proc:
        st.pop("test_live", None)
        return

    # Training or about to start TEST20
    ok = 0
    need = 3
    for f in NINE_FOLDS:
        if is_dpdnet:
            fd = _dpdnet_psma_fold_dir(stamp, f)
        else:
            fd = _nnunet_fold_dir(f"{stamp}_f{f}")
        if fd is not None and (
            (fd / "checkpoint_final.pth").is_file() or (fd / "checkpoint_best.pth").is_file()
        ):
            ok += 1
    # 9-fold competition: need all present fold stamps; classic three-fold: 3
    fold_dirs = 0
    for f in NINE_FOLDS:
        if is_dpdnet:
            fd = _dpdnet_psma_fold_dir(stamp, f)
        else:
            fd = _nnunet_fold_dir(f"{stamp}_f{f}")
        if fd is not None and fd.is_dir():
            fold_dirs += 1
    if fold_dirs >= 4:
        need = fold_dirs
    train_done = ok >= need or phase == "TEST20"
    if train_done or live_proc:
        st["status"] = "running"
        st["phase"] = "TEST20"
        st["epoch"] = total
        st["total_epochs"] = total
        st["eta"] = None
        st["eta_sec"] = None
        st["note"] = f"Train DONE · TEST20 {ok}/{need} · starting…"
        st["test_live"] = {"active": True, "eta": None, "folds": {}, "n_folds": need}
        return

    if status != "running":
        return
    if is_dpdnet:
        # bottleneck across live folds via nnUNet-style logs
        rem: list[float] = []
        epochs: list[int] = []
        for f in NINE_FOLDS:
            fd = _dpdnet_psma_fold_dir(stamp, f)
            info = _eta_from_nnunet_fold(fd, total)
            if info.get("epoch") is not None:
                epochs.append(int(info["epoch"]))
            es = info.get("eta_sec")
            if es is not None and float(es) > 1:
                rem.append(float(es))
        if rem:
            st["eta_sec"] = max(rem)
            st["eta"] = _fmt_eta(st["eta_sec"])
        if epochs:
            st["epoch"] = min(epochs)
        st["total_epochs"] = total
    else:
        info = eta_nnunet_parallel_folds(stamp, total_epochs=total)
        eta_sec = info.get("eta_sec")
        if eta_sec is not None and float(eta_sec) > 1:
            st["eta"] = info.get("eta")
            st["eta_sec"] = eta_sec
        else:
            st["eta"] = None
            st["eta_sec"] = None
        if info.get("epoch") is not None:
            st["epoch"] = info["epoch"]
        st["total_epochs"] = total
        if info.get("phase"):
            st["phase"] = info["phase"]
        else:
            st["phase"] = None
        st.pop("test_live", None)


def refresh_running_etas(board: dict) -> None:
    """Attach eta / epoch progress onto any stage with status=running."""
    _reset_runtime_caches()
    methods = board.get("methods") or {}

    # nnUNet / competition rows (nnUNet-style TEST20 under parent stamp)
    for mkey in (
        "nnunet",
        "nnunet_mim",
        "hemingduo_scratch",
        "chenyixin_scratch",
        "hemingduo",
        "chenyixin",
    ):
        m = methods.get(mkey) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            st = m.get(stage)
            if isinstance(st, dict) and (st.get("stamp") or "").strip():
                _refresh_nnunet_family_psma_stage(st, is_dpdnet=False)

    # DpDNet PSMA fewshot — same nnUNet-style TEST20 eval dir under parent stamp
    for mkey in ("dpdnet", "dpdnet_dualenc"):
        m = methods.get(mkey) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            st = m.get(stage)
            if isinstance(st, dict) and (st.get("stamp") or "").strip():
                _refresh_nnunet_family_psma_stage(st, is_dpdnet=True)

    # MAE / MONAI / SegAnyPET fewshot (fs50/fs10/fs5; extra-fold uses one fold)
    for key, sub in (
        ("mae_swinunetr", "mae"),
        ("mae_scratch", "mae"),
        ("monai_swinvit", "monai"),
        ("monai_scratch", "monai"),
        ("seganypet", "seganypet"),
        ("seganypet_scratch", "seganypet"),
    ):
        if key == "monai_swinvit":
            fdg0 = methods.get("monai_swinvit", {}).get("fdg_pretrain") or {}
            if _monai_fdg_still_active(fdg0):
                continue
        for _n, stage_key, _short in FEWSHOT_VARIANTS:
            st = methods.get(key, {}).get(stage_key) or {}
            _refresh_aligned_fewshot_stage(st, key, sub, int(_n))

    # MAE FDG (SSL) / MAE scratch FDG (random init)
    for _mae_key in ("mae_swinunetr", "mae_scratch"):
        mae_fdg = methods.get(_mae_key, {}).get("fdg_pretrain") or {}
        if mae_fdg.get("status") == "running" and mae_fdg.get("stamp"):
            total = _stage_total_epochs(mae_fdg) or 100
            info = eta_from_metrics(REPO / mae_fdg["stamp"] / "metrics.jsonl", total)
            eta_sec = info.get("eta_sec")
            if eta_sec is not None and float(eta_sec) <= 1:
                mae_fdg["eta"] = None
                mae_fdg["eta_sec"] = None
            else:
                mae_fdg["eta"] = info.get("eta")
                mae_fdg["eta_sec"] = eta_sec
            mae_fdg["epoch"] = info.get("epoch")
            mae_fdg["total_epochs"] = total

    # MONAI FDG (pretrained Tang SSL / scratch random init)
    for _monai_key in ("monai_swinvit", "monai_scratch"):
        monai_fdg = methods.get(_monai_key, {}).get("fdg_pretrain") or {}
        if monai_fdg.get("status") == "running" and monai_fdg.get("stamp"):
            total = _stage_total_epochs(monai_fdg)
            info = eta_from_metrics(REPO / monai_fdg["stamp"] / "metrics.jsonl", total)
            eta_sec = info.get("eta_sec")
            if eta_sec is not None and float(eta_sec) <= 1:
                monai_fdg["eta"] = None
                monai_fdg["eta_sec"] = None
            else:
                monai_fdg["eta"] = info.get("eta")
                monai_fdg["eta_sec"] = eta_sec
            monai_fdg["epoch"] = info.get("epoch")
            monai_fdg["total_epochs"] = total

    # SegAnyPET FDG (lesion init / scratch random init)
    for _seg_key in ("seganypet", "seganypet_scratch"):
        seg_fdg = methods.get(_seg_key, {}).get("fdg_pretrain") or {}
        if seg_fdg.get("status") == "running" and seg_fdg.get("stamp"):
            total = _stage_total_epochs(seg_fdg)
            info = eta_from_metrics(
                REPO / seg_fdg["stamp"] / "seganypet_fdg" / "metrics.jsonl", total
            )
            eta_sec = info.get("eta_sec")
            if eta_sec is not None and float(eta_sec) <= 1:
                seg_fdg["eta"] = None
                seg_fdg["eta_sec"] = None
            else:
                seg_fdg["eta"] = info.get("eta")
                seg_fdg["eta_sec"] = eta_sec
            seg_fdg["epoch"] = info.get("epoch")
            seg_fdg["total_epochs"] = total

    # nnUNet FDG (Dataset228 fold_0; stamp often missing until FDG done — auto-resolve)
    for _nn_key in ("nnunet", "nnunet_mim"):
        nn_fdg = methods.get(_nn_key, {}).get("fdg_pretrain") or {}
        if (nn_fdg.get("status") or "").lower() in ("running", "done", "queued"):
            stamp = _resolve_nnunet_fdg_stamp(nn_fdg) if _nn_key == "nnunet" else (nn_fdg.get("stamp") or "").strip()
            if stamp and not (nn_fdg.get("stamp") or "").strip():
                nn_fdg["stamp"] = stamp
            if stamp:
                total = _stage_total_epochs(nn_fdg) or 169
                info = eta_nnunet_fdg(stamp, total_epochs=total)
                if info.get("done"):
                    nn_fdg["status"] = "done"
                    nn_fdg["eta"] = None
                    nn_fdg["eta_sec"] = None
                    nn_fdg["epoch"] = total
                    nn_fdg["total_epochs"] = total
                    nn_fdg["phase"] = None
                elif (nn_fdg.get("status") or "").lower() == "running":
                    eta_sec = info.get("eta_sec")
                    if eta_sec is not None and float(eta_sec) > 1:
                        nn_fdg["eta"] = info.get("eta")
                        nn_fdg["eta_sec"] = eta_sec
                    else:
                        nn_fdg["eta"] = None
                        nn_fdg["eta_sec"] = None
                    if info.get("epoch") is not None:
                        cur = info["epoch"]
                        try:
                            nn_fdg["epoch"] = min(total, int(cur) + 1)
                        except (TypeError, ValueError):
                            nn_fdg["epoch"] = cur
                    nn_fdg["total_epochs"] = total

    # DpDNet FDG (nnUNet training_log under nnUNet_results/<stamp>/...)
    for _dpd_key in ("dpdnet", "dpdnet_dualenc"):
        dpd_fdg = methods.get(_dpd_key, {}).get("fdg_pretrain") or {}
        if dpd_fdg.get("stamp") and (dpd_fdg.get("status") or "").lower() in (
            "running",
            "done",
            "queued",
        ):
            total = _stage_total_epochs(dpd_fdg) or 169
            info = eta_dpdnet_fdg(str(dpd_fdg["stamp"]), total_epochs=total)
            if info.get("done"):
                dpd_fdg["status"] = "done"
                dpd_fdg["eta"] = None
                dpd_fdg["eta_sec"] = None
                dpd_fdg["epoch"] = total
                dpd_fdg["total_epochs"] = total
                dpd_fdg["phase"] = None
                if not (dpd_fdg.get("note") or "").startswith("DONE"):
                    dpd_fdg["note"] = "FDG done · init PSMA with checkpoint_final"
            elif (dpd_fdg.get("status") or "").lower() == "running":
                eta_sec = info.get("eta_sec")
                if eta_sec is not None and float(eta_sec) > 1:
                    dpd_fdg["eta"] = info.get("eta")
                    dpd_fdg["eta_sec"] = eta_sec
                else:
                    dpd_fdg["eta"] = None
                    dpd_fdg["eta_sec"] = None
                if info.get("epoch") is not None:
                    dpd_fdg["epoch"] = info["epoch"]
                dpd_fdg["total_epochs"] = total

    # PSMA fc70% / FDG TEST / PSMA fs0 — single-run stages (fold0 or eval-only)
    _SINGLE_RUN_ETA: dict[str, tuple[str, str]] = {
        "mae_swinunetr": ("mae", "fold0"),
        "mae_scratch": ("mae", "fold0"),
        "monai_swinvit": ("monai", "fold0"),
        "monai_scratch": ("monai", "fold0"),
        "seganypet": ("seganypet", "fold0"),
        "seganypet_scratch": ("seganypet", "fold0"),
        "dpdnet_dualenc": ("dpdnet", "fold0"),
        "dpdnet": ("dpdnet", "fold0"),
        "nnunet_mim": ("nnunet", "fold0"),
        "nnunet": ("nnunet", "fold0"),
    }
    for stage_key in (FC70_STAGE_KEY, FDG_TEST_STAGE_KEY, PSMA_FS0_STAGE_KEY):
        for mkey, (sub, fold) in _SINGLE_RUN_ETA.items():
            st = methods.get(mkey, {}).get(stage_key) or {}
            if (st.get("status") or "").lower() != "running":
                continue
            stamp = _resolve_stage_stamp(mkey, stage_key, st)
            if stamp:
                st["stamp"] = stamp
            if stage_key in (FDG_TEST_STAGE_KEY, PSMA_FS0_STAGE_KEY):
                _refresh_fdg_test_stage_eta(st, mkey)
                continue
            if not stamp:
                continue
            total = _stage_total_epochs(st)
            info = _single_run_train_eta(mkey, sub, fold, stamp, total)
            _apply_eta_to_stage(st, info)
            if info.get("total_epochs"):
                st["total_epochs"] = info["total_epochs"]
            else:
                st["total_epochs"] = total
            if info.get("phase") == "decline":
                nxt = info.get("total_epochs") or total
                st["note"] = f"decline · next val @{nxt}"

    # MONAI: FDG / PSMA 不同时显示进度（协议顺序：FDG 未完则只刷 FDG）
    _enforce_monai_exclusive_progress(methods)
    _refresh_all_fdg_test20_live(methods)
    _refresh_gpu_idle_waiting(board)
    _finalize_all_running_etas(board)
    _annotate_all_running_devices(board)


def _nnunet_fdg_test20_pred_progress() -> dict[str, Any]:
    """Count unique preds under fdg_test20_eval/nnunet/predict (prefer merged)."""
    root = Path(os.environ.get("WORK_DIR", "/media/ybwang/data1/PSMA-DATA/task1_train_workspace"))
    pred_root = root / "fdg_test20_eval" / "nnunet" / "predict"
    total = 202
    names: set[str] = set()
    if pred_root.is_dir():
        merged = pred_root / "pred"
        if merged.is_dir():
            names.update(p.name for p in merged.glob("*.nii.gz"))
        if len(names) < total:
            shards = pred_root / "shards"
            if shards.is_dir():
                for sh in shards.glob("shard_*/pred"):
                    names.update(p.name for p in sh.glob("*.nii.gz"))
    done = len(names)
    return {"cases_done": done, "cases_total": total, "active": done < total}


def _refresh_all_fdg_test20_live(methods: dict) -> None:
    """If any method's FDG TEST eval/rescore is live, force RUNNING (keep over stale Dice/FP/FN)."""
    for mkey in METHOD_ORDER:
        st = (methods.get(mkey) or {}).get(FDG_TEST_STAGE_KEY)
        if not isinstance(st, dict):
            continue
        pats = STAGE_RUNTIME_PATTERNS.get((mkey, FDG_TEST_STAGE_KEY), ())
        if not pats or not _pgrep_any(pats):
            continue
        st["status"] = "running"
        st["phase"] = "TEST20"
        st["device"] = "gpu"
        live_gpus = _live_gpu_ids_for_patterns(pats)
        if live_gpus:
            st["gpu_ids"] = ",".join(str(i) for i in live_gpus)
        _refresh_fdg_test_stage_eta(st, mkey)


def _refresh_nnunet_fdg_test20_live(methods: dict) -> None:
    """Deprecated alias."""
    _refresh_all_fdg_test20_live(methods)


GPU_IDLE_WAIT_JSON = CTRL / "ICLR2026/vis/gpu_idle_wait.json"
GPU_IDLE_STATE_JSON = CTRL / "ICLR2026/vis/gpu_idle_scheduler_state.json"


def _load_gpu_idle_wait() -> dict[str, Any]:
    for path in (GPU_IDLE_WAIT_JSON, GPU_IDLE_STATE_JSON):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if path == GPU_IDLE_STATE_JSON:
            wait = raw.get("wait") if isinstance(raw.get("wait"), dict) else {}
        else:
            wait = raw if isinstance(raw, dict) else {}
        if wait:
            return wait
    return {}


def _clear_stage_wait_fields(st: dict) -> None:
    for k in ("wait_gpu", "wait_sec", "wait_total", "wait_eta", "wait_from"):
        st.pop(k, None)


def _gpu_mem_mib_board(gpu: int) -> int | None:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        for line in (r.stdout or "").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(float(parts[0])) == gpu:
                return int(float(parts[1]))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def _flip_stage_to_running(st: dict, pats: tuple[str, ...]) -> None:
    st["status"] = "running"
    _clear_stage_wait_fields(st)
    gpus: list[int] = []
    for ln in _pgrep_lines(pats):
        found = _gpu_ids_from_cmdline(ln)
        if found:
            gpus.extend(found)
    if gpus:
        st["device"] = "gpu"
        st["gpu_ids"] = ",".join(str(i) for i in sorted(set(gpus)))
    note = str(st.get("note") or "")
    if note.upper().startswith("WAITING"):
        st["note"] = "running · launched"


def _promote_waiting_to_running(board: dict) -> None:
    """WAITING → RUNNING once the task process is live or its GPU is in use."""
    wait = board.get("gpu_idle_wait") if isinstance(board.get("gpu_idle_wait"), dict) else {}
    if not wait.get("method"):
        wait = _load_gpu_idle_wait()
    mem_thresh = int(os.environ.get("TASK1_GPU_IDLE_MEM_MIB", "2048"))
    methods = board.get("methods") or {}
    cleared = False
    for mkey in METHOD_ORDER:
        m = methods.get(mkey) or {}
        for _n, stage_key, _short in PSMA_BOARD_COLUMNS:
            st = m.get(stage_key)
            if not isinstance(st, dict):
                continue
            if (st.get("status") or "").lower() != "waiting":
                continue
            pats = STAGE_RUNTIME_PATTERNS.get((mkey, stage_key), ())
            is_wait_target = wait.get("method") == mkey and wait.get("stage") == stage_key
            if pats and _pgrep_any(pats):
                _flip_stage_to_running(st, pats)
                if is_wait_target:
                    cleared = True
                continue
            if not is_wait_target:
                continue
            try:
                wg = int(st.get("wait_gpu") if st.get("wait_gpu") is not None else wait.get("gpu"))
            except (TypeError, ValueError):
                continue
            mem = _gpu_mem_mib_board(wg)
            try:
                remain_i = int(st.get("wait_sec") if st.get("wait_sec") is not None else wait.get("remain_sec") or 999)
            except (TypeError, ValueError):
                remain_i = 999
            if mem is not None and mem >= mem_thresh and remain_i <= 0:
                st["status"] = "running"
                _clear_stage_wait_fields(st)
                st["device"] = "gpu"
                st["gpu_ids"] = str(wg)
                note = str(st.get("note") or "")
                if note.upper().startswith("WAITING"):
                    st["note"] = f"running · GPU {wg}"
                cleared = True
    if cleared:
        board["gpu_idle_wait"] = {
            "method": None,
            "stage": None,
            "gpu": None,
            "remain_sec": None,
            "idle_total_sec": wait.get("idle_total_sec", 60),
        }


def _clear_gpu_idle_waiting_marks(board: dict, keep: tuple[str, str] | None = None) -> None:
    methods = board.get("methods") or {}
    for mkey in METHOD_ORDER:
        m = methods.get(mkey) or {}
        if not isinstance(m, dict):
            continue
        for _n, stage_key, _short in PSMA_BOARD_COLUMNS:
            st = m.get(stage_key)
            if not isinstance(st, dict):
                continue
            if (st.get("status") or "").lower() != "waiting":
                continue
            if st.get("wait_from") != "gpu_idle":
                continue
            if keep and (mkey, stage_key) == keep:
                continue
            st["status"] = "pending"
            _clear_stage_wait_fields(st)
            note = str(st.get("note") or "")
            if note.upper().startswith("WAITING"):
                st["note"] = "queued · next after GPU idle 1min"


def _refresh_gpu_idle_waiting(board: dict) -> None:
    """Apply GPU-idle 1min countdown onto the next queued task cell."""
    _promote_waiting_to_running(board)
    wait = board.get("gpu_idle_wait") if isinstance(board.get("gpu_idle_wait"), dict) else {}
    if not wait.get("method"):
        wait = _load_gpu_idle_wait()
    # Live recompute remain from idle_since so PNG countdown advances between scheduler ticks
    try:
        st_raw = json.loads(GPU_IDLE_STATE_JSON.read_text(encoding="utf-8")) if GPU_IDLE_STATE_JSON.is_file() else {}
    except (OSError, json.JSONDecodeError):
        st_raw = {}
    idle_since = st_raw.get("idle_since") if isinstance(st_raw.get("idle_since"), dict) else {}
    idle_total = int(wait.get("idle_total_sec") or st_raw.get("wait", {}).get("idle_total_sec") or 60)
    gpu = wait.get("gpu")
    if gpu is not None and str(gpu) in idle_since:
        try:
            elapsed = max(0.0, time.time() - float(idle_since[str(gpu)]))
            wait["remain_sec"] = int(max(0.0, float(idle_total) - elapsed))
            wait["elapsed_sec"] = int(elapsed)
            wait["idle_total_sec"] = idle_total
        except (TypeError, ValueError):
            pass
    if wait:
        board["gpu_idle_wait"] = wait
    method = wait.get("method")
    stage = wait.get("stage")
    gpu = wait.get("gpu")
    remain = wait.get("remain_sec")
    if not method or not stage or gpu is None or remain is None:
        _clear_gpu_idle_waiting_marks(board, keep=None)
        return
    try:
        remain_i = int(remain)
        gpu_i = int(gpu)
    except (TypeError, ValueError):
        _clear_gpu_idle_waiting_marks(board, keep=None)
        return
    st = (board.get("methods") or {}).get(method, {}).get(stage)
    if not isinstance(st, dict):
        return
    cur = (st.get("status") or "").lower()
    pats = STAGE_RUNTIME_PATTERNS.get((method, stage), ())
    if cur == "running" or (pats and _pgrep_any(pats)):
        if cur == "waiting" and pats:
            _flip_stage_to_running(st, pats)
        _clear_gpu_idle_waiting_marks(board, keep=None)
        return
    if cur in ("paused", "n/a", "na"):
        _clear_gpu_idle_waiting_marks(board, keep=None)
        return
    # allow WAITING on done cells when re-queued (e.g. FDG TEST missing FP/FN)
    _clear_gpu_idle_waiting_marks(board, keep=(method, stage))
    remain_s = _fmt_eta(float(remain_i)) if remain_i > 0 else "0s"
    st["status"] = "waiting"
    st["device"] = "gpu"
    st["gpu_ids"] = str(gpu_i)
    st["wait_gpu"] = gpu_i
    st["wait_sec"] = remain_i
    st["wait_total"] = idle_total
    st["wait_eta"] = remain_s
    st["wait_from"] = "gpu_idle"
    st["note"] = f"WAITING (GPU {gpu_i} · {remain_s})"


def _waiting_status_head(st: dict) -> str:
    gpu = st.get("wait_gpu")
    if gpu is None:
        gids = str(st.get("gpu_ids") or "").strip()
        gpu = gids.split(",")[0] if gids else "?"
    eta = st.get("wait_eta")
    if not eta and isinstance(st.get("wait_sec"), (int, float)):
        eta = _fmt_eta(float(st["wait_sec"]))
    if eta:
        return f"WAITING (GPU {gpu} · {eta})"
    return f"WAITING (GPU {gpu})"


def _parse_gpu_ids_from_text(text: str) -> list[int] | None:
    """Parse stamp/note GPU ids: gpu0 → [0], gpu013 → [0,1,3], gpu0,1,3 / gpu0_1_3 → same."""
    if not text:
        return None
    m = re.search(r"(?:^|[_/-])gpu([0-9]+(?:[,_][0-9]+)*)(?:[_/-]|$)", str(text), re.I)
    if not m:
        m = re.search(r"gpu([0-9]+(?:[,_][0-9]+)*)", str(text), re.I)
    if not m:
        return None
    raw = m.group(1)
    if "," in raw or "_" in raw:
        return [int(p) for p in re.split(r"[,_]", raw) if p.isdigit()]
    # repo convention: each digit is a physical GPU id (gpu013 → 0,1,3)
    return [int(c) for c in raw if c.isdigit()]


def _coerce_gpu_ids(val: Any) -> list[int] | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        out = []
        for x in val:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out or None
    s = str(val).strip()
    if not s:
        return None
    if re.search(r"gpu", s, re.I):
        return _parse_gpu_ids_from_text(s)
    parts = [p for p in re.split(r"[,_\s]+", s) if p.strip().isdigit()]
    return [int(p) for p in parts] if parts else None


def _annotate_stage_device(st: dict, board: dict | None = None) -> str:
    """Set st['device'] / st['gpu_ids']; return display tag 'CPU' or 'GPU 0,1'."""
    if not isinstance(st, dict):
        return "GPU"
    dev = str(st.get("device") or "").lower().strip()
    if dev in ("cpu", "cpu-only"):
        st["device"] = "cpu"
        st.pop("gpu_ids", None)
        return "CPU"
    ids = _coerce_gpu_ids(st.get("gpu_ids")) or _coerce_gpu_ids(st.get("gpus"))
    if not ids:
        ids = _parse_gpu_ids_from_text(str(st.get("stamp") or "")) or _parse_gpu_ids_from_text(
            str(st.get("note") or "")
        )
    if ids:
        st["device"] = "gpu"
        st["gpu_ids"] = ",".join(str(i) for i in ids)
        return f"GPU {st['gpu_ids']}"
    st["device"] = "gpu"
    return "GPU"


def _running_status_head(st: dict, board: dict | None = None) -> str:
    """RUNNING (CPU) or RUNNING (GPU 0,1)."""
    return f"RUNNING ({_annotate_stage_device(st, board)})"


def _annotate_all_running_devices(board: dict) -> None:
    methods = board.get("methods") or {}
    for mkey in METHOD_ORDER:
        m = methods.get(mkey) or {}
        if not isinstance(m, dict):
            continue
        for _n, stage_key, _short in PSMA_BOARD_COLUMNS:
            st = m.get(stage_key)
            if isinstance(st, dict) and (st.get("status") or "").lower() in (
                "running",
                "paused",
                "waiting",
            ):
                _annotate_stage_device(st, board)
        fdg = m.get("fdg_pretrain")
        if isinstance(fdg, dict) and (fdg.get("status") or "").lower() in (
            "running",
            "paused",
            "waiting",
        ):
            _annotate_stage_device(fdg, board)


def _cpu_side_jobs_running() -> list[str]:
    """Non-cell CPU work that should appear as RUNNING (CPU)."""
    jobs: list[str] = []
    if _pgrep_any(("rescore_board_dice_fp_fn.py",)):
        if _pgrep_any(("--proto-fewshot-only", "proto_fdg100_psma")):
            jobs.append("Proto fs50/10/5/0 FP/FN")
        else:
            jobs.append("rescore Dice/FP/FN")
    if _pgrep_any(("run_rescore_proto_fewshot_fp_fn",)):
        if "Proto fs50/10/5/0 FP/FN" not in jobs:
            jobs.append("Proto fs50/10/5/0 FP/FN")
    return jobs


def _clear_stage_live_progress(st: dict) -> None:
    st["eta"] = None
    st["eta_sec"] = None
    st["epoch"] = None
    st["phase"] = None
    st.pop("test_live", None)


def _monai_fdg_still_active(fdg: dict) -> bool:
    """True if FDG supervised is the stage that should own the progress column."""
    stamp = (fdg.get("stamp") or "").strip()
    status = (fdg.get("status") or "").lower()
    total = _stage_total_epochs(fdg) or 100
    if not stamp:
        return status == "running"
    rows = _read_metrics_rows(REPO / stamp / "metrics.jsonl")
    cur = int(rows[-1].get("epoch") or 0) if rows else None
    # docker still named after this stamp?
    docker_alive = False
    try:
        names = subprocess.check_output(
            ["docker", "ps", "--format", "{{.Names}}"], text=True
        )
        docker_alive = any(
            stamp in n or n.startswith(f"monai_fdg_{stamp}") for n in names.splitlines()
        )
        # common name: monai_fdg_<stamp>
        if not docker_alive:
            docker_alive = f"monai_fdg_{stamp}" in names or any(
                "monai_fdg" in n and stamp[:15] in n for n in names.splitlines()
            )
    except (OSError, subprocess.CalledProcessError):
        docker_alive = False
    incomplete = cur is not None and cur + 1 < total  # epoch is 0-based or 1-based?
    # metrics use 1-based completed epoch in this repo (epoch 20 of 100)
    if cur is not None:
        incomplete = cur < total
    if status == "running" and (incomplete or docker_alive or cur is None):
        return True
    if docker_alive and incomplete:
        return True
    return False


def _enforce_monai_exclusive_progress(methods: dict) -> None:
    """Never show live FDG + PSMA progress together for MONAI SwinViT."""
    monai = methods.get("monai_swinvit") or {}
    fdg = monai.get("fdg_pretrain") or {}
    psma = monai.get("psma_fs50_f258") or {}
    if not isinstance(fdg, dict) or not isinstance(psma, dict):
        return

    fdg_active = _monai_fdg_still_active(fdg)
    psma_status = (psma.get("status") or "").lower()
    psma_wants_live = psma_status in ("running", "queued") or bool(psma.get("eta")) or (
        psma.get("epoch") is not None and psma_status != "done"
    )

    if fdg_active:
        # only FDG shows progress; demote PSMA live fields
        if psma_wants_live or (psma.get("status") or "").lower() == "running":
            if (psma.get("status") or "").lower() in ("running", "done"):
                # keep stamp if any, but don't look like concurrent train
                psma["status"] = "queued" if psma.get("stamp") else "pending"
            _clear_stage_live_progress(psma)
            note = str(psma.get("note") or "")
            if "queued" not in note.lower() and not note.strip():
                psma["note"] = "queued"
        # ensure FDG stays the running column
        if (fdg.get("status") or "").lower() != "running":
            fdg["status"] = "running"
        return

    # FDG finished → if still marked running, flip to done when PSMA is the focus
    if (fdg.get("status") or "").lower() == "running":
        stamp = (fdg.get("stamp") or "").strip()
        total = _stage_total_epochs(fdg) or 100
        rows = _read_metrics_rows(REPO / stamp / "metrics.jsonl") if stamp else []
        cur = int(rows[-1].get("epoch") or 0) if rows else 0
        if cur >= total or (psma.get("status") or "").lower() == "running":
            fdg["status"] = "done"
            _clear_stage_live_progress(fdg)
            if not fdg.get("note") or "exited" in str(fdg.get("note")):
                fdg["note"] = fdg.get("note") or "FDG supervised done"


def _bs_label(st: dict) -> str:
    bs = st.get("bs")
    if bs is None:
        return ""
    note = st.get("bs_note") or ""
    if note:
        return f"bs={bs} ({note})"
    return f"bs={bs}"


def _migrate_legacy_stages(m: dict, method_key: str) -> None:
    """Move legacy fdg20_test column data to psma_fs0 / fdg_test20."""
    old = m.pop("fdg20_test", None)
    if not isinstance(old, dict):
        return
    if method_key == "proto_retrieval":
        if FDG_TEST_STAGE_KEY not in m:
            m[FDG_TEST_STAGE_KEY] = old
    elif PSMA_FS0_STAGE_KEY not in m:
        m[PSMA_FS0_STAGE_KEY] = old


def ensure_methods(board: dict) -> None:
    """Ensure all protocol methods exist; keep venue labels in sync."""
    defaults = default_board()["methods"]
    methods = board.setdefault("methods", {})
    for key, tmpl in defaults.items():
        if key not in methods or not isinstance(methods.get(key), dict):
            methods[key] = deepcopy(tmpl)
            continue
        m = methods[key]
        _migrate_legacy_stages(m, key)
        m["label"] = METHOD_LABELS.get(key) or tmpl.get("label", key)
        if tmpl.get("board_policy"):
            m["board_policy"] = tmpl["board_policy"]
        if tmpl.get("repo"):
            m["repo"] = tmpl["repo"]
        if "fdg_pretrain" not in m or not isinstance(m.get("fdg_pretrain"), dict):
            m["fdg_pretrain"] = deepcopy(tmpl.get("fdg_pretrain", {}))
        else:
            # Keep pending competition rows aligned with board policy (no final-submit notes).
            fdg_dst = m["fdg_pretrain"]
            fdg_src = tmpl.get("fdg_pretrain") or {}
            if key in ("hemingduo", "hemingduo_scratch", "chenyixin", "chenyixin_scratch"):
                if str(fdg_dst.get("status", "pending")) in ("pending", "queued", ""):
                    for k in ("note", "total_epochs", "train_iters", "val_iters", "bs", "bs_note"):
                        if k in fdg_src:
                            fdg_dst[k] = fdg_src[k]
        for n, stage, _short in PSMA_BOARD_COLUMNS:
            if stage not in m or not isinstance(m.get(stage), dict):
                if stage == FC70_STAGE_KEY:
                    m[stage] = deepcopy(tmpl.get(stage) or _default_fc70_stage(key == "proto_retrieval"))
                elif stage == PSMA_FS0_STAGE_KEY:
                    m[stage] = deepcopy(tmpl.get(stage) or _default_psma_fs0_stage(key))
                elif stage == FDG_TEST_STAGE_KEY:
                    m[stage] = deepcopy(tmpl.get(stage) or _default_fdg_test_stage(key))
                else:
                    m[stage] = deepcopy(tmpl.get(stage) or _default_psma_stage(n, key == "proto_retrieval"))
                continue
            src = tmpl.get(stage) or (
                _default_fc70_stage(key == "proto_retrieval")
                if stage == FC70_STAGE_KEY
                else _default_psma_fs0_stage(key)
                if stage == PSMA_FS0_STAGE_KEY
                else _default_fdg_test_stage(key)
                if stage == FDG_TEST_STAGE_KEY
                else _default_psma_stage(n, key == "proto_retrieval")
            )
            dst = m[stage]
            for k in ("training_free", "support_pool", "online_val", "train_iters", "val_iters"):
                if k in src and k not in dst:
                    dst[k] = src[k]
    q = board.setdefault("queue", [])
    if isinstance(q, list):
        for item in (
            "nnunet.psma_fs10_f258",
            "nnunet.psma_fs5_f258",
            "mae_swinunetr.psma_fs10_f258",
            "mae_swinunetr.psma_fs5_f258",
            "mae_scratch.fdg_pretrain",
            "mae_scratch.psma_fs50_f258",
            "monai_scratch.fdg_pretrain",
            "monai_scratch.psma_fs50_f258",
            "seganypet_scratch.fdg_pretrain",
            "seganypet_scratch.psma_fs50_f258",
            "hemingduo_scratch.fdg_pretrain",
            "hemingduo_scratch.psma_fs50_f258",
            "hemingduo.fdg_pretrain",
            "hemingduo.psma_fs50_f258",
            "chenyixin_scratch.fdg_pretrain",
            "chenyixin_scratch.psma_fs50_f258",
            "chenyixin.fdg_pretrain",
            "chenyixin.psma_fs50_f258",
        ):
            if not any(item in str(x) for x in q):
                q.append(item)
    board["fewshot_variants"] = board.get("fewshot_variants") or [v[2] for v in FEWSHOT_VARIANTS]
    fv = board.setdefault("fewshot_variants", [v[2] for v in FEWSHOT_VARIANTS])
    if FC70_SHORT not in fv:
        fv.append(FC70_SHORT)
    if PSMA_FS0_SHORT not in fv:
        fv.append(PSMA_FS0_SHORT)
    if FDG_TEST_SHORT not in fv:
        fv.append(FDG_TEST_SHORT)
    _ensure_default_queue(board)


def _ensure_default_queue(board: dict) -> None:
    """Keep board queue reflecting full pipeline including new columns."""
    q = board.setdefault("queue", [])
    if not isinstance(q, list):
        return
    defaults = (
        "fs10/fs5 methods pipeline",
        "nnunet fs10/fs5 rerun FDG169",
        "fc70% PSMA pipeline (queued)",
        "PSMA fs0 eval (queued)",
        "FDG TEST eval (queued)",
        "fs50/fs10/fs5 extra folds → 9fold (queued)",
        "PET/CT MAE scratch 9fold (queued)",
        "MONAI SwinViT scratch 9fold (queued)",
        "SegAnyPET scratch 9fold (queued)",
        "nnUNet MIM + DpDNet dual-enc after scratch (queued)",
    )
    for item in defaults:
        if not any(item.split()[0] in str(x) for x in q):
            q.append(item)


_PGREP_SKIP_SUBSTR = (
    "queue_keeper",
    "run_psma_fc70_after_main_queue",
    "run_fdg_eval_after_fc70_queue",
    "run_nnunet_psma_fs10_fs5_rerun_fdg169_after",
    "run_fdg20_test_after_fc70",
    "pgrep -af",
    "pgrep -f",
    "__CURSOR_SANDBOX",
    "dump_bash_state",
    "monitor_val_dice_decline_stop",
    "monitor_task1_train_auto_resume_guard",
    "run_aligned_board_watch",
    "iclr2026_aligned_fdg_fs50_board.py",
    "run_mae_scratch_after_extra_folds",
    "run_aligned_monai_scratch_9fold_pipeline",
    "run_aligned_seganypet_scratch_9fold_pipeline",
    "run_nnunet_mim_dpdnet_dualenc_after_scratch",
    "gpu_idle_queue_scheduler",
)


def _pgrep_any(patterns: tuple[str, ...]) -> bool:
    return bool(_pgrep_lines(patterns))


def _pgrep_lines(patterns: tuple[str, ...]) -> list[str]:
    """Return matching pgrep -af lines (skip self/sandbox)."""
    out: list[str] = []
    for pat in patterns:
        if not pat:
            continue
        try:
            r = subprocess.run(
                ["pgrep", "-af", pat],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode != 0 or not (r.stdout or "").strip():
                continue
            for line in r.stdout.splitlines():
                if any(s in line for s in _PGREP_SKIP_SUBSTR):
                    continue
                out.append(line)
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def _gpu_ids_from_cmdline(text: str) -> list[int] | None:
    """Parse live GPU from docker/process cmdline (prefer --gpus / CUDA_VISIBLE only)."""
    if not text:
        return None
    m = re.search(r"--gpus\s+[\"']?device=([0-9]+(?:[,+][0-9]+)*)", text, re.I)
    if m:
        return [int(x) for x in re.split(r"[,+]", m.group(1)) if x.isdigit()]
    m = re.search(r"CUDA_VISIBLE_DEVICES=([0-9]+(?:[,:][0-9]+)*)", text)
    if m:
        return [int(x) for x in re.split(r"[,:]", m.group(1)) if x.isdigit()]
    # do NOT fall back to stamp-like gpu013 in paths (misleading pool tag)
    return None


def _live_gpu_ids_for_patterns(patterns: tuple[str, ...]) -> list[int] | None:
    ids: list[int] = []
    seen: set[int] = set()
    for line in _pgrep_lines(patterns):
        got = _gpu_ids_from_cmdline(line)
        if not got:
            continue
        for i in got:
            if i not in seen:
                seen.add(i)
                ids.append(i)
    return ids or None


def _pid_alive(pid_file: Path) -> tuple[bool, int | None]:
    if not pid_file.is_file():
        return False, None
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except OSError:
        return False, pid


def _stage_has_score(st: dict) -> bool:
    if isinstance(st.get("mean"), (int, float)) and st["mean"] == st["mean"]:
        return True
    fd = st.get("fold_dice") or {}
    return any(isinstance(v, (int, float)) and v == v for v in fd.values())


def _stage_fp_fn(st: dict) -> tuple[float | None, float | None]:
    fp = st.get("mean_fp")
    if fp is None:
        fp = st.get("fp_rate")
    fn = st.get("mean_fn")
    if fn is None:
        fn = st.get("fn_rate")
    fp_f = float(fp) if isinstance(fp, (int, float)) and fp == fp else None
    fn_f = float(fn) if isinstance(fn, (int, float)) and fn == fn else None
    return fp_f, fn_f


def _metrics_cell_text(st: dict, dice: float | None = None) -> str:
    """Dice / FP / FN as percentages (3 lines)."""
    if dice is None:
        mean = st.get("mean")
        dice = float(mean) if isinstance(mean, (int, float)) and mean == mean else None
    fp, fn = _stage_fp_fn(st)
    return format_dice_fp_fn(dice, fp, fn, digits=2)


def _apply_fp_fn_from_agg(st: dict, ad: dict) -> None:
    for src, dst in (("fp_rate", "mean_fp"), ("fn_rate", "mean_fn"), ("mean_fp", "mean_fp"), ("mean_fn", "mean_fn")):
        v = ad.get(src)
        if isinstance(v, (int, float)) and v == v:
            st[dst] = float(v)


def _mean_dice_from_score_dict(ad: dict) -> float | None:
    """Canonical Dice for board: empty-GT excluded (prefer mean_dice_positive)."""
    for key in ("mean_dice_positive", "mean_dice", "test_mean", "mean"):
        md = ad.get(key)
        if isinstance(md, (int, float)) and md == md:
            return float(md)
    return None


def _aggregate_json_valid(agg: Path) -> bool:
    """True when aggregate JSON has a finite mean Dice (not NaN / empty run)."""
    if not agg.is_file():
        return False
    try:
        ad = json.loads(agg.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    md = _mean_dice_from_score_dict(ad)
    if md is not None:
        return True
    n_scored = ad.get("n_scored")
    try:
        return n_scored is not None and int(n_scored) > 0
    except (TypeError, ValueError):
        return False


def _stage_aggregate_done(method_key: str, stage_key: str, st: dict) -> bool:
    if stage_key == PSMA_FS0_STAGE_KEY:
        return _aggregate_json_valid(PSMA_FS0_AGG_DIR / f"aggregate_{method_key}.json")
    if stage_key == FDG_TEST_STAGE_KEY:
        return _aggregate_json_valid(FDG_TEST_AGG_DIR / f"aggregate_{method_key}.json")
    if stage_key == FC70_STAGE_KEY:
        stamp = (st.get("stamp") or "").strip()
        if not stamp:
            return False
        for p in (
            CTRL / "ICLR2026/vis" / f"aggregate_nnunet_psma_fc70_{stamp}.json",
            NN_RESULTS / stamp / "aggregate_test20_dice_fc70.json",
            CTRL / "ICLR2026/vis" / f"aggregate_mae_psma_fc70_{stamp}.json",
            CTRL / "ICLR2026/vis" / f"aggregate_monai_psma_fc70_{stamp}.json",
            CTRL / "ICLR2026/vis" / f"aggregate_dpdnet_psma_fc70_{stamp}.json",
            CTRL / "ICLR2026/vis" / f"aggregate_seganypet_psma_test20_f258_{stamp}.json",
            REPO / stamp / "psma_test20_eval" / "aggregate_test20_f258.json",
            NN_RESULTS / stamp / "aggregate_test20_dice_f258.json",
        ):
            if p.is_file() and _aggregate_json_valid(p):
                return True
        return False
    return False


# Runtime pgrep hints per (method, stage) for board self-check.
STAGE_RUNTIME_PATTERNS: dict[tuple[str, str], tuple[str, ...]] = {
    ("nnunet_mim", "psma_fs50_f258"): (
        "run_nnunet_mim_aligned_fdg_psma",
        "nnunet_mim_psma_fs50",
        "TASK1_BOARD_METHOD=nnunet_mim",
    ),
    ("dpdnet_dualenc", "psma_fs50_f258"): (
        "run_dpdnet_dualenc_aligned_fdg_psma",
        "dpdnet_dualenc_psma_fs50",
        "TASK1_BOARD_METHOD=dpdnet_dualenc",
    ),
    ("nnunet", "psma_fs10_f258"): ("run_nnunet_psma_fs10", "nnunet_psma_fs10_f258"),
    ("nnunet", "psma_fs5_f258"): ("run_nnunet_psma_fs5", "nnunet_psma_fs5_f258"),
    ("nnunet", "psma_fc70"): ("run_nnunet_psma_fc70", "nnunet_psma_fc70"),
    ("nnunet", "psma_fs0"): ("psma_fs0_eval/nnunet", "nnunet.*psma_fs0"),
    # Competition scratch / pretrained (nnUNet-style TRAIN + TEST20)
    ("hemingduo_scratch", "psma_fs50_f258"): (
        "hemingduo_scratch_psma_fs50",
        "iclr2026_hemingduo_scratch_psma_fs50",
        "hemingduo_scratch_fs50_test20_resume",
        "TASK1_BOARD_METHOD=hemingduo_scratch",
        "run_competition_scratch",
    ),
    ("hemingduo_scratch", "psma_fs10_f258"): (
        "hemingduo_scratch_psma_fs10",
        "iclr2026_hemingduo_scratch_psma_fs10",
        "TASK1_BOARD_METHOD=hemingduo_scratch",
    ),
    ("hemingduo_scratch", "psma_fs5_f258"): (
        "hemingduo_scratch_psma_fs5",
        "iclr2026_hemingduo_scratch_psma_fs5",
        "TASK1_BOARD_METHOD=hemingduo_scratch",
    ),
    ("chenyixin_scratch", "psma_fs50_f258"): (
        "chenyixin_scratch_psma_fs50",
        "iclr2026_chenyixin_scratch_psma_fs50",
        "TASK1_BOARD_METHOD=chenyixin_scratch",
        "run_competition_scratch",
    ),
    ("chenyixin_scratch", "psma_fs10_f258"): (
        "chenyixin_scratch_psma_fs10",
        "iclr2026_chenyixin_scratch_psma_fs10",
        "TASK1_BOARD_METHOD=chenyixin_scratch",
    ),
    ("chenyixin_scratch", "psma_fs5_f258"): (
        "chenyixin_scratch_psma_fs5",
        "iclr2026_chenyixin_scratch_psma_fs5",
        "TASK1_BOARD_METHOD=chenyixin_scratch",
    ),
    ("hemingduo", "psma_fs50_f258"): (
        "hemingduo_psma_fs50",
        "iclr2026_hemingduo_psma_fs50",
        "TASK1_BOARD_METHOD=hemingduo",
        "run_competition_aligned",
    ),
    ("chenyixin", "psma_fs50_f258"): (
        "chenyixin_psma_fs50",
        "iclr2026_chenyixin_psma_fs50",
        "TASK1_BOARD_METHOD=chenyixin",
        "run_competition_aligned",
    ),
    ("mae_swinunetr", "psma_fs10_f258"): (
        "mae_psma_fs10",
        "mae_fs10_fdgseg",
        "iclr2026_mae_psma_fs10",
    ),
    ("mae_swinunetr", "psma_fs5_f258"): (
        "mae_psma_fs5",
        "mae_fs5_fdgseg",
        "iclr2026_mae_psma_fs5",
    ),
    ("mae_swinunetr", "psma_fc70"): ("run_mae_psma_fc70", "mae_psma_fc70"),
    ("mae_swinunetr", "psma_fs0"): ("psma_fs0_eval/mae", "mae.*psma_fs0"),
    ("mae_scratch", "psma_fs50_f258"): ("mae_scratch_psma_fs50", "iclr2026_mae_scratch_psma_fs50"),
    ("mae_scratch", "psma_fs10_f258"): ("mae_scratch_psma_fs10", "iclr2026_mae_scratch_psma_fs10"),
    ("mae_scratch", "psma_fs5_f258"): ("mae_scratch_psma_fs5", "iclr2026_mae_scratch_psma_fs5"),
    ("mae_scratch", "psma_fc70"): ("mae_scratch_psma_fc70", "iclr2026_mae_scratch_psma_fc70"),
    ("mae_scratch", "psma_fs0"): ("psma_fs0_eval/mae_scratch", "METHOD=mae_scratch"),
    ("monai_scratch", "psma_fs50_f258"): (
        "monai_scratch_psma_fs50",
        "iclr2026_monai_scratch_psma_fs50",
        "TASK1_BOARD_METHOD=monai_scratch",
    ),
    ("monai_scratch", "psma_fs10_f258"): (
        "monai_scratch_psma_fs10",
        "iclr2026_monai_scratch_psma_fs10",
        "TASK1_BOARD_METHOD=monai_scratch",
    ),
    ("monai_scratch", "psma_fs5_f258"): (
        "monai_scratch_psma_fs5",
        "iclr2026_monai_scratch_psma_fs5",
        "TASK1_BOARD_METHOD=monai_scratch",
    ),
    ("monai_scratch", "psma_fc70"): (
        "monai_scratch_psma_fc70",
        "iclr2026_monai_scratch_psma_fc70",
    ),
    ("monai_scratch", "psma_fs0"): ("psma_fs0_eval/monai_scratch", "METHOD=monai_scratch"),
    ("seganypet_scratch", "psma_fs50_f258"): (
        "seganypet_scratch_psma_fs50",
        "iclr2026_seganypet_scratch_psma_fs50",
        "TASK1_BOARD_METHOD=seganypet_scratch",
    ),
    ("seganypet_scratch", "psma_fs10_f258"): (
        "seganypet_scratch_psma_fs10",
        "iclr2026_seganypet_scratch_psma_fs10",
        "TASK1_BOARD_METHOD=seganypet_scratch",
    ),
    ("seganypet_scratch", "psma_fs5_f258"): (
        "seganypet_scratch_psma_fs5",
        "iclr2026_seganypet_scratch_psma_fs5",
        "TASK1_BOARD_METHOD=seganypet_scratch",
    ),
    ("seganypet_scratch", "psma_fc70"): (
        "seganypet_scratch_psma_fc70",
        "iclr2026_seganypet_scratch_psma_fc70",
    ),
    ("seganypet_scratch", "psma_fs0"): (
        "psma_fs0_eval/seganypet_scratch",
        "METHOD=seganypet_scratch",
    ),
    ("monai_swinvit", "psma_fs10_f258"): (
        "monai_psma_fs10",
        "monai_fs10_fdgseg",
        "iclr2026_monai_psma_fs10",
    ),
    ("monai_swinvit", "psma_fs5_f258"): (
        "monai_psma_fs5",
        "monai_fs5_fdgseg",
        "iclr2026_monai_psma_fs5",
    ),
    ("monai_swinvit", "psma_fc70"): ("run_monai_psma_fc70", "monai_psma_fc70"),
    ("monai_swinvit", "psma_fs0"): ("psma_fs0_eval/monai", "monai.*psma_fs0"),
    ("dpdnet", "psma_fs10_f258"): ("dpdnet_psma_fs10", "dpdnet.*fs10"),
    ("dpdnet", "psma_fs5_f258"): ("dpdnet_psma_fs5", "dpdnet.*fs5"),
    ("dpdnet", "psma_fc70"): ("run_dpdnet_psma_fc70", "dpdnet_psma_fc70"),
    ("dpdnet", "psma_fs0"): ("psma_fs0_eval/dpdnet", "dpdnet.*psma_fs0"),
    ("seganypet", "psma_fs10_f258"): ("seganypet.*fs10", "seganypet_fs10"),
    ("seganypet", "psma_fs5_f258"): ("seganypet.*fs5", "seganypet_fs5"),
    ("seganypet", "psma_fc70"): ("run_seganypet_psma_fc70", "seganypet.*fc70", "seganypet_fc70"),
    ("seganypet", "psma_fs0"): ("psma_fs0_eval/seganypet", "seganypet.*psma_fs0"),
    ("nnunet", "fdg_test20"): ("fdg_test20_eval/nnunet", "run_eval_fdg_test20"),
    ("mae_swinunetr", "fdg_test20"): (
        "fdg_test20_eval/mae",
        "run_eval_fdg_test20",
        "METHOD=mae",
        "mae_eval_seg_psma_test",
    ),
    ("mae_scratch", "fdg_test20"): (
        "fdg_test20_eval/mae_scratch",
        "METHOD=mae_scratch",
        "mae_eval_seg_psma_test",
    ),
    ("monai_swinvit", "fdg_test20"): ("fdg_test20_eval/monai", "run_eval_fdg_test20", "METHOD=monai"),
    ("monai_scratch", "fdg_test20"): (
        "fdg_test20_eval/monai_scratch",
        "METHOD=monai_scratch",
        "mae_eval_seg_psma_test",
    ),
    ("dpdnet", "fdg_test20"): ("fdg_test20_eval/dpdnet", "run_eval_fdg_test20", "METHOD=dpdnet"),
    ("seganypet", "fdg_test20"): ("fdg_test20_eval/seganypet", "run_eval_fdg_test20", "METHOD=seganypet"),
    ("seganypet_scratch", "fdg_test20"): (
        "fdg_test20_eval/seganypet_scratch",
        "METHOD=seganypet_scratch",
    ),
    ("proto_retrieval", "fdg_test20"): (
        "fdg_test20_eval/proto_retrieval",
        "fdg80_gallery",
    ),
}

QUEUE_WORKERS: tuple[dict[str, Any], ...] = (
    {
        "name": "fs10_fs5_pipeline",
        "pid_file": CTRL / "ICLR2026/vis/aligned_psma_fs10_fs5_pipeline.pid",
        "patterns": ("run_aligned_psma_fs10_fs5_pipeline_bg.sh",),
        "restart": "bash ICLR2026/run/run_aligned_psma_fs10_fs5_pipeline_bg.sh",
    },
    {
        "name": "nnunet_fs10_fs5_rerun_queue",
        "pid_file": CTRL / "ICLR2026/vis/fs10_fs5_fdg169_rerun_queue.pid",
        "patterns": ("run_nnunet_psma_fs10_fs5_rerun_fdg169",),
        "restart": "bash ICLR2026/run/run_nnunet_psma_fs10_fs5_rerun_fdg169_after_methods_queue_bg.sh",
    },
    {
        "name": "fc70_queue",
        "pid_file": CTRL / "ICLR2026/vis/psma_fc70_queue.pid",
        "patterns": ("run_psma_fc70_after_main_queue", "run_aligned_psma_fc70_pipeline"),
        "restart": "bash ICLR2026/run/run_psma_fc70_after_main_queue_bg.sh",
    },
    {
        "name": "eval_queue",
        "pid_file": CTRL / "ICLR2026/vis/fdg_eval_queue.pid",
        "patterns": (
            "run_fdg_eval_after_fc70_queue",
            "run_eval_fdg_shared_test20",
            "run_eval_fdg_test20",
        ),
        "restart": "bash ICLR2026/run/run_fdg_eval_after_fc70_queue_bg.sh",
    },
    {
        "name": "extra_folds_9fold_queue",
        "pid_file": CTRL / "ICLR2026/vis/extra_folds_9fold_queue.pid",
        "patterns": (
            "run_psma_extra_folds_9fold_after_eval",
            "run_aligned_psma_extra_folds_9fold_pipeline",
        ),
        "restart": "bash ICLR2026/run/run_psma_extra_folds_9fold_after_eval_queue_bg.sh",
    },
    {
        "name": "mae_scratch_9fold_queue",
        "pid_file": CTRL / "ICLR2026/vis/mae_scratch_9fold_queue.pid",
        "patterns": (
            "run_mae_scratch_after_extra_folds_queue",
            "run_aligned_mae_scratch_9fold_pipeline",
        ),
        "restart": "bash ICLR2026/run/run_mae_scratch_after_extra_folds_queue_bg.sh",
    },
    {
        "name": "monai_scratch_9fold_queue",
        "pid_file": CTRL / "ICLR2026/vis/monai_scratch_9fold_pipeline.pid",
        "patterns": ("run_aligned_monai_scratch_9fold_pipeline",),
        "restart": "bash ICLR2026/run/run_aligned_monai_scratch_9fold_pipeline_bg.sh",
    },
    {
        "name": "seganypet_scratch_9fold_queue",
        "pid_file": CTRL / "ICLR2026/vis/seganypet_scratch_9fold_pipeline.pid",
        "patterns": ("run_aligned_seganypet_scratch_9fold_pipeline",),
        "restart": "bash ICLR2026/run/run_aligned_seganypet_scratch_9fold_pipeline_bg.sh",
    },
    {
        "name": "nnunet_mim_dpdnet_dualenc_queue",
        "pid_file": CTRL / "ICLR2026/vis/nnunet_mim_dpdnet_dualenc_after_scratch_queue.pid",
        "patterns": (
            "run_nnunet_mim_dpdnet_dualenc_after_scratch",
            "run_nnunet_mim_aligned_fdg_psma",
            "run_dpdnet_dualenc_aligned_fdg_psma",
        ),
        "restart": "bash ICLR2026/run/run_nnunet_mim_dpdnet_dualenc_after_scratch_queue_bg.sh",
    },
)


def _method_short_label(mkey: str, board: dict | None = None) -> str:
    mapped = {
        "nnunet_mim": "nnUNet MIM",
        "nnunet": "nnUNet",
        "mae_swinunetr": "MAE",
        "mae_scratch": "MAE scratch",
        "monai_scratch": "MONAI scratch",
        "seganypet_scratch": "SegAnyPET scratch",
        "monai_swinvit": "MONAI",
        "dpdnet_dualenc": "DpDNet dual-enc",
        "dpdnet": "DpDNet",
        "seganypet": "SegAnyPET",
        "proto_retrieval": "Proto",
        "hemingduo": "BIRTH",
        "hemingduo_scratch": "BIRTH scratch",
        "chenyixin": "YixinChen",
        "chenyixin_scratch": "YixinChen scratch",
    }
    if mkey in mapped:
        return mapped[mkey]
    if board:
        lbl = (board.get("methods", {}).get(mkey) or {}).get("label")
        if lbl:
            return str(lbl).split()[0]
    return mkey


def _collect_running_tasks(board: dict) -> list[str]:
    """Human-readable RUNNING / WAITING labels for PNG subtitle / health_check."""
    out: list[str] = []
    for name in _cpu_side_jobs_running():
        out.append(f"{name} (CPU)")
    waiting: list[str] = []
    for mkey in METHOD_ORDER:
        m = board.get("methods", {}).get(mkey) or {}
        ml = _method_short_label(mkey, board)
        fdg = m.get("fdg_pretrain")
        if isinstance(fdg, dict) and (fdg.get("status") or "").lower() == "running":
            tag = _annotate_stage_device(fdg, board)
            ep, tot = fdg.get("epoch"), fdg.get("total_epochs")
            bits = [f"{ml}/FDG ({tag})"]
            if ep is not None and tot:
                bits.append(f"ep{ep}/{tot}")
            bits.append(f"ETA {_running_eta_display(fdg)}")
            out.append(" · ".join(bits))
        for _n, stage_key, short in PSMA_BOARD_COLUMNS:
            st = m.get(stage_key)
            if not isinstance(st, dict):
                continue
            status = (st.get("status") or "").lower()
            if status == "waiting":
                waiting.append(f"{ml}/{short} · {_waiting_status_head(st)}")
                continue
            if status != "running":
                continue
            tag = _annotate_stage_device(st, board)
            phase = (st.get("phase") or "").upper()
            ep, tot = st.get("epoch"), st.get("total_epochs")
            bits = [f"{ml}/{short} ({tag})"]
            if phase:
                bits.append(phase)
            if ep is not None and tot:
                bits.append(f"ep{ep}/{tot}")
            bits.append(f"ETA {_running_eta_display(st)}")
            out.append(" · ".join(bits))
    # waiting next tasks first in subtitle (upcoming)
    return waiting + out


def board_self_check(board: dict) -> dict[str, Any]:
    """Detect empty board cells: running/done vs queue gap."""
    _reset_runtime_caches()
    ensure_methods(board)
    _apply_live_extra_fold_running(board)
    _promote_waiting_to_running(board)
    vis = CTRL / "ICLR2026/vis"
    gaps: list[dict[str, Any]] = []
    running_n = done_n = pending_n = 0

    live_slots = _live_fewshot_slots()
    for mkey in METHOD_ORDER:
        m = board.get("methods", {}).get(mkey) or {}
        for _n, stage_key, short in PSMA_BOARD_COLUMNS:
            st = m.get(stage_key)
            if not isinstance(st, dict):
                continue
            status = (st.get("status") or "pending").lower()
            if status in ("n/a", "na"):
                continue
            has_score = _stage_has_score(st)
            agg_done = _stage_aggregate_done(mkey, stage_key, st)
            pats = STAGE_RUNTIME_PATTERNS.get((mkey, stage_key), ())
            runtime = _pgrep_any(pats) or (mkey, stage_key) in live_slots
            # Completed fc70 TEST20: ignore leftover decline monitors / other-stamp jobs
            if runtime and (has_score or agg_done) and stage_key == FC70_STAGE_KEY:
                cell_stamp = (st.get("stamp") or "").strip()
                lines = _pgrep_lines(pats) if pats else []
                live = _live_fc70_stamp(mkey)
                blob = "\n".join(lines)
                if cell_stamp and cell_stamp not in blob and live != cell_stamp:
                    runtime = False

            # Live process wins over stale scores (e.g. FDG TEST re-predict on GPU3)
            if runtime:
                st["status"] = "running"
                _clear_stage_wait_fields(st)
                live_gpus = _live_gpu_ids_for_patterns(pats) if pats else None
                if live_gpus:
                    st["device"] = "gpu"
                    st["gpu_ids"] = ",".join(str(i) for i in live_gpus)
                if stage_key == FDG_TEST_STAGE_KEY:
                    st["phase"] = "TEST20"
                    live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
                    st["test_live"] = {**live, "active": True}
                running_n += 1
                continue
            if status == "waiting":
                # next-task GPU idle countdown — keep, not a gap
                running_n += 1
                continue
            cpu_alive = str(st.get("device") or "").lower() == "cpu" and (
                _pgrep_any(
                    (
                        "rescore_board_dice_fp_fn.py",
                        "run_rescore_proto_fewshot_fp_fn",
                        "score_pred_dice_vs_gt.py",
                    )
                )
                or _pid_alive(CTRL / "ICLR2026/vis/rescore_proto_fewshot_fp_fn.pid")[0]
            )
            if status == "running" and cpu_alive:
                running_n += 1
                continue
            if status == "running":
                # marked running but process gone
                if has_score or agg_done:
                    st["status"] = "done"
                    done_n += 1
                else:
                    st["status"] = "pending"
                    pending_n += 1
                    gaps.append(
                        {
                            "method": mkey,
                            "stage": stage_key,
                            "column": short,
                            "status": "pending",
                            "runtime": False,
                            "aggregate": agg_done,
                        }
                    )
                continue
            if has_score or agg_done:
                # do not clobber an in-flight CPU rescore mark
                if cpu_alive and (st.get("status") or "").lower() == "running":
                    running_n += 1
                    continue
                if status != "done":
                    st["status"] = "done"
                done_n += 1
                continue
            if status in ("done",):
                continue
            pending_n += 1
            gaps.append(
                {
                    "method": mkey,
                    "stage": stage_key,
                    "column": short,
                    "status": status,
                    "runtime": runtime,
                    "aggregate": agg_done,
                }
            )

    workers: dict[str, Any] = {}
    dead_workers: list[str] = []
    for w in QUEUE_WORKERS:
        alive_pid, pid = _pid_alive(w["pid_file"])
        runtime = _pgrep_any(tuple(w["patterns"]))
        alive = alive_pid or runtime
        workers[w["name"]] = {"alive": alive, "pid": pid, "runtime": runtime}
        if not alive:
            dead_workers.append(w["name"])

    # Rebuild queue summary from gaps + workers
    queue_items: list[str] = []
    wait = board.get("gpu_idle_wait") if isinstance(board.get("gpu_idle_wait"), dict) else _load_gpu_idle_wait()
    if wait.get("next_label") and wait.get("gpu") is not None and wait.get("remain_sec") is not None:
        queue_items.append(
            f"WAITING · {wait.get('next_label')} (GPU {wait.get('gpu')} · {_fmt_eta(float(wait['remain_sec']))})"
        )
    if _cpu_side_jobs_running():
        for name in _cpu_side_jobs_running():
            queue_items.append(f"{name} (running · CPU)")
    elif _pgrep_any(("run_rescore_dice_fp_fn_queue_worker",)):
        queue_items.append("rescore GPU MAE/MONAI (queued · wait GPU)")
    if _pgrep_any(("run_aligned_psma_fs10_fs5_pipeline",)):
        queue_items.append("fs10/fs5 pipeline (running · GPU)")
    elif any(g["column"] in ("fs10", "fs5") for g in gaps):
        queue_items.append("fs10/fs5 pipeline")
    if _pgrep_any(("run_nnunet_psma_fs10_fs5_rerun",)):
        queue_items.append("nnunet fs10/fs5 rerun (running · GPU)")
    if workers.get("fc70_queue", {}).get("alive"):
        queue_items.append("fc70% queue (alive · GPU)")
    elif any(g["stage"] == FC70_STAGE_KEY for g in gaps):
        queue_items.append("fc70% pipeline (queued)")
    if workers.get("eval_queue", {}).get("alive") or _pgrep_any(
        (
            "run_eval_fdg_shared_test20_bg.sh",
            "run_eval_fdg_test20_bg.sh",
            "run_fdg_eval_after_fc70_queue_bg.sh",
        )
    ):
        queue_items.append("PSMA fs0 / FDG TEST eval (running · GPU)")
    elif any(g["stage"] in (PSMA_FS0_STAGE_KEY, FDG_TEST_STAGE_KEY) for g in gaps):
        queue_items.append("PSMA fs0 / FDG TEST eval (queued)")
    if workers.get("extra_folds_9fold_queue", {}).get("alive") or _pgrep_any(
        (
            "run_psma_extra_folds_9fold_after_eval",
            "run_aligned_psma_extra_folds_9fold_pipeline",
            "run_aligned_psma_extra_fold_onegpu.sh",
        )
    ):
        queue_items.append("fs50/fs10/fs5 extra folds → 9fold (running · GPU)")
    else:
        queue_items.append("fs50/fs10/fs5 extra folds → 9fold (queued)")
    for g in gaps[:6]:
        queue_items.append(f"{g['method']}/{g['column']}:{g['status']}")

    ok = len(gaps) == 0 or (running_n > 0 and not dead_workers)
    summary_parts = [
        f"done={done_n}",
        f"run={running_n}",
        f"gap={len(gaps)}",
    ]
    if dead_workers:
        summary_parts.append(f"dead_queue={','.join(dead_workers)}")
    running_tasks = _collect_running_tasks(board)
    health: dict[str, Any] = {
        "at": _now(),
        "ok": ok and not dead_workers,
        "summary": " · ".join(summary_parts),
        "done_n": done_n,
        "running_n": running_n,
        "pending_n": pending_n,
        "running_tasks": running_tasks,
        "gaps": gaps,
        "queue_workers": workers,
        "dead_workers": dead_workers,
        "queue_live": queue_items,
    }
    board["health_check"] = health
    if queue_items:
        board["queue"] = queue_items
    return health


def ensure_stage_bs(board: dict) -> None:
    """Fill missing bs fields from protocol defaults (do not overwrite set values)."""
    defaults: dict[tuple[str, str], tuple[int, str]] = {}
    for n, stage, _ in FEWSHOT_VARIANTS:
        defaults[("nnunet", stage)] = (6 if n == 50 else 2, "per-GPU")
        defaults[("nnunet_mim", stage)] = (2, "per-GPU")
        defaults[("mae_swinunetr", stage)] = (2, "per-GPU")
        defaults[("mae_scratch", stage)] = (2, "per-GPU")
        defaults[("monai_swinvit", stage)] = (2, "per-GPU")
        defaults[("monai_scratch", stage)] = (2, "per-GPU")
        defaults[("seganypet", stage)] = (2, "per-GPU")
        defaults[("seganypet_scratch", stage)] = (2, "per-GPU")
        defaults[("dpdnet", stage)] = (2, "per-GPU")
        defaults[("dpdnet_dualenc", stage)] = (2, "per-GPU")
        defaults[("hemingduo", stage)] = (2, "per-GPU")
        defaults[("hemingduo_scratch", stage)] = (2, "per-GPU")
        defaults[("chenyixin", stage)] = (2, "per-GPU")
        defaults[("chenyixin_scratch", stage)] = (2, "per-GPU")
    defaults[("nnunet", "fdg_pretrain")] = (6, "per-GPU")
    defaults[("nnunet_mim", "fdg_pretrain")] = (6, "gbs")
    defaults[("mae_swinunetr", "fdg_pretrain")] = (6, "global 2×3GPU")
    defaults[("mae_scratch", "fdg_pretrain")] = (6, "global 2×3GPU")
    defaults[("monai_swinvit", "fdg_pretrain")] = (6, "global 2×3GPU")
    defaults[("monai_scratch", "fdg_pretrain")] = (6, "global 2×3GPU")
    defaults[("seganypet", "fdg_pretrain")] = (6, "global DP 0,1,3")
    defaults[("seganypet_scratch", "fdg_pretrain")] = (6, "global DP 0,1,3")
    defaults[("dpdnet", "fdg_pretrain")] = (6, "per-GPU")
    defaults[("dpdnet_dualenc", "fdg_pretrain")] = (6, "per-GPU")
    defaults[("hemingduo", "fdg_pretrain")] = (6, "TBD")
    defaults[("hemingduo_scratch", "fdg_pretrain")] = (6, "TBD")
    defaults[("chenyixin", "fdg_pretrain")] = (6, "TBD")
    defaults[("chenyixin_scratch", "fdg_pretrain")] = (6, "TBD")
    methods = board.get("methods") or {}
    for (mk, sk), (bs, note) in defaults.items():
        st = methods.get(mk, {}).get(sk)
        if not isinstance(st, dict):
            continue
        if st.get("bs") is None:
            st["bs"] = bs
        if not st.get("bs_note"):
            st["bs_note"] = note


def _apply_paused_dpdnet_psma(st: dict, stamp: str, total: int) -> None:
    """Keep PSMA row as PAUSED (epoch from logs; hide stale TEST scores)."""
    st["status"] = "paused"
    st["phase"] = None
    st["eta"] = None
    st["eta_sec"] = None
    st.pop("test_live", None)
    st["fold_dice"] = {}
    st["mean"] = None
    st["fold_ckpt_ep"] = {}
    st["eval_done"] = None
    st["eval_total"] = None
    epochs: list[int] = []
    for f in (2, 5, 8):
        fd = _dpdnet_psma_fold_dir(stamp, f)
        if fd is None:
            continue
        ep = _nnunet_log_latest_epoch(fd)
        if ep is not None:
            epochs.append(int(ep))
    if epochs:
        st["epoch"] = min(epochs)
    st["total_epochs"] = total
    st["note"] = "paused · wait nnUNet TEST then -c"


def _status_label(st: dict) -> str:
    status = (st.get("status") or "pending").upper()
    train_t = st.get("train_time")
    if status == "DONE":
        return f"DONE · {train_t}" if train_t else "DONE"
    if status == "PAUSED":
        parts = ["PAUSED"]
        ep, tot = st.get("epoch"), st.get("total_epochs")
        if ep is not None and tot:
            parts.append(f"ep{ep}/{tot}")
        return " · ".join(parts)
    if status == "WAITING":
        return _waiting_status_head(st)
    if status != "RUNNING":
        return status
    parts = [_running_status_head(st)]
    phase = (st.get("phase") or "").upper()
    if phase:
        parts.append(phase)
    if phase == "TEST20":
        done = st.get("eval_done")
        tot = st.get("eval_total") or 3
        if done is not None:
            parts.append(f"{done}/{tot}")
        live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
        note = st.get("note") or ""
        # prefer compact pred progress from note if present
        if "pred " in note:
            frag = note.split("pred ", 1)[-1].strip()
            pred_bit = frag.split("·")[0].strip()
            if pred_bit and pred_bit not in " · ".join(parts):
                parts.append(f"pred {pred_bit}")
        elif live.get("cases_done") is not None and live.get("cases_total"):
            parts.append(f"pred {live['cases_done']}/{live['cases_total']}")
        parts.append(f"ETA {_running_eta_display(st)}")
        if train_t:
            parts.append(f"train {train_t}")
        return " · ".join(parts)
    ep, tot = st.get("epoch"), st.get("total_epochs")
    if ep is not None and tot:
        parts.append(f"ep{ep}/{tot}")
    parts.append(f"ETA {_running_eta_display(st)}")
    # keep status line compact: omit elapsed when ETA already shown
    if train_t and phase != "TEST20":
        parts.append(f"elapsed {train_t}")
    return " · ".join(parts)


def _nnunet_ema_best_epoch(fold_dir: Path) -> int | None:
    """1-based epoch of last finite 'New best EMA' across all training logs."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    import re

    last = None
    for lg in logs:
        try:
            lines = lg.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        cur = None
        for line in lines:
            m = re.search(r"Epoch\s+(\d+)\s*$", line)
            if m:
                cur = int(m.group(1))
                continue
            if "New best EMA" in line and cur is not None:
                if "nan" in line.lower():
                    continue
                last = cur + 1
    return last


def _nnunet_pseudo_dice_best_epoch(fold_dir: Path) -> int | None:
    try:
        from nnunet_pseudo_dice_best import pseudo_dice_best_epoch
    except ImportError:
        import sys

        sys.path.insert(0, str(CTRL / "ICLR2026" / "scripts"))
        from nnunet_pseudo_dice_best import pseudo_dice_best_epoch

    ep, _dice, _series = pseudo_dice_best_epoch(fold_dir)
    return ep


def _nnunet_val_loss_best_epoch(fold_dir: Path) -> int | None:
    """1-based finished epoch of last 'New best val_loss' (matches MAE [ep/total] display)."""
    logs = sorted(fold_dir.glob("training_log*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    cur = None
    last = None
    try:
        lines = logs[-1].read_text(errors="ignore").splitlines()
    except OSError:
        return None
    import re

    for line in lines:
        m = re.search(r"Epoch\s+(\d+)\s*$", line)
        if m:
            cur = int(m.group(1))
            continue
        if "New best val_loss" in line and cur is not None:
            last = cur + 1  # log Epoch is 0-based at epoch start
    return last


def _mae_metrics_summary(metrics: Path) -> dict[str, Any]:
    """best VAL dice / best-ep / last-ep from metrics.jsonl."""
    out: dict[str, Any] = {"best_dice": None, "best_ep": None, "last_ep": None}
    if not metrics.is_file():
        return out
    import math

    prev: float | None = None
    last_improve: int | None = None
    last_ep: int | None = None
    try:
        rows = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
    except (OSError, json.JSONDecodeError):
        return out
    for r in rows:
        ep = r.get("epoch")
        if ep is not None:
            try:
                last_ep = int(ep)
            except (TypeError, ValueError):
                pass
        cand = r.get("val_dice")
        try:
            cand_f = float(cand) if cand is not None else float("nan")
        except (TypeError, ValueError):
            cand_f = float("nan")
        if math.isnan(cand_f):
            cand = r.get("best_dice")
            if cand is None:
                cand = r.get("best_val_dice")
            try:
                cand_f = float(cand)
            except (TypeError, ValueError):
                continue
        if math.isnan(cand_f) or cand_f < 0:
            continue
        if prev is None or cand_f > prev + 1e-12:
            last_improve = int(ep) if ep is not None else last_improve
            prev = cand_f
    out["best_dice"] = prev
    out["best_ep"] = last_improve
    out["last_ep"] = last_ep
    return out


def _mae_best_epoch_from_metrics(metrics: Path) -> int | None:
    """Epoch of last val-Dice improvement. Ignore sentinel best_dice=-1 (pre-first-val)."""
    return _mae_metrics_summary(metrics).get("best_ep")


def _sync_mae_family_fold_displays(st: dict, sub: str) -> None:
    """TEST20 → fold_dice (official). Extra-fold VAL/best → val_monitor_fold_dice for the 3×3."""
    stamp = (st.get("stamp") or "").strip()
    if not stamp:
        return
    test20: dict[str, float] = {}
    eval_root = REPO / stamp / "psma_test20_eval"
    if eval_root.is_dir():
        for f in NINE_FOLD_STRS:
            p = eval_root / f"fold{f}_test20.json"
            if not p.is_file():
                continue
            try:
                d = json.loads(p.read_text())
                md = d.get("mean_dice_positive", d.get("mean_dice"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(md, (int, float)) and md == md:
                test20[f] = float(md)
    val_d: dict[str, float] = {}
    ep_map: dict[str, int] = {}
    for f in NINE_FOLDS:
        summ = _mae_metrics_summary(REPO / stamp / sub / f"fold{f}" / "metrics.jsonl")
        if summ.get("best_dice") is not None:
            val_d[str(f)] = float(summ["best_dice"])
        if summ.get("best_ep") is not None:
            ep_map[str(f)] = int(summ["best_ep"])
        elif summ.get("last_ep") is not None:
            ep_map[str(f)] = int(summ["last_ep"])
    if test20:
        st["fold_dice"] = test20
        st["eval_done"] = len(test20)
        st["eval_total"] = 9
        st["mean"] = _fold_score_mean(test20)
    if val_d:
        st["val_monitor_fold_dice"] = val_d
        st["val_monitor_mean"] = _fold_score_mean(val_d)
    if ep_map:
        st["fold_ckpt_ep"] = ep_map
    n_tr = len(ep_map)
    n_te = len(test20)
    note = str(st.get("note") or "")
    status = (st.get("status") or "").lower()
    stale = (
        note.upper().startswith("WAITING")
        or note.startswith("9fold extra")
        or "TEST20 3/9" in note
        or not note
    )
    if status == "done" and stale and n_tr:
        st["note"] = f"train {n_tr}/9 · TEST20 {n_te}/9 · extra VAL on board"


def _ingest_nnunet_test20_aggregate(nn: dict) -> bool:
    """If current stamp has a finished TEST20 aggregate, fill scores and mark done."""
    stamp = nn.get("stamp") or ""
    if not stamp:
        return False
    status0 = (nn.get("status") or "").lower()
    # queued rerun / pending must not re-ingest the previous stamp's TEST
    if status0 in ("queued", "pending"):
        return False
    if nn.get("test_invalidated") and status0 not in ("done", "running"):
        return False
    agg = CTRL / "ICLR2026/vis" / f"aggregate_nnunet_psma_fs50_f258_{stamp}.json"
    if not agg.is_file():
        for tag in ("fs10", "fs5", "fs50"):
            cand = CTRL / "ICLR2026/vis" / f"aggregate_nnunet_psma_{tag}_f258_{stamp}.json"
            if cand.is_file():
                agg = cand
                break
    if not agg.is_file():
        agg = CTRL / "ICLR2026/vis" / f"aggregate_nnunet_psma_fc70_{stamp}.json"
    if not agg.is_file():
        agg = NN_RESULTS / stamp / "aggregate_test20_dice_f258.json"
    if not agg.is_file():
        agg = NN_RESULTS / stamp / "aggregate_test20_dice_fc70.json"
    if not agg.is_file():
        return False
    try:
        ad = json.loads(agg.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if ad.get("single_run"):
        md = ad.get("mean")
        if md is None:
            md = ad.get("fold_mean")
        if isinstance(md, (int, float)) and md == md:
            nn["fold_dice"] = {"0": float(md)}
            nn["mean"] = float(md)
            nn["status"] = "done"
            nn["test_invalidated"] = False
            nn["phase"] = None
            nn["eval_done"] = 1
            nn["eval_total"] = 1
            _apply_fp_fn_from_agg(nn, ad)
            md_pos = _mean_dice_from_score_dict(ad)
            if md_pos is not None:
                nn["mean"] = float(md_pos)
                nn["fold_dice"] = {"0": float(md_pos)}
            nn["note"] = (
                f"TEST20 DONE · {_pct_fmt(nn['mean'], 2)}/"
                f"{_pct_fmt(nn.get('mean_fp'), 2)}/{_pct_fmt(nn.get('mean_fn'), 2)}"
                if _stage_fp_fn(nn)[0] is not None
                else f"TEST20 DONE · fc70 single · {nn['mean']:.3f}"
            )
            return True
    folds = ad.get("folds") or {}
    incoming: dict[str, float] = {}
    pending: list[str] = []
    for f in NINE_FOLD_STRS:
        fv = folds.get(f) or folds.get(int(f))
        if not isinstance(fv, dict):
            continue
        if fv.get("test_stale"):
            pending.append(f)
            continue
        d = fv.get("test_dice")
        if isinstance(d, (int, float)) and d == d:
            incoming[f] = float(d)
    if not incoming and not pending:
        return False
    fd = _merge_fold_scores(nn.get("fold_dice"), incoming)
    nn["fold_dice"] = fd
    nn["mean"] = _fold_score_mean(fd)
    n_ok = len(fd)
    nn["status"] = "done"
    nn["test_invalidated"] = False
    nn["phase"] = None
    nn["eval_done"] = n_ok
    nn["eval_total"] = 9
    if pending:
        nn["fold_test_pending"] = pending
        nn["note"] = f"TEST20 {n_ok}/9 · pending f{','.join(pending)}"
    elif n_ok >= 9:
        nn.pop("fold_test_pending", None)
        nn["note"] = "TEST20 DONE · 9/9"
    else:
        nn["note"] = f"TEST20 {n_ok}/9"
    if ad.get("ckpt"):
        nn["ckpt"] = ad["ckpt"]
    fep: dict[str, int] = {}
    for f in NINE_FOLD_STRS:
        fv = folds.get(f) or folds.get(int(f))
        if isinstance(fv, dict) and fv.get("ckpt_ep") is not None:
            try:
                fep[f] = int(fv["ckpt_ep"])
            except (TypeError, ValueError):
                pass
    for fk, fv in (ad.get("fold_ckpt_ep") or {}).items():
        try:
            fep[str(fk)] = int(fv)
        except (TypeError, ValueError):
            pass
    if fep:
        nn["fold_ckpt_ep"] = fep
    _apply_fp_fn_from_agg(nn, ad)
    if nn.get("mean_fp") is None or nn.get("mean_fn") is None:
        # Pool from per-fold score_detail under stamp (aggregate may predate FP/FN fields).
        eval_root = NN_RESULTS / stamp / "psma_test20_eval"
        scores: list[dict] = []
        if eval_root.is_dir():
            for f in NINE_FOLD_STRS:
                p = eval_root / f"fold{f}" / "score_detail.json"
                if not p.is_file():
                    continue
                try:
                    sd = json.loads(p.read_text())
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(sd.get("mean_dice"), (int, float)):
                    scores.append(sd)
        if scores:
            sum_fp = sum(int(s.get("sum_fp") or 0) for s in scores)
            sum_fn = sum(int(s.get("sum_fn") or 0) for s in scores)
            sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in scores)
            sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in scores)
            if sum_neg > 0:
                nn["mean_fp"] = float(sum_fp) / float(sum_neg)
            else:
                rates = [float(s["fp_rate"]) for s in scores if isinstance(s.get("fp_rate"), (int, float))]
                if rates:
                    nn["mean_fp"] = sum(rates) / float(len(rates))
            if sum_pos > 0:
                nn["mean_fn"] = float(sum_fn) / float(sum_pos)
            else:
                rates = [float(s["fn_rate"]) for s in scores if isinstance(s.get("fn_rate"), (int, float))]
                if rates:
                    nn["mean_fn"] = sum(rates) / float(len(rates))
    fp, fn = nn.get("mean_fp"), nn.get("mean_fn")
    if isinstance(nn.get("mean"), (int, float)) and isinstance(fp, (int, float)) and isinstance(fn, (int, float)):
        if n_ok >= 9:
            nn["note"] = (
                f"TEST20 DONE · 9/9 · {_pct_fmt(nn['mean'], 2)}/"
                f"{_pct_fmt(fp, 2)}/{_pct_fmt(fn, 2)}"
            )
        else:
            nn["note"] = (
                f"TEST20 {n_ok}/9 · {_pct_fmt(nn['mean'], 2)}/"
                f"{_pct_fmt(fp, 2)}/{_pct_fmt(fn, 2)}"
            )
    return True


def _ingest_dpdnet_test20_aggregate(dpd: dict) -> bool:
    stamp = dpd.get("stamp") or ""
    if not stamp:
        return False
    status0 = (dpd.get("status") or "").lower()
    agg = CTRL / "ICLR2026/vis" / f"aggregate_dpdnet_psma_test20_f258_{stamp}.json"
    if not agg.is_file():
        agg = CTRL / "ICLR2026/vis" / f"aggregate_dpdnet_psma_fc70_{stamp}.json"
    if not agg.is_file():
        agg = NN_RESULTS / stamp / "aggregate_test20_dice_f258.json"
    if not agg.is_file():
        agg = NN_RESULTS / stamp / "aggregate_test20_dice_fc70.json"
    if not agg.is_file():
        hits = sorted((CTRL / "ICLR2026/vis").glob(f"aggregate_dpdnet_psma_fc70_*{stamp}*.json"))
        if hits:
            agg = hits[-1]
    if not agg.is_file():
        return False
    # Pending fewshot must not pick up a leftover aggregate; fc70 single-run may.
    # Exception: current stamp's own aggregate (training finished, board status lagged).
    if status0 in ("queued", "pending") and "fc70" not in agg.name:
        try:
            _peek = json.loads(agg.read_text())
            _ps = str(_peek.get("parent_stamp") or "")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            _ps = ""
        if _ps != stamp and stamp not in agg.name:
            return False
    if dpd.get("test_invalidated") and status0 not in ("done", "running") and "fc70" not in agg.name:
        return False
    try:
        ad = json.loads(agg.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if ad.get("single_run"):
        md = ad.get("mean")
        if md is None:
            md = ad.get("fold_mean")
        if isinstance(md, (int, float)) and md == md:
            dpd["fold_dice"] = {"0": float(md)}
            dpd["mean"] = float(md)
            dpd["status"] = "done"
            dpd["test_invalidated"] = False
            dpd["phase"] = None
            dpd["eval_done"] = 1
            dpd["eval_total"] = 1
            dpd["note"] = f"TEST20 DONE · fc70 single · {md:.3f}"
            if isinstance(ad.get("mean_fp"), (int, float)):
                dpd["mean_fp"] = float(ad["mean_fp"])
            if isinstance(ad.get("mean_fn"), (int, float)):
                dpd["mean_fn"] = float(ad["mean_fn"])
            return True
    incoming: dict[str, float] = {}
    fep: dict[str, int] = {}
    for f in NINE_FOLD_STRS:
        fv = (ad.get("folds") or {}).get(f) or (ad.get("folds") or {}).get(int(f))
        if isinstance(fv, dict):
            d = fv.get("test_dice")
            if isinstance(d, (int, float)) and d == d:
                incoming[f] = float(d)
            if fv.get("ckpt_ep") is not None:
                try:
                    fep[f] = int(fv["ckpt_ep"])
                except (TypeError, ValueError):
                    pass
    for fk, fv in (ad.get("fold_ckpt_ep") or {}).items():
        try:
            fep[str(fk)] = int(fv)
        except (TypeError, ValueError):
            pass
    fd_raw = ad.get("fold_test_dice") or ad.get("fold_dice") or {}
    if isinstance(fd_raw, dict):
        for k, v in fd_raw.items():
            if isinstance(v, (int, float)) and v == v:
                incoming[str(k)] = float(v)
    if not incoming:
        return False
    fd = _merge_fold_scores(dpd.get("fold_dice"), incoming)
    dpd["fold_dice"] = fd
    dpd["mean"] = _fold_score_mean(fd)
    n_ok = len(fd)
    dpd["status"] = "done"
    dpd["test_invalidated"] = False
    dpd["phase"] = None
    dpd["eval_done"] = n_ok
    dpd["eval_total"] = 9
    dpd["note"] = f"TEST20 DONE · {n_ok}/9" if n_ok >= 3 else f"TEST20 {n_ok}/9"
    if ad.get("ckpt_policy"):
        dpd["metric"] = f"TEST20 Dice; best={ad['ckpt_policy'].split('=')[-1].strip()}" if "=" in str(ad["ckpt_policy"]) else "TEST20 Dice; best=max val Dice"
    else:
        dpd["metric"] = "TEST20 Dice; best=max val Pseudo dice"
    if ad.get("ckpt"):
        dpd["ckpt"] = ad["ckpt"]
    if fep:
        dpd["fold_ckpt_ep"] = fep
    return True


def _ingest_seganypet_test20_aggregate(seg: dict) -> bool:
    stamp = seg.get("stamp") or ""
    if not stamp:
        return False
    status0 = (seg.get("status") or "").lower()
    agg = CTRL / "ICLR2026/vis" / f"aggregate_seganypet_psma_test20_f258_{stamp}.json"
    if not agg.is_file():
        agg = REPO / stamp / "psma_test20_eval" / "aggregate_test20_f258.json"
    if not agg.is_file():
        return False
    if status0 in ("queued", "pending"):
        return False
    # Live train (no TEST20 yet) must not be overwritten by a stale aggregate
    if (
        status0 == "running"
        and (seg.get("phase") or "").lower() not in ("", "test", "test20", "decline")
    ):
        return False
    try:
        ad = json.loads(agg.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    fd_raw = ad.get("fold_test_dice") or ad.get("fold_dice") or {}
    incoming: dict[str, float] = {}
    if isinstance(fd_raw, dict):
        for k, v in fd_raw.items():
            if isinstance(v, (int, float)) and v == v:
                incoming[str(k)] = float(v)
    if not incoming:
        return False
    fd = _merge_fold_scores(seg.get("fold_dice"), incoming)
    seg["fold_dice"] = fd
    seg["mean"] = _fold_score_mean(fd) or float(
        ad.get("mean_dice_positive") or ad.get("test_mean") or ad.get("mean_dice") or 0
    )
    for src, dst in (
        ("fp_rate", "mean_fp"),
        ("fn_rate", "mean_fn"),
        ("mean_fp", "mean_fp"),
        ("mean_fn", "mean_fn"),
    ):
        v = ad.get(src)
        if isinstance(v, (int, float)) and v == v:
            seg[dst] = float(v)
    if not (
        isinstance(seg.get("mean_fp"), (int, float))
        and isinstance(seg.get("mean_fn"), (int, float))
    ):
        eval_root = REPO / stamp / "psma_test20_eval"
        sum_fp = sum_fn = sum_neg = sum_pos = 0
        n_sc = 0
        if eval_root.is_dir():
            for f in range(9):
                side = eval_root / f"fold{f}_score_fpfn.json"
                if not side.is_file():
                    continue
                try:
                    sc = json.loads(side.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                sum_fp += int(sc.get("sum_fp") or 0)
                sum_fn += int(sc.get("sum_fn") or 0)
                sum_neg += int(sc.get("sum_neg_voxels") or 0)
                sum_pos += int(sc.get("sum_pos_voxels") or 0)
                n_sc += 1
        if sum_neg > 0 and sum_pos > 0:
            seg["mean_fp"] = float(sum_fp) / float(sum_neg)
            seg["mean_fn"] = float(sum_fn) / float(sum_pos)

    seg["status"] = "done"
    seg["phase"] = None
    fp, fn = seg.get("mean_fp"), seg.get("mean_fn")
    if isinstance(fp, (int, float)) and isinstance(fn, (int, float)):
        seg["note"] = (
            f"TEST20 DONE · {100*float(seg['mean']):.2f}%/"
            f"{100*float(fp):.2f}%/{100*float(fn):.2f}%"
        )
        seg["metric"] = ad.get("metric") or "TEST20 Dice/FP/FN (final)"
    else:
        seg["note"] = f"TEST20 DONE · {len(fd)}/9"
    fep: dict[str, int] = {}
    for fk, fv in (ad.get("fold_ckpt_ep") or {}).items():
        try:
            fep[str(fk)] = int(fv)
        except (TypeError, ValueError):
            pass
    if fep:
        seg["fold_ckpt_ep"] = fep
    return True


def _nnunet_used_test_ckpt(stamp: str, fold: int) -> str | None:
    """Basename of ckpt actually used in the latest TEST20 for this stamp/fold."""
    sidecar = NN_RESULTS / stamp / "psma_test20_eval" / f"fold{fold}" / "ckpt_used.json"
    if sidecar.is_file():
        try:
            name = str(json.loads(sidecar.read_text()).get("ckpt") or "").strip()
            if name:
                return Path(name).name
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    log = CTRL / "ICLR2026/vis" / f"nohup_nnunet_test20_f{fold}_{stamp}.log"
    if not log.is_file():
        return None
    last = None
    try:
        for line in log.read_text(errors="ignore").splitlines():
            if "ckpt=" not in line or "checkpoint_" not in line:
                continue
            part = line.split("ckpt=", 1)[-1].split()[0].strip()
            if "checkpoint_" in part:
                last = Path(part).name
    except OSError:
        return None
    return last


def _recalc_psma_test_mean(board: dict) -> None:
    """TEST mean = arithmetic mean of fold 2/5/8 dice (when all present)."""
    ensure_methods(board)
    for mkey in METHOD_ORDER:
        m = board["methods"].get(mkey) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            st = m.get(stage) or {}
            if (st.get("status") or "").lower() not in ("done", "running", "paused"):
                continue
            fd = st.get("fold_dice") or {}
            vals = []
            if isinstance(fd, dict):
                for _k, v in fd.items():
                    if isinstance(v, (int, float)) and v == v:
                        vals.append(float(v))
            if len(vals) >= 1:
                st["mean"] = sum(vals) / float(len(vals))


def _pull_stamp_fold_test20(st: dict, stamp: str | None = None) -> None:
    """Merge fold0–8 TEST20 jsons from the run dir into fold_dice (does not drop extras)."""
    stamp = (stamp or st.get("stamp") or "").strip()
    if not stamp:
        return
    eval_root = REPO / stamp / "psma_test20_eval"
    if not eval_root.is_dir():
        return
    incoming: dict[str, float] = {}
    for f in NINE_FOLD_STRS:
        p = eval_root / f"fold{f}_test20.json"
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
            md = d.get("mean_dice_positive", d.get("mean_dice"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(md, (int, float)) and md == md:
            incoming[f] = float(md)
    if not incoming:
        return
    fd = _merge_fold_scores(st.get("fold_dice"), incoming)
    st["fold_dice"] = fd
    st["mean"] = _fold_score_mean(fd)
    n_ok = len(fd)
    st["eval_done"] = n_ok
    st["eval_total"] = 9


def _ingest_repo_test20_aggregate(st: dict, method: str) -> bool:
    """Fill fold_dice/mean from MAE/MONAI TEST20 aggregate when missing."""
    stamp = (st.get("stamp") or "").strip()
    if not stamp:
        return False
    status0 = (st.get("status") or "").lower()
    if status0 in ("queued", "pending"):
        return False
    agg = CTRL / "ICLR2026/vis" / f"aggregate_{method}_psma_test20_f258_{stamp}.json"
    if not agg.is_file():
        return False
    try:
        ad = json.loads(agg.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    fd_raw = ad.get("fold_test_dice") or ad.get("fold_dice") or {}
    incoming: dict[str, float] = {}
    if isinstance(fd_raw, dict):
        for k, v in fd_raw.items():
            if isinstance(v, (int, float)) and v == v:
                incoming[str(k)] = float(v)
    if len(incoming) < 1:
        return False
    fd = _merge_fold_scores(st.get("fold_dice"), incoming)
    st["fold_dice"] = fd
    n_fd = len(fd)
    st["mean"] = _fold_score_mean(fd)
    # Extra-fold still trains remaining folds; keep RUNNING instead of stamping DONE.
    if status0 != "running" and not _extra_fold_docker_live(stamp):
        st["status"] = "done"
    st["eval_done"] = n_fd
    st["eval_total"] = 9
    if ad.get("val_monitor_mean") is not None:
        st["val_monitor_mean"] = ad["val_monitor_mean"]
    if isinstance(ad.get("val_monitor_fold_dice"), dict):
        st["val_monitor_fold_dice"] = {str(k): v for k, v in ad["val_monitor_fold_dice"].items()}
    fep: dict[str, int] = {}
    for fk, fv in (ad.get("fold_ckpt_ep") or {}).items():
        try:
            fep[str(fk)] = int(fv)
        except (TypeError, ValueError):
            pass
    if fep:
        st["fold_ckpt_ep"] = fep
    if not st.get("metric"):
        st["metric"] = "TEST20 Dice (final)"
    if not st.get("note") or "mean=" in str(st.get("note")):
        st["note"] = f"TEST20 DONE · 3/3 · mean={st['mean']:.3f}"
    return True


def _fill_mae_family_ckpt_eps(st: dict, sub: str) -> None:
    """Fill fold_ckpt_ep from metrics.jsonl for mae/monai/seganypet stages."""
    stamp = (st.get("stamp") or "").strip()
    if not stamp:
        return
    tot = _stage_total_epochs(st) or 100
    st["total_epochs"] = tot
    ep_map: dict[str, int] = {}
    fold_root = REPO / stamp / sub
    fold_ids = []
    if fold_root.is_dir():
        for p in sorted(fold_root.glob("fold*")):
            if p.is_dir():
                try:
                    fold_ids.append(str(int(p.name.replace("fold", ""))))
                except ValueError:
                    pass
    if not fold_ids:
        fold_ids = ["2", "5", "8"]
    for f in fold_ids:
        metrics = REPO / stamp / sub / f"fold{f}" / "metrics.jsonl"
        ep = _mae_best_epoch_from_metrics(metrics)
        if ep is not None:
            ep_map[f] = ep
    if ep_map:
        st["fold_ckpt_ep"] = ep_map


def _enrich_nnunet_psma_stage(nn: dict) -> None:
    ingested = _ingest_nnunet_test20_aggregate(nn)
    if ingested or (
        (nn.get("status") or "").lower() == "done"
        and (nn.get("mean") is not None or (nn.get("fold_dice") or {}))
    ):
        nn["test_invalidated"] = False
    if (not ingested) and (
        (nn.get("status") or "").lower() in ("pending", "queued")
        or (nn.get("test_invalidated") and (nn.get("status") or "").lower() != "done")
    ):
        nn["fold_ckpt_ep"] = {}
        nn["fold_dice"] = {}
        nn["mean"] = None
        nn["test_live"] = None
        nn["phase"] = None
        nn["eval_done"] = None
        nn["eval_total"] = None
        return
    stamp = nn.get("stamp") or ""
    tot = _stage_total_epochs(nn) or 300
    nn["total_epochs"] = tot
    ckpt_name = (nn.get("ckpt") or "").lower()
    agg = CTRL / "ICLR2026/vis" / f"aggregate_nnunet_psma_fs50_f258_{stamp}.json"
    if not agg.is_file():
        agg = NN_RESULTS / stamp / "aggregate_test20_dice_f258.json"
    agg_ep: dict[str, int] = {}
    if agg.is_file():
        try:
            ad = json.loads(agg.read_text())
            if ad.get("ckpt"):
                ckpt_name = str(ad["ckpt"]).lower()
                nn["ckpt"] = ad["ckpt"]
            for fk, fv in (ad.get("folds") or {}).items():
                if isinstance(fv, dict) and fv.get("ckpt_ep") is not None:
                    agg_ep[str(fk)] = int(fv["ckpt_ep"])
            for fk, fv in (ad.get("fold_ckpt_ep") or {}).items():
                agg_ep[str(fk)] = int(fv)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if len(agg_ep) >= 3:
        nn["fold_ckpt_ep"] = {f: agg_ep[f] for f in ("2", "5", "8") if f in agg_ep}
        return
    fold_ep: dict[str, int] = dict(agg_ep)
    used_names: list[str] = []
    for f in ("2", "5", "8"):
        fd = NN_RESULTS / f"{stamp}_f{f}" / DS228 / TF228 / "fold_0"
        used = _nnunet_used_test_ckpt(stamp, int(f))
        if used:
            used_names.append(used)
        if used and "final" in used.lower():
            fold_ep[f] = tot
            continue
        if f in fold_ep:
            continue
        name = (used or ckpt_name or "").lower()
        if "final" in name:
            fold_ep[f] = tot
            continue
        if not fd.is_dir():
            continue
        ep = _nnunet_ema_best_epoch(fd)
        if ep is not None:
            fold_ep[f] = ep
        elif (fd / "checkpoint_final.pth").is_file():
            fold_ep[f] = tot
    if fold_ep:
        nn["fold_ckpt_ep"] = fold_ep
        if any("final" in x.lower() for x in used_names) or "final" in ckpt_name:
            nn["ckpt"] = "checkpoint_final.pth"
        elif not nn.get("ckpt"):
            nn["ckpt"] = "checkpoint_best.pth"


def enrich_test_ckpt_epochs(board: dict) -> None:
    """Fill fold_dice + fold_ckpt_ep for all PSMA fs50/fs10/fs5 stages."""
    ensure_methods(board)
    methods = board.get("methods") or {}

    # --- nnUNet (+ competition rows that reuse nnUNet TEST20 aggregate layout) ---
    for _nn_key in (
        "nnunet",
        "nnunet_mim",
        "hemingduo_scratch",
        "chenyixin_scratch",
        "hemingduo",
        "chenyixin",
    ):
        nn_m = methods.get(_nn_key) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            st = nn_m.get(stage)
            if isinstance(st, dict):
                _enrich_nnunet_psma_stage(st)

    # --- MAE / MONAI / SegAnyPET ---
    for key, sub, ingest_name in (
        ("mae_swinunetr", "mae", "mae"),
        ("mae_scratch", "mae", "mae_scratch"),
        ("monai_swinvit", "monai", "monai"),
        ("monai_scratch", "monai", "monai_scratch"),
        ("seganypet", "seganypet", "seganypet"),
        ("seganypet_scratch", "seganypet", "seganypet_scratch"),
    ):
        m = methods.get(key) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            st = m.get(stage)
            if not isinstance(st, dict):
                continue
            status = (st.get("status") or "").lower()
            if status in ("pending", "queued"):
                continue
            if key in ("seganypet", "seganypet_scratch"):
                _ingest_seganypet_test20_aggregate(st)
            else:
                # refill missing dice from TEST20 aggregate (fs10/fs5 often only have mean)
                need_dice = len([v for v in (st.get("fold_dice") or {}).values() if isinstance(v, (int, float))]) < 9
                if need_dice:
                    _ingest_repo_test20_aggregate(st, ingest_name)
            _pull_stamp_fold_test20(st)
            _sync_mae_family_fold_displays(st, sub)
            _fill_mae_family_ckpt_eps(st, sub)
            if key in ("monai_swinvit", "monai_scratch") and not st.get("ckpt"):
                st["ckpt"] = "best_seg_*.pth"
            if key in ("seganypet", "seganypet_scratch") and not st.get("ckpt"):
                st["ckpt"] = "best.pth"
    _apply_live_extra_fold_running(board)

    # --- DpDNet: all few-shot stages ---
    for _dpd_key in ("dpdnet", "dpdnet_dualenc"):
        dpd_m = methods.get(_dpd_key) or {}
        for _n, stage, _short in FEWSHOT_VARIANTS:
            dpd = dpd_m.get(stage)
            if not isinstance(dpd, dict):
                continue
            dstatus = (dpd.get("status") or "").lower()
            d_ingested = _ingest_dpdnet_test20_aggregate(dpd)
            if d_ingested or (
                dstatus == "done" and (dpd.get("mean") is not None or (dpd.get("fold_dice") or {}))
            ):
                dpd["test_invalidated"] = False
            if (not d_ingested) and (
                dstatus in ("pending", "queued", "paused")
                or (dpd.get("test_invalidated") and dstatus != "done")
            ):
                dpd["fold_ckpt_ep"] = {}
                dpd["fold_dice"] = {}
                dpd["mean"] = None
                dpd["test_live"] = None
                dpd["phase"] = None
                dpd["eval_done"] = None
                dpd["eval_total"] = None
                continue
            if dstatus in ("pending", "queued"):
                continue
            dstamp = dpd.get("stamp") or ""
            dtot = _stage_total_epochs(dpd) or 100
            dpd["total_epochs"] = dtot
            d_existing = dpd.get("fold_ckpt_ep") if isinstance(dpd.get("fold_ckpt_ep"), dict) else {}
            d_ep: dict[str, int] = {}
            ckpt_name = (dpd.get("ckpt") or "").lower()
            agg_d = CTRL / "ICLR2026/vis" / f"aggregate_dpdnet_psma_test20_f258_{dstamp}.json"
            if not agg_d.is_file():
                agg_d = NN_RESULTS / dstamp / "aggregate_test20_dice_f258.json"
            if agg_d.is_file():
                try:
                    ad = json.loads(agg_d.read_text())
                    if ad.get("ckpt"):
                        ckpt_name = str(ad["ckpt"]).lower()
                        dpd["ckpt"] = ad["ckpt"]
                    for fk, fv in (ad.get("folds") or {}).items():
                        if isinstance(fv, dict) and fv.get("ckpt_ep") is not None:
                            try:
                                d_ep[str(fk)] = int(fv["ckpt_ep"])
                            except (TypeError, ValueError):
                                pass
                    for fk, fv in (ad.get("fold_ckpt_ep") or {}).items():
                        try:
                            d_ep[str(fk)] = int(fv)
                        except (TypeError, ValueError):
                            pass
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            for f in ("2", "5", "8"):
                if f in d_ep:
                    continue
                if d_existing.get(f) is not None:
                    try:
                        d_ep[f] = int(d_existing[f])
                        continue
                    except (TypeError, ValueError):
                        pass
                if "final" in ckpt_name:
                    d_ep[f] = dtot
                    continue
                fd = _dpdnet_psma_fold_dir(dstamp, int(f))
                if fd is None:
                    continue
                ep = _nnunet_pseudo_dice_best_epoch(fd)
                if ep is None:
                    ep = _nnunet_ema_best_epoch(fd)
                if ep is None:
                    ep = _nnunet_val_loss_best_epoch(fd)
                if ep is not None:
                    d_ep[f] = ep
            if d_ep:
                dpd["fold_ckpt_ep"] = d_ep
                if not dpd.get("ckpt"):
                    dpd["ckpt"] = "checkpoint_best.pth"

    # --- PSMA fs0 + FDG TEST: training-free eval columns ---
    for mkey in METHOD_ORDER:
        st0 = (methods.get(mkey) or {}).get(PSMA_FS0_STAGE_KEY)
        if isinstance(st0, dict):
            _ingest_mean_only_stage(st0, mkey, PSMA_FS0_AGG_DIR, PSMA_FS0_STAGE_KEY)
        stf = (methods.get(mkey) or {}).get(FDG_TEST_STAGE_KEY)
        if isinstance(stf, dict):
            _ingest_mean_only_stage(stf, mkey, FDG_TEST_AGG_DIR, FDG_TEST_STAGE_KEY)

    _ingest_fc70_test20_stages(methods)
    _ingest_proto_retrieval_fewshot_stages(methods.get("proto_retrieval") or {})
    _ingest_proto_retrieval_fc70(methods.get("proto_retrieval") or {})


def _finalize_fc70_test20_cell(st: dict) -> None:
    """Keep TEST20 scores; drop leftover decline/idle-queue overlay."""
    if not _stage_has_score(st):
        return
    st["status"] = "done"
    st["test_invalidated"] = False
    st["phase"] = None
    st.pop("test_live", None)
    st["eta"] = None
    st["eta_sec"] = None
    try:
        ep = int(st["epoch"]) if st.get("epoch") is not None else None
    except (TypeError, ValueError):
        ep = None
    if ep is not None and ep > 400:
        st["epoch"] = None
        try:
            if int(st.get("total_epochs") or 0) > 400:
                st["total_epochs"] = 100
        except (TypeError, ValueError):
            pass
    note = str(st.get("note") or "")
    md = st.get("mean")
    if (
        note.startswith("running")
        or "gpu-idle-queue" in note
        or note.startswith("decline")
        or not note
        or "TEST20" not in note
    ) and isinstance(md, (int, float)):
        fp, fn = st.get("mean_fp"), st.get("mean_fn")
        if isinstance(fp, (int, float)) and isinstance(fn, (int, float)):
            st["note"] = (
                f"TEST20 DONE · {100 * float(md):.2f}%/"
                f"{100 * float(fp):.2f}%/{100 * float(fn):.2f}%"
            )
        else:
            st["note"] = f"TEST20 DONE · fc70 single · {float(md):.3f}"


def _ingest_fc70_test20_stages(methods: dict) -> None:
    """Fill missing PSMA fc70% TEST20 cells. Do not clobber complete Dice/FP/FN."""
    for mkey in METHOD_ORDER:
        st = (methods.get(mkey) or {}).get(FC70_STAGE_KEY)
        if not isinstance(st, dict):
            continue
        fp0, fn0 = _stage_fp_fn(st)
        if _stage_has_score(st) and fp0 is not None and fn0 is not None:
            continue
        if mkey in ("dpdnet", "dpdnet_dualenc"):
            _ingest_dpdnet_test20_aggregate(st)
        elif mkey in ("seganypet", "seganypet_scratch"):
            _ingest_seganypet_test20_aggregate(st)
            _pull_stamp_fold_test20(st)
        elif mkey in ("nnunet", "nnunet_mim", "hemingduo_scratch", "chenyixin_scratch", "hemingduo", "chenyixin"):
            _ingest_nnunet_test20_aggregate(st)
        elif mkey in ("mae_swinunetr", "mae_scratch"):
            _ingest_repo_test20_aggregate(st, "mae_scratch" if mkey == "mae_scratch" else "mae")
            _pull_stamp_fold_test20(st)
        elif mkey in ("monai_swinvit", "monai_scratch"):
            _ingest_repo_test20_aggregate(st, "monai_scratch" if mkey == "monai_scratch" else "monai")
            _pull_stamp_fold_test20(st)
        if _stage_has_score(st):
            _finalize_fc70_test20_cell(st)


def _ingest_proto_retrieval_fc70(proto_m: dict) -> None:
    """Ingest Proto PSMA fc70% aggregate (Dice/FP/FN)."""
    if not isinstance(proto_m, dict):
        return
    st = proto_m.get(FC70_STAGE_KEY)
    if not isinstance(st, dict):
        return
    aggs = sorted((CTRL / "ICLR2026/vis").glob("aggregate_proto_retrieval_psma_fc70_*.json"))
    if not aggs:
        return
    try:
        ad = json.loads(aggs[-1].read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    md = _mean_dice_from_score_dict(ad)
    if md is None:
        return
    st["status"] = "done"
    st["training_free"] = True
    st["stamp"] = ad.get("stamp") or st.get("stamp") or ""
    st["fold_dice"] = {"0": float(md)}
    st["mean"] = float(md)
    st["support_pool"] = ad.get("support_pool") or "PSMA70%"
    if ad.get("topk") is not None:
        st["topk"] = ad["topk"]
    _apply_fp_fn_from_agg(st, ad)
    st["note"] = (
        f"TEST20 DONE · {st['support_pool']} · "
        f"{_pct_fmt(md, 2)}/{_pct_fmt(st.get('mean_fp'), 2)}/{_pct_fmt(st.get('mean_fn'), 2)}"
    )
    st["metric"] = ad.get("metric") or "TEST20 Dice/FP/FN; retrieve PSMA70% + prototype"


def _ingest_proto_retrieval_fewshot_stages(proto_m: dict) -> None:
    """Proto fs50/fs10/fs5/fs0 share FDG100% retrieval (training-free) — mirror fs50 eval."""
    if not isinstance(proto_m, dict):
        return
    src = proto_m.get("psma_fs50_f258") or {}
    if not _stage_has_score(src):
        for agg in sorted((CTRL / "ICLR2026/vis").glob("aggregate_proto_retrieval_psma_test20_f258_*.json")):
            try:
                ad = json.loads(agg.read_text())
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            fd = ad.get("fold_test_dice") or ad.get("fold_dice") or {}
            mean = ad.get("test_mean") or ad.get("mean")
            if not isinstance(mean, (int, float)) and fd:
                vals = [float(v) for v in fd.values() if isinstance(v, (int, float))]
                mean = sum(vals) / len(vals) if vals else None
            if not isinstance(mean, (int, float)):
                continue
            src = {
                "stamp": ad.get("stamp", ""),
                "fold_dice": {str(k): float(v) for k, v in fd.items() if isinstance(v, (int, float))},
                "mean": float(mean),
                "support_pool": ad.get("support_pool") or "FDG100%",
                "topk": ad.get("topk", 3),
                "mean_fp": ad.get("mean_fp", ad.get("fp_rate")),
                "mean_fn": ad.get("mean_fn", ad.get("fn_rate")),
            }
            break
    if not _stage_has_score(src):
        return
    for stage in ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258", PSMA_FS0_STAGE_KEY):
        st = proto_m.get(stage)
        if not isinstance(st, dict):
            continue
        # Always keep Proto few-shot / fs0 columns aligned with FDG100% TEST20.
        st["status"] = "done"
        st["training_free"] = True
        st["stamp"] = src.get("stamp") or st.get("stamp") or ""
        st["fold_dice"] = deepcopy(src.get("fold_dice") or {})
        st["mean"] = src.get("mean")
        st["support_pool"] = src.get("support_pool") or "FDG100%"
        if src.get("topk") is not None:
            st["topk"] = src["topk"]
        if isinstance(src.get("mean_fp"), (int, float)):
            st["mean_fp"] = float(src["mean_fp"])
        if isinstance(src.get("mean_fn"), (int, float)):
            st["mean_fn"] = float(src["mean_fn"])
        mean_f = float(st["mean"])
        triple = f"{_pct_fmt(mean_f, 2)}/{_pct_fmt(st.get('mean_fp'), 2)}/{_pct_fmt(st.get('mean_fn'), 2)}"
        if stage == PSMA_FS0_STAGE_KEY:
            st["note"] = f"same as fs50 · FDG100% · {triple}"
            st["metric"] = "TEST20 Dice/FP/FN; retrieve FDG100% + prototype"
        else:
            st["note"] = f"TEST20 DONE · FDG100% · topk={st.get('topk', 3)} · {triple}"


def _ingest_mean_only_stage(
    st: dict, method_key: str, agg_dir: Path, stage_key: str
) -> bool:
    """Ingest single-run mean Dice from eval aggregate JSON."""
    if stage_key == FDG_TEST_STAGE_KEY:
        pats = STAGE_RUNTIME_PATTERNS.get((method_key, stage_key), ())
        if pats and _pgrep_any(pats):
            # Live FDG TEST / rescore on GPU — do not overwrite RUNNING with stale aggregate Dice/FP/FN
            return False
    agg = agg_dir / f"aggregate_{method_key}.json"
    if not agg.is_file() and stage_key == PSMA_FS0_STAGE_KEY and method_key == "nnunet":
        # Fallback: baseline FDG-only zero-shot on PSMA val (~0.143)
        baseline = CTRL / "ICLR2026/vis" / (
            "aggregate_baseline1_fdg_eval_psma_9fold_20260816_002228_"
            "iclr2026_baseline1_fdg_eval_psma_9fold_gpu013.json"
        )
        if baseline.is_file():
            try:
                bd = json.loads(baseline.read_text())
                md = _mean_dice_from_score_dict(bd)
                if md is None and isinstance(bd.get("fold_mean"), (int, float)) and bd["fold_mean"] == bd["fold_mean"]:
                    md = float(bd["fold_mean"])
                if md is not None:
                    st["fold_dice"] = {"0": float(md)}
                    st["mean"] = float(md)
                    st["status"] = "done"
                    st["training_free"] = True
                    st["note"] = f"PSMA fs0 · baseline proxy · {float(md):.3f}"
                    st["support_pool"] = "FDG shared"
                    if bd.get("eval_stamp"):
                        st["stamp"] = bd["eval_stamp"]
                    return True
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
    if not agg.is_file():
        return False
    try:
        ad = json.loads(agg.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    md = _mean_dice_from_score_dict(ad)
    if md is None:
        n_scored = ad.get("n_scored")
        try:
            n_ok = int(n_scored) if n_scored is not None else 0
        except (TypeError, ValueError):
            n_ok = 0
        if stage_key == FDG_TEST_STAGE_KEY and agg.is_file() and n_ok <= 0:
            st["status"] = "pending"
            st["mean"] = None
            st["fold_dice"] = {}
            st["phase"] = None
            st.pop("test_live", None)
            st["note"] = "FDG TEST FAILED · 0/202 · rerun queued"
            return False
        return False
    st["fold_dice"] = {"0": float(md)}
    st["mean"] = float(md)
    st["status"] = "done"
    st["training_free"] = True
    st["phase"] = None
    st.pop("test_live", None)
    st["eval_done"] = 1
    st["eval_total"] = 1
    _apply_fp_fn_from_agg(st, ad)
    if ad.get("ckpt"):
        st["ckpt"] = ad["ckpt"]
    if ad.get("eval_stamp"):
        st["stamp"] = ad["eval_stamp"]
    elif ad.get("stamp"):
        st["stamp"] = ad["stamp"]
    note_md = _pct_fmt(float(md), 2)
    fp_s, fn_s = _pct_fmt(st.get("mean_fp"), 2), _pct_fmt(st.get("mean_fn"), 2)
    if stage_key == FDG_TEST_STAGE_KEY and method_key == "proto_retrieval":
        pool = ad.get("support_pool") or "FDG70%"
        st["note"] = f"FDG TEST DONE · {pool} · {note_md}/{fp_s}/{fn_s}"
        st["support_pool"] = pool
        if ad.get("topk") is not None:
            st["topk"] = ad["topk"]
        if ad.get("n_positive") is not None:
            st["n_test"] = ad["n_positive"]
        elif ad.get("n_scored") is not None:
            st["n_test"] = ad["n_scored"]
    elif stage_key == PSMA_FS0_STAGE_KEY:
        st["note"] = f"PSMA fs0 · FDG ckpt · {note_md}/{fp_s}/{fn_s}"
        st["support_pool"] = ad.get("support_pool") or "FDG shared"
    else:
        st["note"] = f"FDG TEST · {note_md}/{fp_s}/{fn_s}"
        st["support_pool"] = ad.get("support_pool") or "FDG shared"
    return True


def _fdg_board_label(st: dict) -> str:
    """FDG column: static final state (shared across all few-shot sizes). Multi-line for DONE."""
    status = (st.get("status") or "pending").lower()
    if status in ("n/a", "na"):
        pool = st.get("support_pool") or "FDG100%"
        return f"n/a\n{pool}"
    if status == "done":
        lines = ["DONE"]
        tt = st.get("train_time")
        if tt:
            lines.append(str(tt))
        be = st.get("best_ep")
        if be is not None:
            try:
                lines.append(f"best@{int(be)}")
            except (TypeError, ValueError):
                pass
        return "\n".join(lines)
    if status in ("running", "paused", "waiting"):
        return _status_label_multiline(st)
    return status.upper()


def _status_label_multiline(st: dict) -> str:
    """Wrap DONE/RUNNING/WAITING bits onto lines so status stays complete (no truncation)."""
    status = (st.get("status") or "pending").upper()
    if status == "DONE":
        lines = ["DONE"]
        tt = st.get("train_time")
        if tt:
            lines.append(str(tt))
        return "\n".join(lines)
    if status == "WAITING":
        gpu = st.get("wait_gpu", "?")
        eta = st.get("wait_eta") or ""
        if eta:
            return f"WAITING (GPU {gpu})\n{eta}"
        return _waiting_status_head(st)
    # reuse single-line builder then split on · for long RUNNING labels
    one = _status_label(st)
    if " · " not in one:
        return one
    parts = [p.strip() for p in one.split(" · ") if p.strip()]
    if len(parts) <= 1:
        return one
    # first line = STATUS (+ optional phase); rest one-per-line
    return "\n".join(parts)


def _draw_running_cell_bg(ax, cx: float, cy: float, w: float, h: float) -> None:
    """Light orange tint behind RUNNING cells."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=1.0,
            edgecolor="#ef6c00",
            facecolor="#fff3e0",
            alpha=0.92,
            zorder=0,
        )
    )


def _draw_waiting_cell_bg(ax, cx: float, cy: float, w: float, h: float) -> None:
    """Light blue tint behind WAITING (GPU idle 1min countdown) cells."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            linewidth=1.0,
            edgecolor="#0277bd",
            facecolor="#e3f2fd",
            alpha=0.92,
            zorder=0,
        )
    )


def _render_psma_stage_cell(
    ax,
    st: dict,
    x: float,
    y: float,
    col_max: int,
    _fit,
    stage_key: str = "",
) -> None:
    status = st.get("status", "pending")
    color = STATUS_COLOR.get(status, "#757575")
    note = st.get("note") or ""
    is_running = status in ("running", "paused") or bool(
        isinstance(st.get("test_live"), dict) and st["test_live"].get("active")
    )
    is_waiting = status == "waiting"
    if is_running and status not in ("paused",):
        _draw_running_cell_bg(ax, x, y, 0.62, 0.52)
    elif is_waiting:
        _draw_waiting_cell_bg(ax, x, y, 0.62, 0.52)
    # Green DONE (+ train_time) wraps; do not _fit-truncate the status text
    label = _status_label_multiline(st)
    n_lines = label.count("\n") + 1
    ax.text(
        x,
        y + (0.20 if n_lines >= 2 else 0.16),
        label,
        fontsize=7.2 if status in ("running", "paused", "waiting") else 7.8,
        color=color,
        fontweight="bold",
        va="center",
        linespacing=1.05,
        clip_on=True,
    )
    training_free = bool(st.get("training_free"))
    gbs = None if training_free else _stage_gbs(st)
    tot_ep = 0 if training_free else _stage_total_epochs(st)
    bits = []
    if gbs is not None:
        bits.append(f"gbs={gbs}")
    if tot_ep:
        bits.append(f"{tot_ep}ep")
    if training_free:
        bits = [st.get("support_pool") or "FDG100%"]
    mid = _fit(" · ".join(bits), col_max)
    mid_y = y - (0.22 if n_lines >= 2 else 0.10)
    if mid:
        ax.text(x, mid_y, mid, fontsize=6, color="#424242", va="center", clip_on=True)
    if note and status in ("paused", "pending", "queued") and len(str(note)) <= 28:
        ax.text(x, y - 0.32, _fit(str(note), col_max), fontsize=5.5, color="#616161", va="center", clip_on=True)


def _render_mean_only_cell(
    ax,
    st: dict,
    x: float,
    y: float,
    cell_w: float = 0.88,
    method_key: str = "",
    stage_key: str = "",
) -> None:
    """Single merged column: Dice / FP / FN (%) for fc70% / PSMA fs0 / FDG TEST."""
    ps_status = (st.get("status") or "pending").lower()
    mean = st.get("mean")
    fd = st.get("fold_dice") or {}
    val: float | None = None
    if isinstance(mean, (int, float)) and mean == mean:
        val = float(mean)
    else:
        v0 = fd.get("0")
        if isinstance(v0, (int, float)) and v0 == v0:
            val = float(v0)
    live = st.get("test_live") if isinstance(st.get("test_live"), dict) else {}
    pats = STAGE_RUNTIME_PATTERNS.get((method_key, stage_key), ()) if method_key and stage_key else ()
    runtime = bool(pats and _pgrep_any(pats))
    is_running = ps_status == "running" or bool(live.get("active")) or runtime
    is_waiting = ps_status == "waiting"
    if is_running:
        _draw_running_cell_bg(ax, x, y, cell_w, 0.52)
    elif is_waiting:
        _draw_waiting_cell_bg(ax, x, y, cell_w, 0.52)
    if ps_status in ("n/a", "na"):
        text, color, fs = "n/a", "#9e9e9e", 9
    elif is_waiting:
        head = _waiting_status_head(st)
        # split "WAITING (GPU 3 · 7m12s)" onto two lines when possible
        if " · " in head:
            a, b = head.split(" · ", 1)
            text = f"{a})\n{b}" if a.endswith("(") else f"{a}\n{b}"
            # cleaner: WAITING (GPU 3) / 7m12s
            gpu = st.get("wait_gpu", "?")
            eta = st.get("wait_eta") or ""
            text = f"WAITING (GPU {gpu})\n{eta}" if eta else f"WAITING (GPU {gpu})"
        else:
            text = head
        color, fs = "#0277bd", 7.4
    elif is_running:
        head = _running_status_head(st)
        eta = live.get("eta") or st.get("eta")
        ep, tot = st.get("epoch"), st.get("total_epochs")
        pred_bit = None
        if live.get("cases_done") is not None and live.get("cases_total"):
            pred_bit = f"pred {live['cases_done']}/{live['cases_total']}"
        elif "pred " in str(st.get("note") or ""):
            frag = str(st.get("note")).split("pred ", 1)[-1].strip()
            pred_bit = "pred " + frag.split("·")[0].strip()
        if pred_bit:
            text = f"{head}\n{pred_bit}"
        elif eta:
            text = f"{head}\nETA {eta}"
        elif ep is not None and tot:
            text = f"{head}\nep{ep}/{tot}"
        else:
            text = head
        color, fs = "#ef6c00", 7.6
    elif ps_status in ("pending", "queued"):
        text, color, fs = "—", "#9e9e9e", 10
    elif val is not None:
        text, color, fs = _metrics_cell_text(st, val), "#212121", 8.5
    else:
        text, color, fs = "—", "#9e9e9e", 10
    ax.text(
        x,
        y + (0.10 if "\n" in text else 0),
        text,
        fontsize=fs,
        fontweight="bold",
        va="center",
        ha="center",
        family="monospace",
        color=color,
        linespacing=1.05,
    )


def _render_test_block(
    ax,
    st_ps: dict,
    x_test: float,
    x_mean: float,
    y: float,
    _fit,
    stage_key: str = "",
) -> None:
    if stage_key in (FC70_STAGE_KEY, PSMA_FS0_STAGE_KEY, FDG_TEST_STAGE_KEY):
        return
    fd = st_ps.get("fold_dice") or {}
    fep = st_ps.get("fold_ckpt_ep") or {}
    live = st_ps.get("test_live") if isinstance(st_ps.get("test_live"), dict) else {}
    live_active = bool(live.get("active"))
    live_folds = live.get("folds") if isinstance(live.get("folds"), dict) else {}
    test_pending = set(st_ps.get("fold_test_pending") or [])
    ps_status = (st_ps.get("status") or "").lower()
    paused = ps_status == "paused"
    single_run = stage_key in (FC70_STAGE_KEY, PSMA_FS0_STAGE_KEY, FDG_TEST_STAGE_KEY)
    if single_run:
        parts = []
        if paused or ps_status in ("pending", "queued"):
            parts.append("—")
        else:
            v = fd.get("0") if isinstance(fd.get("0"), (int, float)) else st_ps.get("mean")
            if isinstance(v, (int, float)):
                ep_v = fep.get("0")
                parts.append(f"{_pct_fmt(v, 2)}[ep{int(ep_v)}]" if ep_v is not None else _pct_fmt(v, 2))
            else:
                parts.append("—")
        ax.text(x_test, y, "\n".join(parts), fontsize=7.5, va="center", ha="left", family="monospace")
        mean = st_ps.get("mean")
        if isinstance(mean, (int, float)) and mean == mean:
            mean_s = _metrics_cell_text(st_ps, float(mean))
        else:
            mean_s = "—"
        ax.text(
            x_mean,
            y + (0.08 if "\n" in mean_s else 0),
            mean_s,
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="left",
            family="monospace",
            linespacing=1.02,
        )
        return
    cells: list[str] = []
    for f in NINE_FOLD_STRS:
        if paused or ps_status in ("pending", "queued"):
            cells.append(f"f{f}   ·    ")
            continue
        lf = live_folds.get(f) if live_active else None
        if live_active and isinstance(lf, dict) and lf.get("dice") is None:
            done_i = lf.get("done")
            tot_i = lf.get("total") or 120
            prog = f"{done_i}/{tot_i}" if done_i is not None else "…"
            cells.append(f"f{f} {prog:<7}")
            continue
        v = fd.get(f)
        if v is None and isinstance(fd, dict):
            try:
                v = fd.get(int(f))
            except (TypeError, ValueError):
                v = None
        if v is None and f not in test_pending and isinstance(lf, dict) and isinstance(lf.get("dice"), (int, float)):
            v = lf["dice"]
        ep_v = fep.get(f) if isinstance(fep, dict) else None
        src = ""
        if not (isinstance(v, (int, float)) and v == v):
            vd = (st_ps.get("val_monitor_fold_dice") or {}).get(f)
            if vd is None and isinstance(st_ps.get("val_monitor_fold_dice"), dict):
                try:
                    vd = st_ps["val_monitor_fold_dice"].get(int(f))
                except (TypeError, ValueError):
                    vd = None
            if isinstance(vd, (int, float)) and vd == vd:
                v = vd
                src = "~"
        if isinstance(v, (int, float)) and v == v:
            body = f"{src}{100.0 * float(v):4.1f}"
            if ep_v is not None:
                try:
                    body += f"/{int(ep_v)}"
                except (TypeError, ValueError):
                    pass
            cells.append(f"f{f} {body:<7}")
        elif ep_v is not None:
            cells.append(f"f{f} tr/{int(ep_v):<4}")
        else:
            cells.append(f"f{f}   ·    ")
    lines = [" ".join(cells[i : i + 3]) for i in range(0, 9, 3)]
    ax.text(
        x_test,
        y,
        "\n".join(lines),
        fontsize=6.2,
        va="center",
        ha="left",
        family="monospace",
        linespacing=1.15,
        color=(
            STATUS_COLOR["paused"]
            if paused
            else ("#ef6c00" if live_active else "#212121")
        ),
    )
    mean = st_ps.get("mean")
    if paused:
        mean_s, mcolor, mfs = "PAUSED", STATUS_COLOR["paused"], 9
    elif live_active:
        eta_m = live.get("eta") or st_ps.get("eta")
        mean_s, mcolor, mfs = (f"ETA {eta_m}" if eta_m else "ETA …"), "#ef6c00", 9
    elif ps_status in ("pending", "queued"):
        mean_s, mcolor, mfs = "—", "#9e9e9e", 10
    elif isinstance(mean, (int, float)) and mean == mean:
        mean_s = _metrics_cell_text(st_ps, float(mean))
        mcolor, mfs = "#212121", 8
    else:
        mean_s, mcolor, mfs = "—", "#9e9e9e", 10
    ax.text(
        x_mean,
        y + (0.08 if "\n" in str(mean_s) else 0),
        mean_s,
        fontsize=mfs,
        fontweight="bold",
        va="center",
        ha="left",
        family="monospace",
        color=mcolor,
        linespacing=1.02,
    )


def render_png(board: dict, png: Path) -> None:
    if not _HAS_MPL:
        img = "iclr2026_3dmae_petct:cu118"
        # Re-enter this script inside the image (has matplotlib).
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{CTRL}:{CTRL}",
            # host CJK fonts (container matplotlib has Latin-only DejaVu)
            "-v",
            "/usr/share/fonts/opentype/noto:/usr/share/fonts/opentype/noto:ro",
            img,
            "python3",
            str(Path(__file__).resolve()),
            "--board",
            str(board.get("_board_path") or DEFAULT_BOARD),
            "--png",
            str(png),
            "--plot-only",
        ]
        # board already saved on disk; plot-only loads it
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:] if r.stderr else r.stdout[-2000:])
            raise RuntimeError(f"docker plot failed rc={r.returncode}")
        return

    ensure_methods(board)
    enrich_test_ckpt_epochs(board)
    enrich_fdg_best_epochs(board)
    _apply_live_extra_fold_running(board)
    # Do NOT board_self_check here: docker plot cannot pgrep host jobs and would
    # falsely clear RUNNING cells. Host once() saves health_check + status first.
    refresh_running_etas(board)
    refresh_stage_train_times(board)
    methods = [(k, board["methods"][k]) for k in METHOD_ORDER if k in board["methods"]]
    running_cols: set[str] = set()
    waiting_cols: set[str] = set()
    for _k, m in methods:
        for _n, stage_key, short in PSMA_BOARD_COLUMNS:
            st = m.get(stage_key) or {}
            status = (st.get("status") or "").lower()
            if status == "running":
                running_cols.add(short)
            elif status == "waiting":
                waiting_cols.add(short)
    n = len(methods)
    row_h = 1.68
    header_y = 0.45 + n * row_h + 0.35
    fig_h = max(8.0, 1.3 + n * row_h + 0.9)
    x_method = 0.08
    x_fdg = 1.50
    x_group0 = 3.05
    # fs groups: PSMA | TEST f0–8 (3×3) | Dice/FP/FN
    inner_group_w = 3.72
    single_col_w = 1.28
    group_gap = 0.26
    col_psma_off, col_test_off, col_mean_off = 0.0, 0.66, 2.86
    group_layout: list[tuple[float, str, str, str, float]] = []
    divider_xs: list[float] = [x_group0 - 0.14]
    gx = x_group0
    hdr_map = {
        FC70_SHORT: FC70_HDR,
        PSMA_FS0_SHORT: PSMA_FS0_HDR,
        FDG_TEST_SHORT: FDG_TEST_HDR,
    }
    for i, (_fn, sk, short) in enumerate(PSMA_BOARD_COLUMNS):
        if i > 0:
            gx += group_gap
            divider_xs.append(gx - group_gap * 0.5)
        w = single_col_w if short in SINGLE_COL_SHORTS else inner_group_w
        group_layout.append((gx, sk, short, hdr_map.get(short, short), w))
        gx += w
    fig_w = max(16.5, gx + 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, header_y + 0.75)
    ax.axis("off")
    hc_pre = board.get("health_check") if isinstance(board.get("health_check"), dict) else {}
    running_tasks = hc_pre.get("running_tasks") if isinstance(hc_pre.get("running_tasks"), list) else []
    wait_bits = [str(x) for x in running_tasks if "WAITING" in str(x).upper()]
    run_bits = [str(x) for x in running_tasks if "WAITING" not in str(x).upper()]
    run_line = ""
    if wait_bits:
        run_line += f"\n⏳ WAITING: {' · '.join(wait_bits[:4])}"
    if run_bits:
        run_line += f"\n▶ RUNNING: {' · '.join(run_bits[:5])}"
    ax.set_title(
        f"Aligned protocol · FDG → PSMA fs50/fs10/fs5 (TEST f0–8) / PSMA fs0 / fc70% / FDG TEST ({n} methods)\n"
        f"{board.get('updated_at','')} · {board.get('updated_note','')}"
        f"{run_line}\n"
        f"metrics = Dice / FP / FN (%) · Dice excludes empty-GT · "
        f"FP=FP/Neg · FN=FN/Pos (voxel micro-avg) · 3×3: TEST20, ~VAL if extra-fold not yet TEST20",
        fontsize=10,
        pad=6,
    )

    # Method | FDG (shared) | [PSMA | TEST | Dice/FP/FN] × N + single cols
    col_max_method, col_max_fdg, col_max_psma, col_max_test = 15, 17, 11, 11

    def _fit(s: str, nch: int) -> str:
        s = str(s or "")
        if nch <= 0 or len(s) <= nch:
            return s
        if nch <= 1:
            return "…"
        return s[: max(1, nch - 1)] + "…"

    def _wrap_method(label: str, width: int = col_max_method) -> str:
        """Allow Method column to wrap (prefer break before venue parentheses)."""
        s = str(label or "").strip()
        if not s:
            return s
        if " (" in s:
            head, tail = s.split(" (", 1)
            return f"{head}\n({tail}"
        if len(s) <= width:
            return s
        # soft wrap on spaces / plus / slash
        parts: list[str] = []
        cur = ""
        for tok in s.replace("+", "+ ").replace("/", "/ ").split():
            cand = f"{cur} {tok}".strip() if cur else tok
            if len(cand) <= width or not cur:
                cur = cand
            else:
                parts.append(cur)
                cur = tok
        if cur:
            parts.append(cur)
        return "\n".join(parts) if parts else s

    def _method_cell(mkey: str, label: str) -> str:
        wrapped = _wrap_method(label)
        init = METHOD_FDG_INIT.get(mkey, "").strip()
        return f"{wrapped}\n{init}" if init else wrapped

    y_div_bot, y_div_top = 0.32, header_y + 0.55
    for dx in divider_xs:
        ax.plot(
            [dx, dx],
            [y_div_bot, y_div_top],
            color="#bdbdbd",
            linewidth=0.9,
            solid_capstyle="round",
            zorder=0,
        )

    _cjk = _cjk_font(size=8.0, weight="bold")
    _method_hdr_kw = {"va": "center", "linespacing": 1.15}
    if _cjk is not None:
        _method_hdr_kw["fontproperties"] = _cjk
    else:
        _method_hdr_kw.update(fontsize=8.0, fontweight="bold")
    ax.text(x_method, header_y + 0.18, "Method\n(pretrained)", **_method_hdr_kw)
    ax.text(x_fdg, header_y + 0.18, "① FDG\n(shared)", fontsize=8.5, fontweight="bold", va="center")
    for _gi, (gx, _stage_key, short, hdr, gw) in enumerate(group_layout):
        if short in running_cols:
            hdr_run = f"{hdr}\n▶ RUN"
            hdr_color = "#ef6c00"
        elif short in waiting_cols:
            hdr_run = f"{hdr}\n⏳ WAIT"
            hdr_color = "#0277bd"
        else:
            hdr_run = hdr
            hdr_color = "black"
        if short in SINGLE_COL_SHORTS:
            ax.text(
                gx + gw * 0.5,
                header_y + 0.28,
                hdr_run,
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="center",
                color=hdr_color,
            )
            ax.text(
                gx + gw * 0.5,
                header_y - 0.12,
                "Dice\nFP\nFN",
                fontsize=7,
                fontweight="bold",
                va="center",
                ha="center",
                color="#424242",
                linespacing=0.95,
            )
        else:
            ax.text(
                gx + col_psma_off,
                header_y + 0.18,
                f"PSMA\n{short}"
                + ("\n▶ RUN" if short in running_cols else ("\n⏳ WAIT" if short in waiting_cols else "")),
                fontsize=8,
                fontweight="bold",
                va="center",
                color=(
                    "#ef6c00"
                    if short in running_cols
                    else ("#0277bd" if short in waiting_cols else "black")
                ),
            )
            ax.text(gx + col_test_off, header_y + 0.18, "TEST\nf0–8 ~VAL", fontsize=7.4, fontweight="bold", va="center")
            ax.text(
                gx + col_mean_off,
                header_y + 0.18,
                "Dice\nFP\nFN",
                fontsize=7.5,
                fontweight="bold",
                va="center",
                ha="left",
                linespacing=0.95,
            )

    y0 = header_y - 0.88
    box_h = row_h - 0.08
    for i, (key, m) in enumerate(methods):
        y = y0 - i * row_h
        ax.add_patch(
            FancyBboxPatch(
                (0.06, y - box_h / 2),
                fig_w - 0.12,
                box_h,
                boxstyle="round,pad=0.015,rounding_size=0.06",
                linewidth=0.55,
                edgecolor="#bdbdbd",
                facecolor="#fafafa" if i % 2 == 0 else "#f5f5f5",
            )
        )
        _m_cjk = _cjk_font(size=7.4, weight="bold")
        _m_kw = {"va": "center", "linespacing": 1.08, "clip_on": True}
        if _m_cjk is not None:
            _m_kw["fontproperties"] = _m_cjk
        else:
            _m_kw.update(fontsize=7.4, fontweight="bold")
        ax.text(x_method, y, _method_cell(key, m.get("label", key)), **_m_kw)

        st_fdg = m.get("fdg_pretrain") or {}
        fdg_status = (st_fdg.get("status") or "pending").lower()
        fdg_color = STATUS_COLOR.get(st_fdg.get("status", "pending"), "#757575")
        fdg_label = _fdg_board_label(st_fdg)
        fdg_lines = fdg_label.count("\n") + 1
        ax.text(
            x_fdg,
            y + (0.14 if fdg_lines >= 2 else 0.08),
            fdg_label,
            fontsize=7.5,
            color=fdg_color,
            fontweight="bold",
            va="center",
            linespacing=1.05,
            clip_on=True,
        )
        if fdg_status == "done":
            gbs = _stage_gbs(st_fdg)
            mid = f"gbs={gbs}" if gbs is not None else ""
            if st_fdg.get("total_epochs"):
                mid = (mid + " · " if mid else "") + f"{int(st_fdg['total_epochs'])}ep"
            if mid:
                ax.text(
                    x_fdg,
                    y - (0.34 if fdg_lines >= 2 else 0.28),
                    _fit(mid, col_max_fdg),
                    fontsize=6,
                    color="#424242",
                    va="center",
                )

        for gi, (gx, stage_key, short, _hdr, gw) in enumerate(group_layout):
            st_ps = m.get(stage_key) or {}
            if short in SINGLE_COL_SHORTS:
                _render_mean_only_cell(
                    ax, st_ps, gx + gw * 0.5, y, cell_w=gw * 0.92, method_key=key, stage_key=stage_key
                )
            else:
                _render_psma_stage_cell(
                    ax, st_ps, gx + col_psma_off, y, col_max_psma, _fit, stage_key=stage_key
                )
                _render_test_block(
                    ax, st_ps, gx + col_test_off, gx + col_mean_off, y, _fit, stage_key=stage_key
                )

    hc = board.get("health_check") if isinstance(board.get("health_check"), dict) else {}
    hc_sum = str(hc.get("summary") or "")
    hc_ok = hc.get("ok")
    hc_color = "#2e7d32" if hc_ok else "#c62828"
    hc_tag = "OK" if hc_ok else "CHECK"
    q_live = hc.get("queue_live") if isinstance(hc.get("queue_live"), list) else []
    q_txt = " · ".join(str(x) for x in q_live[:4]) if q_live else (board.get("queue") or [])
    if isinstance(q_txt, list):
        q_txt = " · ".join(str(x) for x in q_txt[:4])
    ax.text(
        0.2,
        0.42,
        f"self-check [{hc_tag}] {hc_sum}",
        fontsize=6.8,
        color=hc_color,
    )
    ax.text(
        0.2,
        0.28,
        f"queue: {_fit(str(q_txt), 120)}",
        fontsize=6.5,
        color="#607d8b",
    )
    gaps = hc.get("gaps") if isinstance(hc.get("gaps"), list) else []
    if gaps and not hc_ok:
        gap_txt = ", ".join(f"{g.get('method','?')}/{g.get('column','?')}" for g in gaps[:8])
        ax.text(0.2, 0.14, f"gaps: {_fit(gap_txt, 120)}", fontsize=6.2, color="#ef6c00")
    run_tasks = hc.get("running_tasks") if isinstance(hc.get("running_tasks"), list) else []
    if run_tasks:
        wait_bits = [str(x) for x in run_tasks if "WAITING" in str(x).upper()]
        run_bits = [str(x) for x in run_tasks if "WAITING" not in str(x).upper()]
        y0 = 0.02
        if wait_bits:
            ax.text(
                0.2,
                y0 + (0.12 if run_bits else 0),
                f"⏳ WAITING: {_fit(' · '.join(wait_bits[:4]), 130)}",
                fontsize=7.0,
                color="#0277bd",
                fontweight="bold",
            )
        if run_bits:
            ax.text(
                0.2,
                y0,
                f"▶ RUNNING: {_fit(' · '.join(run_bits[:5]), 130)}",
                fontsize=7.2,
                color="#ef6c00",
                fontweight="bold",
            )

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=DEFAULT_BOARD)
    ap.add_argument("--png", type=Path, default=DEFAULT_PNG)
    ap.add_argument("--init", action="store_true", help="reset board to defaults")
    ap.add_argument("--patch-json", default="", help="JSON fragment to deep-merge")
    ap.add_argument("--ingest-mae-stamp", default="")
    ap.add_argument("--ingest-monai-stamp", default="")
    ap.add_argument("--ingest-seganypet-stamp", default="")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="load board JSON and only render PNG")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds (0=once)")
    args = ap.parse_args()

    def once() -> None:
        if args.plot_only:
            board = load_board(args.board)
            render_png(board, args.png)
            print(f"[board] plot-only {args.png}")
            return

        if args.init or not args.board.is_file():
            board = default_board()
        else:
            board = load_board(args.board)
        if args.patch_json:
            board = deep_merge(board, json.loads(args.patch_json))
        if args.ingest_mae_stamp:
            ingest_mae(board, args.ingest_mae_stamp)
        if args.ingest_monai_stamp:
            ingest_monai(board, args.ingest_monai_stamp)
        if args.ingest_seganypet_stamp:
            ingest_seganypet(board, args.ingest_seganypet_stamp)
        ensure_methods(board)
        ensure_stage_bs(board)
        enrich_test_ckpt_epochs(board)
        _recalc_psma_test_mean(board)
        enrich_fdg_best_epochs(board)
        refresh_running_etas(board)
        refresh_stage_train_times(board)
        health = board_self_check(board)
        # self-check may flip pending↔running via pgrep; recompute ETA after that
        refresh_running_etas(board)
        if health.get("dead_workers"):
            board["updated_note"] = (
                f"self-check: dead queue {','.join(health['dead_workers'])} · run queue_keeper"
            )
        elif health.get("gaps") and not health.get("running_n"):
            board["updated_note"] = board.get("updated_note") or f"self-check: {health.get('summary','')}"
        else:
            board["updated_note"] = board.get("updated_note") or "board refresh · fs50/fs10/fs5/fc70/PSMA fs0/FDG TEST"
        save_board(args.board, board)
        if not args.no_plot:
            # tell docker fallback which board path to load
            board["_board_path"] = str(args.board)
            try:
                render_png(board, args.png)
            finally:
                board.pop("_board_path", None)
                # re-save without helper key
                save_board(args.board, {k: v for k, v in board.items() if k != "_board_path"})
        print(f"[board] updated {args.board} png={args.png}")

    if args.watch and args.watch > 0:
        while True:
            once()
            time.sleep(args.watch)
    else:
        once()


if __name__ == "__main__":
    main()
