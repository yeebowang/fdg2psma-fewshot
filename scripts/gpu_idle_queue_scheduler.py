#!/usr/bin/env python3
"""Launch queued ICLR2026 tasks when a GPU has been idle (low VRAM) for N seconds."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

CTRL = Path(__file__).resolve().parents[2]
VIS = CTRL / "ICLR2026/vis"
BOARD = Path(os.environ.get("TASK1_ALIGN_BOARD_JSON", VIS / "iclr2026_aligned_fdg_fs50_f258_board.json"))
STATE_FILE = Path(os.environ.get("TASK1_GPU_IDLE_STATE", VIS / "gpu_idle_scheduler_state.json"))
LOG_FILE = Path(os.environ.get("TASK1_GPU_IDLE_LOG", VIS / "nohup_gpu_idle_queue_scheduler.log"))

GPU_IDS = [int(x) for x in os.environ.get("TASK1_GPU_IDLE_GPUS", "0,1,3").split(",") if x.strip()]
IDLE_MEM_MIB = int(os.environ.get("TASK1_GPU_IDLE_MEM_MIB", "2048"))
IDLE_SEC = int(os.environ.get("TASK1_GPU_IDLE_SEC", "60"))
POLL_SEC = int(os.environ.get("TASK1_GPU_IDLE_POLL_SEC", "10"))
SKIP_SUBSTR = (
    "pgrep",
    "cursor",
    "gpu_idle_queue",
    "idle-gpu",
    "queue_keeper",
    "__CURSOR_SANDBOX",
)


def _log(msg: str) -> None:
    line = f"[gpu-idle-queue] {datetime.now().strftime('%F %T')} {msg}"
    print(line, flush=True)
    # stdout is already redirected to LOG_FILE by the launcher; extra append doubles lines.


def _pgrep_running(patterns: tuple[str, ...]) -> bool:
    for pat in patterns:
        if not pat:
            continue
        try:
            r = subprocess.run(
                ["pgrep", "-af", pat],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode != 0 or not (r.stdout or "").strip():
            continue
        for line in r.stdout.splitlines():
            if any(s in line for s in SKIP_SUBSTR):
                continue
            return True
    return False


def _gpu_mem_mib(gpu: int) -> int | None:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            capture_output=True,
            text=True,
            timeout=15,
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


def _load_board() -> dict[str, Any]:
    if not BOARD.is_file():
        return {}
    try:
        return json.loads(BOARD.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_status(board: dict, method: str, stage: str) -> str:
    st = (board.get("methods") or {}).get(method, {}).get(stage) or {}
    return (st.get("status") or "pending").lower()


def _aggregate_valid(agg: Path) -> bool:
    if not agg.is_file():
        return False
    try:
        d = json.loads(agg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    md = d.get("mean_dice", d.get("mean"))
    if isinstance(md, (int, float)) and md == md:
        return True
    n = d.get("n_scored")
    try:
        return n is not None and int(n) > 0
    except (TypeError, ValueError):
        return False


def _stage_needs_fp_fn_board(board: dict, method: str, stage: str) -> bool:
    st = (board.get("methods") or {}).get(method, {}).get(stage) or {}
    fp, fn = st.get("mean_fp"), st.get("mean_fn")
    has = (
        isinstance(fp, (int, float))
        and fp == fp
        and isinstance(fn, (int, float))
        and fn == fn
    )
    stamp = (st.get("stamp") or "").strip()
    return bool(stamp) and not has


def _stage_mean_missing(board: dict, method: str, stage: str) -> bool:
    st = (board.get("methods") or {}).get(method, {}).get(stage) or {}
    md = st.get("mean")
    return not (isinstance(md, (int, float)) and md == md)


def _mae_scratch_foundation_ckpt() -> str:
    stamp = ""
    last = VIS / "mae_scratch_fdg_LAST_STAMP.txt"
    if last.is_file():
        try:
            stamp = last.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except OSError:
            stamp = ""
    if not stamp:
        st = ((_load_board().get("methods") or {}).get("mae_scratch") or {}).get("fdg_pretrain") or {}
        stamp = str(st.get("stamp") or st.get("best_ckpt") or "").strip()
        if "/" in stamp:
            p = Path(stamp)
            if p.is_file():
                return str(p)
            stamp = p.parent.name if p.parent.name else stamp
    repo = CTRL / "ICLR2026/3D-MAE-PET-CT/runs" / stamp
    for name in ("best_seg_mae.pth", "best_seg_fdg_mae.pth", "latest_seg_mae.pth"):
        p = repo / name
        if p.is_file():
            return str(p)
    return ""


def _scratch_fdg_ckpt(mkey: str, last_name: str, rel_paths: tuple[str, ...]) -> str:
    stamp = ""
    last = VIS / last_name
    if last.is_file():
        try:
            stamp = last.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except OSError:
            stamp = ""
    if not stamp:
        st = ((_load_board().get("methods") or {}).get(mkey) or {}).get("fdg_pretrain") or {}
        stamp = str(st.get("stamp") or st.get("best_ckpt") or "").strip()
        if "/" in stamp:
            p = Path(stamp)
            if p.is_file():
                return str(p)
            stamp = p.parent.name if p.parent.name else stamp
    repo = CTRL / "ICLR2026/3D-MAE-PET-CT/runs" / stamp
    for rel in rel_paths:
        p = repo / rel
        if p.is_file():
            return str(p)
    return ""


def _monai_scratch_foundation_ckpt() -> str:
    return _scratch_fdg_ckpt(
        "monai_scratch",
        "monai_scratch_fdg_LAST_STAMP.txt",
        ("best_seg_fdg_monai.pth", "latest_seg_fdg_monai.pth"),
    )


def _seganypet_scratch_foundation_ckpt() -> str:
    return _scratch_fdg_ckpt(
        "seganypet_scratch",
        "seganypet_scratch_fdg_LAST_STAMP.txt",
        ("seganypet_fdg/best.pth", "seganypet_fdg/latest.pth"),
    )


def _need_fp_fn_agg(method_key: str) -> bool:
    """True when FDG TEST aggregate lacks FP/FN (needs GPU re-eval)."""
    agg = VIS / "fdg_test20" / f"aggregate_{method_key}.json"
    if not agg.is_file():
        return True
    try:
        d = json.loads(agg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    fp = d.get("fp_rate", d.get("mean_fp"))
    fn = d.get("fn_rate", d.get("mean_fn"))
    return not (
        isinstance(fp, (int, float))
        and fp == fp
        and isinstance(fn, (int, float))
        and fn == fn
    )


@dataclass
class QueueTask:
    task_id: str
    label: str
    pgrep: tuple[str, ...]
    pending: Callable[[dict], bool]
    launch_cmd: Callable[[int], list[str]]
    env: Callable[[int], dict[str, str]] | None = None
    log_name: str = ""
    parallel_per_gpu: bool = False
    need_gpus: int = 1


# Incomplete fc70 jobs yield to extra-fold 9fold; resume after extra folds are done.
FC70_DEFER_WHILE_EXTRA = frozenset(
    {
        "seganypet_fc70",
        "dpdnet_fc70",
        "monai_fc70",
        "mae_fc70",
        "nnunet_fc70",
    }
)
FC70_DEFER_FILE = VIS / "TASK1_FC70_DEFER_UNTIL_EXTRA.json"
SKIPPED_FILE = VIS / "TASK1_IDLE_SKIPPED_TASKS.json"
EXTRA_SKIP_FILE = VIS / "TASK1_EXTRA_FOLD_SKIPPED.txt"
NEVER_SKIP_TASKS = frozenset(
    {
        "extra_fold_9fold",
        "extra_fold_test20",
        "mae_scratch_9fold",
        "monai_scratch_9fold",
        "seganypet_scratch_9fold",
        "nnunet_mim_dpdnet_dualenc",
        "dpdnet_dualenc_fdg",
        "nnunet_mim_remaining",
    }
)
LAUNCH_GRACE_SEC = 25


def _build_tasks() -> list[QueueTask]:
    run = CTRL / "ICLR2026/run"
    return [
        *_extra_fold_9fold_tasks(run),
        *_extra_fold_test20_tasks(run),
        *_monai_scratch_tail_tasks(run),
        *_seganypet_scratch_tail_tasks(run),
        *_nnunet_mim_dualenc_after_scratch_tasks(run),
        *_dpdnet_dualenc_onegpu_tasks(run),
        *_nnunet_mim_remaining_onegpu_tasks(run),
        QueueTask(
            task_id="seganypet_fc70",
            label="SegAnyPET PSMA fc70%",
            pgrep=("run_seganypet_psma_fc70_from_fdg_bg.sh", "seganypet_fewshot_finetune.py.*fc70", "seganypet_eval_psma_test20"),
            pending=lambda b: _stage_status(b, "seganypet", "psma_fc70") in ("pending", "queued", ""),
            launch_cmd=lambda gpu: ["bash", str(run / "run_seganypet_psma_fc70_from_fdg_bg.sh")],
            env=lambda gpu: {"TASK1_PSMA_FC70_GPU": str(gpu), "TASK1_CUDA_VISIBLE_DEVICES": str(gpu)},
            log_name="nohup_seganypet_psma_fc70_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="dpdnet_fc70",
            label="DpDNet PSMA fc70%",
            pgrep=("run_dpdnet_psma_fc70_decline_and_test_bg.sh",),
            pending=lambda b: _stage_status(b, "dpdnet", "psma_fc70") in ("pending", "queued", ""),
            launch_cmd=lambda gpu: ["bash", str(run / "run_dpdnet_psma_fc70_decline_and_test_bg.sh")],
            env=lambda gpu: {"TASK1_PSMA_FC70_GPU": str(gpu)},
            log_name="nohup_dpdnet_psma_fc70_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="monai_fc70",
            label="MONAI PSMA fc70%",
            pgrep=("run_monai_psma_fc70_from_fdg_seg_bg.sh",),
            pending=lambda b: _stage_status(b, "monai_swinvit", "psma_fc70") in ("pending", "queued", ""),
            launch_cmd=lambda gpu: ["bash", str(run / "run_monai_psma_fc70_from_fdg_seg_bg.sh")],
            env=lambda gpu: {"TASK1_PSMA_FC70_GPU": str(gpu)},
            log_name="nohup_monai_psma_fc70_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="mae_fc70",
            label="MAE PSMA fc70%",
            pgrep=("run_mae_psma_fc70_from_fdg_seg_bg.sh",),
            pending=lambda b: _stage_status(b, "mae_swinunetr", "psma_fc70") in ("pending", "queued", ""),
            launch_cmd=lambda gpu: ["bash", str(run / "run_mae_psma_fc70_from_fdg_seg_bg.sh")],
            env=lambda gpu: {"TASK1_PSMA_FC70_GPU": str(gpu)},
            log_name="nohup_mae_psma_fc70_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="mae_scratch_fc70",
            label="MAE-scratch PSMA fc70%",
            pgrep=(
                "run_mae_scratch_psma_fc70_from_fdg_seg_bg.sh",
                "iclr2026_mae_scratch_psma_fc70",
            ),
            pending=lambda b: _stage_mean_missing(b, "mae_scratch", "psma_fc70"),
            launch_cmd=lambda gpu: ["bash", str(run / "run_mae_scratch_psma_fc70_from_fdg_seg_bg.sh")],
            env=lambda gpu: {
                "TASK1_PSMA_FC70_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                "TASK1_BOARD_METHOD": "mae_scratch",
                "TASK1_FC70_EVAL_METHOD": "mae_scratch",
                "TASK1_MAE_FDG_SEG_CKPT": _mae_scratch_foundation_ckpt(),
            },
            log_name="nohup_mae_scratch_psma_fc70_gpu{gpu}.log",
            need_gpus=1,
        ),
        QueueTask(
            task_id="monai_scratch_fc70",
            label="MONAI-scratch PSMA fc70%",
            pgrep=(
                "run_monai_scratch_psma_fc70_from_fdg_seg_bg.sh",
                "iclr2026_monai_scratch_psma_fc70",
            ),
            pending=lambda b: (
                _stage_status(b, "monai_scratch", "fdg_pretrain") == "done"
                and _stage_mean_missing(b, "monai_scratch", "psma_fc70")
            ),
            launch_cmd=lambda gpu: ["bash", str(run / "run_monai_scratch_psma_fc70_from_fdg_seg_bg.sh")],
            env=lambda gpu: {
                "TASK1_PSMA_FC70_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                "TASK1_BOARD_METHOD": "monai_scratch",
                "TASK1_FC70_EVAL_METHOD": "monai_scratch",
                "TASK1_MONAI_FDG_SEG_CKPT": _monai_scratch_foundation_ckpt(),
            },
            log_name="nohup_monai_scratch_psma_fc70_gpu{gpu}.log",
            need_gpus=1,
        ),
        QueueTask(
            task_id="seganypet_scratch_fc70",
            label="SegAnyPET-scratch PSMA fc70%",
            pgrep=(
                "run_seganypet_scratch_psma_fc70_from_fdg_bg.sh",
                "iclr2026_seganypet_scratch_psma_fc70",
            ),
            pending=lambda b: (
                _stage_status(b, "seganypet_scratch", "fdg_pretrain") == "done"
                and _stage_mean_missing(b, "seganypet_scratch", "psma_fc70")
            ),
            launch_cmd=lambda gpu: ["bash", str(run / "run_seganypet_scratch_psma_fc70_from_fdg_bg.sh")],
            env=lambda gpu: {
                "TASK1_PSMA_FC70_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                "TASK1_BOARD_METHOD": "seganypet_scratch",
                "TASK1_SEGANY_CKPT": _seganypet_scratch_foundation_ckpt(),
            },
            log_name="nohup_seganypet_scratch_psma_fc70_gpu{gpu}.log",
            need_gpus=1,
        ),
        QueueTask(
            task_id="nnunet_fc70",
            label="nnUNet PSMA fc70%",
            pgrep=("run_nnunet_psma_fc70_decline_and_test_bg.sh",),
            pending=lambda b: _stage_status(b, "nnunet", "psma_fc70") in ("pending", "queued", ""),
            launch_cmd=lambda gpu: ["bash", str(run / "run_nnunet_psma_fc70_decline_and_test_bg.sh")],
            env=lambda gpu: {"TASK1_PSMA_FC70_GPU": str(gpu)},
            log_name="nohup_nnunet_psma_fc70_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="nnunet_fdg_test",
            label="nnUNet FDG TEST",
            pgrep=("run_eval_fdg_test20_bg.sh", "fdg_test20_eval/nnunet"),
            pending=lambda b: not _aggregate_valid(VIS / "fdg_test20" / "aggregate_nnunet.json")
            and _stage_status(b, "nnunet", "fdg_test20") != "done",
            launch_cmd=lambda gpu: ["bash", str(run / "run_eval_fdg_test20_bg.sh")],
            env=lambda gpu: {
                "METHOD": "nnunet",
                "TASK1_TEST_SKIP_DONE": "0",
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                "TASK1_UDA_PRED_PER_GPU": "1",
            },
            log_name="nohup_fdg_test20_nnunet_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="mae_fdg_test",
            label="MAE FDG TEST",
            pgrep=("fdg_test20_eval/mae", "METHOD=mae"),
            pending=lambda b: _need_fp_fn_agg("mae_swinunetr"),
            launch_cmd=lambda gpu: ["bash", str(run / "run_eval_fdg_test20_bg.sh")],
            env=lambda gpu: {
                "METHOD": "mae",
                "TASK1_TEST_SKIP_DONE": "0",
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
            },
            log_name="nohup_fdg_test20_mae_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="monai_fdg_test",
            label="MONAI FDG TEST",
            pgrep=("fdg_test20_eval/monai", "METHOD=monai"),
            pending=lambda b: _need_fp_fn_agg("monai_swinvit"),
            launch_cmd=lambda gpu: ["bash", str(run / "run_eval_fdg_test20_bg.sh")],
            env=lambda gpu: {
                "METHOD": "monai",
                "TASK1_TEST_SKIP_DONE": "0",
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
            },
            log_name="nohup_fdg_test20_monai_gpu{gpu}.log",
        ),
        QueueTask(
            task_id="proto_fc70",
            label="Proto PSMA fc70%",
            pgrep=("run_proto_retrieval_psma_fc70",),
            pending=lambda b: _stage_status(b, "proto_retrieval", "psma_fc70") in ("pending", "queued", ""),
            launch_cmd=lambda gpu: ["bash", str(run / "run_proto_retrieval_psma_fc70_test20_bg.sh")],
            env=lambda gpu: {"TASK1_CUDA_VISIBLE_DEVICES": str(gpu)},
            log_name="nohup_proto_fc70_gpu{gpu}.log",
        ),
        *_mae_monai_fpfn_tasks(run),
        *_psma_fs0_fpfn_tasks(run),
        QueueTask(
            task_id="nnunet_fc70_test",
            label="nnUNet PSMA fc70 TEST",
            pgrep=("run_nnunet_psma_test20_fc70_bg.sh",),
            pending=lambda b: (
                _stage_status(b, "nnunet", "psma_fc70") in ("done", "running")
                and not isinstance(
                    ((b.get("methods") or {}).get("nnunet") or {}).get("psma_fc70", {}).get("mean"),
                    (int, float),
                )
            ),
            launch_cmd=lambda gpu: ["bash", str(run / "run_nnunet_psma_test20_fc70_bg.sh")],
            env=lambda gpu: {
                "PARENT_STAMP": (
                    ((json.loads(BOARD.read_text()).get("methods") or {}).get("nnunet") or {})
                    .get("psma_fc70")
                    or {}
                ).get("stamp")
                or ""
                if BOARD.is_file()
                else "",
                "TASK1_PSMA_FC70_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
            },
            log_name="nohup_nnunet_psma_fc70_test_gpu{gpu}.log",
        ),
        *_mae_scratch_tail_tasks(run),
    ]


def _mae_monai_fpfn_tasks(run: Path) -> list[QueueTask]:
    out: list[QueueTask] = []
    for method, mkey, stage, few in (
        ("mae", "mae_swinunetr", "psma_fs50_f258", "50"),
        ("mae", "mae_swinunetr", "psma_fs10_f258", "10"),
        ("mae", "mae_swinunetr", "psma_fs5_f258", "5"),
        ("mae_scratch", "mae_scratch", "psma_fs50_f258", "50"),
        ("mae_scratch", "mae_scratch", "psma_fs10_f258", "10"),
        ("mae_scratch", "mae_scratch", "psma_fs5_f258", "5"),
        ("monai_scratch", "monai_scratch", "psma_fs50_f258", "50"),
        ("monai_scratch", "monai_scratch", "psma_fs10_f258", "10"),
        ("monai_scratch", "monai_scratch", "psma_fs5_f258", "5"),
        ("monai", "monai_swinvit", "psma_fs50_f258", "50"),
        ("monai", "monai_swinvit", "psma_fs10_f258", "10"),
        ("monai", "monai_swinvit", "psma_fs5_f258", "5"),
    ):
        tid = f"{method}_psma_fpfn_fs{few}"
        out.append(
            QueueTask(
                task_id=tid,
                label=f"{method.upper()} PSMA fs{few} FP/FN",
                pgrep=("run_eval_psma_test20_fpfn_onegpu.sh", "mae_eval_seg_psma_test.py"),
                pending=lambda b, mk=mkey, stg=stage: _stage_needs_fp_fn_board(b, mk, stg),
                launch_cmd=lambda gpu: ["bash", str(run / "run_eval_psma_test20_fpfn_onegpu.sh")],
                env=lambda gpu, meth=method, stg=stage, few=few: {
                    "METHOD": meth,
                    "TASK1_PSMA_BOARD_STAGE": stg,
                    "TASK1_FEWSHOT_N": few,
                    "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                    "TASK1_PSMA_FC70_GPU": str(gpu),
                    "TASK1_TEST_SKIP_DONE": "0",
                },
                log_name=f"nohup_{method}_psma_fpfn_fs{few}_gpu{{gpu}}.log",
            )
        )
    return out


def _extra_folds_pending(board: dict | None = None) -> bool:
    """True while fs50/fs10/fs5 still miss extra folds 0/1/3/4/6/7."""
    done = VIS / "TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt"
    if done.is_file():
        try:
            if "status=ok" in done.read_text(encoding="utf-8"):
                return False
        except OSError:
            pass
    board = board or _load_board()
    methods = board.get("methods") or {}
    extra = ("0", "1", "3", "4", "6", "7")
    skipped_folds = _extra_skipped_keys()
    repo = CTRL / "ICLR2026/3D-MAE-PET-CT/runs"
    work = Path(os.environ.get("TASK1_BASE", "/media/ybwang/data1/PSMA-DATA")) / "task1_train_workspace/nnUNet_results"
    for mkey, kind in (
        ("mae_swinunetr", "mae"),
        ("monai_swinvit", "monai"),
        ("nnunet", "nnunet"),
        ("dpdnet", "dpdnet"),
        ("seganypet", "seganypet"),
    ):
        for n in (50, 10, 5):
            st = (methods.get(mkey) or {}).get(f"psma_fs{n}_f258") or {}
            stamp = (st.get("stamp") or "").strip()
            if not stamp:
                continue
            fd = st.get("fold_dice") or {}
            for f in extra:
                if f"{kind}|{n}|{f}" in skipped_folds:
                    continue
                if f in fd and isinstance(fd.get(f), (int, float)):
                    continue
                if kind == "mae" and any((repo / stamp / "mae" / f"fold{f}").glob("*.pth")):
                    continue
                if kind == "monai" and any((repo / stamp / "monai" / f"fold{f}").glob("*.pth")):
                    continue
                if kind == "seganypet" and any((repo / stamp / "seganypet" / f"fold{f}").glob("*.pth")):
                    continue
                if kind in ("nnunet", "dpdnet"):
                    if any((work / f"{stamp}_f{f}").glob("**/checkpoint_*.pth")):
                        continue
                return True
    return False


def _extra_fold_9fold_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_aligned_psma_extra_fold_onegpu.sh"
    return [
        QueueTask(
            task_id="extra_fold_9fold",
            label="PSMA fs50/10/5 extra fold → 9fold",
            pgrep=("run_aligned_psma_extra_fold_onegpu.sh --gpu {gpu}",),
            pending=lambda b: _extra_folds_pending(b),
            launch_cmd=lambda gpu: ["bash", str(script), "--gpu", str(gpu)],
            env=lambda gpu: {
                "TASK1_EXTRA_FOLD_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
            },
            log_name="nohup_extra_fold_onegpu_gpu{gpu}.log",
            parallel_per_gpu=True,
        )
    ]


def _extra_fold_test20_pending(board: dict | None = None) -> bool:
    """True while extra folds 0/1/3/4/6/7 still miss TEST20 jsons.

    Ignore TASK1_PSMA_EXTRA_FOLD_TEST20_DONE.txt: that marker was written
    after fail-skip (nnUNet fs50 f6/f7 still empty).
    """
    board = board or _load_board()
    methods = board.get("methods") or {}
    extra = ("0", "1", "3", "4", "6", "7")
    repo = CTRL / "ICLR2026/3D-MAE-PET-CT/runs"
    work = Path(os.environ.get("TASK1_BASE", "/media/ybwang/data1/PSMA-DATA")) / "task1_train_workspace/nnUNet_results"
    for mkey, kind in (
        ("mae_swinunetr", "mae"),
        ("monai_swinvit", "monai"),
        ("nnunet", "nnunet"),
        ("dpdnet", "dpdnet"),
        ("seganypet", "seganypet"),
    ):
        for n in (50, 10, 5):
            st = (methods.get(mkey) or {}).get(f"psma_fs{n}_f258") or {}
            stamp = (st.get("stamp") or "").strip()
            if not stamp:
                continue
            fd = st.get("fold_dice") or {}
            for f in extra:
                if f in fd and isinstance(fd.get(f), (int, float)):
                    continue
                if kind in ("mae", "monai", "seganypet"):
                    if (repo / stamp / "psma_test20_eval" / f"fold{f}_test20.json").is_file():
                        continue
                else:
                    if (work / stamp / "psma_test20_eval" / f"fold{f}" / "score_detail.json").is_file():
                        continue
                return True
    return False


def _extra_fold_test20_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_aligned_psma_extra_fold_test20_onegpu.sh"
    return [
        QueueTask(
            task_id="extra_fold_test20",
            label="PSMA extra-fold TEST20 (1 GPU)",
            pgrep=("run_aligned_psma_extra_fold_test20_onegpu.sh --gpu {gpu}",),
            pending=lambda b: _extra_fold_test20_pending(b),
            launch_cmd=lambda gpu: ["bash", str(script), "--gpu", str(gpu)],
            env=lambda gpu: {
                "TASK1_EXTRA_FOLD_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                "TASK1_UDA_PRED_PER_GPU": "1",
            },
            log_name="nohup_extra_fold_test20_gpu{gpu}.log",
            parallel_per_gpu=True,
            need_gpus=1,
        )
    ]


def _marker_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return "status=ok" in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _scratch_9fold_stages_pending(mkey: str, done: Path, require: Path | None = None) -> bool:
    if _marker_ok(done):
        return False
    if require is not None and not _marker_ok(require):
        return False
    board = _load_board()
    if _extra_folds_pending(board):
        return False
    m = (board.get("methods") or {}).get(mkey) or {}
    for sk in (
        "fdg_pretrain",
        "psma_fs50_f258",
        "psma_fs10_f258",
        "psma_fs5_f258",
        "psma_fc70",
        "psma_fs0",
        "fdg_test20",
    ):
        st = ((m.get(sk) or {}).get("status") or "pending").lower()
        if st in ("pending", "queued", "", "running"):
            return True
    return False


def _mae_scratch_stages_pending(board: dict | None = None) -> bool:
    done = VIS / "TASK1_MAE_SCRATCH_9FOLD_DONE.txt"
    if _marker_ok(done):
        return False
    board = board or _load_board()
    if _extra_folds_pending(board):
        return False
    m = (board.get("methods") or {}).get("mae_scratch") or {}
    for sk in (
        "fdg_pretrain",
        "psma_fs50_f258",
        "psma_fs10_f258",
        "psma_fs5_f258",
        "psma_fc70",
        "psma_fs0",
        "fdg_test20",
    ):
        st = ((m.get(sk) or {}).get("status") or "pending").lower()
        if st in ("pending", "queued", "", "running"):
            return True
    return False


def _mae_scratch_tail_tasks(run: Path) -> list[QueueTask]:
    """Queue tail: PET/CT MAE scratch 9-fold pipeline after extra folds."""
    script = run / "run_mae_scratch_after_extra_folds_queue_bg.sh"
    return [
        QueueTask(
            task_id="mae_scratch_9fold",
            label="PET/CT MAE scratch 9fold",
            pgrep=(
                "run_mae_scratch_after_extra_folds_queue",
                "run_aligned_mae_scratch_9fold_pipeline",
            ),
            pending=lambda b: _mae_scratch_stages_pending(b),
            launch_cmd=lambda gpu: ["bash", str(script)],
            env=lambda gpu: {"TASK1_CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in GPU_IDS)},
            log_name="nohup_mae_scratch_9fold_queue.log",
            need_gpus=len(GPU_IDS) if GPU_IDS else 3,
        )
    ]


def _monai_scratch_tail_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_aligned_monai_scratch_9fold_pipeline_bg.sh"
    return [
        QueueTask(
            task_id="monai_scratch_9fold",
            label="MONAI SwinViT scratch 9fold",
            pgrep=("run_aligned_monai_scratch_9fold_pipeline",),
            pending=lambda b: _scratch_9fold_stages_pending(
                "monai_scratch",
                VIS / "TASK1_MONAI_SCRATCH_9FOLD_DONE.txt",
                VIS / "TASK1_MAE_SCRATCH_9FOLD_DONE.txt",
            ),
            launch_cmd=lambda gpu: ["bash", str(script)],
            env=lambda gpu: {
                "TASK1_CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in GPU_IDS),
                "TASK1_BOARD_METHOD": "monai_scratch",
                "TASK1_MAE_FOUNDATION_KIND": "none",
            },
            log_name="nohup_monai_scratch_9fold_pipeline.log",
            need_gpus=len(GPU_IDS) if GPU_IDS else 3,
        )
    ]


def _seganypet_scratch_tail_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_aligned_seganypet_scratch_9fold_pipeline_bg.sh"
    return [
        QueueTask(
            task_id="seganypet_scratch_9fold",
            label="SegAnyPET scratch 9fold",
            pgrep=("run_aligned_seganypet_scratch_9fold_pipeline",),
            pending=lambda b: _scratch_9fold_stages_pending(
                "seganypet_scratch",
                VIS / "TASK1_SEGANY_SCRATCH_9FOLD_DONE.txt",
                VIS / "TASK1_MONAI_SCRATCH_9FOLD_DONE.txt",
            ),
            launch_cmd=lambda gpu: ["bash", str(script)],
            env=lambda gpu: {
                "TASK1_CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in GPU_IDS),
                "TASK1_BOARD_METHOD": "seganypet_scratch",
                "TASK1_SEGANY_CKPT": "none",
            },
            log_name="nohup_seganypet_scratch_9fold_pipeline.log",
            need_gpus=len(GPU_IDS) if GPU_IDS else 3,
        )
    ]


def _nnunet_mim_dualenc_after_scratch_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_nnunet_mim_dpdnet_dualenc_after_scratch_queue_bg.sh"
    return [
        QueueTask(
            task_id="nnunet_mim_dpdnet_dualenc",
            label="nnUNet MIM + DpDNet dual-enc",
            pgrep=(
                "run_nnunet_mim_dpdnet_dualenc_after_scratch",
                "run_nnunet_mim_aligned_fdg_psma",
                "run_dpdnet_dualenc_aligned_fdg_psma",
            ),
            pending=lambda b: _mim_dualenc_after_scratch_pending(b),
            launch_cmd=lambda gpu: ["bash", str(script)],
            env=lambda gpu: {"TASK1_CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in GPU_IDS)},
            log_name="nohup_nnunet_mim_dpdnet_dualenc_after_scratch_queue.log",
            need_gpus=len(GPU_IDS) if GPU_IDS else 3,
        )
    ]


def _dpdnet_encoders_ready() -> bool:
    root = CTRL / "ICLR2026/3D-MAE-PET-CT/weights/dpdnet"
    return (root / "best_encoder_ct_epoch_94.pth").is_file() and (
        root / "best_encoder_pet_epoch_94.pth"
    ).is_file()


def _dpdnet_dualenc_onegpu_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_dpdnet_dualenc_aligned_fdg_psma_bg.sh"
    return [
        QueueTask(
            task_id="dpdnet_dualenc_fdg",
            label="DpDNet dual-enc FDG→PSMA",
            pgrep=(
                "run_dpdnet_dualenc_aligned_fdg_psma",
                "run_dpdnet_fdg_1gpu_bs6",
                "iclr2026_dpdnet_dualenc",
            ),
            pending=lambda b: _dpdnet_encoders_ready()
            and _stage_status(b, "dpdnet_dualenc", "fdg_pretrain")
            in ("pending", "queued", "", "waiting"),
            launch_cmd=lambda gpu: ["bash", str(script)],
            env=lambda gpu: {
                "TASK1_DPDNET_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
            },
            log_name="nohup_dpdnet_dualenc_aligned_fdg_psma_gpu{gpu}.log",
            need_gpus=1,
        )
    ]


def _nnunet_mim_remaining_pending(board: dict | None = None) -> bool:
    board = board or _load_board()
    mim = (board.get("methods") or {}).get("nnunet_mim") or {}
    if _stage_status(board, "nnunet_mim", "fdg_pretrain") != "done":
        return False
    for stage in ("psma_fs10_f258", "psma_fs5_f258"):
        st = mim.get(stage) or {}
        md = st.get("mean")
        if not (isinstance(md, (int, float)) and md == md):
            return True
    return False


def _nnunet_mim_remaining_onegpu_tasks(run: Path) -> list[QueueTask]:
    script = run / "run_nnunet_mim_remaining_onegpu.sh"
    return [
        QueueTask(
            task_id="nnunet_mim_remaining",
            label="nnUNet MIM fs10/fs5 (1 GPU/fold)",
            pgrep=("run_nnunet_mim_remaining_onegpu.sh --gpu {gpu}",),
            pending=lambda b: _nnunet_mim_remaining_pending(b),
            launch_cmd=lambda gpu: ["bash", str(script), "--gpu", str(gpu)],
            env=lambda gpu: {
                "TASK1_EXTRA_FOLD_GPU": str(gpu),
                "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
            },
            log_name="nohup_nnunet_mim_remaining_gpu{gpu}.log",
            parallel_per_gpu=True,
            need_gpus=1,
        )
    ]


def _mim_dualenc_after_scratch_pending(board: dict | None = None) -> bool:
    if _marker_ok(VIS / "TASK1_NNUNET_MIM_DPDNET_DUALENC_DONE.txt"):
        return False
    if not _marker_ok(VIS / "TASK1_SEGANY_SCRATCH_9FOLD_DONE.txt"):
        return False
    if not _marker_ok(VIS / "TASK1_MONAI_SCRATCH_9FOLD_DONE.txt"):
        return False
    board = board or _load_board()
    for mkey in ("nnunet_mim", "dpdnet_dualenc"):
        m = (board.get("methods") or {}).get(mkey) or {}
        for sk in ("fdg_pretrain", "psma_fs50_f258"):
            st = ((m.get(sk) or {}).get("status") or "pending").lower()
            if st in ("pending", "queued", "", "running"):
                return True
    return False


def _psma_fs0_fpfn_tasks(run: Path) -> list[QueueTask]:
    """PSMA fs0 (FDG ckpt zero-shot) missing FP/FN — MAE/MONAI need GPU; others GPU fallback."""
    out: list[QueueTask] = []
    for method, mkey in (
        ("mae", "mae_swinunetr"),
        ("mae_scratch", "mae_scratch"),
        ("monai_scratch", "monai_scratch"),
        ("monai", "monai_swinvit"),
        ("nnunet", "nnunet"),
        ("dpdnet", "dpdnet"),
        ("seganypet", "seganypet"),
        ("seganypet_scratch", "seganypet_scratch"),
    ):
        out.append(
            QueueTask(
                task_id=f"{method}_psma_fs0_fpfn",
                label=f"{method.upper()} PSMA fs0 FP/FN",
                pgrep=("run_eval_fdg_shared_test20_bg.sh", f"METHOD={method}"),
                pending=lambda b, mk=mkey: _stage_needs_fp_fn_board(b, mk, "psma_fs0"),
                launch_cmd=lambda gpu: ["bash", str(run / "run_eval_fdg_shared_test20_bg.sh")],
                env=lambda gpu, meth=method: {
                    "METHOD": meth,
                    "TASK1_TEST_SKIP_DONE": "0",
                    "TASK1_CUDA_VISIBLE_DEVICES": str(gpu),
                    "TASK1_UDA_PRED_PER_GPU": "1",
                },
                log_name=f"nohup_{method}_psma_fs0_fpfn_gpu{{gpu}}.log",
            )
        )
    return out


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        state: dict[str, Any] = {"idle_since": {}, "launched": {}, "skipped": {}, "active": {}}
    else:
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {"idle_since": {}, "launched": {}, "skipped": {}, "active": {}}
    state.setdefault("skipped", {})
    state.setdefault("active", {})
    if SKIPPED_FILE.is_file() and not state["skipped"]:
        try:
            rec = json.loads(SKIPPED_FILE.read_text(encoding="utf-8"))
            if isinstance(rec, dict):
                state["skipped"] = rec
        except (OSError, json.JSONDecodeError):
            pass
    return state


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _extra_skipped_keys() -> set[str]:
    if not EXTRA_SKIP_FILE.is_file():
        return set()
    try:
        return {ln.strip() for ln in EXTRA_SKIP_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}
    except OSError:
        return set()


def _persist_skipped(state: dict[str, Any]) -> None:
    rec = state.get("skipped") or {}
    try:
        SKIPPED_FILE.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _job_alive(rec: dict[str, Any], task: QueueTask | None, gpu: int) -> bool:
    pid = int(rec.get("pid") or 0)
    if pid and _pid_alive(pid):
        return True
    if task:
        per_gpu = bool(task.parallel_per_gpu) or any("{gpu}" in p for p in task.pgrep)
        if per_gpu and _pgrep_running(_task_pgrep(task, gpu)):
            return True
    mem = _gpu_mem_mib(gpu)
    return mem is not None and mem >= IDLE_MEM_MIB


def _slot_held(state: dict[str, Any], gpu: int, tasks: list[QueueTask], now: float) -> bool:
    rec = (state.get("active") or {}).get(str(gpu))
    if not rec:
        return False
    task = next((t for t in tasks if t.task_id == rec.get("task_id")), None)
    t0 = float(rec.get("t0") or 0.0)
    if t0 and (now - t0) < LAUNCH_GRACE_SEC:
        return True
    return _job_alive(rec, task, gpu)


def _free_gpu_ids(state: dict[str, Any], tasks: list[QueueTask], now: float) -> list[int]:
    """GPUs in the idle pool that are not held and currently low-VRAM."""
    free: list[int] = []
    for gpu in GPU_IDS:
        if _slot_held(state, gpu, tasks, now):
            continue
        mem = _gpu_mem_mib(gpu)
        if mem is not None and mem < IDLE_MEM_MIB:
            free.append(gpu)
    return free


def _hydrate_active(state: dict[str, Any], tasks: list[QueueTask]) -> None:
    active = state.setdefault("active", {})
    for gpu in GPU_IDS:
        if str(gpu) in active:
            rec = active[str(gpu)]
            task = next((t for t in tasks if t.task_id == rec.get("task_id")), None)
            if _job_alive(rec, task, gpu):
                continue
        for t in tasks:
            pf = VIS / f"gpu_idle_{t.task_id}_gpu{gpu}.pid"
            if not pf.is_file():
                continue
            try:
                pid = int((pf.read_text(encoding="utf-8") or "0").strip() or "0")
            except (OSError, ValueError):
                continue
            if _pid_alive(pid):
                active[str(gpu)] = {"task_id": t.task_id, "pid": pid, "t0": time.time(), "at": "hydrate"}
                break
            per_gpu = bool(t.parallel_per_gpu) or any("{gpu}" in p for p in t.pgrep)
            if per_gpu and _pgrep_running(_task_pgrep(t, gpu)):
                active[str(gpu)] = {"task_id": t.task_id, "pid": pid, "t0": time.time(), "at": "hydrate"}
                break


def _patch_skipped_board(task_id: str) -> None:
    slot = TASK_BOARD_MAP.get(task_id)
    if not slot or task_id in NEVER_SKIP_TASKS:
        return
    mkey, stage = slot
    patch = {
        "updated_note": f"SKIP {task_id} (failed) → next task",
        "methods": {
            mkey: {
                stage: {
                    "status": "pending",
                    "note": "skipped failed · gpu-idle next task",
                }
            }
        },
    }
    try:
        subprocess.run(
            [
                "python3",
                str(CTRL / "ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py"),
                "--board",
                str(BOARD),
                "--no-plot",
                "--patch-json",
                json.dumps(patch),
            ],
            capture_output=True,
            timeout=120,
            cwd=str(CTRL),
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _maybe_mark_extra_folds_done(tasks: list[QueueTask] | None = None) -> None:
    if tasks:
        for gpu in GPU_IDS:
            for t in tasks:
                if t.task_id == "extra_fold_9fold" and _pgrep_running(_task_pgrep(t, gpu)):
                    return
    done = VIS / "TASK1_PSMA_EXTRA_FOLDS_9FOLD_DONE.txt"
    try:
        if done.is_file() and "status=ok" in done.read_text(encoding="utf-8"):
            return
    except OSError:
        pass
    skipped = _extra_skipped_keys()
    note = "all extra folds present"
    if skipped:
        note = f"extra folds present or skipped ({len(skipped)} skipped)"
    try:
        done.write_text(
            "done_at="
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + f"\nstatus=ok\nnote={note}\n",
            encoding="utf-8",
        )
        _log(f"extra-fold 9fold DONE ({note})")
    except OSError:
        pass


def _reap_active(
    state: dict[str, Any],
    tasks: list[QueueTask],
    idle_since: dict[str, float],
    now: float,
    board: dict[str, Any],
) -> None:
    """If a launched job died still pending, skip it and free the GPU immediately."""
    active = state.setdefault("active", {})
    by_id = {t.task_id: t for t in tasks}
    for gpu_s, rec in list(active.items()):
        try:
            gpu = int(gpu_s)
        except ValueError:
            active.pop(gpu_s, None)
            continue
        tid = str(rec.get("task_id") or "")
        task = by_id.get(tid)
        t0 = float(rec.get("t0") or 0.0)
        if t0 and (now - t0) < LAUNCH_GRACE_SEC and not _job_alive(rec, task, gpu):
            idle_since.pop(gpu_s, None)
            continue
        if _job_alive(rec, task, gpu):
            idle_since.pop(gpu_s, None)
            continue
        active.pop(gpu_s, None)
        still = bool(task and task.pending(board))
        if tid in NEVER_SKIP_TASKS:
            lived = (now - t0) if t0 else 0.0
            kind = {
                "extra_fold_test20": "extra-fold TEST20",
                "extra_fold_9fold": "extra-fold",
                "mae_scratch_9fold": "mae-scratch 9fold",
                "mae_scratch_fc70": "mae-scratch fc70%",
                "monai_scratch_9fold": "monai-scratch 9fold",
                "seganypet_scratch_9fold": "seganypet-scratch 9fold",
                "nnunet_mim_dpdnet_dualenc": "nnUNet MIM + DpDNet dual-enc",
                "dpdnet_dualenc_fdg": "DpDNet dual-enc",
                "nnunet_mim_remaining": "nnUNet MIM remaining",
            }.get(tid, tid)
            if lived < 45:
                idle_since[gpu_s] = now
                _log(
                    f"GPU {gpu} {kind} died in {int(lived)}s (picker/crash) "
                    f"→ wait {IDLE_SEC}s (not spin)"
                )
            else:
                idle_since[gpu_s] = now - float(IDLE_SEC)
                _log(f"GPU {gpu} {kind} slot free → next fold immediately")
        elif still:
            state.setdefault("skipped", {})[tid] = {
                "at": datetime.now().strftime("%F %T"),
                "gpu": gpu,
                "reason": "exited while still pending",
            }
            _persist_skipped(state)
            idle_since[gpu_s] = now - float(IDLE_SEC)
            _log(f"SKIP {tid} (GPU {gpu} exited pending) → next task immediately")
            _patch_skipped_board(tid)
        else:
            idle_since[gpu_s] = now
            _log(f"GPU {gpu} {tid} finished")
            mem = _gpu_mem_mib(gpu)
            if mem is not None and mem < IDLE_MEM_MIB:
                idle_since[gpu_s] = now - float(IDLE_SEC)
                _log(f"GPU {gpu} already idle after {tid} → assign immediately")


def _task_pgrep(task: QueueTask, gpu: int | None = None) -> tuple[str, ...]:
    """Resolve pgrep patterns; `{gpu}` is per-GPU so extra-fold can run in parallel."""
    out: list[str] = []
    for pat in task.pgrep:
        if "{gpu}" in pat:
            if gpu is None:
                continue
            out.append(pat.format(gpu=gpu))
        else:
            out.append(pat)
    return tuple(out)


def _launch(task: QueueTask, gpu: int) -> int | None:
    cmd = task.launch_cmd(gpu)
    script = next((Path(x) for x in cmd if str(x).endswith(".sh")), Path(cmd[-1]))
    if not script.is_file():
        _log(f"skip {task.label}: missing script {script.name}")
        return None
    if _pgrep_running(_task_pgrep(task, gpu)):
        _log(f"skip {task.label}: already running")
        return None
    log_path = VIS / task.log_name.format(gpu=gpu)
    env = os.environ.copy()
    if task.env:
        env.update(task.env(gpu))
    need = max(1, int(task.need_gpus or 1))
    if need <= 1:
        g = str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = g
        env["TASK1_CUDA_VISIBLE_DEVICES"] = g
        env["TASK1_GPUS"] = g
        env["TASK1_UDA_PRED_PER_GPU"] = "1"
        env["TASK1_MAE_SEQ_GPUS"] = g
        env["TASK1_MAE_FT_GPU_LIST"] = g
        env["TASK1_SEGANY_GPU_LIST"] = g
        env["TASK1_DOCKER_GPUS"] = f"device={g}"
        env["TASK1_PREFLIGHT_GPUS"] = g
    else:
        pool = ",".join(str(x) for x in GPU_IDS)
        env["CUDA_VISIBLE_DEVICES"] = pool
        env["TASK1_CUDA_VISIBLE_DEVICES"] = pool
        env["TASK1_GPUS"] = pool
        env["TASK1_MAE_FT_GPU_LIST"] = pool.replace(",", " ")
        env["TASK1_MAE_SEQ_GPUS"] = pool
        env["TASK1_DOCKER_GPUS"] = f"device={pool}"
    _log(f"LAUNCH gpu={gpu} need={need} {task.label} → {log_path.name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n--- gpu-idle-queue launch {datetime.now().isoformat()} gpu={gpu} ---\n")
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(CTRL),
            start_new_session=True,
        )
    pid_file = VIS / f"gpu_idle_{task.task_id}_gpu{gpu}.pid"
    pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    if task.task_id == "nnunet_fdg_test":
        agg = VIS / "fdg_test20" / "aggregate_nnunet.json"
        if agg.is_file() and not _aggregate_valid(agg):
            agg.unlink(missing_ok=True)
    patch: dict[str, Any] = {"updated_note": f"gpu-idle-queue: {task.label} on GPU {gpu}"}
    board_slot = TASK_BOARD_MAP.get(task.task_id)
    if board_slot and task.task_id != "extra_fold_9fold":
        mkey, stage = board_slot
        patch["methods"] = {
            mkey: {
                stage: {
                    "status": "running",
                    "device": "gpu",
                    "gpu_ids": str(gpu),
                    "note": f"running · gpu-idle-queue GPU {gpu}",
                }
            }
        }
        patch["gpu_idle_wait"] = {
            "method": None,
            "stage": None,
            "gpu": None,
            "remain_sec": None,
            "idle_total_sec": IDLE_SEC,
        }
    try:
        subprocess.run(
            [
                "python3",
                str(CTRL / "ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py"),
                "--board",
                str(BOARD),
                "--no-plot",
                "--patch-json",
                json.dumps(patch),
            ],
            capture_output=True,
            timeout=120,
            cwd=str(CTRL),
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return int(proc.pid)


def _fc70_stage_done(board: dict, task_id: str) -> bool:
    slot = TASK_BOARD_MAP.get(task_id)
    if not slot:
        return False
    mkey, stage = slot
    st = (board.get("methods") or {}).get(mkey, {}).get(stage) or {}
    mean = st.get("mean")
    return isinstance(mean, (int, float)) and mean == mean


def _fc70_already_tried(board: dict, task_id: str, state: dict[str, Any]) -> bool:
    slot = TASK_BOARD_MAP.get(task_id)
    if slot:
        mkey, stage = slot
        st = (board.get("methods") or {}).get(mkey, {}).get(stage) or {}
        if (st.get("stamp") or "").strip():
            return True
    launched = state.get("launched") or {}
    if any(str(k).startswith(f"{task_id}@") for k in launched):
        return True
    return task_id in (state.get("fc70_deferred") or {})


def _should_defer_fc70(task: QueueTask, board: dict, state: dict[str, Any]) -> bool:
    """Skip incomplete fc70 retries until extra-fold 9fold finishes."""
    if task.task_id not in FC70_DEFER_WHILE_EXTRA:
        return False
    if _fc70_stage_done(board, task.task_id):
        return False
    if not _extra_folds_pending(board):
        return False
    return _fc70_already_tried(board, task.task_id, state)


def _mark_fc70_deferred(task: QueueTask, state: dict[str, Any]) -> None:
    rec = state.setdefault("fc70_deferred", {})
    if task.task_id in rec:
        return
    rec[task.task_id] = {
        "at": datetime.now().strftime("%F %T"),
        "reason": "incomplete; extra-fold 9fold first",
    }
    _log(f"DEFER {task.label}: extra-fold 9fold first (fc70 incomplete)")
    try:
        FC70_DEFER_FILE.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    slot = TASK_BOARD_MAP.get(task.task_id)
    if not slot:
        return
    mkey, stage = slot
    try:
        board = json.loads(BOARD.read_text(encoding="utf-8")) if BOARD.is_file() else {}
        st0 = (board.get("methods") or {}).get(mkey, {}).get(stage) or {}
        if (st0.get("status") or "").lower() == "running":
            return
    except (OSError, json.JSONDecodeError):
        pass
    patch = {
        "updated_note": f"DEFER {task.label} · extra-fold 9fold first",
        "methods": {
            mkey: {
                stage: {
                    "status": "pending",
                    "note": "defer · extra-fold 9fold first (fc70 incomplete)",
                }
            }
        },
    }
    try:
        subprocess.run(
            [
                "python3",
                str(CTRL / "ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py"),
                "--board",
                str(BOARD),
                "--no-plot",
                "--patch-json",
                json.dumps(patch),
            ],
            capture_output=True,
            timeout=120,
            cwd=str(CTRL),
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _pick_task(
    board: dict,
    tasks: list[QueueTask],
    state: dict[str, Any] | None = None,
    gpu: int | None = None,
    n_idle: int | None = None,
) -> QueueTask | None:
    state = state if state is not None else {}
    skipped = state.get("skipped") or {}
    for t in tasks:
        if t.task_id in skipped and t.task_id not in NEVER_SKIP_TASKS:
            continue
        if not t.pending(board):
            continue
        if _pgrep_running(_task_pgrep(t, gpu)):
            continue
        if _should_defer_fc70(t, board, state):
            _mark_fc70_deferred(t, state)
            continue
        if t.task_id == "seganypet_fc70":
            script = CTRL / "ICLR2026/run/run_seganypet_psma_fc70_from_fdg_bg.sh"
            if not script.is_file():
                continue
        need = max(1, int(t.need_gpus or 1))
        if n_idle is not None and need > n_idle:
            continue
        return t
    return None


def _blocked_need_msg(
    board: dict, tasks: list[QueueTask], state: dict[str, Any], n_idle: int
) -> str:
    skipped = state.get("skipped") or {}
    for t in tasks:
        if t.task_id in skipped and t.task_id not in NEVER_SKIP_TASKS:
            continue
        if not t.pending(board):
            continue
        if _pgrep_running(_task_pgrep(t)):
            continue
        need = max(1, int(t.need_gpus or 1))
        if need > n_idle:
            return f"next {t.task_id} need={need} idle={n_idle}"
    return "queue empty"


TASK_BOARD_MAP: dict[str, tuple[str, str]] = {
    "seganypet_fc70": ("seganypet", "psma_fc70"),
    "dpdnet_fc70": ("dpdnet", "psma_fc70"),
    "monai_fc70": ("monai_swinvit", "psma_fc70"),
    "mae_fc70": ("mae_swinunetr", "psma_fc70"),
    "mae_scratch_fc70": ("mae_scratch", "psma_fc70"),
    "monai_scratch_fc70": ("monai_scratch", "psma_fc70"),
    "seganypet_scratch_fc70": ("seganypet_scratch", "psma_fc70"),
    "nnunet_fc70": ("nnunet", "psma_fc70"),
    "nnunet_fc70_test": ("nnunet", "psma_fc70"),
    "nnunet_fdg_test": ("nnunet", "fdg_test20"),
    "mae_fdg_test": ("mae_swinunetr", "fdg_test20"),
    "monai_fdg_test": ("monai_swinvit", "fdg_test20"),
    "proto_fc70": ("proto_retrieval", "psma_fc70"),
    "mae_psma_fpfn_fs50": ("mae_swinunetr", "psma_fs50_f258"),
    "mae_psma_fpfn_fs10": ("mae_swinunetr", "psma_fs10_f258"),
    "mae_psma_fpfn_fs5": ("mae_swinunetr", "psma_fs5_f258"),
    "monai_psma_fpfn_fs50": ("monai_swinvit", "psma_fs50_f258"),
    "monai_psma_fpfn_fs10": ("monai_swinvit", "psma_fs10_f258"),
    "monai_psma_fpfn_fs5": ("monai_swinvit", "psma_fs5_f258"),
    "mae_psma_fs0_fpfn": ("mae_swinunetr", "psma_fs0"),
    "mae_scratch_psma_fpfn_fs50": ("mae_scratch", "psma_fs50_f258"),
    "mae_scratch_psma_fpfn_fs10": ("mae_scratch", "psma_fs10_f258"),
    "mae_scratch_psma_fpfn_fs5": ("mae_scratch", "psma_fs5_f258"),
    "monai_scratch_psma_fpfn_fs50": ("monai_scratch", "psma_fs50_f258"),
    "monai_scratch_psma_fpfn_fs10": ("monai_scratch", "psma_fs10_f258"),
    "monai_scratch_psma_fpfn_fs5": ("monai_scratch", "psma_fs5_f258"),
    "mae_scratch_psma_fs0_fpfn": ("mae_scratch", "psma_fs0"),
    "monai_psma_fs0_fpfn": ("monai_swinvit", "psma_fs0"),
    "monai_scratch_psma_fs0_fpfn": ("monai_scratch", "psma_fs0"),
    "nnunet_psma_fs0_fpfn": ("nnunet", "psma_fs0"),
    "dpdnet_psma_fs0_fpfn": ("dpdnet", "psma_fs0"),
    "seganypet_psma_fs0_fpfn": ("seganypet", "psma_fs0"),
    "seganypet_scratch_psma_fs0_fpfn": ("seganypet_scratch", "psma_fs0"),
    "extra_fold_9fold": ("mae_swinunetr", "psma_fs50_f258"),
    "extra_fold_test20": ("mae_swinunetr", "psma_fs50_f258"),
    "mae_scratch_9fold": ("mae_scratch", "fdg_pretrain"),
    "monai_scratch_9fold": ("monai_scratch", "fdg_pretrain"),
    "seganypet_scratch_9fold": ("seganypet_scratch", "fdg_pretrain"),
    "nnunet_mim_dpdnet_dualenc": ("nnunet_mim", "fdg_pretrain"),
    "dpdnet_dualenc_fdg": ("dpdnet_dualenc", "fdg_pretrain"),
    "nnunet_mim_remaining": ("nnunet_mim", "psma_fs10_f258"),
}


def _fmt_remain(secs: float) -> str:
    secs = max(0, int(round(secs)))
    m, s = divmod(secs, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _write_wait_snapshot(
    state: dict[str, Any],
    idle_since: dict[str, float],
    next_task: QueueTask | None,
    now: float,
) -> None:
    """Persist GPU idle countdown + next task for board WAITING display."""
    gpus: list[dict[str, Any]] = []
    for gpu in GPU_IDS:
        key = str(gpu)
        if key not in idle_since:
            continue
        elapsed = max(0.0, now - float(idle_since[key]))
        remain = max(0.0, float(IDLE_SEC) - elapsed)
        mem = _gpu_mem_mib(gpu)
        gpus.append(
            {
                "gpu": gpu,
                "elapsed_sec": int(elapsed),
                "remain_sec": int(remain),
                "idle_total_sec": IDLE_SEC,
                "mem_mib": mem,
            }
        )
    method = stage = None
    if next_task and next_task.task_id in TASK_BOARD_MAP:
        method, stage = TASK_BOARD_MAP[next_task.task_id]
    wait = {
        "at": datetime.now().strftime("%F %T"),
        "idle_total_sec": IDLE_SEC,
        "next_task_id": next_task.task_id if next_task else None,
        "next_label": next_task.label if next_task else None,
        "method": method,
        "stage": stage,
        "gpus": gpus,
    }
    # Prefer GPU with smallest remain (soonest launch)
    if gpus:
        soon = min(gpus, key=lambda g: int(g.get("remain_sec") or 0))
        wait["gpu"] = soon["gpu"]
        wait["remain_sec"] = soon["remain_sec"]
        wait["elapsed_sec"] = soon["elapsed_sec"]
    else:
        wait["gpu"] = None
        wait["remain_sec"] = None
        wait["elapsed_sec"] = None
    state["wait"] = wait
    wait_file = VIS / "gpu_idle_wait.json"
    try:
        wait_file.write_text(json.dumps(wait, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _patch_board_waiting(wait: dict[str, Any]) -> None:
    """Mark next queued cell as WAITING with GPU countdown (or clear)."""
    method = wait.get("method")
    stage = wait.get("stage")
    gpu = wait.get("gpu")
    remain = wait.get("remain_sec")
    label = wait.get("next_label") or ""
    if not method or not stage or gpu is None or remain is None:
        # clear any previous waiting marks via board refresh path
        try:
            subprocess.run(
                [
                    "python3",
                    str(CTRL / "ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py"),
                    "--board",
                    str(BOARD),
                    "--no-plot",
                    "--patch-json",
                    json.dumps({"gpu_idle_wait": wait}),
                ],
                capture_output=True,
                timeout=120,
                cwd=str(CTRL),
            )
        except (OSError, subprocess.SubprocessError):
            pass
        return
    # Do not overwrite RUNNING or a task that already grabbed the GPU
    try:
        board = json.loads(BOARD.read_text(encoding="utf-8"))
        st0 = (board.get("methods") or {}).get(method, {}).get(stage) or {}
        if (st0.get("status") or "").lower() == "running":
            return
    except (OSError, json.JSONDecodeError):
        pass
    tid = wait.get("next_task_id")
    gpu_i = int(gpu) if gpu is not None else None
    if tid:
        for t in _build_tasks():
            if t.task_id == tid and _pgrep_running(_task_pgrep(t, gpu_i)):
                return
    mem = _gpu_mem_mib(int(gpu))
    if mem is not None and mem >= IDLE_MEM_MIB and int(remain) <= 0:
        return
    remain_s = _fmt_remain(float(remain))
    note = f"WAITING (GPU {gpu} · {remain_s})"
    patch = {
        "gpu_idle_wait": wait,
        "updated_note": f"WAITING · {label} → GPU {gpu} · {remain_s}",
        "methods": {
            method: {
                stage: {
                    "status": "waiting",
                    "device": "gpu",
                    "gpu_ids": str(gpu),
                    "wait_gpu": int(gpu),
                    "wait_sec": int(remain),
                    "wait_total": int(wait.get("idle_total_sec") or IDLE_SEC),
                    "wait_eta": remain_s,
                    "wait_from": "gpu_idle",
                    "note": note,
                }
            }
        },
    }
    try:
        subprocess.run(
            [
                "python3",
                str(CTRL / "ICLR2026/scripts/iclr2026_aligned_fdg_fs50_board.py"),
                "--board",
                str(BOARD),
                "--no-plot",
                "--patch-json",
                json.dumps(patch),
            ],
            capture_output=True,
            timeout=120,
            cwd=str(CTRL),
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _tick(state: dict[str, Any], tasks: list[QueueTask]) -> None:
    board = _load_board()
    if not _extra_folds_pending(board) and state.get("fc70_deferred"):
        _log("extra-fold 9fold done — resume deferred fc70")
        state["fc70_deferred"] = {}
        try:
            FC70_DEFER_FILE.unlink(missing_ok=True)
        except OSError:
            pass
    now = time.time()
    idle_since: dict[str, float] = {k: float(v) for k, v in (state.get("idle_since") or {}).items()}
    # drop stale GPU keys not in pool
    for k in list(idle_since.keys()):
        try:
            if int(k) not in GPU_IDS:
                idle_since.pop(k, None)
        except ValueError:
            idle_since.pop(k, None)

    _reap_active(state, tasks, idle_since, now, board)
    if not _extra_folds_pending(board):
        _maybe_mark_extra_folds_done(tasks)

    for gpu in GPU_IDS:
        key = str(gpu)
        if _slot_held(state, gpu, tasks, now):
            idle_since.pop(key, None)
            continue
        mem = _gpu_mem_mib(gpu)
        if mem is None:
            idle_since.pop(key, None)
            continue
        if mem >= IDLE_MEM_MIB:
            if key in idle_since:
                _log(f"GPU {gpu} busy again ({mem} MiB) — reset idle timer")
            idle_since.pop(key, None)
            continue
        if key not in idle_since:
            idle_since[key] = now
            _log(f"GPU {gpu} idle start ({mem} MiB < {IDLE_MEM_MIB} MiB)")
            continue
        elapsed = now - idle_since[key]
        if elapsed < IDLE_SEC:
            remain = int(IDLE_SEC - elapsed)
            if int(elapsed) % max(POLL_SEC, 60) < POLL_SEC:
                _log(f"GPU {gpu} idle {int(elapsed)}s / {IDLE_SEC}s ({mem} MiB) · {remain}s to assign")
            continue
        n_idle = len(_free_gpu_ids(state, tasks, now))
        task = _pick_task(board, tasks, state, gpu, n_idle=n_idle)
        if task is None:
            if int(elapsed) % 60 < POLL_SEC:
                why = _blocked_need_msg(board, tasks, state, n_idle)
                _log(
                    f"GPU {gpu} idle {int(elapsed)}s · idle_gpus={n_idle} · "
                    f"no matching {n_idle}-GPU task ({why})"
                )
            continue
        pid = _launch(task, gpu)
        if pid:
            rec = {
                "task_id": task.task_id,
                "pid": int(pid),
                "t0": now,
                "at": datetime.now().strftime("%F %T"),
            }
            hold = [gpu]
            if int(task.need_gpus or 1) > 1:
                hold = list(dict.fromkeys([gpu, * _free_gpu_ids(state, tasks, now)]))
            for g in hold:
                idle_since.pop(str(g), None)
                state.setdefault("active", {})[str(g)] = dict(rec)
            state.setdefault("launched", {})[
                f"{task.task_id}@gpu{gpu}"
            ] = datetime.now().isoformat(timespec="seconds")
            board = _load_board()

    n_idle_wait = len(_free_gpu_ids(state, tasks, now))
    next_task = _pick_task(board, tasks, state, n_idle=n_idle_wait) if idle_since else None
    _write_wait_snapshot(state, idle_since, next_task, now)
    state["idle_since"] = idle_since
    state["last_tick"] = datetime.now().strftime("%F %T")
    _save_state(state)
    # board WAITING cell + countdown (every tick while idle GPUs exist)
    if idle_since and next_task:
        _patch_board_waiting(state.get("wait") or {})
    elif state.get("wait"):
        # clear wait snapshot display
        empty = {
            "at": datetime.now().strftime("%F %T"),
            "idle_total_sec": IDLE_SEC,
            "next_task_id": None,
            "next_label": None,
            "method": None,
            "stage": None,
            "gpus": [],
            "gpu": None,
            "remain_sec": None,
            "elapsed_sec": None,
        }
        state["wait"] = empty
        _save_state(state)
        _patch_board_waiting(empty)


def main() -> int:
    hold = Path(os.environ.get("TASK1_GPU_IDLE_HOLD", VIS / "TASK1_REMAINING_SCHEDULE_HOLD.txt"))
    if hold.is_file():
        _log(f"hold {hold} — idle queue disabled (explicit remaining schedule)")
        return 0
    _log(
        f"start gpus={GPU_IDS} idle_mem<{IDLE_MEM_MIB}MiB for {IDLE_SEC}s poll={POLL_SEC}s board={BOARD}"
    )
    tasks = _build_tasks()
    state = _load_state()
    _hydrate_active(state, tasks)
    while True:
        try:
            _tick(state, tasks)
        except Exception as exc:
            _log(f"ERROR {exc!r}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _log("stopped")
        raise SystemExit(0)
