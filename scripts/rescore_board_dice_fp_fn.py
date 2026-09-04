#!/usr/bin/env python3
"""Rescore existing TEST preds → Dice (empty-GT excl.) + FP/FN; patch board aggregates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CTRL = Path("/media/ybwang/data1/PSMA-CTRL")
DATA = Path("/media/ybwang/data1/PSMA-DATA")
WORK = DATA / "task1_train_workspace"
GT = DATA / "dataset1/labelsTr"
VIS = CTRL / "ICLR2026/vis"
SCR = CTRL / "ICLR2026/scripts"
IMAGE = "iclr2026_3dmae_petct:cu118"
FDG_CASES = CTRL / "ICLR2026/data/splits_fdg_test20.json"
PSMA_CASES = CTRL / "ICLR2026/data/splits_mae_psma_test20.json"


def _agg_has_fp_fn(method_key: str, agg_dir: Path | None = None) -> bool:
    path = (agg_dir or (VIS / "fdg_test20")) / f"aggregate_{method_key}.json"
    if not path.is_file():
        return False
    try:
        ad = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    fp, fn = ad.get("fp_rate", ad.get("mean_fp")), ad.get("fn_rate", ad.get("mean_fn"))
    return isinstance(fp, (int, float)) and fp == fp and isinstance(fn, (int, float)) and fn == fn


def _merge_nnunet_shards(predict_root: Path) -> Path:
    """Hardlink/copy shard preds into predict/pred (no re-inference)."""
    pred = predict_root / "pred"
    pred.mkdir(parents=True, exist_ok=True)
    shards = predict_root / "shards"
    n_before = len(list(pred.glob("*.nii.gz")))
    if shards.is_dir():
        for src in shards.glob("shard_*/pred/*.nii.gz"):
            dst = pred / src.name
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                import shutil

                shutil.copy2(src, dst)
    n_after = len(list(pred.glob("*.nii.gz")))
    print(f"[merge] nnunet shards → {pred} {n_before}→{n_after}")
    return pred


SEGANY_PSMA_GT = DATA / "task1_train_workspace/seganypet_psma_test20/labelsVal"
SEGANY_RUNS = CTRL / "ICLR2026/3D-MAE-PET-CT/runs"


def _docker_score(
    cases: Path,
    pred: Path,
    out: Path,
    tag: str,
    workers: int = 8,
    gt_dir: Path | None = None,
) -> dict | None:
    if not pred.is_dir():
        print(f"[skip] no pred dir {pred}")
        return None
    n = len(list(pred.glob("*.nii.gz")))
    if n <= 0:
        print(f"[skip] empty pred {pred}")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    gt = gt_dir or GT
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{__import__('os').getuid()}:{__import__('os').getgid()}",
        "-v",
        f"{CTRL}:{CTRL}",
        "-v",
        f"{DATA}:{DATA}",
        "-w",
        str(CTRL),
        IMAGE,
        "python3",
        str(SCR / "score_pred_dice_vs_gt.py"),
        "--cases-json",
        str(cases),
        "--pred-dir",
        str(pred),
        "--gt-dir",
        str(gt),
        "--out-json",
        str(out),
        "--tag",
        tag,
        "--workers",
        str(workers),
    ]
    print(f"[score] {tag} n_pred≈{n} gt={gt} → {out}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:] if r.stderr else r.stdout[-1500:])
        return None
    print(r.stdout.strip().splitlines()[-1] if r.stdout else "")
    return json.loads(out.read_text())


def _write_fdg_agg(method_key: str, score: dict, extra: dict | None = None) -> None:
    agg_dir = VIS / "fdg_test20"
    agg_dir.mkdir(parents=True, exist_ok=True)
    path = agg_dir / f"aggregate_{method_key}.json"
    old = json.loads(path.read_text()) if path.is_file() else {"method": method_key}
    md = float(score["mean_dice"])
    old.update(
        {
            "mean_dice": md,
            "mean_dice_positive": md,
            "fp_rate": score.get("fp_rate"),
            "fn_rate": score.get("fn_rate"),
            "mean_fp": score.get("fp_rate"),
            "mean_fn": score.get("fn_rate"),
            "n_scored": score.get("n_scored"),
            "n_positive": score.get("n_positive"),
            "n_empty_gt": score.get("n_empty_gt"),
            "empty_gt_excluded": True,
            "protocol": old.get("protocol") or "fdg_shared_ckpt_fdg_test20",
            "split": "FDG_TEST20",
        }
    )
    if extra:
        old.update(extra)
    path.write_text(json.dumps(old, indent=2) + "\n")
    print(f"[agg] wrote {path.name} Dice={md:.4f} FP={old.get('fp_rate')} FN={old.get('fn_rate')}")


def _patch_stage(board: dict, method: str, stage: str, score: dict, fold: str | None = None) -> None:
    st = board.setdefault("methods", {}).setdefault(method, {}).setdefault(stage, {})
    md = float(score["mean_dice"])
    if fold is None:
        st["mean"] = md
        st["fold_dice"] = {"0": md}
    else:
        fd = dict(st.get("fold_dice") or {})
        fd[str(fold)] = md
        st["fold_dice"] = fd
        vals = [float(v) for v in fd.values() if isinstance(v, (int, float)) and v == v]
        if len(vals) >= 3 or (len(vals) == 1 and fold == "0"):
            st["mean"] = sum(vals) / len(vals)
    st["mean_fp"] = float(score["fp_rate"]) if score.get("fp_rate") == score.get("fp_rate") else st.get("mean_fp")
    st["mean_fn"] = float(score["fn_rate"]) if score.get("fn_rate") == score.get("fn_rate") else st.get("mean_fn")
    st["status"] = "done"
    st["note"] = (
        f"rescored · Dice={100*md:.2f}% "
        f"FP={100*float(score.get('fp_rate') or float('nan')):.2f}% "
        f"FN={100*float(score.get('fn_rate') or float('nan')):.2f}%"
    )


def _rescore_fdg(board: dict) -> None:
    root = WORK / "fdg_test20_eval"
    # Prefer existing nifti; skip methods that already have fp/fn in aggregate.
    jobs = [
        ("nnunet", root / "nnunet/predict/pred"),
        ("dpdnet", root / "dpdnet/predict"),
        ("seganypet", root / "seganypet/pred"),
        ("proto_retrieval", root / "proto_retrieval/predict"),
    ]
    tmp = VIS / "_tmp_scores"
    for method, pred in jobs:
        if _agg_has_fp_fn(method):
            print(f"[skip] {method} FDG aggregate already has FP/FN")
            continue
        if method == "nnunet":
            pred = _merge_nnunet_shards(root / "nnunet/predict")
        score = _docker_score(FDG_CASES, pred, tmp / f"fdg_{method}.json", f"fdg_{method}")
        if not score or not score.get("n_scored"):
            continue
        _write_fdg_agg(method, score)
        _patch_stage(board, method, "fdg_test20", score)


def _stage_needs_fp_fn(st: dict) -> bool:
    fp, fn = st.get("mean_fp"), st.get("mean_fn")
    return not (
        isinstance(fp, (int, float))
        and fp == fp
        and isinstance(fn, (int, float))
        and fn == fn
    )


def _rescore_nnunet_fewshot(board: dict) -> None:
    nn = board.get("methods", {}).get("nnunet") or {}
    for stage, few in (
        ("psma_fs50_f258", 50),
        ("psma_fs10_f258", 10),
        ("psma_fs5_f258", 5),
    ):
        st0 = nn.get(stage) or {}
        if not _stage_needs_fp_fn(st0) and isinstance(st0.get("mean"), (int, float)):
            print(f"[skip] nnunet {stage} already has FP/FN")
            continue
        stamp = st0.get("stamp") or ""
        if not stamp:
            continue
        eval_root = WORK / "nnUNet_results" / stamp / "psma_test20_eval"
        fold_scores = {}
        for f in (2, 5, 8):
            pred = eval_root / f"fold{f}/predict/pred"
            score = _docker_score(
                PSMA_CASES,
                pred,
                VIS / "_tmp_scores" / f"nn_{few}_f{f}.json",
                f"nn_fs{few}_f{f}",
            )
            if score and score.get("n_scored"):
                fold_scores[str(f)] = score
                _patch_stage(board, "nnunet", stage, score, fold=str(f))
        if len(fold_scores) == 3:
            sum_fp = sum(int(s.get("sum_fp") or 0) for s in fold_scores.values())
            sum_fn = sum(int(s.get("sum_fn") or 0) for s in fold_scores.values())
            sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in fold_scores.values())
            sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in fold_scores.values())
            st = board["methods"]["nnunet"][stage]
            if sum_neg > 0:
                st["mean_fp"] = sum_fp / sum_neg
            if sum_pos > 0:
                st["mean_fn"] = sum_fn / sum_pos
            dices = [float(s["mean_dice"]) for s in fold_scores.values()]
            st["mean"] = sum(dices) / 3.0
            st["status"] = "done"
            print(f"[nnunet fs{few}] mean={st['mean']:.4f} fp={st.get('mean_fp')} fn={st.get('mean_fn')}")


def _rescore_dpdnet_fewshot(board: dict) -> None:
    dpd = board.get("methods", {}).get("dpdnet") or {}
    for stage in ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258"):
        st0 = dpd.get(stage) or {}
        if not _stage_needs_fp_fn(st0) and isinstance(st0.get("mean"), (int, float)):
            print(f"[skip] dpdnet {stage} already has FP/FN")
            continue
        stamp = st0.get("stamp") or ""
        if not stamp:
            continue
        eval_root = WORK / "nnUNet_results" / stamp / "psma_test20_eval"
        fold_scores = {}
        for f in (2, 5, 8):
            pred = eval_root / f"fold{f}/predict/pred"
            if not pred.is_dir():
                pred = eval_root / f"fold{f}/predict"
            score = _docker_score(
                PSMA_CASES,
                pred,
                VIS / "_tmp_scores" / f"dpd_{stage}_f{f}.json",
                f"dpd_{stage}_f{f}",
            )
            if score and score.get("n_scored"):
                fold_scores[str(f)] = score
                _patch_stage(board, "dpdnet", stage, score, fold=str(f))
        if len(fold_scores) == 3:
            sum_fp = sum(int(s.get("sum_fp") or 0) for s in fold_scores.values())
            sum_fn = sum(int(s.get("sum_fn") or 0) for s in fold_scores.values())
            sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in fold_scores.values())
            sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in fold_scores.values())
            st = board["methods"]["dpdnet"][stage]
            if sum_neg > 0:
                st["mean_fp"] = sum_fp / sum_neg
            if sum_pos > 0:
                st["mean_fn"] = sum_fn / sum_pos
            dices = [float(s["mean_dice"]) for s in fold_scores.values()]
            st["mean"] = sum(dices) / 3.0
            st["status"] = "done"


def _rescore_seganypet_psma(board: dict, stages: tuple[str, ...] | None = None) -> None:
    """CPU rescore SegAnyPET fold*_pred vs labelsVal → Dice/FP/FN."""
    seg = board.setdefault("methods", {}).setdefault("seganypet", {})
    todo = stages or (
        "psma_fs50_f258",
        "psma_fs10_f258",
        "psma_fs5_f258",
        "psma_fc70",
    )
    for stage in todo:
        st0 = seg.get(stage) or {}
        if not _stage_needs_fp_fn(st0) and isinstance(st0.get("mean"), (int, float)):
            print(f"[skip] seganypet {stage} already has FP/FN")
            continue
        stamp = (st0.get("stamp") or "").strip()
        if not stamp:
            print(f"[skip] seganypet {stage}: no stamp")
            continue
        eval_root = SEGANY_RUNS / stamp / "psma_test20_eval"
        if not eval_root.is_dir():
            print(f"[skip] seganypet {stage}: no eval_root {eval_root}")
            continue
        # Discover folds from fold*_pred or fold*_test20.json
        fold_ids: list[int] = []
        for p in sorted(eval_root.glob("fold*_pred")):
            try:
                fold_ids.append(int(p.name.replace("fold", "").replace("_pred", "")))
            except ValueError:
                pass
        if not fold_ids:
            for p in sorted(eval_root.glob("fold*_test20.json")):
                try:
                    fold_ids.append(int(p.stem.replace("fold", "").replace("_test20", "")))
                except ValueError:
                    pass
        if not fold_ids:
            print(f"[skip] seganypet {stage}: no fold preds yet")
            continue
        fold_scores: dict[str, dict] = {}
        fd: dict[str, float] = {}
        for f in fold_ids:
            pred = eval_root / f"fold{f}_pred"
            score = _docker_score(
                PSMA_CASES,
                pred,
                VIS / "_tmp_scores" / f"seganypet_{stage}_f{f}.json",
                f"seganypet_{stage}_f{f}",
                workers=10,
                gt_dir=SEGANY_PSMA_GT,
            )
            if score and score.get("n_scored"):
                fold_scores[str(f)] = score
                md = float(score["mean_dice"])
                fd[str(f)] = md
                _patch_stage(board, "seganypet", stage, score, fold=str(f))
        if not fold_scores:
            print(f"[warn] seganypet {stage}: score failed")
            continue
        sum_fp = sum(int(s.get("sum_fp") or 0) for s in fold_scores.values())
        sum_fn = sum(int(s.get("sum_fn") or 0) for s in fold_scores.values())
        sum_neg = sum(int(s.get("sum_neg_voxels") or 0) for s in fold_scores.values())
        sum_pos = sum(int(s.get("sum_pos_voxels") or 0) for s in fold_scores.values())
        st = seg.setdefault(stage, {})
        st["stamp"] = stamp
        st["fold_dice"] = fd
        st["mean"] = sum(fd.values()) / len(fd)
        if sum_neg > 0:
            st["mean_fp"] = sum_fp / sum_neg
        if sum_pos > 0:
            st["mean_fn"] = sum_fn / sum_pos
        st["status"] = "done"
        st["device"] = "cpu"
        st["metric"] = "TEST20 Dice/FP/FN (pred vs labelsVal)"
        fp, fn = st.get("mean_fp"), st.get("mean_fn")
        if isinstance(fp, (int, float)) and isinstance(fn, (int, float)):
            st["note"] = (
                f"TEST20 DONE · {100*float(st['mean']):.2f}%/"
                f"{100*float(fp):.2f}%/{100*float(fn):.2f}%"
            )
        agg = {
            "stamp": stamp,
            "method": "seganypet",
            "split": "PSMA_TEST20",
            "folds": fold_ids,
            "fold_test_dice": fd,
            "test_mean": st["mean"],
            "mean_dice": st["mean"],
            "mean_dice_positive": st["mean"],
            "fp_rate": fp,
            "fn_rate": fn,
            "mean_fp": fp,
            "mean_fn": fn,
            "sum_fp": sum_fp,
            "sum_fn": sum_fn,
            "sum_neg_voxels": sum_neg,
            "sum_pos_voxels": sum_pos,
            "empty_gt_excluded": True,
            "metric": "TEST20 Dice/FP/FN (pred vs labelsVal)",
        }
        (eval_root / "aggregate_test20_f258.json").write_text(json.dumps(agg, indent=2) + "\n")
        (VIS / f"aggregate_seganypet_psma_test20_f258_{stamp}.json").write_text(
            json.dumps(agg, indent=2) + "\n"
        )
        print(
            f"[seganypet {stage}] Dice={st['mean']:.4f} FP={fp} FN={fn} folds={fold_ids}"
        )


def _rescore_proto_fewshot_once(board: dict) -> None:
    """Proto fs50/fs10/fs5/fs0 share one FDG100%→PSMA TEST20 pred set — score once, mirror."""
    proto = board.setdefault("methods", {}).setdefault("proto_retrieval", {})
    stages = ("psma_fs50_f258", "psma_fs10_f258", "psma_fs5_f258", "psma_fs0")
    src = proto.get("psma_fs50_f258") or {}
    if not _stage_needs_fp_fn(src) and all(
        not _stage_needs_fp_fn(proto.get(s) or {}) for s in stages
    ):
        print("[skip] proto fewshot fs50/10/5/0 already have FP/FN")
        return
    stamp = (src.get("stamp") or "").strip()
    if not stamp:
        for agg in sorted(VIS.glob("aggregate_proto_retrieval_psma_test20_f258_*.json")):
            try:
                ad = json.loads(agg.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            stamp = str(ad.get("stamp") or "").strip()
            if stamp:
                break
    if not stamp:
        print("[skip] proto fewshot: no stamp")
        return
    eval_root = CTRL / "ICLR2026/runs/proto_retrieval" / stamp / "psma_test20_eval"
    # identical preds across folds — score fold2 once
    pred = eval_root / "fold2/predict"
    if not pred.is_dir() or not list(pred.glob("*.nii.gz")):
        for f in (5, 8):
            cand = eval_root / f"fold{f}/predict"
            if cand.is_dir() and list(cand.glob("*.nii.gz")):
                pred = cand
                break
    score = _docker_score(
        PSMA_CASES,
        pred,
        VIS / "_tmp_scores" / "proto_fdg100_psma_test20.json",
        "proto_fdg100_psma_once",
        workers=12,
    )
    if not score or not score.get("n_scored"):
        print("[warn] proto fewshot rescore failed")
        return
    md = float(score["mean_dice"])
    fp = score.get("fp_rate")
    fn = score.get("fn_rate")
    support = src.get("support_pool") or "FDG100%"
    topk = src.get("topk", 3)
    fd = {"2": md, "5": md, "8": md}
    note = (
        f"TEST20 DONE · {support} · topk={topk} · "
        f"{100*md:.2f}%/"
        f"{100*float(fp):.2f}%/"
        f"{100*float(fn):.2f}%"
        if isinstance(fp, (int, float)) and isinstance(fn, (int, float))
        else f"TEST20 DONE · {support} · {100*md:.2f}%"
    )
    for stage in stages:
        st = proto.setdefault(stage, {})
        st.update(
            {
                "status": "done",
                "device": "cpu",
                "training_free": True,
                "stamp": stamp,
                "fold_dice": dict(fd),
                "mean": md,
                "mean_fp": float(fp) if isinstance(fp, (int, float)) else None,
                "mean_fn": float(fn) if isinstance(fn, (int, float)) else None,
                "support_pool": support,
                "topk": topk,
                "note": note if stage != "psma_fs0" else f"same as fs50 · {note.split(' · ', 1)[-1]}",
                "metric": "TEST20 Dice/FP/FN; FDG100% retrieve (shared)",
            }
        )
    # update aggregate json for future ingest
    agg_path = VIS / f"aggregate_proto_retrieval_psma_test20_f258_{stamp}.json"
    old = json.loads(agg_path.read_text()) if agg_path.is_file() else {"stamp": stamp, "method": "proto_retrieval"}
    old.update(
        {
            "stamp": stamp,
            "method": "proto_retrieval",
            "split": "PSMA_TEST20",
            "support_pool": support,
            "topk": topk,
            "fold_test_dice": fd,
            "test_mean": md,
            "mean_dice": md,
            "mean_dice_positive": md,
            "fp_rate": fp,
            "fn_rate": fn,
            "mean_fp": fp,
            "mean_fn": fn,
            "n_scored": score.get("n_scored"),
            "n_positive": score.get("n_positive"),
            "empty_gt_excluded": True,
            "metric": "TEST20 Dice/FP/FN; retrieve FDG100% + prototype",
            "shared_once": True,
        }
    )
    agg_path.write_text(json.dumps(old, indent=2) + "\n")
    print(f"[proto fewshot×4] Dice={md:.4f} FP={fp} FN={fn} → mirrored fs50/10/5/0")


def _rescore_psma_fs0(board: dict) -> None:
    """CPU rescore PSMA fs0 nifti preds (nnunet/dpdnet/seganypet). MAE/MONAI have no nifti."""
    root = WORK / "fdg20_test_eval"
    agg_dir = VIS / "psma_fs0"
    agg_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("nnunet", root / "nnunet/predict/pred", PSMA_CASES, GT),
        ("dpdnet", root / "dpdnet/predict", PSMA_CASES, GT),
        ("seganypet", root / "seganypet/pred", PSMA_CASES, SEGANY_PSMA_GT),
    )
    tmp = VIS / "_tmp_scores"
    for method, pred, cases, gt in jobs:
        st0 = (board.get("methods") or {}).get(method, {}).get("psma_fs0") or {}
        if not _stage_needs_fp_fn(st0) and isinstance(st0.get("mean"), (int, float)):
            print(f"[skip] {method} psma_fs0 already has FP/FN")
            continue
        if method == "nnunet":
            pred = _merge_nnunet_shards(root / "nnunet/predict")
        score = _docker_score(
            cases,
            pred,
            tmp / f"psma_fs0_{method}.json",
            f"psma_fs0_{method}",
            workers=12,
            gt_dir=gt,
        )
        if not score or not score.get("n_scored"):
            print(f"[skip] {method} psma_fs0 score failed")
            continue
        agg = {
            "method": method,
            "mean_dice": score.get("mean_dice"),
            "mean_dice_positive": score.get("mean_dice_positive"),
            "fp_rate": score.get("fp_rate"),
            "fn_rate": score.get("fn_rate"),
            "mean_fp": score.get("fp_rate"),
            "mean_fn": score.get("fn_rate"),
            "n_scored": score.get("n_scored"),
            "protocol": "fdg_shared_ckpt_zero_shot_psma_test20",
            "split": "PSMA_TEST20",
        }
        (agg_dir / f"aggregate_{method}.json").write_text(json.dumps(agg, indent=2) + "\n")
        _patch_stage(board, method, "psma_fs0", score)
        st = board["methods"][method]["psma_fs0"]
        st["training_free"] = True
        st["note"] = (
            f"PSMA fs0 · FDG ckpt · {100*float(score['mean_dice']):.2f}%/"
            f"{100*float(score.get('fp_rate') or float('nan')):.2f}%/"
            f"{100*float(score.get('fn_rate') or float('nan')):.2f}%"
        )
        print(f"[psma_fs0 {method}] Dice={score.get('mean_dice')} FP={score.get('fp_rate')} FN={score.get('fn_rate')}")


def _rescore_seganypet_scratch_tail(board: dict) -> None:
    """CPU rescore existing SegAnyPET scratch fs0 / FDG TEST nifti → Dice/FP/FN."""
    tmp = VIS / "_tmp_scores"
    jobs = (
        (
            "psma_fs0",
            WORK / "psma_fs0_eval/seganypet_scratch/pred",
            PSMA_CASES,
            SEGANY_PSMA_GT,
            VIS / "psma_fs0" / "aggregate_seganypet_scratch.json",
        ),
        (
            "fdg_test20",
            WORK / "fdg_test20_eval/seganypet_scratch/pred",
            FDG_CASES,
            Path(os.environ.get("TASK1_MAE_LABELS_TR", str(DATA / "dataset1/labelsTr"))),
            VIS / "fdg_test20" / "aggregate_seganypet_scratch.json",
        ),
    )
    for stage, pred, cases, gt, agg_path in jobs:
        st0 = (board.get("methods") or {}).get("seganypet_scratch", {}).get(stage) or {}
        if not _stage_needs_fp_fn(st0) and isinstance(st0.get("mean"), (int, float)):
            print(f"[skip] seganypet_scratch {stage} already has FP/FN")
            continue
        score = _docker_score(
            cases,
            pred,
            tmp / f"seganypet_scratch_{stage}.json",
            f"seganypet_scratch_{stage}",
            workers=12,
            gt_dir=gt,
        )
        if not score or not score.get("n_scored"):
            print(f"[skip] seganypet_scratch {stage} score failed")
            continue
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(agg_path.read_text()) if agg_path.is_file() else {"method": "seganypet_scratch"}
        old.update(
            {
                "method": "seganypet_scratch",
                "mean_dice": score.get("mean_dice"),
                "mean_dice_positive": score.get("mean_dice_positive", score.get("mean_dice")),
                "fp_rate": score.get("fp_rate"),
                "fn_rate": score.get("fn_rate"),
                "mean_fp": score.get("fp_rate"),
                "mean_fn": score.get("fn_rate"),
                "n_scored": score.get("n_scored"),
                "n_positive": score.get("n_positive"),
                "empty_gt_excluded": True,
                "split": "PSMA_TEST20" if stage == "psma_fs0" else "FDG_TEST20",
            }
        )
        agg_path.write_text(json.dumps(old, indent=2) + "\n")
        _patch_stage(board, "seganypet_scratch", stage, score)
        st = board["methods"]["seganypet_scratch"][stage]
        st["training_free"] = True
        md = float(score["mean_dice"])
        fp, fn = score.get("fp_rate"), score.get("fn_rate")
        if stage == "psma_fs0":
            st["note"] = (
                f"PSMA fs0 · FDG ckpt · {100*md:.2f}%/"
                f"{100*float(fp or float('nan')):.2f}%/{100*float(fn or float('nan')):.2f}%"
            )
        else:
            st["note"] = (
                f"FDG TEST · {100*md:.2f}%/"
                f"{100*float(fp or float('nan')):.2f}%/{100*float(fn or float('nan')):.2f}%"
            )
        print(f"[seganypet_scratch {stage}] Dice={md} FP={fp} FN={fn}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=VIS / "iclr2026_aligned_fdg_fs50_f258_board.json")
    ap.add_argument("--proto-fewshot-only", action="store_true", help="only rescore Proto fs50/10/5/0 once")
    ap.add_argument(
        "--seganypet-only",
        action="store_true",
        help="only rescore SegAnyPET PSMA TEST preds (fs50/10/5 + fc70 if present)",
    )
    ap.add_argument(
        "--seganypet-stages",
        default="",
        help="comma stages for --seganypet-only (default: fs50,fs10,fs5; add psma_fc70 if preds ready)",
    )
    ap.add_argument("--fs0-only", action="store_true", help="only CPU-rescore PSMA fs0 nifti (nnunet/dpdnet/seganypet)")
    ap.add_argument(
        "--seganypet-scratch-tail",
        action="store_true",
        help="CPU rescore seganypet_scratch PSMA fs0 + FDG TEST20 existing preds",
    )
    args = ap.parse_args()
    board = json.loads(args.board.read_text())
    if args.proto_fewshot_only:
        _rescore_proto_fewshot_once(board)
        board["updated_note"] = "CPU · Proto fs50/10/5/0 Dice/FP/FN (shared once)"
    elif args.seganypet_only:
        stages = tuple(s.strip() for s in args.seganypet_stages.split(",") if s.strip()) or (
            "psma_fs50_f258",
            "psma_fs10_f258",
            "psma_fs5_f258",
        )
        _rescore_seganypet_psma(board, stages=stages)
        board["updated_note"] = f"CPU · SegAnyPET {'/'.join(stages)} Dice/FP/FN"
    elif args.seganypet_scratch_tail:
        _rescore_seganypet_scratch_tail(board)
        board["updated_note"] = "CPU · SegAnyPET scratch fs0 + FDG TEST Dice/FP/FN"
    elif args.fs0_only:
        _rescore_psma_fs0(board)
        board["updated_note"] = "CPU · PSMA fs0 nnunet/dpdnet/seganypet Dice/FP/FN"
    else:
        _rescore_fdg(board)
        _rescore_nnunet_fewshot(board)
        _rescore_dpdnet_fewshot(board)
        _rescore_proto_fewshot_once(board)
        _rescore_seganypet_psma(board)
        _rescore_psma_fs0(board)
        board["updated_note"] = "rescored Dice/FP/FN (empty-GT excl. from Dice)"
    args.board.write_text(json.dumps(board, indent=2) + "\n")
    print(f"[done] board updated {args.board}")


if __name__ == "__main__":
    main()
