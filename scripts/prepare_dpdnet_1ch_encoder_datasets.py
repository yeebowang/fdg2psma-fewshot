#!/usr/bin/env python3
"""Build 1-channel FDG raw datasets for local dual-encoder retrain (CT=240, PET=241).

Official UniSeg ``best_encoder_{ct,pet}_epoch_94.pth`` is not published. We retrain
STUNet-small (prompt) encoders on AutoPET FDG CT-only / PET-only, then extract
``conv_blocks_context`` into the expected filenames.

  python3 ICLR2026/scripts/prepare_dpdnet_1ch_encoder_datasets.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def _build_one(
    work: Path,
    src_name: str,
    dst_id: int,
    dst_name: str,
    channel: int,
    cases: list[str],
    pref: str,
) -> dict:
    src_raw = work / "nnUNet_raw" / src_name
    dst_raw = work / "nnUNet_raw" / dst_name
    (dst_raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dst_raw / "labelsTr").mkdir(parents=True, exist_ok=True)
    ch_name = "CT" if channel == 0 else "PET"
    ds = {
        "channel_names": {"0": ch_name},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    (dst_raw / "dataset.json").write_text(json.dumps(ds, indent=2) + "\n")
    missing = []
    renamed = []
    for case in cases:
        new_case = f"{pref}_{case}"
        src_img = src_raw / "imagesTr" / f"{case}_{channel:04d}.nii.gz"
        dst_img = dst_raw / "imagesTr" / f"{new_case}_0000.nii.gz"
        src_lab = src_raw / "labelsTr" / f"{case}.nii.gz"
        dst_lab = dst_raw / "labelsTr" / f"{new_case}.nii.gz"
        if not src_img.is_file() or not src_lab.is_file():
            missing.append(case)
            continue
        _link(src_img, dst_img)
        _link(src_lab, dst_lab)
        renamed.append(new_case)
    ds["numTraining"] = len(renamed)
    (dst_raw / "dataset.json").write_text(json.dumps(ds, indent=2) + "\n")
    meta = {
        "dst_id": dst_id,
        "dst": dst_name,
        "channel": channel,
        "channel_name": ch_name,
        "n": len(renamed),
        "n_missing": len(missing),
        "missing": missing[:10],
        "splits_n": len(renamed),
    }
    (dst_raw / "prepare_1ch_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    if missing:
        raise SystemExit(f"[error] {dst_name}: missing {len(missing)} source files")
    return {"train": renamed, "val": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace"))
    ap.add_argument(
        "--splits",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_fdg_nnunet.json"),
    )
    ap.add_argument("--src-id", type=int, default=228)
    ap.add_argument("--ct-id", type=int, default=250)
    ap.add_argument("--pet-id", type=int, default=251)
    ap.add_argument("--prompt-prefix", default="lymp")
    args = ap.parse_args()
    src_name = f"Dataset{args.src_id}_AutoPETIV_Task1_2ch"
    splits = json.loads(args.splits.read_text())
    fold0 = splits[0] if isinstance(splits, list) else splits
    cases = list(fold0.get("train") or []) + list(fold0.get("val") or [])
    if not cases:
        raise SystemExit("[error] empty FDG split")
    n_tr = max(1, int(0.9 * len(cases)))
    train, val = cases[:n_tr], cases[n_tr:]
    pref = args.prompt_prefix
    assert len(pref) == 4
    ct_name = f"Dataset{args.ct_id}_DpDNet_FDG_CT1ch"
    pet_name = f"Dataset{args.pet_id}_DpDNet_FDG_PET1ch"
    _build_one(args.work, src_name, args.ct_id, ct_name, 0, cases, pref)
    _build_one(args.work, src_name, args.pet_id, pet_name, 1, cases, pref)
    # 90/10 split written after preprocess; keep raw-side hint
    hint = {
        "train": [f"{pref}_{c}" for c in train],
        "val": [f"{pref}_{c}" for c in val],
    }
    for name in (ct_name, pet_name):
        (args.work / "nnUNet_raw" / name / "splits_hint.json").write_text(
            json.dumps([hint], indent=2) + "\n"
        )
    print(f"[ok] raw 1ch CT={ct_name} PET={pet_name} n={len(cases)} tr/val={len(train)}/{len(val)}")


if __name__ == "__main__":
    main()
