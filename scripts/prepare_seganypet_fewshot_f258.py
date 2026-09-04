#!/usr/bin/env python3
"""Prepare SegAnyPET nnUNet-style PET-only folders for fewshot50 folds 2/5/8.

Creates:
  <out>/fold{K}/imagesTr|labelsTr  (train)
  <out>/fold{K}/imagesVal|labelsVal (shared val)
PET channel = *_0001.nii.gz from Dataset221.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_split(path: Path) -> dict:
    d = json.loads(path.read_text())
    if isinstance(d, list):
        d = d[0]
    return d


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def export_fold(
    fold: int,
    split_path: Path,
    raw_root: Path,
    out_root: Path,
) -> None:
    sp = _load_split(split_path)
    train_cases = list(sp["train"])
    val_cases = list(sp["val"])
    fold_dir = out_root / f"fold{fold}"
    n_ok = 0
    for split_name, cases, img_key, lab_key in [
        ("train", train_cases, "imagesTr", "labelsTr"),
        ("val", val_cases, "imagesVal", "labelsVal"),
    ]:
        for case in cases:
            pet = raw_root / "imagesTr" / f"{case}_0001.nii.gz"
            lab = raw_root / "labelsTr" / f"{case}.nii.gz"
            if not pet.is_file() or not lab.is_file():
                print(f"[warn] missing {case} pet={pet.exists()} lab={lab.exists()}")
                continue
            _link(pet, fold_dir / img_key / f"{case}.nii.gz")
            _link(lab, fold_dir / lab_key / f"{case}.nii.gz")
            n_ok += 1
    print(f"[prep] fold{fold} linked={n_ok} train={len(train_cases)} val={len(val_cases)} → {fold_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_mae_psma_fewshot50_9fold"),
    )
    ap.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_raw/Dataset221_AutoPETIV_Task1_4ch"
        ),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/seganypet_fewshot50_f258"),
    )
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    for f in folds:
        export_fold(f, args.split_dir / f"fold{f}_nnunet.json", args.raw_root, args.out_root)


if __name__ == "__main__":
    main()
