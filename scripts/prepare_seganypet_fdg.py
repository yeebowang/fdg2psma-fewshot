#!/usr/bin/env python3
"""Prepare SegAnyPET PET-only folders for FDG supervised pretrain (Baseline1 splits).

Creates:
  <out>/imagesTr|labelsTr  (FDG train)
  <out>/imagesVal|labelsVal (FDG val)
PET = *_0001.nii.gz from Dataset221 (or Dataset228 PET channel if present).
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


def _find_pet_lab(raw_root: Path, case: str) -> tuple[Path | None, Path | None]:
    # Prefer 4ch Dataset221 naming
    pet = raw_root / "imagesTr" / f"{case}_0001.nii.gz"
    lab = raw_root / "labelsTr" / f"{case}.nii.gz"
    if pet.is_file() and lab.is_file():
        return pet, lab
    # Fallback: Dataset228 2ch may still use _0001 for PET
    for ch in ("_0001", "_0000"):
        p2 = raw_root / "imagesTr" / f"{case}{ch}.nii.gz"
        if p2.is_file() and lab.is_file():
            return p2, lab
    return None, None


def export_split(cases: list[str], raw_root: Path, out_root: Path, img_key: str, lab_key: str) -> int:
    n_ok = 0
    for case in cases:
        pet, lab = _find_pet_lab(raw_root, case)
        if pet is None or lab is None:
            print(f"[warn] missing {case}")
            continue
        _link(pet, out_root / img_key / f"{case}.nii.gz")
        _link(lab, out_root / lab_key / f"{case}.nii.gz")
        n_ok += 1
    return n_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits-json",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_fdg_nnunet.json"),
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
        default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/seganypet_fdg_baseline1"),
    )
    args = ap.parse_args()
    sp = _load_split(args.splits_json)
    train_cases = list(sp["train"])
    val_cases = list(sp["val"])
    args.out_root.mkdir(parents=True, exist_ok=True)
    n_tr = export_split(train_cases, args.raw_root, args.out_root, "imagesTr", "labelsTr")
    n_va = export_split(val_cases, args.raw_root, args.out_root, "imagesVal", "labelsVal")
    print(
        f"[prep-fdg] linked train={n_tr}/{len(train_cases)} val={n_va}/{len(val_cases)} → {args.out_root}"
    )


if __name__ == "__main__":
    main()
