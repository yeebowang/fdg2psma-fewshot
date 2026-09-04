#!/usr/bin/env python3
"""Prepare SegAnyPET PET-only TEST20 folder (imagesVal/labelsVal) for click/autoseg eval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases-json",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_mae_psma_test20.json"),
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
        default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/seganypet_psma_test20"),
    )
    args = ap.parse_args()
    d = json.loads(args.cases_json.read_text())
    cases = list(d["cases"] if isinstance(d, dict) else d)
    n_ok = 0
    for case in cases:
        pet = args.raw_root / "imagesTr" / f"{case}_0001.nii.gz"
        lab = args.raw_root / "labelsTr" / f"{case}.nii.gz"
        if not pet.is_file() or not lab.is_file():
            print(f"[warn] missing {case}")
            continue
        _link(pet, args.out_root / "imagesVal" / f"{case}.nii.gz")
        _link(lab, args.out_root / "labelsVal" / f"{case}.nii.gz")
        n_ok += 1
    meta = {"n_cases": len(cases), "n_linked": n_ok, "out": str(args.out_root)}
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[prep-test20] linked={n_ok}/{len(cases)} → {args.out_root}")


if __name__ == "__main__":
    main()
