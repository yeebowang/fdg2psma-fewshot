#!/usr/bin/env python3
"""Inject DpDNet prompt field ``type`` into preprocessed case *.pkl.

Official DpDNet preprocessor writes properties['type'] = [filename[:4], ...]
(e.g. lymp). Autopet nnUNetv2 plan_and_preprocess omits it, so
STUNetTrainer_small_prompt dies on KeyError: 'type' at train_step.
Idempotent.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

DEFAULT_DIRS = (
    Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_preprocessed")
    / "Dataset250_DpDNet_FDG_CT1ch"
    / "nnUNetPlans_3d_fullres",
    Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_preprocessed")
    / "Dataset251_DpDNet_FDG_PET1ch"
    / "nnUNetPlans_3d_fullres",
)


def inject_dir(folder: Path) -> tuple[int, int]:
    n_ok = n_add = 0
    for pkl in sorted(folder.glob("*.pkl")):
        with pkl.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, dict):
            continue
        if "type" in obj:
            n_ok += 1
            continue
        code = pkl.stem[:4] if len(pkl.stem) >= 4 else "lymp"
        obj["type"] = [code]
        with pkl.open("wb") as f:
            pickle.dump(obj, f)
        n_add += 1
    return n_add, n_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path)
    args = ap.parse_args()
    dirs = args.dirs or [d for d in DEFAULT_DIRS if d.is_dir()]
    if not dirs:
        raise SystemExit("[inject-type] no preprocessed dirs")
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"[inject-type] missing {d}")
        added, ok = inject_dir(d)
        print(f"[inject-type] {d} added={added} already={ok}")


if __name__ == "__main__":
    main()
