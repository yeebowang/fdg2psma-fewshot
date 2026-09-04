#!/usr/bin/env python3
"""Build Dataset239 for DpDNet FDG pretrain from Dataset228 FDG splits.

DpDNet reads cancer type from the first 4 chars of the case id. AutoPET FDG
cases are mixed-diagnosis under ``fdg_``; for a runnable FDG stage we map all
FDG cases to prompt prefix ``lymp`` (single-task dual-prompt training).

Preprocessed ``.b2nd`` / ``.pkl`` are hard-linked from Dataset228 (same blosc2
pipeline as Baseline1 nnU-Net). Do **not** unpack to ``.npy`` — that regresses
to full-volume mmap IO. Pkls get ``type=['lymp','lymp']`` injected.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--work",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace"),
    )
    ap.add_argument(
        "--splits",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_fdg_nnunet.json"),
    )
    ap.add_argument("--src-id", type=int, default=228)
    ap.add_argument("--dst-id", type=int, default=239)
    ap.add_argument("--prompt-prefix", default="lymp")
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()

    src_name = f"Dataset{args.src_id}_AutoPETIV_Task1_2ch"
    dst_name = f"Dataset{args.dst_id}_DpDNet_FDG_2ch"
    pref = args.prompt_prefix
    assert len(pref) == 4, "DpDNet uses first 4 chars as cancer type"

    src_raw = args.work / "nnUNet_raw" / src_name
    src_pp = args.work / "nnUNet_preprocessed" / src_name
    src_cfg = src_pp / "nnUNetPlans_3d_fullres"
    dst_raw = args.work / "nnUNet_raw" / dst_name
    dst_pp = args.work / "nnUNet_preprocessed" / dst_name
    dst_cfg = dst_pp / "nnUNetPlans_3d_fullres"

    splits = json.loads(args.splits.read_text())
    fold0 = splits[0] if isinstance(splits, list) else splits
    train = list(fold0["train"])
    val = list(fold0["val"])
    cases = train + val

    # raw dataset.json (minimal; training reads preprocessed)
    ds_src = json.loads((src_raw / "dataset.json").read_text())
    ds = {
        "channel_names": ds_src.get("channel_names") or {"0": "CT", "1": "PET"},
        "labels": ds_src.get("labels") or {"background": 0, "lesion": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": ds_src.get("overwrite_image_reader_writer"),
    }
    for sub in ("imagesTr", "labelsTr"):
        (dst_raw / sub).mkdir(parents=True, exist_ok=True)
    (dst_raw / "dataset.json").write_text(json.dumps(ds, indent=2) + "\n")

    # plans + fingerprint
    dst_pp.mkdir(parents=True, exist_ok=True)
    dst_cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pp / "nnUNetPlans.json", dst_pp / "nnUNetPlans.json")
    if (src_pp / "dataset_fingerprint.json").is_file():
        shutil.copy2(src_pp / "dataset_fingerprint.json", dst_pp / "dataset_fingerprint.json")
    (dst_pp / "dataset.json").write_text(json.dumps(ds, indent=2) + "\n")

    plans = json.loads((dst_pp / "nnUNetPlans.json").read_text())
    plans["dataset_name"] = dst_name
    cfg = plans.setdefault("configurations", {}).setdefault("3d_fullres", {})
    cfg["batch_size"] = int(args.batch_size)
    # DpDNet STUNet expects legacy top-level pool/conv kernels (not only architecture.*)
    arch_kwargs = ((cfg.get("architecture") or {}).get("arch_kwargs") or {})
    if arch_kwargs.get("strides") and not cfg.get("pool_op_kernel_sizes"):
        cfg["pool_op_kernel_sizes"] = arch_kwargs["strides"]
    if arch_kwargs.get("kernel_sizes") and not cfg.get("conv_kernel_sizes"):
        cfg["conv_kernel_sizes"] = arch_kwargs["kernel_sizes"]
    if cfg.get("pool_op_kernel_sizes") and not cfg.get("num_pool_per_axis"):
        strides = cfg["pool_op_kernel_sizes"]
        axes = len(strides[0])
        cfg["num_pool_per_axis"] = [
            sum(1 for s in strides if s[a] > 1) for a in range(axes)
        ]
    (dst_pp / "nnUNetPlans.json").write_text(json.dumps(plans, indent=2) + "\n")

    renamed_train, renamed_val = [], []
    missing = []
    for split_name, src_list, out_list in (
        ("train", train, renamed_train),
        ("val", val, renamed_val),
    ):
        for case in src_list:
            new_case = f"{pref}_{case}"
            out_list.append(new_case)
            # raw softlinks (optional, for integrity tools)
            for ch in (0, 1):
                s = src_raw / "imagesTr" / f"{case}_{ch:04d}.nii.gz"
                d = dst_raw / "imagesTr" / f"{new_case}_{ch:04d}.nii.gz"
                if s.is_file():
                    _link(s, d)
            sl = src_raw / "labelsTr" / f"{case}.nii.gz"
            dl = dst_raw / "labelsTr" / f"{new_case}.nii.gz"
            if sl.is_file():
                _link(sl, dl)

            # preprocessed
            ok = False
            for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
                s = src_cfg / f"{case}{suffix}"
                d = dst_cfg / f"{new_case}{suffix}"
                if not s.is_file():
                    continue
                _link(s, d)
                ok = True
            if not ok:
                missing.append(case)
                continue
            # inject cancer-type prompt into properties pkl
            pkl = dst_cfg / f"{new_case}.pkl"
            with open(pkl, "rb") as f:
                props = pickle.load(f)
            if not isinstance(props, dict):
                raise TypeError(f"unexpected pkl type for {pkl}: {type(props)}")
            props["type"] = [pref, pref]
            with open(pkl, "wb") as f:
                pickle.dump(props, f)

    splits_out = [{"train": renamed_train, "val": renamed_val}]
    (dst_pp / "splits_final.json").write_text(json.dumps(splits_out, indent=2) + "\n")

    meta = {
        "dst": dst_name,
        "n_train": len(renamed_train),
        "n_val": len(renamed_val),
        "prompt_prefix": pref,
        "batch_size": args.batch_size,
        "missing": missing[:20],
        "n_missing": len(missing),
    }
    (dst_pp / "dpdnet_prepare_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    if missing:
        raise SystemExit(f"[error] missing preprocessed for {len(missing)} cases")


if __name__ == "__main__":
    main()
