#!/usr/bin/env python3
"""Add legacy nnUNet plan keys that DpDNet's STUNetTrainer still reads.

Newer nnUNetv2 (autopet image) only writes architecture.arch_kwargs.strides /
kernel_sizes. DpDNet ConfigurationManager expects top-level
pool_op_kernel_sizes / conv_kernel_sizes / num_pool_per_axis.
Idempotent: already-adapted files are left unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_PLANS = (
    Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_preprocessed")
    / "Dataset250_DpDNet_FDG_CT1ch"
    / "nnUNetPlans.json",
    Path("/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_preprocessed")
    / "Dataset251_DpDNet_FDG_PET1ch"
    / "nnUNetPlans.json",
)


def _num_pool_per_axis(strides: list) -> list[int]:
    if not strides:
        return []
    ndim = len(strides[0])
    out = [0] * ndim
    for s in strides[1:]:
        for i, v in enumerate(s):
            if int(v) > 1:
                out[i] += 1
    return out


def _write_plans(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=4) + "\n"
    try:
        path.write_text(text)
        return
    except PermissionError:
        pass
    import os
    import subprocess
    import tempfile

    image = os.environ.get("TASK1_NNUNET_IMAGE", "autopet_baseline:latest")
    tmp = Path(tempfile.mkstemp(prefix="nnunet_plans_", suffix=".json")[1])
    tmp.write_text(text)
    subprocess.check_call(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "-v",
            f"{tmp}:{tmp}",
            "-v",
            f"{path.parent}:{path.parent}",
            "--entrypoint",
            "bash",
            image,
            "-lc",
            f"cp -f '{tmp}' '{path}' && chmod a+rw '{path}'",
        ]
    )
    tmp.unlink(missing_ok=True)


def adapt_plans(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    cfgs = data.get("configurations") or {}
    changed: list[str] = []
    for name, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            continue
        arch = ((cfg.get("architecture") or {}).get("arch_kwargs")) or {}
        strides = arch.get("strides")
        kernels = arch.get("kernel_sizes")
        if "pool_op_kernel_sizes" not in cfg and isinstance(strides, list) and strides:
            cfg["pool_op_kernel_sizes"] = strides
            changed.append(f"{name}.pool_op_kernel_sizes")
        if "conv_kernel_sizes" not in cfg and isinstance(kernels, list) and kernels:
            cfg["conv_kernel_sizes"] = kernels
            changed.append(f"{name}.conv_kernel_sizes")
        if "num_pool_per_axis" not in cfg and isinstance(strides, list) and strides:
            cfg["num_pool_per_axis"] = _num_pool_per_axis(strides)
            changed.append(f"{name}.num_pool_per_axis")
    if changed:
        _write_plans(path, data)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="*", type=Path, help="nnUNetPlans.json paths")
    args = ap.parse_args()
    paths = args.plans or [p for p in DEFAULT_PLANS if p.is_file()]
    if not paths:
        raise SystemExit("[adapt-plans] no plans files")
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"[adapt-plans] missing {p}")
        ch = adapt_plans(p)
        extra = f" +{','.join(ch)}" if ch else " (already ok)"
        print(f"[adapt-plans] {p}{extra}")


if __name__ == "__main__":
    main()
