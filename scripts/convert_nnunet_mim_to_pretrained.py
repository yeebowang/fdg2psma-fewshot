#!/usr/bin/env python3
"""Convert PET/CT MAE nnUNet MIM ckpt → nnUNetv2 -pretrained_weights format.

Official nnUNetv2_train -pretrained_weights requires:
  saved_model['network_weights'] with keys matching the target network.

The released nnunet_v2_mim_best.pth instead stores:
  model_state_dict['nnunet.*']  (MAE wrapper)

Passing the raw MIM file therefore crashes (KeyError: network_weights).
Even after renaming, Dataset228 3d_fullres has one stride mismatch on
decoder.transpconvs.0 (MIM 2×2×2 vs plans 1×2×2). This script:
  - strips the nnunet. prefix
  - skips reconstruction seg_layers (nnUNet re-inits the head)
  - adapts the one mismatched transpose-conv by averaging the extra Z kernel
  - writes a checkpoint nnUNet can load
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _adapt(src: torch.Tensor, dst_shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(src.shape) == dst_shape:
        return src
    # (C_out, C_in, D, H, W) transpose-conv: average extra depth taps
    if src.ndim == 5 and len(dst_shape) == 5 and src.shape[0] == dst_shape[0] and src.shape[1] == dst_shape[1]:
        out = src
        for dim in (2, 3, 4):
            if out.shape[dim] == dst_shape[dim]:
                continue
            if out.shape[dim] > dst_shape[dim] and dst_shape[dim] == 1:
                out = out.mean(dim=dim, keepdim=True)
            elif out.shape[dim] > dst_shape[dim]:
                # center crop
                extra = out.shape[dim] - dst_shape[dim]
                start = extra // 2
                out = out.narrow(dim, start, dst_shape[dim])
            else:
                # repeat / pad
                reps = [1] * out.ndim
                reps[dim] = int((dst_shape[dim] + out.shape[dim] - 1) // out.shape[dim])
                out = out.repeat(*reps).narrow(dim, 0, dst_shape[dim])
        if tuple(out.shape) == dst_shape:
            return out.contiguous()
    raise ValueError(f"cannot adapt {tuple(src.shape)} -> {dst_shape}")


def convert(mim_path: Path, template_ckpt: Path, out_path: Path) -> dict:
    mim = torch.load(mim_path, map_location="cpu", weights_only=False)
    if not isinstance(mim, dict) or "model_state_dict" not in mim:
        raise SystemExit(f"unexpected MIM ckpt keys: {list(mim)[:20] if isinstance(mim, dict) else type(mim)}")
    raw = mim["model_state_dict"]
    body: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        nk = k[len("nnunet.") :] if k.startswith("nnunet.") else k
        if ".seg_layers." in nk:
            continue
        body[nk] = v

    tmpl = torch.load(template_ckpt, map_location="cpu", weights_only=False)
    tmpl_w = tmpl.get("network_weights") or {}
    if not tmpl_w:
        raise SystemExit(f"template missing network_weights: {template_ckpt}")

    out_w: dict[str, torch.Tensor] = {}
    adapted = []
    missing = []
    for k, v in tmpl_w.items():
        if ".seg_layers." in k:
            continue
        if k not in body:
            missing.append(k)
            continue
        src = body[k]
        if tuple(src.shape) != tuple(v.shape):
            src = _adapt(src, tuple(v.shape))
            adapted.append(f"{k} {tuple(body[k].shape)}->{tuple(v.shape)}")
        out_w[k] = src.contiguous()

    if missing:
        raise SystemExit(f"MIM missing {len(missing)} keys e.g. {missing[:8]}")

    ckpt = {
        "network_weights": out_w,
        "nnunet_mim_source": str(mim_path),
        "nnunet_mim_template": str(template_ckpt),
        "nnunet_mim_adapted": adapted,
        "nnunet_mim_epoch": mim.get("epoch"),
        "nnunet_mim_best_loss": mim.get("best_loss"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    return {"n": len(out_w), "adapted": adapted, "out": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mim",
        type=Path,
        default=Path(
            "/media/ybwang/data1/PSMA-CTRL/ICLR2026/3D-MAE-PET-CT/weights/nnunetv2/nnunet_v2_mim_best.pth"
        ),
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=Path(
            "/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/"
            "20260817_225543_iclr2026_baseline1_fdg_2ch_fullres_gpu013_bs6_tr70_val0_169ep/"
            "Dataset228_AutoPETIV_Task1_2ch/nnUNetTrainer_Task1StdTrainVal50__nnUNetPlans__3d_fullres/"
            "fold_0/checkpoint_final.pth"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/media/ybwang/data1/PSMA-CTRL/ICLR2026/3D-MAE-PET-CT/weights/nnunetv2/"
            "nnunet_v2_mim_best_nnunetformat.pth"
        ),
    )
    args = ap.parse_args()
    info = convert(args.mim, args.template, args.out)
    print("[ok]", info)


if __name__ == "__main__":
    main()
