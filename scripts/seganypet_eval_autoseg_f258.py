#!/usr/bin/env python3
"""SegAnyPET auto-seg eval (nnUNet-comparable protocol).

Differences vs official GT-click val:
  - No GT used for ROI crop (center CropOrPad 128³)
  - No GT used for prompts (single center positive click)
  - GT only for final binary Dice on shared PSMA val
  - Empty-GT cases still inferred (same as nnUNet)

Note: architecture still PET-only + fixed 128 ROI (not full-volume SW like nnUNet);
this removes the GT-oracle click/localization advantage for a fairer Dice compare.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from tqdm import tqdm

_CODE = Path(__file__).resolve().parents[1] / "third_party" / "SegAnyPET" / "code"
_PIP = Path(__file__).resolve().parents[1] / "third_party" / "seganypet_pip"
sys.path.insert(0, str(_PIP))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from segment_anything.build_sam3D import sam_model_registry3D  # noqa: E402
from utils.infer_utils import (  # noqa: E402
    data_postprocess,
    get_roi_from_subject,
    get_subject_and_meta_info,
    read_arr_from_nifti,
    sam_model_infer,
    save_numpy_to_nifti,
)
from seganypet_fewshot_finetune import _binary_dice_nifti  # noqa: E402


def _preprocess_auto(subject, meta_info, target_spacing, crop_size: int):
    """Resample + canonical + center crop (NO GT mask for crop)."""
    # binary label for subject completeness (not used for crop center)
    lab = subject.label.data.clone()
    subject.label.set_data((lab > 0).to(lab.dtype))

    meta_info["original_subject_affine"] = subject.image.affine.copy()
    meta_info["original_subject_spatial_shape"] = subject.image.spatial_shape

    subject = tio.Resample(target=target_spacing)(subject)
    subject = tio.ToCanonical()(subject)

    # center crop/pad — do NOT pass mask_name so GT does not drive ROI
    crop_transform = tio.CropOrPad(target_shape=(crop_size, crop_size, crop_size))
    norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)
    roi_image, roi_label, meta_info = get_roi_from_subject(
        subject, meta_info, crop_transform, norm_transform
    )
    return roi_image, roi_label, meta_info


def validate_auto_no_gt(
    model,
    img_path: str,
    gt_path: str,
    output_path: str,
    num_clicks: int = 1,
    crop_size: int = 128,
    target_spacing=(1.5, 1.5, 1.5),
    seed: int = 233,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(osp.dirname(output_path) or ".", exist_ok=True)

    _, gt_meta = read_arr_from_nifti(gt_path, get_meta_info=True)
    zero = np.zeros(gt_meta["original_numpy_shape"], dtype=np.uint8)
    subject, meta_info = get_subject_and_meta_info(img_path, gt_path)
    subject = copy.deepcopy(subject)
    meta_info = copy.deepcopy(meta_info)

    roi_image, _roi_label, meta_info = _preprocess_auto(
        subject, meta_info, target_spacing=target_spacing, crop_size=crop_size
    )
    # roi_gt=None → center click path inside sam_model_infer
    roi_pred, _ = sam_model_infer(
        model,
        roi_image,
        roi_gt=None,
        num_clicks=max(1, int(num_clicks)),
        prev_low_res_mask=None,
    )
    pred_grid = data_postprocess(roi_pred, meta_info)
    zero[pred_grid == 1] = 1
    save_numpy_to_nifti(zero, output_path, gt_meta)


def eval_fold(
    model,
    data_root: Path,
    out_pred_dir: Path,
    num_clicks: int,
    crop_size: int,
    max_cases: int | None,
    seed: int,
) -> dict:
    img_dir = data_root / "imagesVal"
    lab_dir = data_root / "labelsVal"
    pairs = []
    for lab in sorted(lab_dir.glob("*.nii.gz")):
        img = img_dir / lab.name
        if img.is_file():
            pairs.append((img, lab))
    if max_cases is not None:
        pairs = pairs[:max_cases]

    out_pred_dir.mkdir(parents=True, exist_ok=True)
    dices = []
    per_case = {}
    for img_path, lab_path in tqdm(pairs, desc=f"auto@{data_root.name}", leave=False):
        pred_path = out_pred_dir / lab_path.name
        try:
            validate_auto_no_gt(
                model=model,
                img_path=str(img_path),
                gt_path=str(lab_path),
                output_path=str(pred_path),
                num_clicks=num_clicks,
                crop_size=crop_size,
                seed=seed,
            )
            d = _binary_dice_nifti(str(lab_path), str(pred_path))
            if d == d:
                dices.append(float(d))
                per_case[lab_path.name] = float(d)
        except Exception as e:
            print(f"[warn] {lab_path.name}: {e}")
            per_case[lab_path.name] = None
    mean = float(np.mean(dices)) if dices else float("nan")
    return {"mean_dice": mean, "n_scored": len(dices), "n_total": len(pairs), "per_case": per_case}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True, help=".../seganypet_fewshot50_f258")
    ap.add_argument("--folds", default="2,5,8")
    ap.add_argument("--num-clicks", type=int, default=1, help="center clicks (no GT)")
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--ckpt", choices=["best", "latest"], default="best")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vmax = None if args.max_cases <= 0 else args.max_cases

    fold_dice = {}
    details = {}
    for fold in folds:
        fold_dir = args.run_root / "seganypet" / f"fold{fold}"
        ckpt = fold_dir / f"{args.ckpt}.pth"
        if not ckpt.is_file():
            alt = fold_dir / ("latest.pth" if args.ckpt == "best" else "best.pth")
            ckpt = alt if alt.is_file() else ckpt
        if not ckpt.is_file():
            print(f"[skip] fold{fold} no ckpt")
            fold_dice[fold] = None
            continue
        print(f"[auto-eval] fold{fold} ← {ckpt}")
        model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = state.get("model_state_dict", state)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device).eval()

        det = eval_fold(
            model,
            args.data_root / f"fold{fold}",
            fold_dir / "val_pred_autoseg",
            num_clicks=args.num_clicks,
            crop_size=args.crop_size,
            max_cases=vmax,
            seed=42,
        )
        fold_dice[fold] = det["mean_dice"]
        details[str(fold)] = det
        print(f"[auto-eval] fold{fold} mean_dice={det['mean_dice']:.4f} n={det['n_scored']}")

    ok = [fold_dice[f] for f in folds if fold_dice.get(f) is not None and fold_dice[f] == fold_dice[f]]
    mean = float(sum(ok) / len(ok)) if ok else None
    std = float((sum((x - mean) ** 2 for x in ok) / len(ok)) ** 0.5) if ok and mean is not None else None
    summary = {
        "protocol": "autoseg_no_gt_center_crop_center_click",
        "comparable_to": "nnUNet_shared_psma_val_binary_dice",
        "note": (
            "No GT for ROI/clicks; GT only for Dice. "
            "Still PET-only + single 128³ center ROI (not nnUNet full-volume SW)."
        ),
        "folds": folds,
        "fold_best_dice": [fold_dice.get(f) for f in folds],
        "mean": mean,
        "std": std,
        "n_ok": len(ok),
        "num_clicks": args.num_clicks,
        "crop_size": args.crop_size,
        "ckpt": args.ckpt,
        "details": {k: {"mean_dice": v["mean_dice"], "n_scored": v["n_scored"]} for k, v in details.items()},
    }
    out = args.run_root / "aggregate_val_dice_f258_autoseg.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    vis = Path(__file__).resolve().parents[1] / "vis" / f"aggregate_seganypet_autoseg_{args.run_root.name}.json"
    vis.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))
    print(f"[auto-eval] wrote {out}")


if __name__ == "__main__":
    main()
