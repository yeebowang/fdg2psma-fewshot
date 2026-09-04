#!/usr/bin/env python3
"""Sliding-window Dice/HD95 on PSMA test20 for a finetuned SwinUNETR seg ckpt."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mae_finetune_fdg_swinbase import ROI  # noqa: E402
from seg_voxel_metrics import aggregate_case_metrics, confusion_counts  # noqa: E402


class LoadCachedSegNpzd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with np.load(d["npz_path"]) as z:
            d["image"] = z["image"].astype(np.float32)
            d["label"] = z["label"].astype(np.uint8)
        return d


def build_model(
    depths: tuple[int, ...] = (2, 2, 6, 2),
    use_v2: bool = True,
) -> torch.nn.Module:
    downsample = "mergingv2" if use_v2 else "merging"
    return SwinUNETR(
        img_size=ROI,
        in_channels=2,
        out_channels=2,
        feature_size=48,
        depths=depths,
        num_heads=(3, 6, 12, 24),
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.05,
        normalize=True,
        use_checkpoint=True,
        spatial_dims=3,
        downsample=downsample,
        use_v2=use_v2,
    )


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-json", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--seg-ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--sw-batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--depths", default="2,2,6,2")
    ap.add_argument("--use-v2", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    payload = json.loads(Path(args.cases_json).read_text())
    cases = list(payload.get("cases") or payload.get("test") or [])
    files = []
    missing = []
    for c in cases:
        p = Path(args.cache_dir) / f"{c}.npz"
        if not p.is_file():
            missing.append(c)
            continue
        files.append({"npz_path": str(p), "case_id": c})
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} cache e.g. {missing[:3]}")

    tf = Compose(
        [
            LoadCachedSegNpzd(keys=["image", "label"]),
            SpatialPadd(keys=["image", "label"], spatial_size=ROI, mode="constant", constant_values=0),
            # full-volume SW; pad only if smaller than ROI
            ToTensord(keys=["image", "label"]),
        ]
    )
    # For SW we want full volume — do NOT center-crop. Only pad min size.
    loader = DataLoader(
        Dataset(files, tf),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = tuple(int(x) for x in args.depths.split(",") if x.strip())
    model = build_model(depths=depths, use_v2=bool(args.use_v2)).to(device)
    ckpt = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    cleaned = {}
    for k, v in sd.items():
        kk = k[7:] if k.startswith("module.") else k
        cleaned[kk] = v
    missing_k, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"==> seg {args.seg_ckpt} missing={len(missing_k)} unexpected={len(unexpected)} "
        f"epoch={ckpt.get('epoch')} best_dice={ckpt.get('best_dice')}"
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    per_case = []
    for batch in tqdm(loader, desc=f"seg-test[{args.tag or 'ckpt'}]"):
        cid = batch["case_id"][0] if isinstance(batch["case_id"], (list, tuple)) else str(batch["case_id"][0])
        vi = batch["image"].to(device, non_blocking=True)
        vl = batch["label"].to(device, non_blocking=True)
        # label may be (B,1,Z,Y,X)
        with autocast("cuda"):
            vo = sliding_window_inference(
                inputs=vi,
                roi_size=ROI,
                sw_batch_size=args.sw_batch_size,
                predictor=model,
                overlap=0.5,
            )
        vo_l = [post_pred(i) for i in vo]
        vl_l = [post_label(i) for i in vl]
        dice_metric.reset()
        hd95_metric.reset()
        dice_metric(y_pred=vo_l, y=vl_l)
        d = float(dice_metric.aggregate().reshape(-1)[0].item())
        try:
            hd95_metric(y_pred=vo_l, y=vl_l)
            h = float(hd95_metric.aggregate().reshape(-1)[0].item())
        except Exception:
            h = float("nan")
        # FG channel (1) for voxel FP/FN; empty GT → pos=0, still counts toward FP.
        pred_fg = (vo_l[0][1].detach().cpu().numpy() > 0.5)
        gt_fg = (vl_l[0][1].detach().cpu().numpy() > 0.5)
        cm = confusion_counts(gt_fg, pred_fg)
        if d == d:
            cm["dice"] = d  # keep MONAI dice when finite
        cm["case_id"] = cid
        cm["hd95"] = h
        per_case.append(cm)

    by_id = {r["case_id"]: r for r in per_case}
    agg = aggregate_case_metrics(by_id)
    hd95s = [r["hd95"] for r in per_case if r.get("hd95") == r.get("hd95") and r["hd95"] < 1e6]
    pos_d = [
        r["dice"]
        for r in per_case
        if int(r.get("pos_voxels") or 0) > 0 and isinstance(r.get("dice"), (int, float)) and r["dice"] == r["dice"]
    ]
    summary = {
        "tag": args.tag,
        "seg_ckpt": str(args.seg_ckpt),
        "n_cases": len(per_case),
        **agg,
        "median_dice": float(np.median(pos_d)) if pos_d else float("nan"),
        "mean_hd95": float(np.mean(hd95s)) if hd95s else float("nan"),
        "per_case": per_case,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[eval-seg] tag={args.tag} Dice={summary['mean_dice']:.4f} "
        f"FP={summary['fp_rate']:.4f} FN={summary['fn_rate']:.4f} "
        f"hd95={summary['mean_hd95']:.2f} n={summary['n_cases']} (pos={summary['n_positive']})"
    )


if __name__ == "__main__":
    main()
