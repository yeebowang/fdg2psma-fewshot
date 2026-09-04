#!/usr/bin/env python3
"""Evaluate MAE reconstruction on PSMA test20 (deterministic center crop).

Compares checkpoints (e.g. FDG MAE before SSL vs continued SSL after).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.transforms import CenterSpatialCropd, Compose, SpatialPadd, ToTensord
from torch.amp import autocast
from tqdm import tqdm

# reuse SSL model definition
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mae_continued_ssl_psma import (  # noqa: E402
    ROI,
    TRACER_PSMA,
    LoadCachedImageNpzd,
    MultiModalSwinMAE,
    build_file_list,
    load_fdg_mae_weights,
    _unwrap,
)


def make_eval_loader(files: list[dict], batch_size: int, num_workers: int) -> DataLoader:
    tf = Compose(
        [
            LoadCachedImageNpzd(keys=["image"]),
            SpatialPadd(keys=["image"], spatial_size=ROI, mode="constant", constant_values=0),
            CenterSpatialCropd(keys=["image"], roi_size=ROI),
            ToTensord(keys=["image"]),
        ]
    )
    return DataLoader(
        Dataset(files, tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def eval_ckpt(
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
    *,
    is_ssl_ckpt: bool,
    mask_ratio: float,
    seed: int,
) -> dict:
    model = MultiModalSwinMAE(mask_ratio=mask_ratio).to(device)
    if is_ssl_ckpt:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cleaned = {}
        for k, v in sd.items():
            kk = k[7:] if k.startswith("module.") else k
            cleaned[kk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(
            f"==> SSL ckpt {ckpt_path.name} missing={len(missing)} unexpected={len(unexpected)} "
            f"epoch={ckpt.get('epoch')} best_loss={ckpt.get('best_loss')}"
        )
    else:
        load_fdg_mae_weights(model, ckpt_path)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # fixed mask seed for fair before/after compare
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    losses, masked, unmasked = [], [], []
    for batch in tqdm(loader, desc=f"recon[{ckpt_path.stem[:24]}]"):
        x = batch["image"].to(device, non_blocking=True)
        bsz = x.shape[0]
        tracer = torch.full((bsz,), TRACER_PSMA, device=device, dtype=torch.long)
        with autocast("cuda"):
            # call unwrap forward with deterministic mask
            m = _unwrap(model)
            x_t = m._apply_tracer(x, tracer)
            B, C, Z, Y, X = x_t.shape
            pz, py, px = m.patch_size
            grid_z, grid_y, grid_x = Z // pz, Y // py, X // px
            noise = torch.rand(B, C, grid_z, grid_y, grid_x, device=device, generator=g)
            mask = (noise < m.mask_ratio).float()
            mask_expanded = (
                mask.repeat_interleave(pz, dim=2)
                .repeat_interleave(py, dim=3)
                .repeat_interleave(px, dim=4)
            )
            x_masked = x_t * (1.0 - mask_expanded)
            x_rec = m.encoder_decoder(x_masked)
            mse = F.mse_loss(x_rec, x_t, reduction="none")
            loss_m = (mse * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            loss_u = (mse * (1.0 - mask_expanded)).sum() / ((1.0 - mask_expanded).sum() + 1e-8)
            loss = loss_m + 0.2 * loss_u
        losses.append(float(loss.item()))
        masked.append(float(loss_m.item()))
        unmasked.append(float(loss_u.item()))

    return {
        "ckpt": str(ckpt_path),
        "n_batches": len(losses),
        "mae_loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "mae_loss_std": float(np.std(losses)) if losses else float("nan"),
        "masked_mse_mean": float(np.mean(masked)) if masked else float("nan"),
        "unmasked_mse_mean": float(np.mean(unmasked)) if unmasked else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-json", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--before-ckpt", required=True, help="FDG MAE (before SSL)")
    ap.add_argument("--after-ckpt", required=True, help="SSL continued (after)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    payload = json.loads(Path(args.cases_json).read_text())
    cases = list(payload.get("cases") or payload.get("test") or [])
    files = build_file_list(cases, Path(args.cache_dir), TRACER_PSMA)
    print(f"[eval-recon] n_cases={len(files)} bs={args.batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_eval_loader(files, args.batch_size, args.num_workers)

    before = eval_ckpt(
        Path(args.before_ckpt),
        loader,
        device,
        is_ssl_ckpt=False,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    after = eval_ckpt(
        Path(args.after_ckpt),
        loader,
        device,
        is_ssl_ckpt=True,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )

    delta = before["mae_loss_mean"] - after["mae_loss_mean"]
    rel = delta / max(before["mae_loss_mean"], 1e-8) * 100.0
    summary = {
        "n_cases": len(files),
        "mask_ratio": args.mask_ratio,
        "seed": args.seed,
        "before_ssl": before,
        "after_ssl": after,
        "delta_mae_loss_before_minus_after": delta,
        "rel_improve_pct": rel,
        "ssl_helped_recon": bool(delta > 0),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(
        f"[eval-recon] before={before['mae_loss_mean']:.4f} "
        f"after={after['mae_loss_mean']:.4f} "
        f"Δ={delta:+.4f} ({rel:+.1f}%) helped={delta > 0}"
    )


if __name__ == "__main__":
    main()
