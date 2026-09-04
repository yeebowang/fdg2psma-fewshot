#!/usr/bin/env python3
"""Continued MAE SSL on unlabeled PSMA (≤70% train), initialized from FDG MAE.

Adds:
  - tracer token (FDG=0 / PSMA=1) as additive channel bias
  - feature alignment (MMD on SwinViT bottleneck GAP) between FDG & PSMA batches

Saves latest/best MAE ckpts for subsequent few-shot finetune.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.data import DataLoader, Dataset
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    MapTransform,
    RandSpatialCropd,
    RandZoomd,
    SpatialPadd,
    ToTensord,
)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROI = (96, 128, 128)
TRACER_FDG = 0
TRACER_PSMA = 1


class LoadCachedImageNpzd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with np.load(d["npz_path"]) as z:
            d["image"] = z["image"].astype(np.float32)
        return d


class MultiModalSwinMAE(nn.Module):
    def __init__(
        self,
        img_size=ROI,
        patch_size=(12, 16, 16),
        in_channels=2,
        mask_ratio=0.5,
        n_tracers=2,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder_decoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=in_channels,
            feature_size=48,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.05,
            normalize=True,
            use_checkpoint=True,
            spatial_dims=3,
            downsample="mergingv2",
            use_v2=True,
        )
        # tracer token: additive bias on PET/CT channels (keeps in_channels=2 for weight load)
        self.tracer_embed = nn.Embedding(n_tracers, in_channels)
        nn.init.zeros_(self.tracer_embed.weight)

    def _apply_tracer(self, x: torch.Tensor, tracer_id: torch.Tensor) -> torch.Tensor:
        # tracer_id: (B,) long
        emb = self.tracer_embed(tracer_id).view(-1, x.shape[1], 1, 1, 1)
        return x + emb

    def _bottleneck_feat(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.encoder_decoder.swinViT(x, self.encoder_decoder.normalize)
        # last stage (B, 768, z, y, x)
        return hs[-1].mean(dim=(2, 3, 4))

    def forward(self, x: torch.Tensor, tracer_id: torch.Tensor):
        """Always returns (loss, x_rec, feat) so DataParallel can scatter cleanly."""
        x = self._apply_tracer(x, tracer_id)
        B, C, Z, Y, X = x.shape
        pz, py, px = self.patch_size
        grid_z, grid_y, grid_x = Z // pz, Y // py, X // px
        noise = torch.rand(B, C, grid_z, grid_y, grid_x, device=x.device)
        mask = (noise < self.mask_ratio).float()
        mask_expanded = (
            mask.repeat_interleave(pz, dim=2)
            .repeat_interleave(py, dim=3)
            .repeat_interleave(px, dim=4)
        )
        x_masked = x * (1.0 - mask_expanded)
        x_rec = self.encoder_decoder(x_masked)
        mse = F.mse_loss(x_rec, x, reduction="none")
        loss_masked = (mse * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
        loss_unmasked = (mse * (1.0 - mask_expanded)).sum() / ((1.0 - mask_expanded).sum() + 1e-8)
        loss = loss_masked + 0.2 * loss_unmasked
        feat = self._bottleneck_feat(x_masked)
        return loss, x_rec, feat


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Biased MMD^2 with RBF kernel; x,y: (N,D)/(M,D)."""
    xx = x @ x.t()
    yy = y @ y.t()
    xy = x @ y.t()
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    dxx = torch.clamp(rx.t() + rx - 2 * xx, min=0.0)
    dyy = torch.clamp(ry.t() + ry - 2 * yy, min=0.0)
    dxy = torch.clamp(xx.diag().unsqueeze(1) + yy.diag().unsqueeze(0) - 2 * xy, min=0.0)
    kxx = torch.exp(-dxx / (2 * sigma**2))
    kyy = torch.exp(-dyy / (2 * sigma**2))
    kxy = torch.exp(-dxy / (2 * sigma**2))
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


def build_file_list(cases: list[str], cache_dir: Path, tracer_id: int) -> list[dict]:
    files, missing = [], []
    for c in cases:
        p = cache_dir / f"{c}.npz"
        if not p.is_file():
            missing.append(c)
            continue
        files.append({"npz_path": str(p), "case_id": c, "tracer_id": tracer_id})
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} cache (e.g. {missing[:3]}); run mae_preprocess first"
        )
    return files


def make_loader(files: list[dict], batch_size: int, num_workers: int) -> DataLoader:
    tf = Compose(
        [
            LoadCachedImageNpzd(keys=["image"]),
            RandZoomd(
                keys=["image"],
                prob=0.5,
                min_zoom=0.9,
                max_zoom=1.1,
                mode="trilinear",
                keep_size=False,
            ),
            SpatialPadd(keys=["image"], spatial_size=ROI, mode="constant", constant_values=0),
            RandSpatialCropd(keys=["image"], roi_size=ROI, random_size=False),
            ToTensord(keys=["image"]),
        ]
    )
    return DataLoader(
        Dataset(files, tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def freeze_early_swin(model: nn.Module, n_stages: int) -> int:
    """Freeze patch_embed + early Swin stages to preserve FDG MAE features.

    n_stages=2 freezes layers1/1c + layers2/2c (+ patch_embed).
    """
    if n_stages <= 0:
        return 0
    core = _unwrap(model).encoder_decoder.swinViT
    frozen = 0
    # always freeze patch embed when freezing any early stage
    for p in core.patch_embed.parameters():
        p.requires_grad = False
        frozen += p.numel()
    for i in range(1, n_stages + 1):
        for attr in (f"layers{i}", f"layers{i}c"):
            mod = getattr(core, attr, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = False
                frozen += p.numel()
    return frozen


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def load_fdg_mae_weights(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    cleaned = {}
    for k, v in sd.items():
        kk = k[7:] if k.startswith("module.") else k
        # skip tracer_embed if absent in old ckpt
        if kk.startswith("tracer_embed"):
            continue
        cleaned[kk] = v
    missing, unexpected = _unwrap(model).load_state_dict(cleaned, strict=False)
    print(
        f"==> Loaded FDG MAE {ckpt_path} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"epoch={ckpt.get('epoch')} best_loss={ckpt.get('best_loss')}"
    )


def _fmt_eta_hms(eta_s: float) -> str:
    h, rem = divmod(max(0, int(eta_s)), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _fmt_finish_local(eta_s: float) -> str:
    from datetime import datetime, timedelta, timezone

    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Asia/Shanghai")
    except Exception:
        tz = timezone(timedelta(hours=8))
    return (datetime.now(tz) + timedelta(seconds=max(0, int(eta_s)))).strftime("%m-%d %H:%M")


def _eta_title_suffix(cur_ep: int, total_epochs: int, epoch_secs: list[float]) -> str:
    """``ep cur/total | ETA 1h23m → 08-12 15:30``"""
    if cur_ep >= total_epochs:
        return f"ep {cur_ep}/{total_epochs} | done"
    secs = [s for s in epoch_secs if s == s and s > 0]
    if not secs:
        return f"ep {cur_ep}/{total_epochs} | ETA …"
    window = secs[-min(10, len(secs)) :]
    avg_s = float(sum(window) / len(window))
    remain = total_epochs - cur_ep
    eta_s = avg_s * remain
    return f"ep {cur_ep}/{total_epochs} | ETA {_fmt_eta_hms(eta_s)} → {_fmt_finish_local(eta_s)}"


def update_plot(metrics_log: Path, out_png: Path, total_epochs: int) -> None:
    if not metrics_log.is_file():
        return
    xs, tot, mae_p, mae_f, align, epoch_secs = [], [], [], [], [], []
    for line in metrics_log.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        xs.append(int(row["epoch"]))
        tot.append(float(row["loss"]))
        mae_p.append(float(row.get("mae_psma", float("nan"))))
        mae_f.append(float(row.get("mae_fdg", float("nan"))))
        align.append(float(row.get("align", float("nan"))))
        es = row.get("epoch_sec", None)
        if es is not None:
            try:
                epoch_secs.append(float(es))
            except (TypeError, ValueError):
                epoch_secs.append(float("nan"))
        else:
            epoch_secs.append(float("nan"))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=120)
    ax.plot(xs, tot, label="total", color="#1f77b4", lw=1.6)
    ax.plot(xs, mae_p, label="mae_psma", color="#ff7f0e", lw=1.4)
    ax.plot(xs, mae_f, label="mae_fdg", color="#2ca02c", lw=1.4)
    ax.plot(xs, align, label="align_mmd", color="#9467bd", lw=1.4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(1, max(total_epochs, max(xs)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.92)
    cur = max(xs)
    eta_suf = _eta_title_suffix(cur, total_epochs, epoch_secs)
    ax.set_title(f"MAE continued SSL (PSMA+FDG align)\n{eta_suf}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(
        f"[ssl] device={device} n_gpu={n_gpu} epochs={args.epochs} "
        f"bs={args.batch_size} align_w={args.align_weight} "
        f"lr={args.lr} freeze_stages={args.freeze_swin_stages}"
    )

    psma_splits = json.loads(Path(args.psma_splits_json).read_text())
    fold0 = psma_splits[0] if isinstance(psma_splits, list) else psma_splits
    fdg_splits = json.loads(Path(args.fdg_splits_json).read_text())
    fdg0 = fdg_splits[0] if isinstance(fdg_splits, list) else fdg_splits

    psma_files = build_file_list(list(fold0["train"]), Path(args.psma_cache_dir), TRACER_PSMA)
    fdg_files = build_file_list(list(fdg0["train"]), Path(args.fdg_cache_dir), TRACER_FDG)
    print(f"[ssl] psma_unlabeled={len(psma_files)} fdg_align={len(fdg_files)}")

    psma_loader = make_loader(psma_files, args.batch_size, args.num_workers)
    fdg_loader = make_loader(fdg_files, args.batch_size, args.num_workers)

    model = MultiModalSwinMAE(mask_ratio=args.mask_ratio).to(device)
    load_fdg_mae_weights(model, Path(args.fdg_mae_ckpt))
    n_frozen = freeze_early_swin(model, int(args.freeze_swin_stages))
    n_train = sum(p.numel() for p in trainable_params(model))
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[ssl] freeze_early_swin stages={args.freeze_swin_stages} "
          f"frozen_params={n_frozen} trainable={n_train}/{n_all}")
    if n_gpu > 1:
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(trainable_params(model), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    latest = out_dir / "swin_mae_psma_continued_latest.pth"
    best = out_dir / "swin_mae_psma_continued_best.pth"
    metrics_log = out_dir / "metrics.jsonl"
    loss_png = Path(args.loss_png) if args.loss_png else (out_dir / "loss_curve.png")

    start_epoch = 0
    best_loss = float("inf")
    if latest.is_file() and not args.fresh:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"]
        # normalize to match current wrapping
        has_mod = any(k.startswith("module.") for k in sd)
        if isinstance(model, nn.DataParallel) and not has_mod:
            sd = {f"module.{k}": v for k, v in sd.items()}
        elif (not isinstance(model, nn.DataParallel)) and has_mod:
            sd = {k[7:]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_loss = float(ckpt.get("best_loss", float("inf")))
        print(f"==> Resume SSL epoch {start_epoch + 1} best={best_loss:.4f}")

    fdg_iter = iter(fdg_loader)

    def next_fdg():
        nonlocal fdg_iter
        try:
            return next(fdg_iter)
        except StopIteration:
            fdg_iter = iter(fdg_loader)
            return next(fdg_iter)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        sum_tot = sum_p = sum_f = sum_a = 0.0
        n_steps = 0
        lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(psma_loader, desc=f"SSL {epoch + 1}/{args.epochs} [lr={lr:.2e}]")
        for batch_p in pbar:
            batch_f = next_fdg()
            xp = batch_p["image"].to(device, non_blocking=True)
            xf = batch_f["image"].to(device, non_blocking=True)
            # tracer ids
            if "tracer_id" in batch_p:
                tid_p = batch_p["tracer_id"].to(device).long()
            else:
                tid_p = torch.full((xp.shape[0],), TRACER_PSMA, device=device, dtype=torch.long)
            if "tracer_id" in batch_f:
                tid_f = batch_f["tracer_id"].to(device).long()
            else:
                tid_f = torch.full((xf.shape[0],), TRACER_FDG, device=device, dtype=torch.long)
            # Dataset may return tracer_id as scalar list via default collate
            if tid_p.ndim == 0:
                tid_p = tid_p.expand(xp.shape[0])
            if tid_f.ndim == 0:
                tid_f = tid_f.expand(xf.shape[0])

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss_p, _, feat_p = model(xp, tid_p)
                loss_f, _, feat_f = model(xf, tid_f)
                loss_p = loss_p.mean()
                loss_f = loss_f.mean()
                align = mmd_rbf(feat_p.float(), feat_f.float(), sigma=args.mmd_sigma)
                loss = loss_p + loss_f + args.align_weight * align

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            sum_tot += float(loss.item())
            sum_p += float(loss_p.item())
            sum_f += float(loss_f.item())
            sum_a += float(align.item())
            n_steps += 1
            pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                P=f"{loss_p.item():.3f}",
                F=f"{loss_f.item():.3f}",
                A=f"{align.item():.3f}",
            )

        scheduler.step()
        avg = sum_tot / max(1, n_steps)
        avg_p = sum_p / max(1, n_steps)
        avg_f = sum_f / max(1, n_steps)
        avg_a = sum_a / max(1, n_steps)
        print(
            f"[Epoch {epoch + 1}] total={avg:.4f} mae_psma={avg_p:.4f} "
            f"mae_fdg={avg_f:.4f} align={avg_a:.4f} time={time.time() - t0:.1f}s"
        )

        state = {
            "epoch": epoch + 1,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "avg_loss": avg,
        }
        torch.save(state, latest)
        if avg < best_loss:
            best_loss = avg
            state["best_loss"] = best_loss
            torch.save(state, best)
            print(f"*** new best SSL -> {best} loss={best_loss:.4f}")

        with metrics_log.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "loss": avg,
                        "mae_psma": avg_p,
                        "mae_fdg": avg_f,
                        "align": avg_a,
                        "lr": lr,
                        "epoch_sec": time.time() - t0,
                    }
                )
                + "\n"
            )
        try:
            update_plot(metrics_log, loss_png, args.epochs)
        except Exception as e:
            print(f"[warn] plot failed: {e}")

    print(f"[ssl] done best={best_loss:.4f} latest={latest} best_ckpt={best}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fdg-mae-ckpt",
        default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/3D-MAE-PET-CT/weights/swinv2base/swin_mae_best_v2.pth",
        help="FDG MAE init (released best_v2; local latest unavailable)",
    )
    ap.add_argument(
        "--psma-splits-json",
        default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline2_psma_uda_nnunet.json",
    )
    ap.add_argument(
        "--fdg-splits-json",
        default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_fdg_nnunet.json",
    )
    ap.add_argument(
        "--psma-cache-dir",
        default="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/mae_cache/psma_baseline2_70_10",
    )
    ap.add_argument(
        "--fdg-cache-dir",
        default="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/mae_cache/fdg_baseline1_70_10",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.5)
    ap.add_argument("--align-weight", type=float, default=0.1)
    ap.add_argument("--mmd-sigma", type=float, default=1.0)
    ap.add_argument(
        "--freeze-swin-stages",
        type=int,
        default=0,
        help="freeze patch_embed + first N Swin stages (e.g. 2) to limit FDG-feature drift",
    )
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--loss-png", default="")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
