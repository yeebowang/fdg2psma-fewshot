#!/usr/bin/env python3
"""FDG SwinUNETR-Base segmentation finetune from MAE foundation weights.

Defaults match the upstream notebook (100 ep, AdamW 1e-4, cosine, DiceCE),
adapted to ICLR2026 FDG 70/10 splits and multi-GPU DataParallel bs=2*3=6.

Late phase (last N epochs, default 20): every epoch runs FDG + PSMA dual val
and live loss plot becomes 3 curves (train / FDG val_loss / PSMA_val_loss).
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
import torch.optim as optim
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    MapTransform,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ToTensord,
    AsDiscrete,
)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROI = (96, 128, 128)


class LoadCachedNpzd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        with np.load(d["npz_path"]) as z:
            d["image"] = z["image"].astype(np.float32)
            d["label"] = z["label"].astype(np.float32)
        return d


def build_file_list(cases: list[str], cache_dir: Path) -> list[dict]:
    files = []
    missing = []
    for c in cases:
        p = cache_dir / f"{c}.npz"
        if not p.is_file():
            missing.append(c)
            continue
        files.append({"npz_path": str(p), "case_id": c})
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} cached cases (e.g. {missing[:3]}); run mae_preprocess_fdg_cache.py first"
        )
    return files


def _load_val_cases(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and "val" in payload:
        return list(payload["val"])
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "val" in payload[0]:
            return list(payload[0]["val"])
        return [str(x) for x in payload]
    raise ValueError(f"unsupported val cases json: {path}")


def make_loader(files: list[dict], batch_size: int, num_workers: int, train: bool) -> DataLoader:
    if train:
        tf = Compose(
            [
                LoadCachedNpzd(keys=["image", "label"]),
                RandSpatialCropd(keys=["image", "label"], roi_size=ROI, random_size=False),
                ToTensord(keys=["image", "label"]),
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
    # val loss: pad + center crop (fast, stable); SWI dice uses full volume separately
    tf = Compose(
        [
            LoadCachedNpzd(keys=["image", "label"]),
            SpatialPadd(keys=["image", "label"], spatial_size=ROI, mode="constant"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=ROI),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return DataLoader(
        Dataset(files, tf),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )


def make_sw_loader(files: list[dict], num_workers: int) -> DataLoader:
    tf = Compose(
        [
            LoadCachedNpzd(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return DataLoader(
        Dataset(files, tf),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )


def get_loaders(
    cache_dir: Path,
    splits_json: Path,
    batch_size: int,
    num_workers: int,
    cross_val_json: Path | None,
    cross_cache_dir: Path | None,
):
    splits = json.loads(Path(splits_json).read_text())
    fold0 = splits[0] if isinstance(splits, list) else splits
    train_files = build_file_list(list(fold0["train"]), cache_dir)
    val_files = build_file_list(list(fold0["val"]), cache_dir)
    print(f"[data] train={len(train_files)} primary_val={len(val_files)} bs={batch_size}")

    train_loader = make_loader(train_files, batch_size, num_workers, train=True)
    primary_val_loader = make_loader(val_files, 1, num_workers, train=False)
    primary_sw_loader = make_sw_loader(val_files, num_workers)

    cross_val_loader = None
    if cross_val_json is not None:
        ccache = cross_cache_dir or cache_dir
        cross_cases = _load_val_cases(cross_val_json)
        cross_files = build_file_list(cross_cases, ccache)
        cross_val_loader = make_loader(cross_files, 1, num_workers, train=False)
        print(f"[data] cross_val={len(cross_files)} cache={ccache}")
    return train_loader, primary_val_loader, primary_sw_loader, cross_val_loader


def build_param_groups(model: nn.Module, lr: float, backbone_lr_mult: float):
    """Lower LR for swinViT/encoder*; full LR for decoder/out."""
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nn = name[7:] if name.startswith("module.") else name
        if nn.startswith("swinViT") or nn.startswith("encoder"):
            backbone.append(p)
        else:
            head.append(p)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": lr * float(backbone_lr_mult)})
    if head:
        groups.append({"params": head, "lr": lr})
    if not groups:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    return groups


def set_encoder_requires_grad(model: nn.Module, enabled: bool) -> None:
    core = model.module if isinstance(model, nn.DataParallel) else model
    for name, p in core.named_parameters():
        if name.startswith("swinViT") or name.startswith("encoder"):
            p.requires_grad = enabled


def build_model(
    foundation_ckpt: Path | None,
    use_foundation: bool,
    *,
    foundation_kind: str = "mae",
    in_channels: int = 2,
    depths: tuple[int, ...] = (2, 2, 6, 2),
    use_v2: bool = True,
) -> nn.Module:
    downsample = "mergingv2" if use_v2 else "merging"
    model = SwinUNETR(
        img_size=ROI,
        in_channels=in_channels,
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
    if not use_foundation or foundation_ckpt is None:
        return model
    if not foundation_ckpt.is_file():
        raise FileNotFoundError(f"foundation ckpt missing: {foundation_ckpt}")

    print(f"==> Loading Foundation ({foundation_kind}): {foundation_ckpt}")
    checkpoint = torch.load(foundation_ckpt, map_location="cpu", weights_only=False)

    if foundation_kind == "monai_swinvit":
        # Tang et al. / MONAI SwinViT SSL: expects weights["state_dict"]["module.*"]
        weights = checkpoint if isinstance(checkpoint, dict) and "state_dict" in checkpoint else {"state_dict": checkpoint}
        model.load_from(weights)
        print("Weight transfer OK via SwinUNETR.load_from (monai_swinvit)")
        return model

    if foundation_kind == "seg":
        # Full SwinUNETR supervised seg ckpt (best_*.pth = raw sd; latest_*.pth = wrapped)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            mae_sd = checkpoint["model_state_dict"]
        else:
            mae_sd = checkpoint
        seg_sd = {}
        for k, v in mae_sd.items():
            kk = k[7:] if k.startswith("module.") else k
            seg_sd[kk] = v
        missing, unexpected = model.load_state_dict(seg_sd, strict=False)
        print(
            f"Weight transfer OK (seg) missing={len(missing)} unexpected={len(unexpected)}"
        )
        return model

    if foundation_kind == "monai_ssl":
        # Disruptive Autoencoders SSL (filter_swinunetr); 1ch→2ch patch embed
        from monai.networks.nets.swin_unetr import filter_swinunetr
        from monai.networks.utils import copy_model_state

        ssl = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        tmp = SwinUNETR(
            img_size=ROI,
            in_channels=1,
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
        copy_model_state(tmp, ssl, filter_func=filter_swinunetr)
        sd1, sd2 = tmp.state_dict(), model.state_dict()
        n = 0
        for k, v in sd1.items():
            if k in sd2 and sd2[k].shape == v.shape:
                sd2[k] = v
                n += 1
            elif (
                k == "swinViT.patch_embed.proj.weight"
                and v.ndim == 5
                and v.shape[1] == 1
                and sd2[k].shape[1] == in_channels
            ):
                sd2[k] = v.repeat(1, in_channels, 1, 1, 1) / float(in_channels)
                n += 1
                print(f"expanded patch_embed 1->{in_channels}")
        model.load_state_dict(sd2)
        print(f"Weight transfer OK monai_ssl tensors={n}")
        return model

    # default: MAE / continued-SSL style (encoder_decoder.*)
    mae_sd = checkpoint.get("model_state_dict", checkpoint)
    seg_sd = {}
    for k, v in mae_sd.items():
        kk = k[7:] if k.startswith("module.") else k
        if "encoder_decoder." in kk:
            seg_sd[kk.split("encoder_decoder.", 1)[1]] = v
    missing, unexpected = model.load_state_dict(seg_sd, strict=False)
    print(f"Weight transfer OK missing={len(missing)} unexpected={len(unexpected)}")
    return model


def _unwrap_sd(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _load_sd(model: nn.Module, sd: dict) -> None:
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(sd)
    else:
        model.load_state_dict(sd)


@torch.no_grad()
def eval_val_loss(model, loader, loss_fn, device, desc: str) -> float:
    model.eval()
    losses = []
    for batch in tqdm(loader, desc=desc):
        inputs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast("cuda"):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_sw_dice_hd95(model, loader, device, sw_batch_size: int, desc: str):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    for val_data in tqdm(loader, desc=desc):
        vi = val_data["image"].to(device, non_blocking=True)
        vl = val_data["label"].to(device, non_blocking=True)
        with autocast("cuda"):
            vo = sliding_window_inference(
                inputs=vi,
                roi_size=ROI,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.5,
            )
        vo = [post_pred(i) for i in vo]
        vl = [post_label(i) for i in vl]
        dice_metric(y_pred=vo, y=vl)
        try:
            hd95_metric(y_pred=vo, y=vl)
        except Exception:
            pass
    mean_dice = float(dice_metric.aggregate().item())
    try:
        mean_hd95 = float(hd95_metric.aggregate().item())
    except Exception:
        mean_hd95 = float("inf")
    return mean_dice, mean_hd95


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
    # prefer recent window
    window = secs[-min(10, len(secs)) :]
    avg_s = float(sum(window) / len(window))
    remain = total_epochs - cur_ep
    eta_s = avg_s * remain
    return f"ep {cur_ep}/{total_epochs} | ETA {_fmt_eta_hms(eta_s)} → {_fmt_finish_local(eta_s)}"


def update_live_loss_plot(
    metrics_log: Path,
    out_png: Path,
    total_epochs: int,
    val_from_epoch: int,
    *,
    title_tag: str = "MAE finetune",
    cross_label: str = "PSMA_val_loss",
) -> None:
    if not metrics_log.is_file():
        return
    xs, train, primary, cross, epoch_secs = [], [], [], [], []
    for line in metrics_log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = int(row["epoch"])
        xs.append(ep)
        train.append(float(row.get("train_loss", float("nan"))))
        es = row.get("epoch_sec", None)
        if es is not None:
            try:
                epoch_secs.append(float(es))
            except (TypeError, ValueError):
                epoch_secs.append(float("nan"))
        else:
            epoch_secs.append(float("nan"))
        vl = row.get("val_loss", None)
        # accept both new cross_val_loss and legacy psma_val_loss
        pl = row.get("cross_val_loss", row.get("psma_val_loss", None))
        if ep < val_from_epoch:
            primary.append(float("nan"))
            cross.append(float("nan"))
        else:
            primary.append(float(vl) if vl is not None and vl == vl else float("nan"))
            cross.append(float(pl) if pl is not None and pl == pl else float("nan"))

    if not xs:
        return

    has_primary = any(v == v for v in primary)
    has_cross = any(v == v for v in cross)
    cur_ep = max(xs)
    eta_suf = _eta_title_suffix(cur_ep, total_epochs, epoch_secs)

    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=120)
    ax.plot(xs, train, label="train_loss", color="#1f77b4", linewidth=1.6)
    if has_primary:
        ax.plot(xs, primary, label="val_loss", color="#ff7f0e", linewidth=1.6)
    if has_cross:
        ax.plot(xs, cross, label=cross_label, color="#9467bd", linewidth=1.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xlim(1, max(total_epochs, max(xs)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.92)
    n_curves = 1 + int(has_primary) + int(has_cross)
    title = f"{title_tag} ({n_curves} curves)\n{eta_suf}"
    ax.set_title(title, fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"[train] device={device} n_gpu={n_gpu} epochs={args.epochs} global_bs={args.batch_size}")

    cross_json_raw = args.cross_val_json or args.psma_val_json
    cross_json = Path(cross_json_raw) if cross_json_raw else None
    cross_cache = Path(args.cross_cache_dir or args.psma_cache_dir or args.cache_dir)
    dual_from = int(args.cross_val_from_epoch or args.psma_val_from_epoch)
    if dual_from <= 0:
        dual_from = max(1, args.epochs - args.late_dual_epochs + 1)
    cross_label = args.cross_val_label or "PSMA_val_loss"
    title_tag = args.title_tag or "MAE SwinBase finetune"
    ckpt_stem = args.ckpt_stem or "seg_mae"
    print(
        f"[train] dual_val from_ep>={dual_from} "
        f"(last {args.epochs - dual_from + 1} ep) cross_json={cross_json} label={cross_label}"
    )

    train_loader, primary_val_loader, primary_sw_loader, cross_val_loader = get_loaders(
        Path(args.cache_dir),
        Path(args.splits_json),
        args.batch_size,
        args.num_workers,
        cross_json,
        cross_cache,
    )

    latest_ckpt = out_dir / f"latest_{ckpt_stem}.pth"
    best_ckpt = out_dir / f"best_{ckpt_stem}.pth"
    metrics_log = out_dir / "metrics.jsonl"
    loss_png = Path(args.loss_png) if args.loss_png else (out_dir / "loss_curve.png")

    model = build_model(
        None,
        use_foundation=False,
        foundation_kind=args.foundation_kind,
        depths=tuple(int(x) for x in args.depths.split(",")),
        use_v2=bool(args.use_v2),
    )
    model = model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = optim.AdamW(
        build_param_groups(model, args.lr, args.backbone_lr_mult),
        weight_decay=1e-5,
    )
    scaler = GradScaler("cuda")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_metric = -1.0
    if latest_ckpt.is_file() and not args.fresh:
        print(f"==> Resume from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        _load_sd(model, ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = int(ckpt["epoch"])
        best_metric = float(ckpt.get("best_metric", -1.0))
        print(f"==> Resumed at epoch {start_epoch + 1}, best_dice={best_metric:.4f}")
    else:
        raw = build_model(
            Path(args.foundation_ckpt) if args.foundation_ckpt else None,
            use_foundation=(args.foundation_kind != "none" and bool(args.foundation_ckpt)),
            foundation_kind=args.foundation_kind,
            depths=tuple(int(x) for x in args.depths.split(",")),
            use_v2=bool(args.use_v2),
        )
        if args.foundation_kind == "none" or not args.foundation_ckpt:
            print(f"[train] random init depths={args.depths} use_v2={args.use_v2}")

        if n_gpu > 1:
            model = nn.DataParallel(raw.to(device))
        else:
            model = raw.to(device)
        optimizer = optim.AdamW(
            build_param_groups(model, args.lr, args.backbone_lr_mult),
            weight_decay=1e-5,
        )
        scaler = GradScaler("cuda")
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(
            f"[train] backbone_lr_mult={args.backbone_lr_mult} "
            f"freeze_encoder_epochs={args.freeze_encoder_epochs} "
            f"param_groups_lr={[g['lr'] for g in optimizer.param_groups]}"
        )

    freeze_ep = int(args.freeze_encoder_epochs)
    encoder_frozen = False
    if freeze_ep > 0 and start_epoch < freeze_ep:
        set_encoder_requires_grad(model, False)
        encoder_frozen = True
        print(f"[train] encoder frozen for first {freeze_ep} epochs")

    for epoch in range(start_epoch, args.epochs):
        if encoder_frozen and epoch >= freeze_ep:
            set_encoder_requires_grad(model, True)
            encoder_frozen = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # rebuild optimizer so newly enabled params get grads with correct LR
            optimizer = optim.AdamW(
                build_param_groups(model, args.lr, args.backbone_lr_mult),
                weight_decay=1e-5,
            )
            # keep cosine progress roughly
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
            for _ in range(epoch):
                scheduler.step()
            print(f"[train] encoder unfrozen at epoch {epoch + 1}")

        model.train()
        epoch_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [lr={lr:.2e}]")
        t0 = time.time()
        for batch in pbar:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1} train_loss={avg_loss:.4f} time={time.time() - t0:.1f}s")

        ep1 = epoch + 1
        late = ep1 >= dual_from
        do_primary_val = late or (ep1 % args.val_interval == 0) or (ep1 == args.epochs)
        do_cross_val = late and cross_val_loader is not None
        do_sw_dice = ((ep1 % args.val_interval == 0) or (ep1 == args.epochs)) and (
            not late or args.late_sw_dice
        )

        mean_dice = float("nan")
        mean_hd95 = float("nan")
        primary_val_loss = float("nan")
        cross_val_loss = float("nan")

        if do_primary_val:
            primary_val_loss = eval_val_loss(
                model, primary_val_loader, loss_fn, device, desc=f"Epoch {ep1} [val]"
            )
            print(f"val_loss {primary_val_loss:.4f}")

        if do_cross_val:
            cross_val_loss = eval_val_loss(
                model, cross_val_loader, loss_fn, device, desc=f"Epoch {ep1} [cross_val]"
            )
            print(f"{cross_label} {cross_val_loss:.4f}")

        if do_sw_dice:
            mean_dice, mean_hd95 = eval_sw_dice_hd95(
                model, primary_sw_loader, device, args.sw_batch_size, desc=f"Epoch {ep1} [sw_dice]"
            )
            print(f"👉 Val Dice={mean_dice:.4f} HD95={mean_hd95:.4f}")
            if mean_dice > best_metric:
                best_metric = mean_dice
                torch.save(_unwrap_sd(model), best_ckpt)
                print(f"🏆 new best -> {best_ckpt} Dice={best_metric:.4f}")

        epoch_sec = float(time.time() - t0)
        state = {
            "epoch": ep1,
            "model_state_dict": _unwrap_sd(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "avg_train_loss": avg_loss,
            "val_loss": primary_val_loss,
            "cross_val_loss": cross_val_loss,
        }
        torch.save(state, latest_ckpt)

        row = {
            "epoch": ep1,
            "train_loss": avg_loss,
            "val_loss": primary_val_loss if do_primary_val else None,
            "cross_val_loss": cross_val_loss if do_cross_val else None,
            # legacy alias for old FDG-run plotters
            "psma_val_loss": cross_val_loss if do_cross_val else None,
            "val_dice": mean_dice,
            "val_hd95": mean_hd95,
            "best_dice": best_metric,
            "lr": lr,
            "late_dual_val": late,
            "epoch_sec": epoch_sec,
        }
        with metrics_log.open("a") as f:
            f.write(json.dumps(row) + "\n")

        try:
            update_live_loss_plot(
                metrics_log,
                loss_png,
                args.epochs,
                dual_from,
                title_tag=title_tag,
                cross_label=cross_label,
            )
        except Exception as e:
            print(f"[warn] loss plot failed: {e}")

    print(f"[train] done best_dice={best_metric:.4f} out={out_dir} loss_png={loss_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        default="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/mae_cache/fdg_baseline1_70_10",
    )
    ap.add_argument(
        "--splits-json",
        default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_fdg_nnunet.json",
    )
    ap.add_argument(
        "--foundation-ckpt",
        default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/3D-MAE-PET-CT/weights/swinv2base/swin_mae_best_v2.pth",
    )
    ap.add_argument(
        "--foundation-kind",
        default="mae",
        choices=["mae", "seg", "monai_swinvit", "monai_ssl", "none"],
        help="mae=FDG/continued MAE encoder_decoder; seg=full SwinUNETR supervised; "
        "monai_swinvit=Tang SSL; monai_ssl=DA SSL; none=scratch",
    )
    ap.add_argument(
        "--depths",
        default="2,2,6,2",
        help="SwinUNETR depths; monai_swinvit typically 2,2,2,2",
    )
    ap.add_argument(
        "--use-v2",
        type=int,
        default=1,
        help="1=use_v2+mergingv2 (MAE base); 0=classic SwinUNETR for monai_swinvit",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=6, help="global batch (2*3 GPUs)")
    ap.add_argument("--sw-batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=1.0,
        help="LR multiplier for swinViT/encoder* vs decoder/out (use 0.1 for SSL transfer)",
    )
    ap.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=0,
        help="freeze swinViT+encoder for first N epochs then unfreeze",
    )
    ap.add_argument("--val-interval", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="ignore latest ckpt, reload foundation")
    # dual val / 3-curve loss
    ap.add_argument(
        "--cross-val-json",
        default="",
        help="late dual-val cases JSON; empty falls back to --psma-val-json",
    )
    ap.add_argument(
        "--cross-cache-dir",
        default="",
        help="cache for cross-val cases (default: --cache-dir)",
    )
    ap.add_argument("--cross-val-label", default="PSMA_val_loss")
    ap.add_argument("--cross-val-from-epoch", type=int, default=0)
    ap.add_argument("--title-tag", default="MAE SwinBase finetune")
    ap.add_argument("--ckpt-stem", default="seg_mae")
    # legacy aliases (FDG-run scripts)
    ap.add_argument(
        "--psma-val-json",
        default="/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_psma_val.json",
    )
    ap.add_argument("--psma-cache-dir", default="")
    ap.add_argument("--psma-val-from-epoch", type=int, default=0)
    ap.add_argument("--late-dual-epochs", type=int, default=20)
    ap.add_argument(
        "--late-sw-dice",
        action="store_true",
        help="also run SWI dice during late dual-val phase (slow)",
    )
    ap.add_argument("--loss-png", default="")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
