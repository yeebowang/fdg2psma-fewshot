#!/usr/bin/env python3
"""SegAnyPET fewshot finetune (SAM-Med3D-style click training).

Official train_cpcl.py is broken (IndentationError / missing get_next_click3D_torch_3).
This script does standard multi-click DiceCE finetune from seganypet_v2 / lesion ckpt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

# SegAnyPET code + optional site-packages overlay
_CODE = Path(__file__).resolve().parents[1] / "third_party" / "SegAnyPET" / "code"
_PIP = Path(__file__).resolve().parents[1] / "third_party" / "seganypet_pip"
if _PIP.is_dir():
    sys.path.insert(0, str(_PIP))
sys.path.insert(0, str(_CODE))

from segment_anything.build_sam3D import sam_model_registry3D  # noqa: E402
from utils.click_method import get_next_click3D_torch_2  # noqa: E402
from utils.data_loader import Dataset_Union_ALL  # noqa: E402
from utils.infer_utils import validate_paired_img_gt, read_arr_from_nifti  # noqa: E402


def _binary_dice_nifti(gt_path: str, pred_path: str) -> float:
    """Dice without surface_distance dependency (official metric_utils needs it)."""
    gt = read_arr_from_nifti(gt_path) > 0
    pred = read_arr_from_nifti(pred_path) > 0
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def build_loader(data_root: Path, img_size: int, batch_size: int, num_workers: int, threshold: int):
    ds = Dataset_Union_ALL(
        paths=[str(data_root)],
        mode="train",
        data_type="Tr",
        image_size=img_size,
        threshold=threshold,
        transform=tio.Compose(
            [
                tio.ToCanonical(),
                tio.CropOrPad(mask_name="label", target_shape=(img_size, img_size, img_size)),
                tio.RandomFlip(axes=(0, 1, 2)),
            ]
        ),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(ds) >= batch_size,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


class ClickSegTrainModule(torch.nn.Module):
    """Wrap encode+click-interaction so DataParallel can split the batch."""

    def __init__(self, sam: torch.nn.Module, img_size: int):
        super().__init__()
        self.sam = sam
        self.img_size = img_size
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")

    def forward(self, image3D: torch.Tensor, gt3D: torch.Tensor, num_clicks: torch.Tensor):
        nc = int(num_clicks.reshape(-1)[0].item())
        image_embedding = self.sam.image_encoder(image3D)
        prev_masks, loss = interaction(
            self.sam, image_embedding, gt3D, nc, self.img_size, self.seg_loss, image3D.device
        )
        return prev_masks, loss


def state_dict_cpu(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in unwrap(model).state_dict().items()}


def load_model(ckpt: Path | None, device: torch.device) -> torch.nn.Module:
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    if ckpt is None or not Path(ckpt).is_file():
        print("[load] random ViT-B (scratch) — no checkpoint")
        return model.to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    # strip module.
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt.name} missing={len(miss)} unexpected={len(unexp)}")
    return model.to(device)


def batch_forward(model, image_embedding, gt3D, low_res_masks, points, img_size: int):
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points, boxes=None, masks=low_res_masks
    )
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    prev_masks = F.interpolate(low_res_masks, size=gt3D.shape[-3:], mode="trilinear", align_corners=False)
    return low_res_masks, prev_masks


def interaction(model, image_embedding, gt3D, num_clicks: int, img_size: int, seg_loss, device):
    prev_masks = torch.zeros_like(gt3D, device=device, dtype=torch.float)
    low_res = F.interpolate(
        prev_masks,
        size=(img_size // 4, img_size // 4, img_size // 4),
        mode="trilinear",
        align_corners=False,
    )
    total_loss = 0.0
    click_points, click_labels = [], []
    for _ in range(num_clicks):
        batch_points, batch_labels = get_next_click3D_torch_2(prev_masks, gt3D)
        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)
        click_points.append(points_co)
        click_labels.append(points_la)
        points_multi = torch.cat(click_points, dim=1)
        labels_multi = torch.cat(click_labels, dim=1)
        low_res, prev_masks = batch_forward(
            model, image_embedding, gt3D, low_res, points=[points_multi, labels_multi], img_size=img_size
        )
        total_loss = total_loss + seg_loss(prev_masks, gt3D)
    return prev_masks, total_loss


def dice_score(prev_masks, gt3D) -> float:
    pred = torch.sigmoid(prev_masks) > 0.5
    gt = gt3D > 0
    scores = []
    for i in range(gt.shape[0]):
        inter = (pred[i] & gt[i]).sum().float()
        denom = pred[i].sum().float() + gt[i].sum().float()
        if denom.item() == 0:
            scores.append(float("nan"))
        else:
            scores.append((2 * inter / denom).item())
    ok = [s for s in scores if s == s]
    return float(np.mean(ok)) if ok else float("nan")


@torch.no_grad()
def eval_val_set(
    model,
    data_root: Path,
    out_pred_dir: Path,
    num_clicks: int,
    crop_size: int,
    max_cases: int | None,
    seed: int,
) -> float:
    """Full-volume click eval on imagesVal/labelsVal (official protocol)."""
    img_dir = data_root / "imagesVal"
    lab_dir = data_root / "labelsVal"
    if not lab_dir.is_dir():
        return float("nan")
    pairs = []
    for name in sorted(lab_dir.glob("*.nii.gz")):
        img = img_dir / name.name
        if img.is_file():
            pairs.append((img, name))
    if max_cases is not None:
        pairs = pairs[:max_cases]
    if not pairs:
        return float("nan")

    out_pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    dices = []
    for img_path, lab_path in tqdm(pairs, desc="val", leave=False):
        pred_path = out_pred_dir / lab_path.name
        try:
            validate_paired_img_gt(
                model=model,
                img_path=str(img_path),
                gt_path=str(lab_path),
                output_path=str(pred_path),
                num_clicks=num_clicks,
                crop_size=crop_size,
                target_spacing=(1.5, 1.5, 1.5),
                seed=seed,
            )
            metrics = None
            try:
                d = _binary_dice_nifti(str(lab_path), str(pred_path))
            except Exception as e:
                print(f"[val-metric] {lab_path.name}: {e}")
                continue
            if d == d:
                dices.append(float(d))
        except Exception as e:
            print(f"[val-warn] {lab_path.name}: {e}")
            continue
    return float(np.mean(dices)) if dices else float("nan")


def _optional_ckpt(s: str) -> Path | None:
    t = (s or "").strip()
    if not t or t.lower() in ("none", "scratch", "-", "random"):
        return None
    return Path(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True, help="fold dir with imagesTr/labelsTr/imagesVal/labelsVal")
    ap.add_argument("--checkpoint", type=_optional_ckpt, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accumulation-steps", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=8e-5)
    ap.add_argument("--lr-other", type=float, default=8e-4)
    ap.add_argument(
        "--lr-mode",
        default="finetune",
        choices=["finetune", "official"],
        help="official=train_cpcl style: encoder lr=--lr, prompt/decoder=0.1*lr",
    )
    ap.add_argument("--lr", type=float, default=8e-4, help="base lr for --lr-mode official")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--milestones", default="60,85", help="MultiStepLR milestones")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-interval", type=int, default=20)
    ap.add_argument("--val-clicks", type=int, default=5)
    ap.add_argument("--val-max-cases", type=int, default=0, help="0=all val cases")
    ap.add_argument("--label-threshold", type=int, default=50)
    ap.add_argument("--click-max", type=int, default=11, help="train clicks ~ U[1, click_max)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-dataparallel", action="store_true", help="disable DP even if multi-GPU visible")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sam = load_model(args.checkpoint, device)
    train_mod = ClickSegTrainModule(sam, args.img_size).to(device)
    use_dp = (not args.no_dataparallel) and n_gpu > 1
    if use_dp:
        train_mod = torch.nn.DataParallel(train_mod)
        print(f"[train] DataParallel on {n_gpu} GPUs (global_bs={args.batch_size} ≈ {args.batch_size}/{n_gpu}/GPU)")

    norm = tio.ZNormalization(masking_method=lambda x: x > 0)

    raw = unwrap(train_mod).sam
    if args.lr_mode == "official":
        # match train_cpcl.py: encoder @ lr; prompt_encoder & mask_decoder @ 0.1*lr
        param_groups = [
            {"params": list(raw.image_encoder.parameters()), "lr": args.lr},
            {"params": list(raw.prompt_encoder.parameters()), "lr": args.lr * 0.1},
            {"params": list(raw.mask_decoder.parameters()), "lr": args.lr * 0.1},
        ]
        print(f"[train] lr_mode=official lr={args.lr} (prompt/decoder x0.1)")
    else:
        enc_ids = set(id(p) for p in raw.image_encoder.parameters())
        enc_params = [p for p in raw.parameters() if id(p) in enc_ids]
        other_params = [p for p in raw.parameters() if id(p) not in enc_ids]
        param_groups = [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": other_params, "lr": args.lr_other},
        ]
        print(f"[train] lr_mode=finetune enc={args.lr_encoder} other={args.lr_other}")

    milestones = [int(x) for x in args.milestones.split(",") if x.strip()]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr if args.lr_mode == "official" else args.lr_other, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = amp.GradScaler()

    start_ep = 0
    best_val = -1.0
    latest = args.out_dir / "latest.pth"
    if (not args.fresh) and latest.is_file():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        unwrap(train_mod).sam.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_ep = int(ck.get("epoch", 0))
        best_val = float(ck.get("best_val_dice", -1.0))
        print(f"[resume] epoch={start_ep} best_val={best_val}")

    loader = build_loader(args.data_root, args.img_size, args.batch_size, args.num_workers, args.label_threshold)
    print(
        f"[train] n_train≈{len(loader.dataset)} device={device} n_gpu={n_gpu} "
        f"bs={args.batch_size} accum={args.accumulation_steps} milestones={milestones}",
        flush=True,
    )

    for epoch in range(start_ep, args.epochs):
        t0 = time.time()
        train_mod.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)
        num_clicks = int(np.random.randint(1, max(2, args.click_max)))

        pbar = tqdm(loader, desc=f"ep{epoch+1}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            image3D = batch["image"]
            gt3D = batch["label"].long()
            image3D = norm(image3D.squeeze(dim=1))
            image3D = image3D.unsqueeze(dim=1).to(device)
            gt3D = (gt3D > 0).float().to(device)
            clicks_t = torch.full((image3D.shape[0],), num_clicks, device=device, dtype=torch.long)

            with amp.autocast():
                prev_masks, loss = train_mod(image3D, gt3D, clicks_t)
                if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                    loss = loss.mean()
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * args.accumulation_steps
            epoch_dice += dice_score(prev_masks.detach(), gt3D)
            n_steps += 1
            pbar.set_postfix(loss=epoch_loss / n_steps, dice=epoch_dice / n_steps, clicks=num_clicks)

        scheduler.step()
        train_loss = epoch_loss / max(n_steps, 1)
        train_dice = epoch_dice / max(n_steps, 1)
        epoch_sec = time.time() - t0

        val_dice = None
        do_val = ((epoch + 1) % args.val_interval == 0) or (epoch + 1 == args.epochs)
        if do_val:
            if (epoch + 1) == args.epochs or args.val_max_cases <= 0:
                vmax = None
            else:
                vmax = args.val_max_cases
            val_dice = eval_val_set(
                unwrap(train_mod).sam,
                args.data_root,
                args.out_dir / f"val_pred_ep{epoch+1}",
                num_clicks=args.val_clicks,
                crop_size=args.img_size,
                max_cases=vmax,
                seed=args.seed,
            )
            if (
                val_dice is not None
                and val_dice == val_dice
                and val_dice > 1e-6
                and val_dice > best_val
            ):
                best_val = val_dice
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": state_dict_cpu(unwrap(train_mod).sam),
                        "best_val_dice": best_val,
                        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    },
                    args.out_dir / "best.pth",
                )
                print(f"[best] ep={epoch+1} val_dice={best_val:.4f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict_cpu(unwrap(train_mod).sam),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": best_val,
            },
            latest,
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
            "best_val_dice": best_val if best_val >= 0 else None,
            "epoch_sec": epoch_sec,
            "num_clicks_train": num_clicks,
            "n_gpu": n_gpu,
            "batch_size": args.batch_size,
        }
        _append_jsonl(metrics_path, row)
        print(
            f"[ep {epoch+1}/{args.epochs}] loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val={val_dice} best={best_val:.4f} sec={epoch_sec:.1f}",
            flush=True,
        )

    summary = {
        "best_val_dice": best_val if best_val >= 0 else None,
        "epochs": args.epochs,
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "lr_mode": args.lr_mode,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "n_gpu": n_gpu,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("[done]", summary)


if __name__ == "__main__":
    main()
