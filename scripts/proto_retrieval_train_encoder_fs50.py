#!/usr/bin/env python3
"""Fine-tune retrieval encoder on PSMA fs50 (SimCLR on augmented case_embedding views)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto_retrieval_core import case_embedding, load_ct_pet_label
from proto_retrieval_encoder import RetrievalEncoder, augment_volumes, augment_embedding, nt_xent_loss, save_encoder


def _load_fs50_train(fold: int, split_dir: Path) -> list[str]:
    p = split_dir / f"fold{fold}_nnunet.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = raw[0]
    return [str(x) for x in raw.get("train", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split-dir", type=Path, default=Path("ICLR2026/data/splits_mae_psma_fewshot50_9fold"))
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--feature-aug", action="store_true", default=True, help="augment in embedding space (fast)")
    ap.add_argument("--volume-aug", action="store_true", help="augment volumes (slow, more faithful)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)

    cases = _load_fs50_train(args.fold, args.split_dir)
    print(f"[encoder-train] fold{args.fold} n={len(cases)} device={device}")

    # cache base embeddings (+ optional volumes for volume-aug)
    base_embs: dict[str, np.ndarray] = {}
    vols: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    in_dim = None
    for i, cid in enumerate(cases):
        ct, pet, _ = load_ct_pet_label(cid, args.img_dir, None)
        emb = case_embedding(ct, pet)
        base_embs[cid] = emb
        if args.volume_aug:
            vols[cid] = (ct, pet)
        in_dim = emb.shape[0]
        if (i + 1) % 10 == 0:
            print(f"[encoder-train] cached {i+1}/{len(cases)}", flush=True)

    assert in_dim is not None
    model = RetrievalEncoder(in_dim, hidden=args.hidden, out_dim=args.out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs = max(2, min(args.batch_size, len(cases)))

    for ep in range(1, args.epochs + 1):
        model.train()
        order = list(cases)
        rng.shuffle(order)
        ep_loss = []
        for start in range(0, len(order), bs):
            batch = order[start : start + bs]
            zs: list[torch.Tensor] = []
            for cid in batch:
                if args.volume_aug:
                    ct, pet = vols[cid]
                    for _ in range(args.views):
                        act, apet = augment_volumes(ct, pet, rng)
                        emb = case_embedding(act, apet)
                        zs.append(torch.from_numpy(emb))
                else:
                    base = base_embs[cid]
                    for _ in range(args.views):
                        emb = augment_embedding(base, rng)
                        zs.append(torch.from_numpy(emb))
            z = torch.stack(zs, dim=0).to(device)
            z = model(z)
            loss = nt_xent_loss(z, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(float(loss.item()))
        if ep == 1 or ep % 20 == 0 or ep == args.epochs:
            print(f"[encoder-train] fold{args.fold} ep={ep}/{args.epochs} loss={np.mean(ep_loss):.4f}", flush=True)

    meta = {
        "fold": args.fold,
        "in_dim": in_dim,
        "hidden": args.hidden,
        "out_dim": args.out_dim,
        "epochs": args.epochs,
        "n_train": len(cases),
        "train_cases": cases,
        "loss": "SimCLR/NT-Xent on PSMA fs50 augmented views",
        "base_features": "case_embedding (PET/CT hist + coarse grid)",
    }
    save_encoder(args.out, model.cpu(), meta)
    print(f"[encoder-train] saved {args.out}")


if __name__ == "__main__":
    main()
