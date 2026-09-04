#!/usr/bin/env python3
"""Lightweight retrieval encoder: MLP on handcrafted case_embedding + SimCLR loss."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


def augment_embedding(emb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = emb.astype(np.float32, copy=True)
    x *= float(rng.uniform(0.9, 1.1))
    x += rng.normal(0.0, 0.05, size=x.shape).astype(np.float32)
    n = np.linalg.norm(x) + 1e-8
    return (x / n).astype(np.float32)


def augment_volumes(ct: np.ndarray, pet: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ct = ct.astype(np.float32, copy=True)
    pet = pet.astype(np.float32, copy=True)
    pet_scale = float(rng.uniform(0.85, 1.15))
    ct_scale = float(rng.uniform(0.90, 1.10))
    pet *= pet_scale
    ct *= ct_scale
    pet += rng.normal(0.0, 0.02 * (float(pet.max()) + 1e-6), size=pet.shape).astype(np.float32)
    ct += rng.normal(0.0, 0.02 * (float(ct.std()) + 1e-6), size=ct.shape).astype(np.float32)
    if rng.random() < 0.5:
        shift = rng.integers(-2, 3, size=3)
        pet = np.roll(pet, shift, axis=(0, 1, 2))
        ct = np.roll(ct, shift, axis=(0, 1, 2))
    return ct, pet


def nt_xent_loss(z: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SimCLR loss; z shape (2N, D) with views [0..N-1] and [N..2N-1] paired."""
    n2 = z.shape[0]
    n = n2 // 2
    sim = (z @ z.t()) / temperature
    mask = torch.eye(n2, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    pos = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    labels = pos
    return F.cross_entropy(sim, labels)


def save_encoder(path: Path, model: RetrievalEncoder, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_encoder(path: Path, map_location: str | torch.device = "cpu") -> tuple[RetrievalEncoder, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt.get("meta") or {}
    model = RetrievalEncoder(
        int(meta.get("in_dim", 512)),
        hidden=int(meta.get("hidden", 256)),
        out_dim=int(meta.get("out_dim", 128)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


@torch.no_grad()
def encode_numpy(model: RetrievalEncoder, feats: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(feats.astype(np.float32)).to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    z = model(x).cpu().numpy()
    return z.astype(np.float32)


def encode_gallery(model: RetrievalEncoder, raw_embs: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(raw_embs), batch):
        chunk = raw_embs[i : i + batch]
        out.append(encode_numpy(model, chunk, device))
    return np.concatenate(out, axis=0)
