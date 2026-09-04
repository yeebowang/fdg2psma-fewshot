#!/usr/bin/env python3
"""Prototype + Retrieval few-shot seg: retrieve support from FDG pool, ALPNet-lite prototype match."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


@dataclass
class Gallery:
    case_ids: list[str]
    embeddings: np.ndarray  # (N, D) float32, L2-normalized rows

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            case_ids=np.array(self.case_ids, dtype=object),
            embeddings=self.embeddings.astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> Gallery:
        d = np.load(path, allow_pickle=True)
        return cls(list(d["case_ids"]), d["embeddings"])


def load_fdg100_case_ids(stratified_json: Path) -> list[str]:
    return _load_modality_case_ids(stratified_json, "fdg_")


def load_fdg80_case_ids(stratified_json: Path) -> list[str]:
    """FDG train+val (70%+10%) from stratified 70/10/20 split."""
    raw = json.loads(stratified_json.read_text(encoding="utf-8"))
    out: list[str] = []
    for key in ("train", "val"):
        for c in raw.get(key, []):
            cid = str(c)
            if cid.startswith("fdg_"):
                out.append(cid)
    return sorted(set(out))


def load_fdg70_case_ids(stratified_json: Path) -> list[str]:
    """FDG train only (70%) from stratified 70/10/20 split — support pool for FDG TEST."""
    raw = json.loads(stratified_json.read_text(encoding="utf-8"))
    return sorted({str(c) for c in raw.get("train", []) if str(c).startswith("fdg_")})


def load_fdg20_test_case_ids(stratified_json: Path) -> list[str]:
    """FDG test (20%) from stratified 70/10/20 split."""
    raw = json.loads(stratified_json.read_text(encoding="utf-8"))
    return sorted({str(c) for c in raw.get("test", []) if str(c).startswith("fdg_")})


def load_psma100_case_ids(stratified_json: Path) -> list[str]:
    return _load_modality_case_ids(stratified_json, "psma_")


def load_psma70_case_ids(stratified_json: Path) -> list[str]:
    """PSMA train only (70%) — support pool for Proto+ fc70% → PSMA TEST20."""
    raw = json.loads(stratified_json.read_text(encoding="utf-8"))
    return sorted({str(c) for c in raw.get("train", []) if str(c).startswith("psma_")})


def _load_modality_case_ids(stratified_json: Path, prefix: str) -> list[str]:
    raw = json.loads(stratified_json.read_text(encoding="utf-8"))
    out: list[str] = []
    for key in ("train", "val", "test"):
        for c in raw.get(key, []):
            if str(c).startswith(prefix):
                out.append(str(c))
    return sorted(set(out))


def _read_vol(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def load_ct_pet_label(
    case_id: str,
    img_dir: Path,
    lab_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ct = _read_vol(img_dir / f"{case_id}_0000.nii.gz")
    pet = _read_vol(img_dir / f"{case_id}_0001.nii.gz")
    lab = None
    if lab_dir is not None:
        lp = lab_dir / f"{case_id}.nii.gz"
        if lp.is_file():
            lab = np.asarray(nib.load(str(lp)).dataobj) > 0
    return ct, pet, lab


def body_mask(pet: np.ndarray, ct: np.ndarray, thr_frac: float = 0.05) -> np.ndarray:
    ref = pet if np.any(pet > 0) else ct
    mx = float(ref.max()) if ref.size else 0.0
    if mx <= 0:
        return np.ones_like(ref, dtype=bool)
    return ref > (thr_frac * mx)


def crop_to_mask(
    ct: np.ndarray, pet: np.ndarray, mask: np.ndarray | None, lab: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    bm = body_mask(pet, ct)
    idx = np.where(bm)
    if idx[0].size == 0:
        return ct, pet, lab
    slc = tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)
    ct_c = ct[slc]
    pet_c = pet[slc]
    lab_c = lab[slc] if lab is not None else None
    return ct_c, pet_c, lab_c


def resize_vol(vol: np.ndarray, shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    if vol.shape == shape:
        return vol
    factors = [t / s for t, s in zip(shape, vol.shape)]
    return zoom(vol, factors, order=order)


def case_embedding(
    ct: np.ndarray,
    pet: np.ndarray,
    grid: int = 8,
    hist_bins: int = 16,
) -> np.ndarray:
    """Global retrieval embedding: PET/CT histograms + coarse grid means."""
    ct, pet, _ = crop_to_mask(ct, pet, None)
    shape = (grid * 4, grid * 4, grid * 4)
    pet_r = resize_vol(pet, shape, order=1)
    ct_r = resize_vol(ct, shape, order=1)
    bm = pet_r > (0.05 * float(pet_r.max() + 1e-6))
    pet_v = pet_r[bm]
    ct_v = ct_r[bm]
    if pet_v.size == 0:
        pet_v = pet_r.ravel()
        ct_v = ct_r.ravel()
    pet_hist, _ = np.histogram(pet_v, bins=hist_bins, range=(0, float(pet_v.max() + 1e-3)))
    ct_hist, _ = np.histogram(ct_v, bins=hist_bins, range=(float(ct_v.min()), float(ct_v.max() + 1e-3)))
    pet_hist = pet_hist.astype(np.float64) / (pet_hist.sum() + 1e-8)
    ct_hist = ct_hist.astype(np.float64) / (ct_hist.sum() + 1e-8)
    # coarse grid cell means
    cell = grid
    step = shape[0] // cell
    grid_feats = []
    for i in range(cell):
        for j in range(cell):
            for k in range(cell):
                p = pet_r[i * step : (i + 1) * step, j * step : (j + 1) * step, k * step : (k + 1) * step]
                c = ct_r[i * step : (i + 1) * step, j * step : (j + 1) * step, k * step : (k + 1) * step]
                grid_feats.extend([float(p.mean()), float(c.mean()), float(p.std()), float(c.std())])
    stats = [
        float(pet_v.mean()),
        float(pet_v.std()),
        float(np.percentile(pet_v, 95)),
        float(ct_v.mean()),
        float(ct_v.std()),
    ]
    emb = np.concatenate([pet_hist, ct_hist, np.array(grid_feats), np.array(stats)], axis=0)
    n = np.linalg.norm(emb) + 1e-8
    return (emb / n).astype(np.float32)


def retrieve(gallery: Gallery, query_emb: np.ndarray, topk: int = 1) -> list[tuple[str, float]]:
    q = query_emb.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    sims = gallery.embeddings @ q
    order = np.argsort(-sims)[:topk]
    return [(gallery.case_ids[i], float(sims[i])) for i in order]


def patch_features(pet: np.ndarray, ct: np.ndarray, grid: int = 16) -> np.ndarray:
    """Return (grid,grid,grid,F) feature map."""
    shape = (grid * 4, grid * 4, grid * 4)
    pet_r = resize_vol(pet, shape, order=1)
    ct_r = resize_vol(ct, shape, order=1)
    step = shape[0] // grid
    feats = np.zeros((grid, grid, grid, 4), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                p = pet_r[i * step : (i + 1) * step, j * step : (j + 1) * step, k * step : (k + 1) * step]
                c = ct_r[i * step : (i + 1) * step, j * step : (j + 1) * step, k * step : (k + 1) * step]
                feats[i, j, k] = [p.mean(), c.mean(), p.std(), c.std()]
    return feats


def prototype_predict(
    q_pet: np.ndarray,
    q_ct: np.ndarray,
    s_pet: np.ndarray,
    s_ct: np.ndarray,
    s_lab: np.ndarray,
    grid: int = 16,
) -> np.ndarray:
    """ALPNet-lite: fg/bg prototypes from support supervoxels, classify query supervoxels."""
    q_pet, q_ct, _ = crop_to_mask(q_pet, q_ct, None)
    s_pet, s_ct, s_lab = crop_to_mask(s_pet, s_ct, None, s_lab)
    if s_lab is None:
        s_lab = np.zeros_like(s_pet, dtype=bool)
    s_lab_r = resize_vol(s_lab.astype(np.float32), (grid, grid, grid), order=0) > 0.5
    qf = patch_features(q_pet, q_ct, grid=grid)
    sf = patch_features(s_pet, s_ct, grid=grid)
    fg = sf[s_lab_r]
    bg = sf[~s_lab_r]
    if fg.size == 0:
        fg_proto = sf.reshape(-1, 4).mean(0)
    else:
        fg_proto = fg.reshape(-1, 4).mean(0)
    if bg.size == 0:
        bg_proto = np.zeros(4, dtype=np.float32)
    else:
        bg_proto = bg.reshape(-1, 4).mean(0)
    q_flat = qf.reshape(-1, 4)
    d_fg = np.linalg.norm(q_flat - fg_proto[None], axis=1)
    d_bg = np.linalg.norm(q_flat - bg_proto[None], axis=1)
    pred_grid = (d_fg < d_bg).reshape(grid, grid, grid).astype(np.float32)
    # upsample to original query shape
    pred_lo = resize_vol(pred_grid, q_pet.shape, order=0)
    return pred_lo > 0.5


def ensemble_prototype_predict(
    q_pet: np.ndarray,
    q_ct: np.ndarray,
    supports: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    grid: int = 16,
    weights: list[float] | None = None,
) -> np.ndarray:
    if not supports:
        raise ValueError("supports must be non-empty")
    if weights is None:
        weights = [1.0] * len(supports)
    if len(weights) != len(supports):
        raise ValueError("weights length must match supports")
    vote_sum: np.ndarray | None = None
    total_w = float(sum(weights))
    for (sp, sc, sl), w in zip(supports, weights):
        p = prototype_predict(q_pet, q_ct, sp, sc, sl, grid=grid).astype(np.float32)
        vote_sum = p * w if vote_sum is None else vote_sum + p * w
    assert vote_sum is not None
    return vote_sum >= (total_w / 2.0)


def save_pred_nifti(pred: np.ndarray, ref_path: Path, out_path: Path) -> None:
    ref = nib.load(str(ref_path))
    out = nib.Nifti1Image(pred.astype(np.uint8), ref.affine, ref.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))
