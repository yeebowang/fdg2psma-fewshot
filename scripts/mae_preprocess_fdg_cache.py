#!/usr/bin/env python3
"""Offline preprocess for MAE FDG finetune (spacing / body-crop / z-score).

Saves per-case npz under --out-dir:
  image: float32 (2, Z, Y, X) with channel order [PET, CT] (MAE convention)
  label: uint8  (1, Z, Y, X)
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
from tqdm import tqdm


TARGET_SPACING_ZYX = np.array([3.0, 2.0, 2.0], dtype=np.float64)


def _z_score(vol: np.ndarray) -> np.ndarray:
    mu, sigma = float(np.mean(vol)), max(float(np.std(vol)), 1e-8)
    return ((vol - mu) / sigma).astype(np.float32)


def preprocess_one(case_id: str, images_tr: Path, labels_tr: Path, out_dir: Path) -> str:
    out_path = out_dir / f"{case_id}.npz"
    if out_path.is_file():
        return f"skip:{case_id}"

    ct_path = images_tr / f"{case_id}_0000.nii.gz"
    pet_path = images_tr / f"{case_id}_0001.nii.gz"
    seg_path = labels_tr / f"{case_id}.nii.gz"
    for p in (ct_path, pet_path, seg_path):
        if not p.is_file():
            raise FileNotFoundError(str(p))

    pet_img = sitk.ReadImage(str(pet_path))
    ct_img = sitk.ReadImage(str(ct_path))
    seg_img = sitk.ReadImage(str(seg_path))

    spacing_xyz = np.round(pet_img.GetSpacing()).astype(float)
    orig_spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float64)

    pet_array = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    ct_array = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    seg_array = (sitk.GetArrayFromImage(seg_img) > 0).astype(np.uint8)

    zoom_pet = orig_spacing_zyx / TARGET_SPACING_ZYX
    pet_r = ndi.zoom(pet_array, zoom_pet, order=1)
    zoom_ct = np.array(pet_r.shape, dtype=np.float64) / np.array(ct_array.shape, dtype=np.float64)
    ct_r = ndi.zoom(ct_array, zoom_ct, order=1)
    seg_r = ndi.zoom(seg_array, zoom_ct, order=0)

    body = ct_r > -500
    labels, nfeat = ndi.label(body)
    if nfeat > 0:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        clean = labels == counts.argmax()
        z_idx, y_idx, x_idx = np.where(clean)
        pad = 3
        z0, z1 = max(0, z_idx.min() - pad), min(pet_r.shape[0], z_idx.max() + pad + 1)
        y0, y1 = max(0, y_idx.min() - pad), min(pet_r.shape[1], y_idx.max() + pad + 1)
        x0, x1 = max(0, x_idx.min() - pad), min(pet_r.shape[2], x_idx.max() + pad + 1)
        sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        pet_c, ct_c, seg_c = pet_r[sl], ct_r[sl], seg_r[sl]
    else:
        pet_c, ct_c, seg_c = pet_r, ct_r, seg_r

    # MAE channel order: [PET, CT]
    image = np.stack([_z_score(pet_c), _z_score(ct_c)], axis=0).astype(np.float32)
    label = np.expand_dims(seg_c.astype(np.uint8), axis=0)

    tmp = out_dir / f".{case_id}.part.npz"
    np.savez_compressed(tmp, image=image, label=label, case_id=case_id)
    os.replace(tmp, out_path)
    return f"ok:{case_id}"


def _load_cases(splits_json: Path | None, cases_json: Path | None) -> list[str]:
    cases: list[str] = []
    if cases_json is not None:
        payload = json.loads(Path(cases_json).read_text())
        if isinstance(payload, dict):
            if "val" in payload:
                cases.extend(list(payload["val"]))
            if "train" in payload:
                cases.extend(list(payload["train"]))
            if "test" in payload:
                cases.extend(list(payload["test"]))
            if "cases" in payload:
                cases.extend(list(payload["cases"]))
        elif isinstance(payload, list):
            if payload and isinstance(payload[0], dict) and ("train" in payload[0] or "val" in payload[0]):
                cases.extend(list(payload[0].get("train", [])))
                cases.extend(list(payload[0].get("val", [])))
            else:
                cases.extend([str(x) for x in payload])
    if splits_json is not None:
        splits = json.loads(Path(splits_json).read_text())
        fold0 = splits[0] if isinstance(splits, list) else splits
        cases.extend(list(fold0.get("train", [])))
        cases.extend(list(fold0.get("val", [])))
    return list(dict.fromkeys(cases))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits-json",
        default="",
        help="nnUNet-style splits with train/val (optional if --cases-json set)",
    )
    ap.add_argument(
        "--cases-json",
        default="",
        help='JSON with {"val":[...]} / {"cases":[...]} / plain list (e.g. PSMA val)',
    )
    ap.add_argument("--images-tr", default="/media/ybwang/data1/PSMA-DATA/dataset1/imagesTr")
    ap.add_argument("--labels-tr", default="/media/ybwang/data1/PSMA-DATA/dataset1/labelsTr")
    ap.add_argument(
        "--out-dir",
        default="/media/ybwang/data1/PSMA-DATA/task1_train_workspace/mae_cache/fdg_baseline1_70_10",
    )
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    splits_json = Path(args.splits_json) if args.splits_json.strip() else None
    cases_json = Path(args.cases_json) if args.cases_json.strip() else None
    if splits_json is None and cases_json is None:
        splits_json = Path(
            "/media/ybwang/data1/PSMA-CTRL/ICLR2026/data/splits_baseline1_fdg_nnunet.json"
        )
    cases = _load_cases(splits_json, cases_json)
    if not cases:
        raise SystemExit("[error] empty case list")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_tr, labels_tr = Path(args.images_tr), Path(args.labels_tr)

    print(f"[mae-prep] cases={len(cases)} out={out_dir} workers={args.workers}")
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(preprocess_one, c, images_tr, labels_tr, out_dir): c for c in cases
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="mae-prep"):
            try:
                msg = fut.result()
                if msg.startswith("skip"):
                    skip += 1
                else:
                    ok += 1
            except Exception as e:
                err += 1
                print(f"[error] {futs[fut]}: {e}")
    print(f"[mae-prep] done ok={ok} skip={skip} err={err}")
    if err:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
