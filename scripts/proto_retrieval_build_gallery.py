#!/usr/bin/env python3
"""Build FDG 100% retrieval gallery for prototype+retrieval f258."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto_retrieval_core import (
    Gallery,
    case_embedding,
    load_ct_pet_label,
    load_fdg100_case_ids,
    load_fdg70_case_ids,
    load_fdg80_case_ids,
    load_psma100_case_ids,
    load_psma70_case_ids,
)


def _embed_one(args: tuple[str, str, str]) -> tuple[str, list[float] | None]:
    case_id, img_dir, lab_dir = args
    img_dir_p, lab_dir_p = Path(img_dir), Path(lab_dir)
    try:
        ct, pet, _ = load_ct_pet_label(case_id, img_dir_p, lab_dir_p)
        emb = case_embedding(ct, pet)
        return case_id, emb.tolist()
    except Exception as e:
        print(f"[gallery] skip {case_id}: {e}")
        return case_id, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stratified-json",
        type=Path,
        default=Path("ICLR2026/data/splits_stratified_70_10_20.json"),
    )
    ap.add_argument(
        "--img-dir",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-DATA/dataset1/imagesTr"),
    )
    ap.add_argument(
        "--lab-dir",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-DATA/dataset1/labelsTr"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/media/ybwang/data1/PSMA-DATA/task1_train_workspace/proto_retrieval/fdg100_gallery.npz"
        ),
    )
    ap.add_argument(
        "--pool",
        choices=("fdg100", "fdg80", "fdg70", "psma100", "psma70"),
        default="fdg100",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.pool == "psma100":
        cases = load_psma100_case_ids(args.stratified_json)
        pool_label = "PSMA100%"
    elif args.pool == "psma70":
        cases = load_psma70_case_ids(args.stratified_json)
        pool_label = "PSMA70%"
    elif args.pool == "fdg80":
        cases = load_fdg80_case_ids(args.stratified_json)
        pool_label = "FDG80%"
    elif args.pool == "fdg70":
        cases = load_fdg70_case_ids(args.stratified_json)
        pool_label = "FDG70%"
    else:
        cases = load_fdg100_case_ids(args.stratified_json)
        pool_label = "FDG100%"
    print(f"[gallery] {pool_label} n={len(cases)}")

    jobs = [(c, str(args.img_dir), str(args.lab_dir)) for c in cases]
    embs: dict[str, list[float]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_embed_one, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            cid, emb = fut.result()
            done += 1
            if emb is not None:
                embs[cid] = emb
            if done % 100 == 0 or done == len(jobs):
                print(f"[gallery] embedded {done}/{len(jobs)} ok={len(embs)}")

    ids = sorted(embs.keys())
    mat = np.array([embs[i] for i in ids], dtype=np.float32)
    gal = Gallery(ids, mat)
    gal.save(args.out)
    meta = {
        "n_cases": len(ids),
        "pool": pool_label,
        "stratified_json": str(args.stratified_json),
        "embedding": "PET/CT hist + coarse grid (retrieval)",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[gallery] saved {args.out} n={len(ids)}")


if __name__ == "__main__":
    main()
