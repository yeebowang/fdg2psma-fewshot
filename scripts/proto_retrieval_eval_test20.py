#!/usr/bin/env python3
"""Prototype+Retrieval PSMA TEST20 eval (support pool = FDG 100%, optional +fs50)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proto_retrieval_core import (
    Gallery,
    case_embedding,
    ensemble_prototype_predict,
    load_ct_pet_label,
    retrieve,
    save_pred_nifti,
)
from proto_retrieval_encoder import encode_gallery, encode_numpy, load_encoder
from seg_voxel_metrics import aggregate_case_metrics, confusion_counts


def _build_psma_gallery(
    fs50_ids: list[str],
    img_dir: Path,
    encoder,
    device,
) -> Gallery:
    import torch

    raw = []
    for cid in fs50_ids:
        ct, pet, _ = load_ct_pet_label(cid, img_dir, None)
        raw.append(case_embedding(ct, pet))
    mat = np.stack(raw, axis=0)
    if encoder is not None:
        mat = encode_gallery(encoder, mat, device)
    return Gallery(list(fs50_ids), mat)


def _encode_query(q_ct, q_pet, encoder, device) -> np.ndarray:
    q_raw = case_embedding(q_ct, q_pet)
    if encoder is not None:
        return encode_numpy(encoder, q_raw, device)[0]
    return q_raw


def _hits_to_supports(hits, img_dir, lab_dir):
    supports, hit_meta = [], []
    for sid, sim in hits:
        s_ct, s_pet, s_lab = load_ct_pet_label(sid, img_dir, lab_dir)
        if s_lab is None:
            s_lab = np.zeros_like(s_pet, dtype=bool)
        supports.append((s_pet, s_ct, s_lab))
        hit_meta.append(
            {
                "support_id": sid,
                "similarity": sim,
                "from_fdg": sid.startswith("fdg_"),
                "from_psma": sid.startswith("psma_"),
            }
        )
    return supports, hit_meta


def _build_encoded_gallery(
    gallery: Gallery,
    encoder,
    device,
    extra_ids: list[str] | None,
    img_dir: Path,
) -> Gallery:
    enc_fdg = encode_gallery(encoder, gallery.embeddings, device)
    ids = list(gallery.case_ids)
    embs = enc_fdg
    if extra_ids:
        extra_raw = []
        for cid in extra_ids:
            if cid in ids:
                continue
            ct, pet, _ = load_ct_pet_label(cid, img_dir, None)
            extra_raw.append(case_embedding(ct, pet))
            ids.append(cid)
        if extra_raw:
            extra_enc = encode_gallery(encoder, np.stack(extra_raw, axis=0), device)
            embs = np.concatenate([embs, extra_enc], axis=0)
    return Gallery(ids, embs)


def _load_cases_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [str(x) for x in raw.get("cases", raw.get("test", []))]
    return [str(x) for x in raw]


def _load_fs50_train(fold: int, split_dir: Path) -> list[str]:
    p = split_dir / f"fold{fold}_nnunet.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = raw[0]
    return [str(x) for x in raw.get("train", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--cases-json", type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--lab-dir", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0, help="0 = no fs50 union; else fold id")
    ap.add_argument("--split-dir", type=Path, default=Path("ICLR2026/data/splits_mae_psma_fewshot50_9fold"))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--psma-topk", type=int, default=2, help="dual mode: top-K from PSMA fs50")
    ap.add_argument("--fdg-topk", type=int, default=2, help="dual mode: top-K from FDG100%")
    ap.add_argument("--psma-vote-weight", type=float, default=2.0)
    ap.add_argument("--fdg-vote-weight", type=float, default=1.0)
    ap.add_argument(
        "--pool-mode",
        choices=(
            "fdg100",
            "fdg80",
            "fdg70",
            "fdg100_psma50",
            "psma50",
            "psma70",
            "psma100",
            "dual_psma_fdg",
        ),
        default="fdg100",
        help="retrieval pool; dual_psma_fdg = PSMA top-k + FDG top-k weighted vote",
    )
    ap.add_argument("--encoder-ckpt", type=Path, default=None, help="fold-specific retrieval encoder")
    ap.add_argument("--tag", default="proto_retrieval_test20")
    ap.add_argument("--stamp", default="")
    args = ap.parse_args()

    gallery = Gallery.load(args.gallery)
    cases = _load_cases_json(args.cases_json)
    args.pred_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = None
    if args.encoder_ckpt is not None:
        encoder, enc_meta = load_encoder(args.encoder_ckpt, map_location=device)
        encoder = encoder.to(device)
        print(f"[proto-retrieval] encoder fold={enc_meta.get('fold')} from {args.encoder_ckpt}")

    fs50_ids: list[str] = []
    if args.fold > 0 and args.pool_mode in ("fdg100_psma50", "psma50", "dual_psma_fdg"):
        fs50_ids = _load_fs50_train(args.fold, args.split_dir)

    fdg_gallery = gallery
    psma_gallery: Gallery | None = None

    if args.pool_mode in ("psma100", "psma70", "fdg70", "fdg80", "fdg100"):
        # Gallery already restricted to the requested pool at build time.
        sub_gallery = gallery
    elif args.pool_mode == "dual_psma_fdg":
        if not fs50_ids:
            raise SystemExit("pool-mode dual_psma_fdg requires --fold > 0")
        if encoder is not None:
            fdg_gallery = Gallery(gallery.case_ids, encode_gallery(encoder, gallery.embeddings, device))
        psma_gallery = _build_psma_gallery(fs50_ids, args.img_dir, encoder, device)
        sub_gallery = fdg_gallery  # unused in dual loop
    elif args.pool_mode == "psma50":
        if not fs50_ids:
            raise SystemExit("pool-mode psma50 requires --fold > 0")
        raw = []
        ids = []
        for cid in fs50_ids:
            ct, pet, _ = load_ct_pet_label(cid, args.img_dir, None)
            raw.append(case_embedding(ct, pet))
            ids.append(cid)
        raw_mat = np.stack(raw, axis=0)
        if encoder is not None:
            sub_gallery = Gallery(ids, encode_gallery(encoder, raw_mat, device))
        else:
            sub_gallery = Gallery(ids, raw_mat)
    elif encoder is not None:
        extra = fs50_ids if args.pool_mode == "fdg100_psma50" else None
        sub_gallery = _build_encoded_gallery(gallery, encoder, device, extra, args.img_dir)
    elif args.pool_mode == "fdg100_psma50" and fs50_ids:
        pool_ids = sorted(set(gallery.case_ids) | set(fs50_ids))
        mask = np.array([cid in pool_ids for cid in gallery.case_ids])
        sub_ids = [gallery.case_ids[i] for i in np.where(mask)[0]]
        sub_embs = gallery.embeddings[mask]
        extra = [c for c in fs50_ids if c not in sub_ids]
        if extra:
            extra_raw = []
            for cid in extra:
                ct, pet, _ = load_ct_pet_label(cid, args.img_dir, None)
                extra_raw.append(case_embedding(ct, pet))
            sub_ids = sub_ids + extra
            sub_embs = np.concatenate([sub_embs, np.stack(extra_raw, axis=0)], axis=0)
        sub_gallery = Gallery(sub_ids, sub_embs)
    else:
        sub_gallery = gallery

    pool_labels = {
        "fdg100": "FDG100%",
        "fdg80": "FDG80%",
        "fdg70": "FDG70%",
        "fdg100_psma50": "FDG100%+PSMAfs50",
        "psma50": "PSMAfs50",
        "psma70": "PSMA70%",
        "psma100": "PSMA100%",
        "dual_psma_fdg": f"PSMAfs50 top{args.psma_topk} + FDG100% top{args.fdg_topk} weighted",
    }

    per_case = {}
    retrieval_log = {}
    for qi, qid in enumerate(cases):
        q_ct, q_pet, _ = load_ct_pet_label(qid, args.img_dir, None)
        q_emb = _encode_query(q_ct, q_pet, encoder, device)

        if args.pool_mode == "dual_psma_fdg":
            assert psma_gallery is not None
            psma_hits = retrieve(psma_gallery, q_emb, topk=args.psma_topk)
            fdg_hits = retrieve(fdg_gallery, q_emb, topk=args.fdg_topk)
            psma_sup, psma_meta = _hits_to_supports(psma_hits, args.img_dir, args.lab_dir)
            fdg_sup, fdg_meta = _hits_to_supports(fdg_hits, args.img_dir, args.lab_dir)
            supports = psma_sup + fdg_sup
            weights = [args.psma_vote_weight] * len(psma_sup) + [args.fdg_vote_weight] * len(fdg_sup)
            hit_meta = {
                "psma": psma_meta,
                "fdg": fdg_meta,
                "vote_weights": {"psma": args.psma_vote_weight, "fdg": args.fdg_vote_weight},
            }
            pred = ensemble_prototype_predict(q_pet, q_ct, supports, weights=weights)
        else:
            hits = retrieve(sub_gallery, q_emb, topk=args.topk)
            supports, hit_meta = _hits_to_supports(hits, args.img_dir, args.lab_dir)
            pred = ensemble_prototype_predict(q_pet, q_ct, supports)
        ref_img = args.img_dir / f"{qid}_0000.nii.gz"
        pred_path = args.pred_dir / f"{qid}.nii.gz"
        save_pred_nifti(pred, ref_img, pred_path)
        gt_path = args.lab_dir / f"{qid}.nii.gz"
        gt = np.asarray(nib.load(str(gt_path)).dataobj) > 0
        pred_loaded = pred
        if gt.shape != pred_loaded.shape:
            factors = [g / p for g, p in zip(gt.shape, pred_loaded.shape)]
            pred_loaded = zoom(pred_loaded.astype(np.float32), factors, order=0) > 0.5
        metrics = confusion_counts(gt, pred_loaded)
        dice = float(metrics["dice"])
        per_case[qid] = metrics
        retrieval_log[qid] = hit_meta
        if (qi + 1) % 20 == 0 or qi + 1 == len(cases):
            if args.pool_mode == "dual_psma_fdg":
                p1 = hit_meta["psma"][0]["support_id"][:36] if hit_meta["psma"] else "?"
                f1 = hit_meta["fdg"][0]["support_id"][:36] if hit_meta["fdg"] else "?"
                print(f"[proto-retrieval] {qi+1}/{len(cases)} q={qid} psma1={p1}... fdg1={f1}... dice={dice:.3f}")
            else:
                print(f"[proto-retrieval] {qi+1}/{len(cases)} q={qid} top1={hit_meta[0]['support_id'][:40]}... dice={dice:.3f}")

    agg = aggregate_case_metrics(per_case)
    summary = {
        "tag": args.tag,
        "stamp": args.stamp,
        "method": "proto_retrieval",
        "protocol": "retrieve support -> prototype match",
        "fold": args.fold,
        "topk": args.topk if args.pool_mode != "dual_psma_fdg" else None,
        "psma_topk": args.psma_topk if args.pool_mode == "dual_psma_fdg" else None,
        "fdg_topk": args.fdg_topk if args.pool_mode == "dual_psma_fdg" else None,
        "psma_vote_weight": args.psma_vote_weight if args.pool_mode == "dual_psma_fdg" else None,
        "fdg_vote_weight": args.fdg_vote_weight if args.pool_mode == "dual_psma_fdg" else None,
        "pool_mode": args.pool_mode,
        "encoder_ckpt": str(args.encoder_ckpt) if args.encoder_ckpt else "",
        "support_pool": pool_labels[args.pool_mode],
        "n_cases": len(cases),
        **agg,
        "per_case": per_case,
        "retrieval": retrieval_log,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"[proto-retrieval] DONE Dice={summary['mean_dice']:.4f} "
        f"FP={summary['fp_rate']:.4f} FN={summary['fn_rate']:.4f} "
        f"pos_n={summary['n_positive']} n={summary['n_scored']}"
    )


if __name__ == "__main__":
    main()
