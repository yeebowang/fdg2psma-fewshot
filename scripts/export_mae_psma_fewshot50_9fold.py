#!/usr/bin/env python3
"""Export PSMA fewshot=50 × 9 folds covering all 70% train (421).

Design:
  - Each fold: 50 labeled train + shared val=59 (PSMA 10%)
  - Stratified quota ≈ 65/25/10 → small=32, large=12, negative=6
  - Phase1: round-robin assign all 421 into 9 folds (union covers 100%)
  - Phase2: top-up each fold to 50 while respecting quota (minimal overlap)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROLES = ("small", "large", "negative")


def _quota(n_shot: int) -> dict[str, int]:
    # PSMA mixture ≈ 65% / 25% / 10%
    q = {
        "small": int(round(n_shot * 0.65)),
        "large": int(round(n_shot * 0.25)),
        "negative": 0,
    }
    q["negative"] = n_shot - q["small"] - q["large"]
    while sum(q.values()) > n_shot:
        for k in ROLES:
            if q[k] > 0 and sum(q.values()) > n_shot:
                q[k] -= 1
    while sum(q.values()) < n_shot:
        q["small"] += 1
    return q


def _role_of(meta: dict, case_id: str) -> str:
    return str(meta.get(case_id, {}).get("role", "small"))


def build_covering_folds(
    train_pool: list[str],
    meta: dict,
    *,
    n_folds: int,
    n_shot: int,
    seed: int,
) -> tuple[list[list[str]], dict[str, int]]:
    rng = random.Random(seed)
    by_role: dict[str, list[str]] = defaultdict(list)
    for c in train_pool:
        by_role[_role_of(meta, c)].append(c)
    for role in ROLES:
        rng.shuffle(by_role[role])

    quota = _quota(n_shot)
    folds: list[list[str]] = [[] for _ in range(n_folds)]

    # Phase 1: cover all cases via round-robin within each role
    for role in ROLES:
        for i, c in enumerate(by_role[role]):
            folds[i % n_folds].append(c)

    # sanity: full cover
    covered = set().union(*folds)
    if covered != set(train_pool):
        missing = sorted(set(train_pool) - covered)
        raise RuntimeError(f"cover incomplete missing={len(missing)} e.g. {missing[:3]}")

    # Phase 2: top-up to n_shot with quota preference
    for fi in range(n_folds):
        have = set(folds[fi])
        counts = Counter(_role_of(meta, c) for c in folds[fi])
        for role in ROLES:
            need = quota[role] - counts[role]
            if need <= 0:
                continue
            cand = [c for c in by_role[role] if c not in have]
            # deterministic order from shuffled pool
            for c in cand[:need]:
                folds[fi].append(c)
                have.add(c)
                counts[role] += 1
        # residual top-up (any role) if still short
        if len(folds[fi]) < n_shot:
            rest = [c for c in train_pool if c not in have]
            rng.shuffle(rest)
            for c in rest:
                if len(folds[fi]) >= n_shot:
                    break
                folds[fi].append(c)
                have.add(c)
        folds[fi] = sorted(folds[fi][:n_shot])
        if len(folds[fi]) != n_shot:
            raise RuntimeError(f"fold{fi} size={len(folds[fi])} != {n_shot}")

    return folds, quota


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stratified-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "splits_stratified_70_10_20.json",
    )
    ap.add_argument(
        "--psma-splits-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "splits_baseline2_psma_uda_nnunet.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "splits_mae_psma_fewshot50_9fold",
    )
    ap.add_argument("--n-shot", type=int, default=50)
    ap.add_argument("--n-folds", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    strat = json.loads(args.stratified_json.read_text(encoding="utf-8"))
    meta = strat["case_meta"]
    psma = json.loads(args.psma_splits_json.read_text(encoding="utf-8"))[0]
    train_pool = sorted(psma["train"])
    val = sorted(psma["val"])

    folds, quota = build_covering_folds(
        train_pool,
        meta,
        n_folds=args.n_folds,
        n_shot=args.n_shot,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_files = []
    union: set[str] = set()
    fold_summaries = []
    for i, tr in enumerate(folds):
        union.update(tr)
        roles = Counter(_role_of(meta, c) for c in tr)
        payload = [{"train": tr, "val": val}]
        out_i = args.out_dir / f"fold{i}_nnunet.json"
        out_i.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        fold_files.append(str(out_i))
        fold_summaries.append(
            {
                "fold": i,
                "n_train": len(tr),
                "roles": {r: roles[r] for r in ROLES},
                "path": str(out_i),
            }
        )
        print(
            f"[fewshot50-9fold] fold{i} n={len(tr)} "
            f"roles={{small:{roles['small']}, large:{roles['large']}, negative:{roles['negative']}}}"
        )

    overlap_slots = sum(len(f) for f in folds) - len(train_pool)
    meta_out = {
        "name": "iclr2026_mae_psma_fewshot50_9fold_cover",
        "n_shot": args.n_shot,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "quota_target": quota,
        "n_train_pool": len(train_pool),
        "n_val": len(val),
        "union_size": len(union),
        "covers_all_train70": union == set(train_pool),
        "overlap_slots": overlap_slots,
        "folds": fold_summaries,
        "note": (
            "9 folds × 50-shot; phase1 round-robin covers all PSMA 70% train; "
            "phase2 top-up to 50 with stratified quota; shared val=PSMA 10% labeled"
        ),
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = {
        "n_folds": args.n_folds,
        "n_shot": args.n_shot,
        "folds": fold_files,
        "meta": str(meta_path),
        "covers_all_train70": meta_out["covers_all_train70"],
    }
    index_path = args.out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"[fewshot50-9fold] union={len(union)}/{len(train_pool)} "
        f"cover={meta_out['covers_all_train70']} overlap_slots={overlap_slots}"
    )
    print(f"[fewshot50-9fold] index={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
