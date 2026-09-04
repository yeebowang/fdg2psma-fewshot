#!/usr/bin/env python3
"""Export PSMA few-shot train (10 cases) + labeled val (59) for MAE finetune."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


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
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "splits_mae_psma_fewshot10_nnunet.json",
    )
    ap.add_argument("--n-shot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    strat = json.loads(args.stratified_json.read_text(encoding="utf-8"))
    meta = strat["case_meta"]
    psma = json.loads(args.psma_splits_json.read_text(encoding="utf-8"))[0]
    train_pool = list(psma["train"])
    val = list(psma["val"])

    by_role: dict[str, list[str]] = defaultdict(list)
    for c in train_pool:
        role = str(meta.get(c, {}).get("role", "small"))
        by_role[role].append(c)
    rng = random.Random(args.seed)
    for role in by_role:
        rng.shuffle(by_role[role])

    # PSMA mixture ≈ 65% small / 25% large / 10% negative → 6/3/1 for n=10
    quota = {"small": 6, "large": 3, "negative": 1}
    if args.n_shot != 10:
        # scale roughly
        quota = {
            "small": max(1, int(round(args.n_shot * 0.65))),
            "large": max(1, int(round(args.n_shot * 0.25))),
            "negative": max(0, args.n_shot - int(round(args.n_shot * 0.65)) - int(round(args.n_shot * 0.25))),
        }
        # fix sum
        while sum(quota.values()) > args.n_shot:
            for k in ("small", "large", "negative"):
                if quota[k] > 0 and sum(quota.values()) > args.n_shot:
                    quota[k] -= 1
        while sum(quota.values()) < args.n_shot:
            quota["small"] += 1

    picked: list[str] = []
    for role, n in quota.items():
        pool = by_role.get(role, [])
        take = min(n, len(pool))
        picked.extend(pool[:take])
    # top-up if short
    remain = [c for c in train_pool if c not in set(picked)]
    rng.shuffle(remain)
    while len(picked) < args.n_shot and remain:
        picked.append(remain.pop())
    picked = picked[: args.n_shot]

    payload = [{"train": sorted(picked), "val": sorted(val)}]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta_out = {
        "name": "iclr2026_mae_psma_fewshot10",
        "n_shot": args.n_shot,
        "seed": args.seed,
        "quota": quota,
        "n_train": len(picked),
        "n_val": len(val),
        "train_roles": {
            r: sum(1 for c in picked if meta.get(c, {}).get("role") == r)
            for r in ("small", "large", "negative")
        },
        "note": "few-shot labeled train from PSMA 70% pool; val=PSMA 10% labeled",
    }
    meta_path = args.out_json.with_name("splits_mae_psma_fewshot10_meta.json")
    meta_path.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")
    print(f"[fewshot10] wrote {args.out_json} train={len(picked)} val={len(val)}")
    print(f"[fewshot10] roles={meta_out['train_roles']} meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
