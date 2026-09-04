#!/usr/bin/env python3
"""从分层划分导出 baseline1：仅 FDG train/val 的 nnU-Net splits_final 列表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stratified-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "splits_stratified_70_10_20.json",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "splits_baseline1_fdg_nnunet.json",
    )
    args = ap.parse_args()
    split = json.loads(args.stratified_json.read_text(encoding="utf-8"))
    fdg_tr = sorted(x for x in split["train"] if str(x).startswith("fdg_"))
    fdg_va = sorted(x for x in split["val"] if str(x).startswith("fdg_"))
    payload = [{"train": fdg_tr, "val": fdg_va}]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "name": "iclr2026_baseline1_fdg_train_val",
        "source": str(args.stratified_json),
        "filter": "tracer=FDG (case id startswith fdg_)",
        "n_train": len(fdg_tr),
        "n_val": len(fdg_va),
        "note": "nnU-Net splits_final fold0; stratified test held out",
    }
    meta_path = args.out_json.with_name("splits_baseline1_fdg_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[baseline1-splits] wrote {args.out_json} train={len(fdg_tr)} val={len(fdg_va)}")
    print(f"[baseline1-splits] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
