#!/usr/bin/env python3
"""从分层划分导出 baseline1 后 1/3 用的 PSMA-only val case 列表。"""
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
        default=Path(__file__).resolve().parents[1] / "data" / "splits_baseline1_psma_val.json",
    )
    args = ap.parse_args()
    split = json.loads(args.stratified_json.read_text(encoding="utf-8"))
    psma_va = sorted(x for x in split["val"] if str(x).startswith("psma_"))
    payload = {"val": psma_va}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    meta = {
        "name": "iclr2026_baseline1_psma_val_monitor",
        "source": str(args.stratified_json),
        "filter": "tracer=PSMA (case id startswith psma_) from stratified val",
        "n_val": len(psma_va),
        "note": "monitor-only from epoch>=TASK1_PSMA_VAL_FROM_EPOCH; does not affect FDG best",
    }
    meta_path = args.out_json.with_name("splits_baseline1_psma_val_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[baseline1-psma-val] wrote {args.out_json} val={len(psma_va)}")
    print(f"[baseline1-psma-val] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
