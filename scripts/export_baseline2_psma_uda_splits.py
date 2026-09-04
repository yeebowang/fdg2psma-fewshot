#!/usr/bin/env python3
"""从分层划分导出 baseline2 PSMA UDA：train=421（伪标）/ val=59（真 GT 监控）。"""
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
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "splits_baseline2_psma_uda_nnunet.json",
    )
    args = ap.parse_args()
    split = json.loads(args.stratified_json.read_text(encoding="utf-8"))
    psma_tr = sorted(x for x in split["train"] if str(x).startswith("psma_"))
    psma_va = sorted(x for x in split["val"] if str(x).startswith("psma_"))
    payload = [{"train": psma_tr, "val": psma_va}]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    meta = {
        "name": "iclr2026_baseline2_psma_uda_train_val",
        "source": str(args.stratified_json),
        "filter": "tracer=PSMA (case id startswith psma_)",
        "n_train": len(psma_tr),
        "n_val": len(psma_va),
        "note": (
            "nnU-Net splits_final fold0; train labels replaced by UDA pseudo masks; "
            "val keeps GT for monitoring/best"
        ),
    }
    meta_path = args.out_json.with_name("splits_baseline2_psma_uda_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[baseline2-splits] wrote {args.out_json} train={len(psma_tr)} val={len(psma_va)}"
    )
    print(f"[baseline2-splits] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
