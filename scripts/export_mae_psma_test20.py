#!/usr/bin/env python3
"""Export PSMA stratified test (20%) case list for MAE/SSL eval."""
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
        default=Path(__file__).resolve().parents[1] / "data" / "splits_mae_psma_test20.json",
    )
    args = ap.parse_args()
    split = json.loads(args.stratified_json.read_text(encoding="utf-8"))
    cases = sorted(x for x in split["test"] if str(x).startswith("psma_"))
    payload = {"cases": cases, "test": cases, "n": len(cases), "tracer": "PSMA", "split": "test20"}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[psma-test20] wrote {args.out_json} n={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
