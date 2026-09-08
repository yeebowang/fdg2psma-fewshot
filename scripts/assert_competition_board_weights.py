#!/usr/bin/env python3
"""Refuse GC final-submission weights for ICLR2026 aligned board rows.

Allowed pretrain init for hemingduo / chenyixin (pretrained rows only):
  Zenodo 13753413 Dataset619_nativemultistem / MultiTalentV2_challengeversion.zip

Forbidden for board train/eval/scoring:
  Zenodo 14007247 LesionTracer final
  BIRTH EDT interactive finals
  YixinChen LocalEdit / TACE submission ckpts
  Any GC Docker bake-in champion ensemble used as board metric
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FORBIDDEN_PATTERNS = (
    r"14007247",
    r"autoPET-3-LesionTracer",
    r"LesionTracer(?!.*Dataset619)",
    r"/EDT/",
    r"EDT.*checkpoint",
    r"LocalEdit",
    r"TACE",
    r"weights-v0\.3\.0",
    r"HoleGuard",
)

ALLOWED_HINTS = (
    "Dataset619",
    "nativemultistem",
    "13753413",
    "MultiTalentV2_challengeversion",
)


def _is_forbidden(path: str) -> bool:
    s = path.replace("\\", "/")
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, s, flags=re.I):
            # Dataset619 paths that mention MultiTalent are allowed even if 'Talent' matches nothing else
            if "Dataset619" in s or "13753413" in s or "nativemultistem" in s:
                if re.search(r"14007247|LocalEdit|TACE|/EDT/|weights-v0\.3\.0|HoleGuard", s, flags=re.I):
                    return True
                if "14007247" in s or "LesionTracer" in s and "Dataset619" not in s:
                    return True
                continue
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="*",
        help="Checkpoint / weight paths to check (empty → only print policy)",
    )
    ap.add_argument(
        "--require-dataset619",
        action="store_true",
        help="Also require ICLR2026/weights/Dataset619_nativemultistem/PRETRAIN_CHECKPOINT.txt",
    )
    ap.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    args = ap.parse_args()
    root = Path(args.repo_root)
    bad: list[str] = []
    for p in args.paths:
        if _is_forbidden(p):
            bad.append(p)
    if args.require_dataset619:
        marker = root / "ICLR2026/weights/Dataset619_nativemultistem/PRETRAIN_CHECKPOINT.txt"
        if not marker.is_file():
            print(f"[error] missing Dataset619 marker: {marker}", file=sys.stderr)
            print("  run: bash ICLR2026/run/download_dataset619_multitalent_pretrain_bg.sh", file=sys.stderr)
            return 2
        ckpt = marker.read_text(encoding="utf-8").strip()
        if not ckpt or not Path(ckpt).is_file():
            print(f"[error] invalid PRETRAIN_CHECKPOINT.txt → {ckpt!r}", file=sys.stderr)
            return 2
        if _is_forbidden(ckpt):
            bad.append(ckpt)
        elif not any(h.lower() in ckpt.lower() for h in ALLOWED_HINTS):
            print(f"[warn] Dataset619 path lacks expected hints: {ckpt}", file=sys.stderr)
    if bad:
        print("[error] FORBIDDEN for aligned board (final submission / interactive GC):", file=sys.stderr)
        for b in bad:
            print(f"  - {b}", file=sys.stderr)
        print(
            "Allowed pretrain only: Zenodo 13753413 Dataset619_nativemultistem",
            file=sys.stderr,
        )
        return 1
    print("[ok] competition board weight policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
