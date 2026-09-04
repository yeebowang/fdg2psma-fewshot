#!/usr/bin/env python3
"""Safe board JSON patch for aligned pipeline (avoid fragile bash-quoted JSON).

Usage:
  python3 aligned_board_patch.py --board PATH --patch '{"updated_note":"..."}'
  python3 aligned_board_patch.py --board PATH --set-stage nnunet.psma_fs50_f258 running --stamp STAMP --extra '{"bs":2}'
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, required=True)
    ap.add_argument("--patch", default="", help="JSON object to deep-merge")
    ap.add_argument("--set-stage", default="", help="e.g. nnunet.psma_fs50_f258")
    ap.add_argument("--status", default="")
    ap.add_argument("--stamp", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--queue", default="", help="comma-separated queue items")
    ap.add_argument("--updated-note", default="")
    ap.add_argument("--extra", default="", help="JSON object merged into stage")
    args = ap.parse_args()

    board: dict[str, Any] = {}
    if args.board.is_file():
        board = json.loads(args.board.read_text())

    if args.patch:
        board = deep_merge(board, json.loads(args.patch))

    if args.set_stage:
        method, stage = args.set_stage.split(".", 1)
        st = board.setdefault("methods", {}).setdefault(method, {}).setdefault(stage, {})
        if args.status:
            st["status"] = args.status
        if args.stamp:
            st["stamp"] = args.stamp
        if args.note:
            st["note"] = args.note
        if args.extra:
            st.update(json.loads(args.extra))

    if args.queue:
        board["queue"] = [x.strip() for x in args.queue.split(",") if x.strip()]
    if args.updated_note:
        board["updated_note"] = args.updated_note
    board["updated_at"] = _now()

    args.board.parent.mkdir(parents=True, exist_ok=True)
    args.board.write_text(json.dumps(board, indent=2) + "\n")
    print(f"[board-patch] wrote {args.board}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
