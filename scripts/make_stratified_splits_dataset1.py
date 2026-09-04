#!/usr/bin/env python3
"""dataset1 分层抽样：按 tracer×病变负荷 分层，再切 70% train / 10% val / 20% test。

类别定义（与目标比例一致）：
  - negative: GT 为空（无病灶体素）
  - small / large: 非空病例按总病灶体积（mm³）分位切分
      * PSMA 目标 small:large:negative = 65:25:10
        → 非空内 small 占比 65/(65+25)=72.22%（低体积端）
      * FDG  目标 small:large:negative = 15:35:50
        → 非空内 small 占比 15/(15+35)=30%（低体积端；对应既往 tiny 语义）

用法：
  python3 ICLR2026/scripts/make_stratified_splits_dataset1.py \\
    --dataset-dir /media/ybwang/data1/PSMA-DATA/dataset1 \\
    --out-json ICLR2026/data/splits_stratified_70_10_20.json \\
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


TARGET = {
    "psma": {"small": 65, "large": 25, "negative": 10},
    "fdg": {"small": 15, "large": 35, "negative": 50},
}
SPLIT_RATIOS = ("train", "val", "test")
SPLIT_WEIGHTS = (0.7, 0.1, 0.2)


def _tracer(case_id: str) -> str:
    if case_id.startswith("fdg_"):
        return "fdg"
    if case_id.startswith("psma_"):
        return "psma"
    raise ValueError(f"unknown tracer for case_id={case_id!r}")


def _scan_case(args: tuple[str, str]) -> dict:
    case_id, label_path = args
    img = nib.load(label_path)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    mask = arr > 0
    nvox = int(mask.sum())
    zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
    vox_mm3 = float(np.prod(zooms))
    vol_mm3 = float(nvox * vox_mm3)
    return {
        "case": case_id,
        "tracer": _tracer(case_id),
        "nvox": nvox,
        "volume_mm3": vol_mm3,
        "spacing_mm": list(zooms),
        "empty": nvox == 0,
    }


def _assign_roles(rows: list[dict]) -> None:
    """原地写入 role。"""
    by_tr: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_tr[r["tracer"]].append(r)

    for tr, items in by_tr.items():
        tgt = TARGET[tr]
        pos_small_frac = tgt["small"] / (tgt["small"] + tgt["large"])
        nonempty = [r for r in items if not r["empty"]]
        for r in items:
            if r["empty"]:
                r["role"] = "negative"
        if not nonempty:
            continue
        # 按体积升序；体积相同用 case id 打破平局，保证可复现
        nonempty_sorted = sorted(nonempty, key=lambda r: (r["volume_mm3"], r["case"]))
        n_small = int(round(len(nonempty_sorted) * pos_small_frac))
        n_small = max(0, min(len(nonempty_sorted), n_small))
        for i, r in enumerate(nonempty_sorted):
            r["role"] = "small" if i < n_small else "large"
        # 记录切分阈值（small 的最大体积）
        thr = (
            nonempty_sorted[n_small - 1]["volume_mm3"]
            if n_small > 0
            else float("-inf")
        )
        for r in items:
            r["small_volume_threshold_mm3"] = thr
            r["positive_small_frac_target"] = pos_small_frac


def _split_stratum(cases: list[str], seed: int) -> dict[str, list[str]]:
    """对单个 stratum 做 70/10/20；余数优先补给 train，再 test。"""
    rng = random.Random(seed)
    xs = list(cases)
    rng.shuffle(xs)
    n = len(xs)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        return {"train": xs, "val": [], "test": []}
    if n == 2:
        return {"train": [xs[0]], "val": [], "test": [xs[1]]}

    n_train = int(math.floor(n * 0.7))
    n_val = int(math.floor(n * 0.1))
    n_test = int(math.floor(n * 0.2))
    rem = n - (n_train + n_val + n_test)
    # 余数：先 train，再 test，再 val
    order = ["train", "test", "val"]
    buckets = {"train": n_train, "val": n_val, "test": n_test}
    i = 0
    while rem > 0:
        buckets[order[i % 3]] += 1
        rem -= 1
        i += 1
    # 保证至少 1 train（n>=3 已保证）
    out = {
        "train": xs[: buckets["train"]],
        "val": xs[buckets["train"] : buckets["train"] + buckets["val"]],
        "test": xs[buckets["train"] + buckets["val"] :],
    }
    assert len(out["train"]) + len(out["val"]) + len(out["test"]) == n
    return out


def _mixture_report(rows: list[dict], ids: set[str] | None = None) -> dict:
    subset = [r for r in rows if ids is None or r["case"] in ids]
    by_tr: dict[str, Counter] = defaultdict(Counter)
    for r in subset:
        by_tr[r["tracer"]][r["role"]] += 1
    report = {}
    for tr, cnt in by_tr.items():
        total = sum(cnt.values())
        report[tr] = {
            "n": total,
            "counts": dict(cnt),
            "fractions_pct": {
                k: round(100.0 * cnt[k] / total, 2) if total else 0.0
                for k in ("small", "large", "negative")
            },
            "target_pct": {
                k: round(100.0 * TARGET[tr][k] / sum(TARGET[tr].values()), 2)
                for k in ("small", "large", "negative")
            },
        }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/media/ybwang/data1/PSMA-DATA/dataset1"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "splits_stratified_70_10_20.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    labels = args.dataset_dir / "labelsTr"
    paths = sorted(labels.glob("*.nii.gz"))
    if not paths:
        raise SystemExit(f"no labels under {labels}")

    jobs = [(p.name[: -len(".nii.gz")], str(p)) for p in paths]
    print(f"[iclr2026-split] scanning {len(jobs)} labels …", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_scan_case, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(futs):
                print(f"[iclr2026-split] scanned {i}/{len(futs)}", flush=True)

    rows.sort(key=lambda r: r["case"])
    _assign_roles(rows)

    # stratum key = tracer|role
    strata: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        strata[f"{r['tracer']}|{r['role']}"].append(r["case"])

    split = {"train": [], "val": [], "test": []}
    stratum_splits = {}
    for key in sorted(strata):
        # 稳定子种子：主 seed + stratum 名 hash
        sub_seed = (args.seed * 1_000_003 + sum(ord(c) for c in key)) % (2**31 - 1)
        part = _split_stratum(sorted(strata[key]), sub_seed)
        stratum_splits[key] = {k: len(v) for k, v in part.items()}
        for sp in SPLIT_RATIOS:
            split[sp].extend(part[sp])

    for sp in SPLIT_RATIOS:
        split[sp] = sorted(split[sp])

    # sanity: partition
    all_ids = [r["case"] for r in rows]
    assert sorted(split["train"] + split["val"] + split["test"]) == sorted(all_ids)

    case_meta = {r["case"]: r for r in rows}
    payload = {
        "name": "iclr2026_dataset1_stratified_70_10_20",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "seed": args.seed,
        "split_ratios": {"train": 0.7, "val": 0.1, "test": 0.2},
        "target_mixture": TARGET,
        "protocol": (
            "negative = empty GT; among non-empty, small vs large by volume_mm3 quantile "
            "so that within each tracer the positive split matches "
            "small/(small+large) from the target mixture; then stratified 70/10/20 "
            "within each (tracer, role) stratum."
        ),
        "n_total": len(rows),
        "n_splits": {k: len(v) for k, v in split.items()},
        "mixture_overall": _mixture_report(rows),
        "mixture_by_split": {
            sp: _mixture_report(rows, set(ids)) for sp, ids in split.items()
        },
        "stratum_sizes": {k: len(v) for k, v in sorted(strata.items())},
        "stratum_split_counts": stratum_splits,
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        # nnU-Net 风格单折视图（无 test；test 单独字段）
        "splits_final_style": [{"train": split["train"], "val": split["val"]}],
        "case_meta": case_meta,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[iclr2026-split] wrote {args.out_json}", flush=True)
    print(f"[iclr2026-split] n={payload['n_splits']}", flush=True)
    for tr, rep in payload["mixture_overall"].items():
        print(
            f"  {tr}: n={rep['n']} fractions={rep['fractions_pct']} target={rep['target_pct']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
