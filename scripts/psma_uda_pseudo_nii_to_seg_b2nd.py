#!/usr/bin/env python3
"""
把伪标签 nii 写成预处理网格上的 ``{case}_seg.b2nd``（旁路目录），并写出
``{case}_classloc.npz``（重算 class_locations）供训练 Dataset 覆盖 GT。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import blosc2
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def _write_b2nd(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.ascontiguousarray(arr)
    if path.is_file():
        path.unlink()
    blosc2.asarray(
        out,
        urlpath=str(path),
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8},
        mmap_mode="w+",
    )


def _resample_seg_to(seg: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    if tuple(seg.shape) == tuple(target):
        return seg.astype(np.uint8, copy=False)
    factors = [t / s for t, s in zip(target, seg.shape)]
    return zoom(seg.astype(np.float32), factors, order=0).astype(np.uint8)


def _class_locations_from_seg(seg_chw: np.ndarray, max_samples: int = 10000) -> dict:
    """seg_chw: (1,D,H,W)；返回 {1: (N,4) int64}。"""
    coords = np.argwhere(seg_chw == 1)
    if coords.size == 0:
        return {1: np.zeros((0, 4), dtype=np.int64)}
    if coords.shape[0] > max_samples:
        rs = np.random.RandomState(0)
        idx = rs.choice(coords.shape[0], size=max_samples, replace=False)
        coords = coords[idx]
    return {1: coords.astype(np.int64, copy=False)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out-seg-b2nd-dir", type=Path, required=True)
    ap.add_argument("--cases-json", type=Path, default=None)
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    cases: list[str]
    if args.cases_json and args.cases_json.is_file():
        import json

        raw = json.loads(args.cases_json.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            cases = [str(x) for x in raw[0]["train"]]
        elif isinstance(raw, dict) and "train" in raw:
            cases = [str(x) for x in raw["train"]]
        else:
            cases = [str(x) for x in raw]
    else:
        cases = sorted(p.stem for p in args.labels_dir.glob("*.nii.gz"))

    prep = args.prep_dir.resolve()
    out_dir = args.out_seg_b2nd_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(cases)
    done = failed = 0
    pe = max(1, args.progress_every)
    t0 = time.monotonic()
    print(f"[pseudo-b2nd] cases={n} prep={prep} out={out_dir}", flush=True)

    for i, case in enumerate(cases, 1):
        try:
            main_b2nd = prep / f"{case}.b2nd"
            if not main_b2nd.is_file():
                raise FileNotFoundError(f"missing image b2nd: {main_b2nd}")
            main = np.asarray(blosc2.open(urlpath=str(main_b2nd), mode="r"))
            target = tuple(int(x) for x in main.shape[1:])

            lab = args.labels_dir / f"{case}.nii.gz"
            if not lab.is_file():
                raise FileNotFoundError(f"missing pseudo label: {lab}")
            d = np.asarray(nib.load(str(lab)).get_fdata())
            while d.ndim > 3 and d.shape[-1] == 1:
                d = d[..., 0]
            if d.ndim == 4:
                d = d[..., 0]
            u8 = (d > 0).astype(np.uint8)
            u8 = _resample_seg_to(u8, target)

            seg_chw = u8[None, ...].astype(np.int16)
            _write_b2nd(out_dir / f"{case}_seg.b2nd", seg_chw)
            cl = _class_locations_from_seg(seg_chw)
            np.savez_compressed(
                out_dir / f"{case}_classloc.npz",
                **{f"c{k}": v for k, v in cl.items()},
            )
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[pseudo-b2nd] FAIL {case}: {exc}", file=sys.stderr, flush=True)
        if i % pe == 0 or i == n:
            print(
                f"[pseudo-b2nd] {i}/{n} done={done} fail={failed} "
                f"elapsed={time.monotonic() - t0:.0f}s",
                flush=True,
            )

    print(f"[pseudo-b2nd] finished done={done} fail={failed}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
