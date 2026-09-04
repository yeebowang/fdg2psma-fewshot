#!/usr/bin/env python3
"""RAM 友好训练分片：全量 case 随机均分为 N 片，每片连训若干 epoch，周期重新洗牌。

默认语义（可由 env 覆盖）：
  TASK1_RAM_SHARD_NUM=10
  TASK1_RAM_SHARD_EPOCHS=50          # 每片连训 epoch 数
  TASK1_RAM_SHARD_RESHUFFLE_EVERY=500  # 每隔多少 epoch 重新随机分片
  TASK1_RAM_SHARD_SEED=20260730

epoch e 时：
  cycle = e // RESHUFFLE_EVERY
  shard = (e % RESHUFFLE_EVERY) // SHARD_EPOCHS
  seed = BASE_SEED + cycle
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence


def env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(str(v).strip())


def ram_shard_enabled() -> bool:
    return env_truthy("TASK1_RAM_SHARD_ENABLE", "0")


def ram_shard_params() -> dict:
    return {
        "num_shards": max(1, env_int("TASK1_RAM_SHARD_NUM", 10)),
        "epochs_per_shard": max(1, env_int("TASK1_RAM_SHARD_EPOCHS", 50)),
        "reshuffle_every": max(1, env_int("TASK1_RAM_SHARD_RESHUFFLE_EVERY", 500)),
        "base_seed": env_int("TASK1_RAM_SHARD_SEED", 20260730),
    }


def shard_meta_for_epoch(epoch: int, params: dict | None = None) -> dict:
    p = params or ram_shard_params()
    e = max(0, int(epoch))
    period = int(p["reshuffle_every"])
    ep_per = int(p["epochs_per_shard"])
    n = int(p["num_shards"])
    # 若 10*50 != 500，仍按 reshuffle_every 切 cycle，shard 在周期内取模
    cycle = e // period
    within = e % period
    shard_id = (within // ep_per) % n
    seed = int(p["base_seed"]) + cycle
    return {
        "epoch": e,
        "cycle": cycle,
        "shard_id": shard_id,
        "seed": seed,
        "num_shards": n,
        "epochs_per_shard": ep_per,
        "reshuffle_every": period,
    }


def should_rebuild_dataloaders(epoch: int, params: dict | None = None) -> bool:
    """每片起点（含 reshuffle 后的 shard0）重建；epoch0 由 on_train_start 首次构建。"""
    e = int(epoch)
    if e <= 0:
        return False
    p = params or ram_shard_params()
    return (e % int(p["epochs_per_shard"])) == 0


def split_cases_into_shards(
    case_ids: Sequence[str],
    *,
    num_shards: int,
    seed: int,
) -> list[list[str]]:
    ids = sorted(str(x) for x in case_ids)
    import random

    rng = random.Random(int(seed))
    order = list(ids)
    rng.shuffle(order)
    n = max(1, int(num_shards))
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, cid in enumerate(order):
        shards[i % n].append(cid)
    return shards


def cases_for_epoch(
    all_case_ids: Sequence[str],
    epoch: int,
    params: dict | None = None,
) -> tuple[list[str], dict]:
    p = params or ram_shard_params()
    meta = shard_meta_for_epoch(epoch, p)
    shards = split_cases_into_shards(
        all_case_ids, num_shards=meta["num_shards"], seed=meta["seed"]
    )
    chosen = list(shards[meta["shard_id"]])
    meta = {
        **meta,
        "n_total": len(all_case_ids),
        "n_shard": len(chosen),
    }
    return chosen, meta


def write_state(path: Path, meta: dict, case_ids: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "case_ids": list(case_ids)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
