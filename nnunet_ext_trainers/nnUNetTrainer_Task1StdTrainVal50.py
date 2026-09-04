"""
Task1：与排查 val bug 前一致的标准 train（继承 nnUNetTrainer 默认 250 iter/ep、1000 ep、save_every 等），
仅显式固定每 epoch 的 val iteration 数（默认 50，与 PETV baseline / 原 nnUNet 默认 val 一致），
并整合 demo 侧行为：默认跳过训练结束后的全量 ``perform_actual_validation``，便于只依赖 epoch 内 val。

环境变量:
  TASK1_TRAIN_ITERS_PER_EPOCH  未设置或为空则保持 nnUNetTrainer 默认 250；否则覆盖每 epoch train 迭代数
  TASK1_VAL_ITERS_PER_EPOCH  默认 50
  TASK1_VAL_EVERY_N_EPOCHS  默认 1：每 N 个 finished-epoch（1-based：20/40/…）才跑 epoch 内 val；
      末轮始终 val。设 0/1=每轮都 val（与历史一致）。
  TASK1_NUM_EPOCHS  设置则覆盖总训练轮数（默认与 nnUNetTrainer 一致为 1000）；续训时与 nnUNetv2_train --c 联用
  TASK1_LR_SCHEDULE_NUM_EPOCHS  PolyLR 分母（默认=num_epochs）。续训须与首轮一致（如 cascade 0→3000 则固定 3000），
      否则 LR 会阶跃、loss 曲线出现断崖。nnUNet 每 epoch 用 lr_scheduler.step(current_epoch) 按绝对 epoch 算 LR。
  TASK1_LR_SCHEDULE_RESET_AT_LR  设后启用分段 PolyLR：在绝对 epoch 达到该 LR（默认按 PRE_RESET 段推算）时，
      以 TASK1_LR_SCHEDULE_RESET_INITIAL_LR（默认同 RESET_AT_LR）和 TASK1_LR_SCHEDULE_NUM_EPOCHS 重算后续 PolyLR。
  TASK1_LR_SCHEDULE_PRE_RESET_MAX_EPOCHS  RESET 前 PolyLR 分母（默认 1000，对应当前 0→1000ep 首轮）。
  TASK1_LR_SCHEDULE_RESET_INITIAL_LR  RESET 后 PolyLR 起点 LR（默认 = RESET_AT_LR）。
  TASK1_LR_SCHEDULE_RESET_EPOCH  可选：手动指定 RESET 绝对 epoch；不设则按 PRE_RESET 段自动推算。
  TASK1_ALLOW_FULL_VOLUME_VAL  设为 1/true 则跑全量滑窗验证（与 demo 的 TASK1_DEMO_ALLOW_FULL_VAL 语义类似）
  TASK1_SEGMENT_CHECKPOINT  默认 1：每 TASK1_SEGMENT_CHECKPOINT_EVERY（默认 100）epoch 为一段，
      段内 best 创新高时额外写入 ``checkpoint_best_seg{lo}_{hi}.pth``（fold 折：EMA 伪 Dice 最高；fold=all：train_loss 最低）；
      且每完成整百 epoch（100、200、…）写入 ``checkpoint_final_ep{epoch}.pth`` 全量快照（与 checkpoint_final 内容同类）。
      设为 0/false 关闭上述额外文件（仍保留 nnUNet 默认 best/latest/final 行为）。
  TASK1_SEGMENT_CHECKPOINT_EVERY  段长，默认 100；须为正整数。
  TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH  默认 99：本轮训练结束时的 epoch 索引 < 99（即前 99 轮 ep0..ep98）
      禁用一切 checkpoint 落盘；第 100 轮（finished index 99）起恢复原有 best/latest/段快照逻辑。
      设为 0 关闭延迟（与历史行为一致）。
  TASK1_TRAIN_LIVE_PROGRESS  默认 1：每 epoch 写 fold 目录 ``task1_train_live_progress.json``（供 ETA 监控读 iter）
  TASK1_TRAIN_PROGRESS_INTERVAL  进度刷新间隔（iter 数），默认 1
  TASK1_PERIODIC_CHECKPOINT_EVERY  每 N 轮写入 ``checkpoint_final_ep{epoch}.pth``（默认 0=沿用 SEGMENT 逻辑；显式 0 关闭）
  TASK1_ALWAYS_SAVE_LATEST  默认 1：启用 ``checkpoint_latest.pth`` 写入
  TASK1_SAVE_LATEST_EVERY  与 ALWAYS_SAVE_LATEST 联用：每 N 轮写 latest（默认 1=每轮）；末轮始终写
  TASK1_SEGMENT_CHECKPOINT_MIN_EPOCH / TASK1_SEGMENT_CHECKPOINT_MAX_EPOCH  可选：仅在该 finished-epoch 半开区间
      [MIN, MAX) 内写 ``checkpoint_best_seg*``（如 2900/3000 表示 ep2900..2999）
  TASK1_CHECKPOINT_EMPTY_CACHE  默认 1：每次 save_checkpoint 后 ``torch.cuda.empty_cache()``。
      设为 0 则保留 PyTorch 显存缓存供后续 iter 复用（XL 大模型训练更满显存、少碎片）。
  TASK1_RAM_SHARD_ENABLE  设 1：将全量 case 随机均分为 N 片，训练时每片连训若干 ep（利于 page cache 常驻）。
  TASK1_RAM_SHARD_NUM  分片数，默认 10
  TASK1_RAM_SHARD_EPOCHS  每片连训 epoch 数，默认 50
  TASK1_RAM_SHARD_RESHUFFLE_EVERY  每隔多少 epoch 重新随机分片，默认 500
  TASK1_RAM_SHARD_SEED  基础随机种子，默认 20260730（每 cycle 用 seed+cycle）
  TASK1_TRAIN_CASE_ALLOWLIST  逗号/空白分隔的 case id；仅用这些例训练（demo/predemo 子集）
  TASK1_TRAIN_CASE_LIST_FILE  可选：每行一个 case id 的文本文件（与 ALLOWLIST 二选一或并用）
  TASK1_SPLITS_FINAL_JSON  可选：覆盖 ``splits_final.json``（nnU-Net 列表 ``[{train,val}, ...]``）；
      须 ``fold=0..4``（不可 fold=all）。用于 ICLR 等自定义划分而不改写共享 prep。
  TASK1_BEST_BY  可选覆盖 best 规则：``val_loss`` | ``train_loss`` | ``ema_fg_dice``；
      未设则沿用下方默认。``val_loss`` 要求 fold 折且 ``val_iters>0``。
  TASK1_VAL_LOSS_ONLY  设 1：epoch 内 val 只算 loss，跳过 hard Pseudo dice（one-hot/tp-fp-fn）；
      默认：当 ``TASK1_BEST_BY=val_loss`` 时自动开启；显式 ``0`` 可强制仍算 dice。
  TASK1_PSMA_VAL_ENABLE  设 1：从 ``TASK1_PSMA_VAL_FROM_EPOCH`` 起每 epoch 额外跑 PSMA-only val（监控用）
  TASK1_PSMA_VAL_FROM_EPOCH  默认 2000（``current_epoch >=`` 时启用）
  TASK1_PSMA_VAL_ITERS_PER_EPOCH  PSMA val 每 epoch step；未设则用当期 FDG val iters
  TASK1_PSMA_VAL_CASES_JSON  PSMA val case 列表 JSON（``{"val":[...]}`` / nnU-Net splits 列表 / 纯 list）
      仅写日志 ``PSMA_val_loss``，**不**写入 ``val_losses``、不参与 best。
  TASK1_VAL_ITERS_LATE_FROM_EPOCH  设后：``current_epoch >=`` 时 FDG val step 改用 LATE 值（与 PSMA 监控同段）
  TASK1_VAL_ITERS_LATE_PER_EPOCH  后半段 FDG val step（如 baseline1 后 1/3=50）
  TASK1_PSEUDO_SEG_B2ND_DIR  可选：旁路 ``{case}_seg.b2nd`` 目录；**仅 train** 用伪标覆盖 GT，val 仍读预处理真标。
  TASK1_INITIAL_LR  可选：覆盖 nnUNet 默认 ``initial_lr``（常为 0.01）；UDA 微调建议 1e-3。

best 判定（默认）：
  - **fold=all（全量训练，不使用折划分）**：``checkpoint_best`` / ``checkpoint_best_seg*`` 一律按 **train_loss 最低**；
    若仍配置 val step，仅作日志监控，不参与 best。
  - **fold=0..4 且 val_iters>0**：与 nnUNet 默认一致，按 **val ema_fg_dice 最高**。
  - **fold=0..4 且 val_iters=0**：同 fold=all，按 **train_loss 最低**（并跳过 val 计算）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from time import time
from typing import List

import numpy as np
import torch
from torch import autocast

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


def _ram_shard_import():
    """延迟导入：predict / 未启用分片时不强制依赖 task1_ram_shard.py。"""
    try:
        from task1_ram_shard import (  # type: ignore
            cases_for_epoch as _ram_cases_for_epoch,
            ram_shard_enabled as _ram_shard_enabled,
            ram_shard_params as _ram_shard_params,
            should_rebuild_dataloaders as _ram_should_rebuild,
            write_state as _ram_write_state,
        )
    except ImportError:
        from nnunetv2.training.nnUNetTrainer.task1_ram_shard import (  # type: ignore
            cases_for_epoch as _ram_cases_for_epoch,
            ram_shard_enabled as _ram_shard_enabled,
            ram_shard_params as _ram_shard_params,
            should_rebuild_dataloaders as _ram_should_rebuild,
            write_state as _ram_write_state,
        )
    return (
        _ram_cases_for_epoch,
        _ram_shard_enabled,
        _ram_shard_params,
        _ram_should_rebuild,
        _ram_write_state,
    )


def _ram_shard_enabled_env() -> bool:
    v = os.environ.get("TASK1_RAM_SHARD_ENABLE", "0")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int_nonempty(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return max(1, int(str(v).strip()))


def _env_int_allow_zero(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return max(0, int(str(v).strip()))


def _is_finite_num(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


class nnUNetTrainer_Task1StdTrainVal50(nnUNetTrainer):
    """标准 train；val 步数可配（默认 50）；默认跳过训练后全量 val。"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        v_train = os.environ.get("TASK1_TRAIN_ITERS_PER_EPOCH")
        if v_train is not None and str(v_train).strip() != "":
            self.num_iterations_per_epoch = max(1, int(str(v_train).strip()))
        self.num_val_iterations_per_epoch = _env_int_allow_zero(
            "TASK1_VAL_ITERS_PER_EPOCH", 50
        )
        v_epochs = os.environ.get("TASK1_NUM_EPOCHS")
        if v_epochs is not None and str(v_epochs).strip() != "":
            self.num_epochs = max(1, int(str(v_epochs).strip()))
        v_lr = os.environ.get("TASK1_INITIAL_LR")
        if v_lr is not None and str(v_lr).strip() != "":
            self.initial_lr = float(str(v_lr).strip())
        if self.local_rank == 0:
            best_rule = self._task1_best_rule_label()
            defer_ckpt = self._defer_checkpoint_until_epoch()
            defer_msg = (
                f"defer_checkpoint_until_finished_ep={defer_ckpt}"
                if defer_ckpt > 0
                else "defer_checkpoint=off"
            )
            self.print_to_log_file(
                "[Task1StdTrainVal50] "
                f"num_iterations_per_epoch={self.num_iterations_per_epoch}, "
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}, "
                f"val_every_n_epochs={self._task1_val_every_n_epochs()}, "
                f"num_epochs={self.num_epochs}, initial_lr={self.initial_lr}, "
                f"best_rule={best_rule}, "
                f"val_loss_only={int(self._task1_val_loss_only())}, {defer_msg}"
            )
        self._task1_seg_base: int | None = None
        self._task1_seg_best_ema: float | None = None
        self._task1_seg_best_train_loss: float | None = None
        self._task1_seg_best_val_loss: float | None = None
        self._best_train_loss: float | None = None
        self._best_val_loss: float | None = None
        self._defer_checkpoint_writes = False
        self._task1_epoch_start_unix: float | None = None
        self._task1_lr_reset_epoch_resolved: int | None = None
        self._task1_ram_shard_all_keys: list[str] | None = None
        self._task1_ram_shard_last_key: tuple[int, int] | None = None
        self.dataloader_psma_val = None
        self._task1_last_psma_val_loss: float | None = None
        self._task1_fullcase_val_dl = None
        self._task1_fullcase_val_gen = None
        self._task1_fullcase_val_queue: list | None = None
        if self.local_rank == 0 and _ram_shard_enabled_env():
            _, _, _ram_shard_params, _, _ = _ram_shard_import()
            p = _ram_shard_params()
            self.print_to_log_file(
                "[Task1StdTrainVal50] RAM shard ON: "
                f"num={p['num_shards']} epochs_per_shard={p['epochs_per_shard']} "
                f"reshuffle_every={p['reshuffle_every']} seed={p['base_seed']}"
            )

    def _task1_finish_dataloaders(self) -> None:
        if self.dataloader_train is None and self.dataloader_val is None:
            return
        old_stdout = sys.stdout
        with open(os.devnull, "w") as f:
            sys.stdout = f
            try:
                if self.dataloader_train is not None and isinstance(
                    self.dataloader_train,
                    (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                ):
                    self.dataloader_train._finish()
                if self.dataloader_val is not None and isinstance(
                    self.dataloader_val,
                    (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                ):
                    self.dataloader_val._finish()
                if getattr(self, "dataloader_psma_val", None) is not None and isinstance(
                    self.dataloader_psma_val,
                    (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
                ):
                    self.dataloader_psma_val._finish()
            finally:
                sys.stdout = old_stdout
        self.dataloader_train = self.dataloader_val = None
        self.dataloader_psma_val = None

    def _task1_case_allowlist(self) -> list[str] | None:
        ids: list[str] = []
        raw = os.environ.get("TASK1_TRAIN_CASE_ALLOWLIST", "").strip()
        if raw:
            for tok in raw.replace(",", " ").split():
                t = tok.strip()
                if t:
                    ids.append(t)
        path = os.environ.get("TASK1_TRAIN_CASE_LIST_FILE", "").strip()
        if path:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        t = line.strip()
                        if t and not t.startswith("#"):
                            ids.append(t)
            except OSError as e:
                if self.local_rank == 0:
                    self.print_to_log_file(
                        f"[Task1StdTrainVal50] WARN case list file unreadable: {path} ({e})"
                    )
        if not ids:
            return None
        # 保序去重
        seen: set[str] = set()
        out: list[str] = []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _task1_load_override_splits(self, path: str):
        if self._is_full_dataset_fold(self.fold):
            raise RuntimeError(
                "TASK1_SPLITS_FINAL_JSON requires fold=0..4 (not fold=all)"
            )
        raw = load_json(path)
        if isinstance(raw, dict) and "splits_final_style" in raw:
            splits = raw["splits_final_style"]
        elif isinstance(raw, dict) and "train" in raw and "val" in raw:
            splits = [{"train": raw["train"], "val": raw["val"]}]
        else:
            splits = raw
        if not isinstance(splits, list) or not splits:
            raise RuntimeError(f"TASK1_SPLITS_FINAL_JSON invalid list: {path}")
        fold_id = int(self.fold)
        if fold_id < 0 or fold_id >= len(splits):
            raise RuntimeError(
                f"TASK1_SPLITS_FINAL_JSON fold={fold_id} out of range "
                f"(n_folds={len(splits)}): {path}"
            )
        entry = splits[fold_id]
        tr_keys = list(entry["train"])
        val_keys = list(entry["val"])
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1Splits] "
                f"override={path} fold={fold_id} train={len(tr_keys)} val={len(val_keys)}"
            )
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        """若设置 TASK1_PSEUDO_SEG_B2ND_DIR：train 用伪标 Dataset，val 用原 Dataset。"""
        pseudo_dir = os.environ.get("TASK1_PSEUDO_SEG_B2ND_DIR", "").strip()
        if not pseudo_dir:
            return super().get_tr_and_val_datasets()
        tr_keys, val_keys = self.do_split()
        if self.dataset_class is None:
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        try:
            from task1_pseudo_seg_dataset import (  # type: ignore
                nnUNetDatasetBlosc2_Task1PseudoSeg,
            )
        except ImportError:
            from nnunetv2.training.nnUNetTrainer.task1_pseudo_seg_dataset import (  # type: ignore
                nnUNetDatasetBlosc2_Task1PseudoSeg,
            )
        if self.local_rank == 0:
            self.print_to_log_file(
                f"[Task1PseudoSeg] train uses pseudo b2nd from {pseudo_dir}; "
                f"val keeps GT ({len(tr_keys)}/{len(val_keys)})"
            )
        dataset_tr = nnUNetDatasetBlosc2_Task1PseudoSeg(
            self.preprocessed_dataset_folder, tr_keys
        )
        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder, val_keys
        )
        return dataset_tr, dataset_val

    def do_split(self):
        override = os.environ.get("TASK1_SPLITS_FINAL_JSON", "").strip()
        if override:
            tr_keys, val_keys = self._task1_load_override_splits(override)
        else:
            tr_keys, val_keys = super().do_split()
        allow = self._task1_case_allowlist()
        if allow is not None:
            allow_set = set(allow)
            tr_f = [k for k in tr_keys if k in allow_set]
            val_f = [k for k in val_keys if k in allow_set]
            missing = sorted(allow_set - set(tr_keys) - set(val_keys))
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1CaseAllowlist] "
                    f"requested={len(allow)} train={len(tr_f)} val={len(val_f)} "
                    f"missing={len(missing)}"
                )
                if missing:
                    self.print_to_log_file(
                        "[Task1CaseAllowlist] missing ids (first 8): "
                        + ", ".join(missing[:8])
                    )
            if not tr_f:
                raise RuntimeError(
                    "TASK1_TRAIN_CASE_ALLOWLIST/LIST_FILE matched 0 training cases"
                )
            tr_keys, val_keys = tr_f, (val_f if val_f else tr_f)

        if not _ram_shard_enabled_env():
            return tr_keys, val_keys
        (
            _ram_cases_for_epoch,
            _,
            _,
            _,
            _ram_write_state,
        ) = _ram_shard_import()
        # 全量 fold=all 时 tr==val；分片后训练/监控仍用同一子集
        all_keys = sorted(set(list(tr_keys) + list(val_keys)))
        self._task1_ram_shard_all_keys = all_keys
        shard_keys, meta = _ram_cases_for_epoch(all_keys, int(self.current_epoch))
        self._task1_ram_shard_last_key = (int(meta["cycle"]), int(meta["shard_id"]))
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1RamShard] "
                f"ep={meta['epoch']} cycle={meta['cycle']} shard={meta['shard_id']}/"
                f"{meta['num_shards']} n={meta['n_shard']}/{meta['n_total']} "
                f"seed={meta['seed']}"
            )
            try:
                from pathlib import Path

                out = Path(self.output_folder) / "task1_ram_shard_state.json"
                _ram_write_state(out, meta, shard_keys)
            except Exception as e:
                self.print_to_log_file(f"[Task1RamShard] WARN write state failed: {e}")
        return shard_keys, shard_keys

    def _task1_lr_schedule_max_epochs(self) -> int:
        """PolyLR 分母；续训时须与首轮 num_epochs 一致（可用 TASK1_LR_SCHEDULE_NUM_EPOCHS 钉死）。"""
        v = os.environ.get("TASK1_LR_SCHEDULE_NUM_EPOCHS")
        if v is not None and str(v).strip() != "":
            return max(1, int(str(v).strip()))
        return max(1, int(self.num_epochs))

    @staticmethod
    def _task1_poly_lr(initial_lr: float, epoch: int, max_epochs: int, exponent: float = 0.9) -> float:
        ep = max(0, min(int(epoch), int(max_epochs)))
        return float(initial_lr) * (1.0 - ep / float(max_epochs)) ** exponent

    def _task1_lr_reset_at(self) -> float | None:
        v = os.environ.get("TASK1_LR_SCHEDULE_RESET_AT_LR")
        if v is None or str(v).strip() == "":
            return None
        return float(str(v).strip())

    def _task1_lr_pre_reset_max_epochs(self) -> int:
        v = os.environ.get("TASK1_LR_SCHEDULE_PRE_RESET_MAX_EPOCHS")
        if v is not None and str(v).strip() != "":
            return max(1, int(str(v).strip()))
        return 1000

    def _task1_lr_reset_initial(self) -> float:
        v = os.environ.get("TASK1_LR_SCHEDULE_RESET_INITIAL_LR")
        if v is not None and str(v).strip() != "":
            return float(str(v).strip())
        ra = self._task1_lr_reset_at()
        return ra if ra is not None else float(self.initial_lr)

    def _task1_uses_lr_reset_schedule(self) -> bool:
        return self._task1_lr_reset_at() is not None

    def _task1_resolve_reset_epoch(self) -> int | None:
        if not self._task1_uses_lr_reset_schedule():
            return None
        if self._task1_lr_reset_epoch_resolved is not None:
            return self._task1_lr_reset_epoch_resolved
        v = os.environ.get("TASK1_LR_SCHEDULE_RESET_EPOCH")
        if v is not None and str(v).strip() != "":
            self._task1_lr_reset_epoch_resolved = max(0, int(str(v).strip()))
            return self._task1_lr_reset_epoch_resolved
        reset_at = self._task1_lr_reset_at()
        pre_max = self._task1_lr_pre_reset_max_epochs()
        for ep in range(pre_max + 1):
            if self._task1_poly_lr(float(self.initial_lr), ep, pre_max) <= float(reset_at) + 1e-9:
                self._task1_lr_reset_epoch_resolved = ep
                return ep
        self._task1_lr_reset_epoch_resolved = pre_max
        return self._task1_lr_reset_epoch_resolved

    def _task1_compute_lr(self, epoch: int) -> float:
        if not self._task1_uses_lr_reset_schedule():
            max_ep = self._task1_lr_schedule_max_epochs()
            return self._task1_poly_lr(float(self.initial_lr), epoch, max_ep)
        reset_ep = self._task1_resolve_reset_epoch()
        pre_max = self._task1_lr_pre_reset_max_epochs()
        if reset_ep is not None and int(epoch) >= int(reset_ep):
            max_ep = self._task1_lr_schedule_max_epochs()
            rel = int(epoch) - int(reset_ep)
            return self._task1_poly_lr(self._task1_lr_reset_initial(), rel, max_ep)
        return self._task1_poly_lr(float(self.initial_lr), epoch, pre_max)

    def _task1_apply_lr(self, epoch: int) -> None:
        lr = self._task1_compute_lr(epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        max_ep = self._task1_lr_schedule_max_epochs()
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, max_ep)
        return optimizer, lr_scheduler

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if (
            self.local_rank != 0
            or not hasattr(self, "lr_scheduler")
            or self.lr_scheduler is None
        ):
            return
        max_ep = self._task1_lr_schedule_max_epochs()
        self.lr_scheduler.max_steps = max_ep
        ep = int(self.current_epoch)
        if self._task1_uses_lr_reset_schedule():
            reset_ep = self._task1_resolve_reset_epoch()
            self._task1_apply_lr(ep)
            expected = self._task1_compute_lr(ep)
            actual = float(self.optimizer.param_groups[0]["lr"])
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1StdTrainVal50] resume LR reset-schedule sync "
                    f"ep={ep} reset_ep={reset_ep} lr={actual:.5f} "
                    f"(pre_max={self._task1_lr_pre_reset_max_epochs()}, "
                    f"post_max={max_ep}, train_num_epochs={self.num_epochs})"
                )
            return
        if _env_truthy("TASK1_LR_RESUME_PRESERVE_OPTIMIZER", "0"):
            actual = float(self.optimizer.param_groups[0]["lr"])
            self.print_to_log_file(
                "[Task1StdTrainVal50] resume LR preserve optimizer "
                f"ep={ep}/{max_ep} lr={actual:.5f} "
                f"(schedule_epochs={max_ep}, train_num_epochs={self.num_epochs})"
            )
            return
        self.lr_scheduler.step(ep)
        expected = self._task1_poly_lr(float(self.initial_lr), ep, max_ep)
        actual = float(self.optimizer.param_groups[0]["lr"])
        self.print_to_log_file(
            "[Task1StdTrainVal50] resume LR sync "
            f"ep={ep}/{max_ep} poly_expected={expected:.5f} actual={actual:.5f} "
            f"(schedule_epochs={max_ep}, train_num_epochs={self.num_epochs})"
        )
        if abs(expected - actual) > 1e-4:
            self.print_to_log_file(
                "[Task1StdTrainVal50] WARN LR mismatch after resume; "
                "check TASK1_LR_SCHEDULE_NUM_EPOCHS matches original run"
            )

    def on_train_epoch_start(self):
        ep = int(self.current_epoch)
        if _ram_shard_enabled_env():
            _, _, _, _ram_should_rebuild, _ = _ram_shard_import()
            if _ram_should_rebuild(ep):
                self._task1_finish_dataloaders()
                self.dataloader_train, self.dataloader_val = self.get_dataloaders()
                _ = next(self.dataloader_train)
                if self.dataloader_val is not None:
                    _ = next(self.dataloader_val)
                if self.local_rank == 0:
                    self.print_to_log_file(
                        f"[Task1RamShard] rebuilt dataloaders at epoch {ep}"
                    )
        if self._task1_uses_lr_reset_schedule():
            reset_ep = self._task1_resolve_reset_epoch()
            if reset_ep is not None and ep == reset_ep and self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1StdTrainVal50] LR schedule reset at "
                    f"ep={ep}: poly initial={self._task1_lr_reset_initial():.5f} "
                    f"max_epochs={self._task1_lr_schedule_max_epochs()}"
                )
            self._task1_apply_lr(ep)
        else:
            max_ep = self._task1_lr_schedule_max_epochs()
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.max_steps = max_ep
            lr_ep = min(ep, max_ep)
            if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
                self.lr_scheduler.step(lr_ep)
        self.network.train()
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)
        if self.local_rank == 0:
            self._task1_epoch_start_unix = time()
            if _env_truthy("TASK1_TRAIN_LIVE_PROGRESS", "1"):
                self._task1_write_live_progress(
                    "train", 0, self.num_iterations_per_epoch
                )

    def _defer_checkpoint_until_epoch(self) -> int:
        return _env_int_allow_zero("TASK1_DEFER_CHECKPOINT_UNTIL_EPOCH", 99)

    def _checkpoint_save_allowed_for_finished_epoch(self, finished_epoch: int) -> bool:
        defer_until = self._defer_checkpoint_until_epoch()
        if defer_until <= 0:
            return True
        return int(finished_epoch) >= defer_until

    def _task1_save_latest_every(self) -> int:
        v = os.environ.get("TASK1_SAVE_LATEST_EVERY")
        if v is None or str(v).strip() == "":
            return 1
        return max(1, int(str(v).strip()))

    def _task1_should_save_latest(self, finished_epoch: int) -> bool:
        if not _env_truthy("TASK1_ALWAYS_SAVE_LATEST", "1"):
            return False
        fe = int(finished_epoch)
        if fe >= int(self.num_epochs) - 1:
            return True
        return (fe + 1) % self._task1_save_latest_every() == 0

    def _task1_seg_checkpoint_epoch_in_range(self, finished_epoch: int) -> bool:
        vmin = os.environ.get("TASK1_SEGMENT_CHECKPOINT_MIN_EPOCH")
        vmax = os.environ.get("TASK1_SEGMENT_CHECKPOINT_MAX_EPOCH")
        if (vmin is None or str(vmin).strip() == "") and (
            vmax is None or str(vmax).strip() == ""
        ):
            return True
        lo = int(str(vmin).strip()) if vmin and str(vmin).strip() else 0
        if vmax is not None and str(vmax).strip() != "":
            hi_excl = int(str(vmax).strip())
        else:
            hi_excl = int(self.num_epochs)
        return lo <= int(finished_epoch) < hi_excl

    @staticmethod
    def _is_full_dataset_fold(fold) -> bool:
        return str(fold).strip().lower() == "all"

    def _task1_best_by_env(self) -> str | None:
        raw = os.environ.get("TASK1_BEST_BY", "").strip().lower()
        if not raw:
            return None
        aliases = {
            "val_loss": "val_loss",
            "valloss": "val_loss",
            "min_val_loss": "val_loss",
            "train_loss": "train_loss",
            "trainloss": "train_loss",
            "min_train_loss": "train_loss",
            "ema_fg_dice": "ema_fg_dice",
            "ema": "ema_fg_dice",
            "dice": "ema_fg_dice",
            "val_dice": "ema_fg_dice",
        }
        if raw not in aliases:
            raise RuntimeError(
                f"TASK1_BEST_BY={raw!r} invalid; use val_loss|train_loss|ema_fg_dice"
            )
        return aliases[raw]

    def _task1_resolved_best_by(self) -> str:
        override = self._task1_best_by_env()
        if override == "val_loss":
            if self._is_full_dataset_fold(self.fold):
                raise RuntimeError("TASK1_BEST_BY=val_loss requires fold=0..4 (not all)")
            if self.num_val_iterations_per_epoch <= 0:
                raise RuntimeError("TASK1_BEST_BY=val_loss requires TASK1_VAL_ITERS_PER_EPOCH>0")
            return "val_loss"
        if override == "train_loss":
            return "train_loss"
        if override == "ema_fg_dice":
            if self.num_val_iterations_per_epoch <= 0:
                raise RuntimeError(
                    "TASK1_BEST_BY=ema_fg_dice requires TASK1_VAL_ITERS_PER_EPOCH>0"
                )
            return "ema_fg_dice"
        if self._is_full_dataset_fold(self.fold) or self.num_val_iterations_per_epoch <= 0:
            return "train_loss"
        return "ema_fg_dice"

    def _task1_best_rule_label(self) -> str:
        by = self._task1_resolved_best_by()
        env = self._task1_best_by_env()
        suffix = f" (TASK1_BEST_BY={env})" if env else ""
        if by == "val_loss":
            return f"min val_loss{suffix}"
        if by == "train_loss":
            if self._is_full_dataset_fold(self.fold):
                return f"min train_loss (fold=all full-data){suffix}"
            if self.num_val_iterations_per_epoch <= 0:
                return f"min train_loss (val_iters=0){suffix}"
            return f"min train_loss{suffix}"
        return f"max val ema_fg_dice{suffix}"

    def _uses_train_loss_best(self) -> bool:
        return self._task1_resolved_best_by() == "train_loss"

    def _uses_val_loss_best(self) -> bool:
        return self._task1_resolved_best_by() == "val_loss"

    def _task1_val_loss_only(self) -> bool:
        """epoch 内 val 是否跳过 hard Pseudo dice（仅 loss）。"""
        raw = os.environ.get("TASK1_VAL_LOSS_ONLY", "")
        if raw is not None and str(raw).strip() != "":
            return _env_truthy("TASK1_VAL_LOSS_ONLY", "0")
        return self._task1_resolved_best_by() == "val_loss"

    def _task1_psma_val_enabled(self) -> bool:
        return _env_truthy("TASK1_PSMA_VAL_ENABLE", "0")

    def _task1_psma_val_from_epoch(self) -> int:
        return max(0, _env_int_allow_zero("TASK1_PSMA_VAL_FROM_EPOCH", 2000))

    def _task1_val_iters_late_from_epoch(self) -> int | None:
        """后半段 FDG val step 切换起点；未设则不切换。可与 PSMA from_epoch 对齐。"""
        raw = os.environ.get("TASK1_VAL_ITERS_LATE_FROM_EPOCH", "").strip()
        if not raw:
            # 若开了 PSMA 监控且设了 LATE iters，默认与 PSMA from 对齐
            if (
                self._task1_psma_val_enabled()
                and os.environ.get("TASK1_VAL_ITERS_LATE_PER_EPOCH", "").strip()
            ):
                return self._task1_psma_val_from_epoch()
            return None
        return max(0, int(raw))

    def _task1_fdg_val_iters(self) -> int:
        """当期 FDG val step：前半用 ``num_val_iterations_per_epoch``，后半可切到 LATE。"""
        base = int(self.num_val_iterations_per_epoch)
        late_from = self._task1_val_iters_late_from_epoch()
        if late_from is None or int(self.current_epoch) < late_from:
            return base
        raw = os.environ.get("TASK1_VAL_ITERS_LATE_PER_EPOCH", "").strip()
        if not raw:
            return base
        return max(0, int(raw))

    def _task1_val_every_n_epochs(self) -> int:
        """1=every epoch; N>1 → val on finished ep N,2N,… and final epoch (1-based)."""
        return max(0, _env_int_allow_zero("TASK1_VAL_EVERY_N_EPOCHS", 1))

    def _task1_n_val_cases(self) -> int:
        try:
            _tr, val_keys = self.do_split()
            return len(list(val_keys))
        except Exception:
            return 0

    def _task1_fullcase_val_iters(self) -> int:
        """Iters needed to visit every val case once (ceil(n_cases / batch_size))."""
        n = self._task1_n_val_cases()
        bs = max(1, int(self.batch_size))
        configured = max(0, int(self._task1_fdg_val_iters()))
        if n > 0:
            return max(configured, int(math.ceil(n / float(bs))))
        return max(1, configured)

    def _task1_ensure_fullcase_val_gen(self) -> None:
        """Single-threaded val loader so we can iterate all val identifiers once."""
        if self._task1_fullcase_val_gen is not None:
            return
        from batchgenerators.dataloading.single_threaded_augmenter import (
            SingleThreadedAugmenter,
        )

        try:
            from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        except ImportError:
            from nnunetv2.training.dataloading.data_loader_3d import (  # type: ignore
                nnUNetDataLoader3D as nnUNetDataLoader,
            )

        _tr, dataset_val = self.get_tr_and_val_datasets()
        deep_supervision_scales = self._get_deep_supervision_scales()
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )
        dl_kw = dict(
            data=dataset_val,
            batch_size=self.batch_size,
            patch_size=self.configuration_manager.patch_size,
            final_patch_size=self.configuration_manager.patch_size,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )
        try:
            dl = nnUNetDataLoader(
                **dl_kw,
                transforms=val_transforms,
                probabilistic_oversampling=self.probabilistic_oversampling,
            )
            tf_for_aug = None
        except TypeError:
            dl = nnUNetDataLoader(**dl_kw)
            tf_for_aug = val_transforms
        self._task1_fullcase_val_dl = dl
        self._task1_fullcase_val_gen = SingleThreadedAugmenter(dl, tf_for_aug)
        _ = next(self._task1_fullcase_val_gen)

    def _task1_run_fullcase_val_outputs(self) -> list:
        """One shuffled pass over all val cases (not random-with-replacement patches)."""
        self._task1_ensure_fullcase_val_gen()
        dl = self._task1_fullcase_val_dl
        ids = list(getattr(dl, "indices", []) or [])
        if not ids:
            n = max(1, int(self._task1_fdg_val_iters()))
            return [self.validation_step(next(self.dataloader_val)) for _ in range(n)]
        rng = np.random.RandomState(int(self.current_epoch) + 20260818)
        order = ids[:]
        rng.shuffle(order)
        bs = max(1, int(getattr(dl, "batch_size", self.batch_size)))
        queue = {"q": order}

        def _get_indices():
            q = queue["q"]
            if len(q) >= bs:
                chunk, queue["q"] = q[:bs], q[bs:]
                return chunk
            chunk = list(q)
            pad_from = ids
            need = bs - len(chunk)
            chunk.extend(pad_from[:need])
            queue["q"] = []
            return chunk[:bs]

        orig = dl.get_indices
        dl.get_indices = _get_indices
        n_iters = int(math.ceil(len(ids) / float(bs)))
        if self.local_rank == 0:
            self.print_to_log_file(
                f"[Task1StdTrainVal50] fullcase val: n_cases={len(ids)} "
                f"bs={bs} iters={n_iters}"
            )
        try:
            outputs = []
            for batch_id in range(n_iters):
                outputs.append(self.validation_step(next(self._task1_fullcase_val_gen)))
                if self._task1_should_write_live_progress(batch_id, n_iters):
                    self._task1_write_live_progress("val", batch_id + 1, n_iters)
            return outputs
        finally:
            dl.get_indices = orig

    def _task1_last_finite_scalar(self, key: str) -> float:
        lst = self.logger.my_fantastic_logging.get(key, [])
        for v in reversed(lst):
            if _is_finite_num(v):
                return float(v)
        return float("nan")

    def _task1_repair_ema_after_val(self) -> None:
        """EMA must not be 0.9*nan + 0.1*dice. First real val initializes EMA = dice."""
        ema = self.logger.my_fantastic_logging.get("ema_fg_dice")
        mean = self.logger.my_fantastic_logging.get("mean_fg_dice")
        if not ema or not mean:
            return
        ep = int(self.current_epoch)
        if ep >= len(mean) or not _is_finite_num(mean[ep]):
            return
        cur = float(mean[ep])
        prev = float(ema[ep - 1]) if ep > 0 and len(ema) >= ep else float("nan")
        if _is_finite_num(prev):
            fixed = 0.9 * prev + 0.1 * cur
        else:
            fixed = cur
        if len(ema) == ep:
            ema.append(fixed)
        elif len(ema) > ep:
            ema[ep] = fixed
        else:
            while len(ema) < ep:
                ema.append(float("nan"))
            ema.append(fixed)

    def _task1_should_run_val_this_epoch(self) -> bool:
        every = self._task1_val_every_n_epochs()
        if every <= 1:
            return True
        # current_epoch is 0-based index of the epoch currently finishing
        ep1 = int(self.current_epoch) + 1
        if ep1 >= int(self.num_epochs):
            return True
        return (ep1 % every) == 0

    def _task1_psma_val_iters(self) -> int:
        raw = os.environ.get("TASK1_PSMA_VAL_ITERS_PER_EPOCH", "").strip()
        if raw:
            return max(1, int(raw))
        # 未显式指定：跟当期 FDG val step（后半段通常已是 50）
        return max(1, self._task1_fdg_val_iters())

    def _task1_load_psma_val_keys(self) -> list[str]:
        path = os.environ.get("TASK1_PSMA_VAL_CASES_JSON", "").strip()
        if not path:
            raise RuntimeError(
                "TASK1_PSMA_VAL_ENABLE=1 requires TASK1_PSMA_VAL_CASES_JSON"
            )
        raw = load_json(path)
        keys: list[str]
        if isinstance(raw, dict) and "val" in raw:
            keys = list(raw["val"])
        elif isinstance(raw, dict) and "cases" in raw:
            keys = list(raw["cases"])
        elif isinstance(raw, list) and raw and isinstance(raw[0], dict) and "val" in raw[0]:
            fold_id = int(self.fold) if not self._is_full_dataset_fold(self.fold) else 0
            if fold_id < 0 or fold_id >= len(raw):
                fold_id = 0
            keys = list(raw[fold_id]["val"])
        elif isinstance(raw, list):
            keys = [str(x) for x in raw]
        else:
            raise RuntimeError(f"TASK1_PSMA_VAL_CASES_JSON unsupported format: {path}")
        keys = [str(k) for k in keys if str(k).strip()]
        if not keys:
            raise RuntimeError(f"TASK1_PSMA_VAL_CASES_JSON empty val list: {path}")
        return keys

    def _task1_ensure_psma_val_dataloader(self) -> None:
        if self.dataloader_psma_val is not None:
            return
        from batchgenerators.dataloading.single_threaded_augmenter import (
            SingleThreadedAugmenter,
        )
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        keys = self._task1_load_psma_val_keys()
        dataset_psma = self.dataset_class(
            self.preprocessed_dataset_folder,
            keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
        )
        deep_supervision_scales = self._get_deep_supervision_scales()
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions
            else None,
            ignore_label=self.label_manager.ignore_label,
        )
        dl_val = nnUNetDataLoader(
            dataset_psma,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        _ = next(mt_gen_val)
        self.dataloader_psma_val = mt_gen_val
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1PsmaVal] dataloader ready "
                f"n_cases={len(keys)} iters/ep={self._task1_psma_val_iters()} "
                f"from_epoch>={self._task1_psma_val_from_epoch()}"
            )

    def _task1_run_psma_val_if_needed(self) -> float | None:
        """额外 PSMA val；返回 mean loss，未启用则 None。不写 logger.val_losses。"""
        self._task1_last_psma_val_loss = None
        if not self._task1_psma_val_enabled():
            return None
        if int(self.current_epoch) < self._task1_psma_val_from_epoch():
            return None
        if self._task1_fdg_val_iters() <= 0 and int(self.num_val_iterations_per_epoch) <= 0:
            return None
        self._task1_ensure_psma_val_dataloader()
        n_iters = self._task1_psma_val_iters()
        with torch.no_grad():
            val_outputs = []
            for batch_id in range(n_iters):
                val_outputs.append(
                    self.validation_step(next(self.dataloader_psma_val))
                )
                if self._task1_should_write_live_progress(batch_id, n_iters):
                    self._task1_write_live_progress(
                        "psma_val", batch_id + 1, n_iters
                    )
        outputs_collated = collate_outputs(val_outputs)
        if self.is_ddp:
            import torch.distributed as dist

            world_size = dist.get_world_size()
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = float(np.vstack(losses_val).mean())
        else:
            loss_here = float(np.mean(outputs_collated["loss"]))
        self._task1_last_psma_val_loss = loss_here
        return loss_here

    def _task1_print_psma_val_loss_if_any(self) -> None:
        if self._task1_last_psma_val_loss is None:
            return
        self.print_to_log_file(
            "PSMA_val_loss",
            np.round(self._task1_last_psma_val_loss, decimals=4),
        )

    def validation_step(self, batch: dict) -> dict:
        if not self._task1_val_loss_only():
            return super().validation_step(batch)

        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        use_cuda = self.device.type == "cuda"
        with autocast(self.device.type, enabled=True) if use_cuda else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        # 占位：与 collate / 旧 on_validation_epoch_end 兼容；loss-only 路径不读这些
        z = np.zeros((1,), dtype=np.float64)
        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": z,
            "fp_hard": z,
            "fn_hard": z,
        }

    def _task1_pad_log_key(self, key: str, value, epoch: int) -> None:
        """Pad logger list to ``epoch`` then set value — no EMA side effects.

        nnUNetLogger.log('mean_fg_dice') auto-writes ema_fg_dice via
        ``ema[epoch-1]``, which IndexErrors when online val is every-N
        (sparse epochs). Online val must not use that path.
        """
        lst = self.logger.my_fantastic_logging[key]
        fill = value
        while len(lst) < epoch:
            lst.append(fill if not isinstance(fill, list) else list(fill))
        if len(lst) == epoch:
            lst.append(value)
        elif len(lst) == epoch + 1:
            lst[epoch] = value
        else:
            raise RuntimeError(
                f"logger[{key}] length {len(lst)} incompatible with epoch={epoch}"
            )

    def _on_epoch_end_skip_val_bookkeeping(self) -> None:
        """Val skipped (every-N): log train + pad val/dice lists; do not update best.

        Must pad ``mean_fg_dice`` / ``ema_fg_dice`` / ``val_losses`` every skipped
        epoch. Otherwise the next real val calls ``logger.log('mean_fg_dice')``
        which does ``ema[epoch-1]`` and IndexErrors on sparse every-N schedules.
        """
        finished_epoch = int(self.current_epoch)
        allow_ckpt = self._checkpoint_save_allowed_for_finished_epoch(finished_epoch)
        # keep plot_progress_png length-aligned; carry last finite EMA so the
        # next real val does not do 0.9*nan + 0.1*dice.
        carry_loss = self._task1_last_finite_scalar("val_losses")
        carry_mean = self._task1_last_finite_scalar("mean_fg_dice")
        carry_ema = self._task1_last_finite_scalar("ema_fg_dice")
        self._task1_pad_log_key("val_losses", carry_loss, self.current_epoch)
        self._task1_pad_log_key("mean_fg_dice", carry_mean, self.current_epoch)
        self._task1_pad_log_key("ema_fg_dice", carry_ema, self.current_epoch)
        last_dice = self.logger.my_fantastic_logging.get("dice_per_class_or_region", [])
        pad_dice = last_dice[-1] if last_dice else [float("nan")]
        if isinstance(pad_dice, list):
            pad_dice = [
                (float(x) if _is_finite_num(x) else float("nan")) for x in pad_dice
            ] or [float("nan")]
        else:
            pad_dice = [float("nan")]
        self._task1_pad_log_key(
            "dice_per_class_or_region", list(pad_dice), self.current_epoch
        )
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        train_loss = float(self.logger.my_fantastic_logging["train_losses"][-1])
        self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))
        every = self._task1_val_every_n_epochs()
        self.print_to_log_file(
            f"[Task1StdTrainVal50] online val skipped "
            f"(every_n={every}; next at ep % {every} == 0 or final)"
        )
        self.print_to_log_file(
            "Epoch time: "
            f"{np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )
        current_epoch = self.current_epoch
        if (
            allow_ckpt
            and self._task1_should_save_latest(current_epoch)
            and self.local_rank == 0
            and not self.disable_checkpointing
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        self.current_epoch += 1
        self._task1_maybe_save_periodic_checkpoint(allow_ckpt)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        if self._task1_val_loss_only():
            outputs_collated = collate_outputs(val_outputs)
            if self.is_ddp:
                import torch.distributed as dist

                world_size = dist.get_world_size()
                losses_val = [None for _ in range(world_size)]
                dist.all_gather_object(losses_val, outputs_collated["loss"])
                loss_here = np.vstack(losses_val).mean()
            else:
                loss_here = np.mean(outputs_collated["loss"])

            # Online val = val_loss only. Do NOT logger.log('mean_fg_dice'):
            # that triggers ema_fg_dice and crashes on sparse every-N schedules.
            self._task1_pad_log_key("mean_fg_dice", float("nan"), self.current_epoch)
            self._task1_pad_log_key("ema_fg_dice", float("nan"), self.current_epoch)
            self._task1_pad_log_key(
                "dice_per_class_or_region", [float("nan")], self.current_epoch
            )
            self.logger.log("val_losses", loss_here, self.current_epoch)
            return

        # Dice/EMA best + every-N: ensure ema list is contiguous before parent
        # ``log('mean_fg_dice')`` reads ``ema[epoch-1]``.
        ep = int(self.current_epoch)
        ema = self.logger.my_fantastic_logging.get("ema_fg_dice", [])
        while len(ema) < ep:
            ema.append(float("nan"))
        mean = self.logger.my_fantastic_logging.get("mean_fg_dice", [])
        while len(mean) < ep:
            mean.append(float("nan"))
        ret = super().on_validation_epoch_end(val_outputs)
        self._task1_repair_ema_after_val()
        return ret

    def save_checkpoint(self, filename: str) -> None:
        if self._defer_checkpoint_writes:
            return
        super().save_checkpoint(filename)
        if (
            self.local_rank == 0
            and torch.cuda.is_available()
            and _env_truthy("TASK1_CHECKPOINT_EMPTY_CACHE", "1")
        ):
            torch.cuda.empty_cache()

    def _on_epoch_end_train_loss_best(self) -> None:
        """fold=all 或 val_iters=0：best 按 train_loss 最低；val 指标仅可选日志。"""
        finished_epoch = int(self.current_epoch)
        allow_ckpt = self._checkpoint_save_allowed_for_finished_epoch(finished_epoch)
        if not allow_ckpt and self.local_rank == 0:
            self.print_to_log_file(
                "[Task1StdTrainVal50] checkpoint save deferred "
                f"(finished ep {finished_epoch}, defer until finished ep >= "
                f"{self._defer_checkpoint_until_epoch()})"
            )

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        train_loss = float(self.logger.my_fantastic_logging["train_losses"][-1])
        self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))

        val_losses = self.logger.my_fantastic_logging.get("val_losses", [])
        if val_losses:
            self.print_to_log_file("val_loss", np.round(val_losses[-1], decimals=4))
        self._task1_print_psma_val_loss_if_any()
        dice_list = self.logger.my_fantastic_logging.get("dice_per_class_or_region", [])
        if dice_list:
            self.print_to_log_file(
                "Pseudo dice",
                [np.round(i, decimals=4) for i in dice_list[-1]],
            )

        self.print_to_log_file(
            "Epoch time: "
            f"{np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        seg_enabled = _env_truthy("TASK1_SEGMENT_CHECKPOINT", "1")
        seg_every = _env_int_nonempty("TASK1_SEGMENT_CHECKPOINT_EVERY", 100)
        improved_seg = False
        seg_lo = seg_hi = 0
        if seg_enabled:
            ce = int(self.current_epoch)
            seg_lo = (ce // seg_every) * seg_every
            seg_hi = seg_lo + seg_every - 1
            if self._task1_seg_base != seg_lo:
                self._task1_seg_base = seg_lo
                self._task1_seg_best_train_loss = None
            improved_seg = (
                self._task1_seg_best_train_loss is None
                or train_loss < self._task1_seg_best_train_loss
            )
            if improved_seg:
                self._task1_seg_best_train_loss = train_loss

        current_epoch = self.current_epoch
        if (
            allow_ckpt
            and self._task1_should_save_latest(current_epoch)
            and self.local_rank == 0
            and not self.disable_checkpointing
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        if self._best_train_loss is None or train_loss < self._best_train_loss:
            self._best_train_loss = train_loss
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"Yayy! New best train_loss: {np.round(self._best_train_loss, decimals=4)}"
                )
                if allow_ckpt and not self.disable_checkpointing:
                    self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        self.current_epoch += 1

        self._task1_maybe_save_periodic_checkpoint(allow_ckpt)

        if (
            not allow_ckpt
            or self.local_rank != 0
            or self.disable_checkpointing
            or not seg_enabled
        ):
            return

        if improved_seg and self._task1_seg_checkpoint_epoch_in_range(current_epoch):
            fn = join(
                self.output_folder,
                f"checkpoint_best_seg{seg_lo:04d}_{seg_hi:04d}.pth",
            )
            self.save_checkpoint(fn)
            self.print_to_log_file(
                f"[Task1StdTrainVal50] segment best (seg {seg_lo}-{seg_hi}, "
                f"train_loss={train_loss:.4f}) -> {fn}"
            )

    def _on_epoch_end_val_loss_best(self) -> None:
        """fold 折 + TASK1_BEST_BY=val_loss：best 按 val_loss 最低。"""
        finished_epoch = int(self.current_epoch)
        allow_ckpt = self._checkpoint_save_allowed_for_finished_epoch(finished_epoch)
        if not allow_ckpt and self.local_rank == 0:
            self.print_to_log_file(
                "[Task1StdTrainVal50] checkpoint save deferred "
                f"(finished ep {finished_epoch}, defer until finished ep >= "
                f"{self._defer_checkpoint_until_epoch()})"
            )

        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        train_loss = float(self.logger.my_fantastic_logging["train_losses"][-1])
        self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))

        val_losses = self.logger.my_fantastic_logging.get("val_losses", [])
        if not val_losses:
            raise RuntimeError(
                "TASK1_BEST_BY=val_loss but logger has no val_losses "
                "(check TASK1_VAL_ITERS_PER_EPOCH>0)"
            )
        val_loss = float(val_losses[-1])
        self.print_to_log_file("val_loss", np.round(val_loss, decimals=4))
        self._task1_print_psma_val_loss_if_any()
        dice_list = self.logger.my_fantastic_logging.get("dice_per_class_or_region", [])
        if dice_list:
            self.print_to_log_file(
                "Pseudo dice",
                [np.round(i, decimals=4) for i in dice_list[-1]],
            )

        self.print_to_log_file(
            "Epoch time: "
            f"{np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        seg_enabled = _env_truthy("TASK1_SEGMENT_CHECKPOINT", "1")
        seg_every = _env_int_nonempty("TASK1_SEGMENT_CHECKPOINT_EVERY", 100)
        improved_seg = False
        seg_lo = seg_hi = 0
        if seg_enabled:
            ce = int(self.current_epoch)
            seg_lo = (ce // seg_every) * seg_every
            seg_hi = seg_lo + seg_every - 1
            if self._task1_seg_base != seg_lo:
                self._task1_seg_base = seg_lo
                self._task1_seg_best_val_loss = None
            improved_seg = (
                self._task1_seg_best_val_loss is None
                or val_loss < self._task1_seg_best_val_loss
            )
            if improved_seg:
                self._task1_seg_best_val_loss = val_loss

        current_epoch = self.current_epoch
        if (
            allow_ckpt
            and self._task1_should_save_latest(current_epoch)
            and self.local_rank == 0
            and not self.disable_checkpointing
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        if _is_finite_num(val_loss) and (
            self._best_val_loss is None
            or not _is_finite_num(self._best_val_loss)
            or val_loss < float(self._best_val_loss)
        ):
            self._best_val_loss = val_loss
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"Yayy! New best val_loss: {np.round(self._best_val_loss, decimals=4)}"
                )
                if allow_ckpt and not self.disable_checkpointing:
                    self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

        self._task1_maybe_save_periodic_checkpoint(allow_ckpt)

        if (
            not allow_ckpt
            or self.local_rank != 0
            or self.disable_checkpointing
            or not seg_enabled
        ):
            return

        if improved_seg and self._task1_seg_checkpoint_epoch_in_range(current_epoch):
            fn = join(
                self.output_folder,
                f"checkpoint_best_seg{seg_lo:04d}_{seg_hi:04d}.pth",
            )
            self.save_checkpoint(fn)
            self.print_to_log_file(
                f"[Task1StdTrainVal50] segment best (seg {seg_lo}-{seg_hi}, "
                f"val_loss={val_loss:.4f}) -> {fn}"
            )

    def on_epoch_end(self):
        """在父类逻辑前后增加：段内 best 快照、整百 epoch 全量快照（可关）。"""
        finished_epoch = int(self.current_epoch)
        allow_ckpt = self._checkpoint_save_allowed_for_finished_epoch(finished_epoch)
        self._defer_checkpoint_writes = not allow_ckpt
        try:
            # train_loss best：每轮可更新；val 仅作可选日志
            if self._uses_train_loss_best():
                self._on_epoch_end_train_loss_best()
                return

            # ema / val_loss best：非 val 轮次不碰 best（避免 stale metric）
            if not getattr(self, "_task1_val_ran_this_epoch", True):
                self._on_epoch_end_skip_val_bookkeeping()
                return

            if self._uses_val_loss_best():
                self._on_epoch_end_val_loss_best()
                return

            self._on_epoch_end_ema_dice_best(allow_ckpt)
        finally:
            self._defer_checkpoint_writes = False

    def _on_epoch_end_ema_dice_best(self, allow_ckpt: bool) -> None:
        """max val ema_fg_dice; ignore non-finite EMA (skip-val nan must not win)."""
        self._task1_repair_ema_after_val()
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        train_loss = float(self.logger.my_fantastic_logging["train_losses"][-1])
        self.print_to_log_file("train_loss", np.round(train_loss, decimals=4))
        val_losses = self.logger.my_fantastic_logging.get("val_losses", [])
        if val_losses:
            self.print_to_log_file("val_loss", np.round(val_losses[-1], decimals=4))
        self._task1_print_psma_val_loss_if_any()
        dice_list = self.logger.my_fantastic_logging.get("dice_per_class_or_region", [])
        if dice_list:
            self.print_to_log_file(
                "Pseudo dice",
                [np.round(i, decimals=4) for i in dice_list[-1]],
            )
        self.print_to_log_file(
            "Epoch time: "
            f"{np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (
            allow_ckpt
            and self._task1_should_save_latest(current_epoch)
            and self.local_rank == 0
            and not self.disable_checkpointing
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        ema_list = self.logger.my_fantastic_logging.get("ema_fg_dice", [])
        ema = float(ema_list[-1]) if ema_list else float("nan")
        if _is_finite_num(ema) and (
            self._best_ema is None
            or not _is_finite_num(self._best_ema)
            or ema > float(self._best_ema)
        ):
            self._best_ema = ema
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
                )
                if allow_ckpt and not self.disable_checkpointing:
                    self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))
        elif self.local_rank == 0 and not _is_finite_num(ema):
            self.print_to_log_file(
                "[Task1StdTrainVal50] skip best update (ema_fg_dice is non-finite)"
            )

        if self.local_rank == 0:
            try:
                self.logger.plot_progress_png(self.output_folder)
            except Exception as e:
                self.print_to_log_file(f"plot_progress_png skipped: {e}")

        finished_epoch = int(self.current_epoch)
        self.current_epoch += 1
        self._task1_maybe_save_periodic_checkpoint(allow_ckpt)

        if not _env_truthy("TASK1_SEGMENT_CHECKPOINT", "1"):
            return
        if self.local_rank != 0 or self.disable_checkpointing or not allow_ckpt:
            return
        ce = int(finished_epoch)
        seg_every = _env_int_nonempty("TASK1_SEGMENT_CHECKPOINT_EVERY", 100)
        seg_lo = (ce // seg_every) * seg_every
        seg_hi = seg_lo + seg_every - 1
        if self._task1_seg_base != seg_lo:
            self._task1_seg_base = seg_lo
            self._task1_seg_best_ema = None
        improved_seg = _is_finite_num(ema) and (
            self._task1_seg_best_ema is None
            or not _is_finite_num(self._task1_seg_best_ema)
            or ema > float(self._task1_seg_best_ema)
        )
        if improved_seg:
            self._task1_seg_best_ema = ema
        if improved_seg and self._task1_seg_checkpoint_epoch_in_range(finished_epoch):
            fn = join(
                self.output_folder,
                f"checkpoint_best_seg{seg_lo:04d}_{seg_hi:04d}.pth",
            )
            self.save_checkpoint(fn)
            self.print_to_log_file(
                f"[Task1StdTrainVal50] segment best (seg {seg_lo}-{seg_hi}, ema={ema:.4f}) -> {fn}"
            )
        if self.current_epoch > 0 and self.current_epoch % seg_every == 0:
            fn = join(
                self.output_folder,
                f"checkpoint_final_ep{self.current_epoch:04d}.pth",
            )
            self.save_checkpoint(fn)
            self.print_to_log_file(
                f"[Task1StdTrainVal50] periodic full snapshot ep{self.current_epoch} -> {fn}"
            )

    def _task1_live_progress_path(self) -> str:
        return join(self.output_folder, "task1_train_live_progress.json")

    def _task1_live_progress_interval(self) -> int:
        v = os.environ.get("TASK1_TRAIN_PROGRESS_INTERVAL")
        if v is None or str(v).strip() == "":
            return 1
        return max(1, int(str(v).strip()))

    def _task1_should_write_live_progress(self, iter_idx: int, iter_total: int) -> bool:
        if not _env_truthy("TASK1_TRAIN_LIVE_PROGRESS", "1"):
            return False
        if self.local_rank != 0:
            return False
        iv = self._task1_live_progress_interval()
        done = iter_idx + 1
        return done == 1 or done >= iter_total or done % iv == 0

    def _task1_write_live_progress(self, phase: str, iter_done: int, iter_total: int) -> None:
        payload = {
            "epoch": int(self.current_epoch),
            "phase": phase,
            "iter": int(iter_done),
            "iter_total": int(iter_total),
            "num_epochs": int(self.num_epochs),
            "updated_unix": time(),
        }
        if self._task1_epoch_start_unix is not None:
            payload["epoch_start_unix"] = float(self._task1_epoch_start_unix)
        try:
            with open(self._task1_live_progress_path(), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        except OSError:
            pass

    def _task1_periodic_checkpoint_every(self) -> int:
        v = os.environ.get("TASK1_PERIODIC_CHECKPOINT_EVERY")
        if v is not None and str(v).strip() != "":
            return max(0, int(str(v).strip()))
        if _env_truthy("TASK1_SEGMENT_CHECKPOINT", "1"):
            return _env_int_nonempty("TASK1_SEGMENT_CHECKPOINT_EVERY", 100)
        return 0

    def _task1_maybe_save_periodic_checkpoint(self, allow_ckpt: bool) -> None:
        if (
            not allow_ckpt
            or self.local_rank != 0
            or self.disable_checkpointing
        ):
            return
        periodic = self._task1_periodic_checkpoint_every()
        if periodic <= 0:
            return
        ce = int(self.current_epoch)
        if ce > 0 and ce % periodic == 0:
            fn = join(
                self.output_folder,
                f"checkpoint_final_ep{ce:04d}.pth",
            )
            self.save_checkpoint(fn)
            self.print_to_log_file(
                f"[Task1StdTrainVal50] periodic full snapshot ep{ce} -> {fn}"
            )

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
                if self._task1_should_write_live_progress(
                    batch_id, self.num_iterations_per_epoch
                ):
                    self._task1_write_live_progress(
                        "train", batch_id + 1, self.num_iterations_per_epoch
                    )
            self.on_train_epoch_end(train_outputs)

            n_val = self._task1_fdg_val_iters()
            run_val = n_val > 0 and self._task1_should_run_val_this_epoch()
            self._task1_val_ran_this_epoch = False
            if run_val:
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = self._task1_run_fullcase_val_outputs()
                    self.on_validation_epoch_end(val_outputs)
                self._task1_val_ran_this_epoch = True
            elif self.local_rank == 0:
                every = self._task1_val_every_n_epochs()
                reason = (
                    f"fdg_val_iters={n_val}, base={self.num_val_iterations_per_epoch}"
                    if n_val <= 0
                    else (
                        f"every_n={every}, finished_ep={int(self.current_epoch) + 1}"
                    )
                )
                self.print_to_log_file(
                    f"[Task1StdTrainVal50] skipping validation ({reason})"
                )

            self._task1_run_psma_val_if_needed()
            self.on_epoch_end()

        self.on_train_end()

    def perform_actual_validation(self, save_probabilities: bool = False):
        if not _env_truthy("TASK1_ALLOW_FULL_VOLUME_VAL", default="0"):
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1StdTrainVal50] Skipping perform_actual_validation (full-volume val). "
                    "Set TASK1_ALLOW_FULL_VOLUME_VAL=1 to enable."
                )
            return
        return super().perform_actual_validation(save_probabilities)
