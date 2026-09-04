"""
Task1 方案一：冻结 nnUNet 教师（PlainConvUNet checkpoint）+ 默认管线知识蒸馏训练 3D 学生。

学生网络由 ``task1_mamba.student_3d.build_task1_student_3d`` 构建；默认 ``TASK1_MAMBA_STUDENT=vmamba3d``（``task1_mamba/vmamba3d_student.py``，三正交平面 2D 混合 + 3D U-Net）。
可选 ``plain_shrink``（与 plans 同拓扑通道减半 PlainConvUNet）、``tri_scan_unet``（三轴深度可分混合块）。

3D 体素下的 VMamba 风格学生见 ``vmamba3d`` 实现；CIPA 双模态 Sigma（dual_vmamba）仍为 2D 切片管线：
``bash run_task1_vmamba_student_cipa.sh``。

无特别说明时，与 ``run_task1_mamba_kd.sh`` / ``task1_train_nnunet_from_dataset1.sh``（本 TRAINER）对齐的约定默认：
300 epoch、3d_fullres patch ``[112,160,128]``、全局 batch ``2×`` 训练卡数（默认 3 卡→6）、每 epoch train/val 迭代 ``83/10``（均可被对应 ``TASK1_*`` 环境变量覆盖）。

环境变量（常用）:
  TASK1_MAMBA_TEACHER_CKPT   教师 ``checkpoint_final.pth``（或 latest），必填
  TASK1_MAMBA_KD_WEIGHT      蒸馏项权重，默认 1.0；0 关闭 KD（仅分割 loss）
  TASK1_MAMBA_KD_TEMP        温度 T，默认 2.0
  TASK1_MAMBA_KD_MODE        kl | mse，默认 kl
  TASK1_MAMBA_INITIAL_LR     PolyLR 初始学习率，默认 **1e-4**（较 nnUNet 默认 0.01 更稳，利于 KD+新骨干）
  TASK1_MAMBA_LOSS_NORM      1/0：是否对 seg / KD 分项做 detach 量级归一后再加权求和（默认 1，避免标量 loss 极大引发 FP16/梯度问题）
  TASK1_MAMBA_TEACHER_DS     1/0 教师是否按 plans 开 deep supervision（与 checkpoint 一致，默认 1）
  TASK1_MAMBA_STUDENT        vmamba3d（默认）| plain_shrink | tri_scan_unet
  TASK1_MAMBA_TRISCAN_BASE    tri_scan_unet / vmamba3d 的 base 通道，默认 32
  TASK1_MAMBA_VMAMBA3D_BASE   仅 vmamba3d：覆盖 base（未设时同 TRISCAN_BASE）
  TASK1_MAMBA_VMAMBA3D_PLANAR_K  仅 vmamba3d：平面 DW 卷积核，默认 5

训练节奏与 ``nnUNetTrainer_Task1StdTrainVal50`` 对齐（本文件独立继承 ``nnUNetTrainer``，便于 Docker 仅挂载单文件）。
"""

from __future__ import annotations

import os
from copy import deepcopy

import torch
from torch import autocast, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_int_nonempty(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return max(1, int(str(v).strip()))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(str(raw).strip())


def _env_loss_norm_on() -> bool:
    """默认开启 loss 归一；TASK1_MAMBA_LOSS_NORM=0|false 关闭。"""
    v = os.environ.get("TASK1_MAMBA_LOSS_NORM", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _unwrap_network(net: nn.Module) -> nn.Module:
    mod = net.module if isinstance(net, DDP) else net
    if hasattr(mod, "_orig_mod"):
        mod = mod._orig_mod  # type: ignore[assignment]
    return mod


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class nnUNetTrainer_Task1MambaKD(nnUNetTrainer):
    """标准 Task1 训练节奏 + 可选 logits 蒸馏（冻结教师）。"""

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
        self.num_val_iterations_per_epoch = _env_int_nonempty(
            "TASK1_VAL_ITERS_PER_EPOCH", 10
        )
        v_epochs = os.environ.get("TASK1_NUM_EPOCHS")
        if v_epochs is not None and str(v_epochs).strip() != "":
            self.num_epochs = max(1, int(str(v_epochs).strip()))

        self.enable_deep_supervision = False
        self._teacher_network: nn.Module | None = None
        self._kd_weight = _env_float("TASK1_MAMBA_KD_WEIGHT", 1.0)
        self._kd_temp = _env_float("TASK1_MAMBA_KD_TEMP", 2.0)
        self._kd_mode = os.environ.get("TASK1_MAMBA_KD_MODE", "kl").strip().lower() or "kl"
        self._loss_norm = _env_loss_norm_on()
        # 覆盖 nnUNet 默认 initial_lr（常为 0.01）；KD + 新学生骨干建议更小起步
        self.initial_lr = _env_float("TASK1_MAMBA_INITIAL_LR", 1e-4)
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1MambaKD] "
                f"num_iterations_per_epoch={self.num_iterations_per_epoch}, "
                f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}, "
                f"num_epochs={self.num_epochs}, "
                f"enable_deep_supervision={self.enable_deep_supervision}, "
                f"initial_lr={self.initial_lr}, loss_norm={self._loss_norm}, "
                f"kd_weight={self._kd_weight}, kd_temp={self._kd_temp}, kd_mode={self._kd_mode}"
            )

    def perform_actual_validation(self, save_probabilities: bool = False):
        if not _env_truthy("TASK1_ALLOW_FULL_VOLUME_VAL", default="0"):
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1MambaKD] Skipping perform_actual_validation (full-volume val). "
                    "Set TASK1_ALLOW_FULL_VOLUME_VAL=1 to enable."
                )
            return
        return super().perform_actual_validation(save_probabilities)

    @staticmethod
    def build_network_architecture(*args, **kwargs) -> nn.Module:
        """
        nnUNet ``initialize`` 用 ``inspect.signature`` 判断是否走「新」5 参 API；部分环境下对子类
        ``@staticmethod`` 检测不到 ``plans_manager``，会退回旧 6 参调用。此处两种都支持。
        """
        del kwargs  # nnUNet 当前不传 kwargs
        from task1_mamba.student_3d import (
            build_task1_student_3d,
            build_tri_scan_unet_student,
        )
        from task1_mamba.vmamba3d_student import build_vmamba3d_unet_student

        if len(args) == 6 and isinstance(args[0], str):
            (
                network_arch_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                _enable_deep_supervision,
            ) = args
            kind = str(os.environ.get("TASK1_MAMBA_STUDENT", "vmamba3d")).strip().lower()
            if kind in ("tri_scan", "tri_scan_unet", "mamba_lite"):
                return build_tri_scan_unet_student(num_input_channels, num_output_channels)
            if kind in ("vmamba", "vmamba3d", "vmamba_3d"):
                return build_vmamba3d_unet_student(num_input_channels, num_output_channels)
            arch = deepcopy(arch_init_kwargs) if isinstance(arch_init_kwargs, dict) else arch_init_kwargs
            if isinstance(arch, dict):
                fs = arch.get("features_per_stage")
                if fs is not None and kind in ("plain_shrink", "plain", "shrink", ""):
                    arch["features_per_stage"] = [max(8, int(c) // 2) for c in fs]
            return get_network_from_plans(
                network_arch_class_name,
                arch,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                allow_init=True,
                deep_supervision=False,
            )

        if len(args) == 5:
            _pm, configuration_manager, num_input_channels, num_output_channels, _eds = args
            if not isinstance(configuration_manager, ConfigurationManager):
                raise TypeError(
                    "build_network_architecture(5-arg): 期望 args[1] 为 ConfigurationManager，"
                    f"实为 {type(configuration_manager).__name__}"
                )
            return build_task1_student_3d(
                configuration_manager, num_input_channels, num_output_channels
            )

        raise TypeError(
            "build_network_architecture: 不支持的参数组合 "
            f"len(args)={len(args)} first={type(args[0]).__name__ if args else None}"
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = _unwrap_network(self.network)
        if hasattr(mod, "decoder") and hasattr(mod.decoder, "deep_supervision"):
            return super().set_deep_supervision_enabled(enabled)
        if self.local_rank == 0:
            self.print_to_log_file(
                "[Task1MambaKD] skip set_deep_supervision_enabled (student has no decoder.deep_supervision)"
            )
        return None

    def initialize(self):
        super().initialize()
        # 学生子图 / KD 与部分 rank 上未参与 loss 的参数会触发 DDP「未完成 reduction」；重包一层 find_unused_parameters。
        if self.is_ddp and isinstance(self.network, DDP):
            inner = self.network.module
            self.network = DDP(
                inner,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )
            if self.local_rank == 0:
                self.print_to_log_file(
                    "[Task1MambaKD] DDP re-wrapped with find_unused_parameters=True"
                )
        ckpt = os.environ.get("TASK1_MAMBA_TEACHER_CKPT", "").strip()
        if not ckpt:
            raise RuntimeError(
                "TASK1_MAMBA_TEACHER_CKPT 未设置：请指向教师 nnUNet checkpoint_final.pth（含 network_weights）。"
            )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"TASK1_MAMBA_TEACHER_CKPT 不是文件: {ckpt}")

        teacher_ds = _env_truthy("TASK1_MAMBA_TEACHER_DS", "1")
        self._teacher_network = get_network_from_plans(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.label_manager.num_segmentation_heads,
            allow_init=True,
            deep_supervision=teacher_ds,
        ).to(self.device)
        blob = _torch_load(ckpt, map_location=str(self.device))
        if "network_weights" not in blob:
            raise KeyError(
                f"checkpoint 缺少 network_weights 键: {ckpt}（键: {list(blob.keys())[:12]}...）"
            )
        missing, unexpected = self._teacher_network.load_state_dict(
            blob["network_weights"], strict=False
        )
        if self.local_rank == 0:
            if missing or unexpected:
                self.print_to_log_file(
                    f"[Task1MambaKD] teacher load_state_dict strict=False "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            self.print_to_log_file(
                f"[Task1MambaKD] loaded teacher from {ckpt} teacher_deep_supervision={teacher_ds}"
            )
        self._teacher_network.eval()
        for p in self._teacher_network.parameters():
            p.requires_grad_(False)

    def _combine_seg_kd_loss(
        self, l_seg: torch.Tensor, l_kd: torch.Tensor | None
    ) -> torch.Tensor:
        """分割项与 KD 项组合；可选按 detach 绝对值缩放到 O(1) 量级再反传，减轻超大标量 loss。"""
        if not self._loss_norm:
            if l_kd is None or self._kd_weight == 0.0:
                return l_seg
            return l_seg + self._kd_weight * l_kd
        if l_kd is None or self._kd_weight == 0.0:
            s = torch.clamp(l_seg.detach().abs(), min=1.0)
            return l_seg / s
        s_seg = torch.clamp(l_seg.detach().abs(), min=1.0)
        s_kd = torch.clamp(l_kd.detach().abs(), min=1.0)
        return (l_seg / s_seg) + self._kd_weight * (l_kd / s_kd)

    def train_step(self, batch: dict) -> dict:
        from task1_mamba.kd_loss import kd_loss_logits

        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        use_cuda = self.device.type == "cuda"
        with autocast(self.device.type, enabled=use_cuda) if use_cuda else dummy_context():
            output = self.network(data)
            l_seg = self.loss(output, target)
            if self._kd_weight != 0.0 and self._teacher_network is not None:
                with torch.no_grad():
                    t_out = self._teacher_network(data)
                t_logits = t_out[0] if isinstance(t_out, (list, tuple)) else t_out
                s_logits = output[0] if isinstance(output, (list, tuple)) else output
                l_kd = kd_loss_logits(
                    s_logits,
                    t_logits,
                    mode=self._kd_mode,
                    temperature=self._kd_temp,
                )
                l = self._combine_seg_kd_loss(l_seg, l_kd)
            else:
                l = self._combine_seg_kd_loss(l_seg, None)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}
