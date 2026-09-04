# ICLR2026

新研究工作区（与 autoPET~V 提交管线分离）。

## dataset1 分层划分

脚本：`scripts/make_stratified_splits_dataset1.py`  
输出：`data/splits_stratified_70_10_20.json`

### 类别（每个 tracer 三类）

| Tracer | small | large | negative | 含义 |
|--------|------:|------:|---------:|------|
| PSMA   | 65%   | 25%   | 10%      | negative=空 GT；非空按体积分位，低体积端为 small |
| FDG    | 15%   | 35%   | 50%      | 同上（FDG 的 small 对应既往 tiny 语义） |

比例通过「空 GT → negative + 非空体积分位」自然逼近目标混合比（dataset1 上空 GT 比例本身接近 PSMA 10% / FDG 50%）。

### 划分

每个 `(tracer, role)` 层内独立随机打乱后按 **70% train / 10% val / 20% test** 切分（`seed=42`）。

### 生成

```bash
docker run --rm --entrypoint python3 \
  -v /media/ybwang/data1/PSMA-CTRL:/media/ybwang/data1/PSMA-CTRL \
  -v /media/ybwang/data1/PSMA-DATA:/media/ybwang/data1/PSMA-DATA \
  -w /media/ybwang/data1/PSMA-CTRL \
  autopet_baseline \
  ICLR2026/scripts/make_stratified_splits_dataset1.py \
  --dataset-dir /media/ybwang/data1/PSMA-DATA/dataset1 \
  --out-json ICLR2026/data/splits_stratified_70_10_20.json \
  --seed 42 --workers 16
```

## baseline1（FDG train）

定义：仅用分层划分中的 **FDG train**（+ FDG val 选优），训练设定对齐 2 通道 `3d_fullres` 主训。

| 项 | 值 |
|----|----|
| 数据 | FDG train=711 / FDG val=101（`data/splits_baseline1_fdg_nnunet.json`） |
| 骨干 | PlainConvUNet · Dataset228 · CT+PET · `3d_fullres` |
| GPU / batch | 0,1,3 · 全局 bs=6（2×3） |
| iters | 70 train / 10 val per epoch |
| epochs | 3000 |
| best | **min val_loss**（`TASK1_BEST_BY=val_loss`，`TASK1_VAL_LOSS_ONLY=1`） |
| PSMA monitor | **ep≥2000** 额外 `PSMA_val_loss`（分层 PSMA val=59；不参与 best） |
| val steps | 前半 FDG val=**10**；**ep≥2000** FDG+PSMA val 均 **50** |
| loss 图 | `ICLR2026/vis/`：前段 train+val；≥2000 起三折线（+`PSMA_val_loss`） |

导出 splits：

```bash
python3 ICLR2026/scripts/export_baseline1_fdg_splits.py
python3 ICLR2026/scripts/export_baseline1_psma_val.py
```

启动：

```bash
bash ICLR2026/run/run_baseline1_fdg_2ch_fullres_3000ep_bg.sh
```

停止（训练 + 续训 guard）：

```bash
TASK1_NNUNET_RESULTS_STAMP_NAME=<STAMP> bash scripts/task1_stop_train_and_resume.sh
```

## baseline2（FDG→PSMA 伪标微调）

定义：FDG `checkpoint_best` 仅作初始化；对分层 **PSMA train=421** 做**一次**伪标，再微调 **2000 ep**（tr70/val50）；**PSMA val=59 真 GT** 监控/选优。

| 项 | 值 |
|----|----|
| 启动 | `ICLR2026/run/run_baseline2_psma_uda_oneshot_2000ep_bg.sh` |
| splits | `data/splits_baseline2_psma_uda_nnunet.json` |
| 初始化 | baseline1 FDG `checkpoint_best.pth` |
| 伪标 | 一次 CC+μ/h/λ（可 `TASK1_UDA_SKIP_PSEUDO=1` 复用已有 b2nd） |
| 训练 | **2000 ep**，tr70/val50，`TASK1_INITIAL_LR=1e-3`，best=`val_loss` |
| loss 图 | `ICLR2026/vis/loss_curve_iclr2026_baseline2_oneshot_<STAMP>.png` |

```bash
# 复用已有 round0 伪标 b2nd 直接开训：
export TASK1_UDA_SKIP_PSEUDO=1
export TASK1_PSEUDO_SEG_B2ND_DIR=/media/ybwang/data1/PSMA-DATA/task1_train_workspace/nnUNet_results/<OLD>/round_000/pseudo_seg_b2nd
bash ICLR2026/run/run_baseline2_psma_uda_oneshot_2000ep_bg.sh
```

旧版 200-round 自训练脚本仍保留：`run_baseline2_psma_uda_selftrain_bg.sh`（一般不再用）。

停止：

```bash
TASK1_NNUNET_RESULTS_STAMP_NAME=<STAMP> bash scripts/task1_stop_train_and_resume.sh
```

产物：`nnUNet_results/<STAMP>/`（含 fold checkpoint）+ `ICLR2026/vis/loss_curve_iclr2026_baseline2_oneshot_<STAMP>.png`。
