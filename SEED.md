# Reproducibility seeds

Aligned board protocol uses **seed = 42** everywhere by default:

| Knob | Default | Where |
|------|--------:|-------|
| Stratified / few-shot split export | 42 | `scripts/make_stratified_splits_dataset1.py`, `export_mae_psma_fewshot*.py` |
| `TASK1_TRAIN_SEED` | 42 | `nnUNetTrainer_Task1StdTrainVal50` (python/numpy/torch/CUDA) |
| `TASK1_RAM_SHARD_SEED` | 42 | `task1_ram_shard.py` |
| `TASK1_FEWSHOT_SPLIT_SEED` | 42 | `run/_aligned_seed_env.sh` |

Override only when intentionally ablating seed; set `TASK1_TRAIN_SEED=none` to disable the trainer global seed.
