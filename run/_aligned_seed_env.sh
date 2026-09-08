# Aligned board reproducibility — source from run scripts.
# All RNG seeds default to 42 unless explicitly overridden.
export TASK1_TRAIN_SEED="${TASK1_TRAIN_SEED:-42}"
export TASK1_RAM_SHARD_SEED="${TASK1_RAM_SHARD_SEED:-42}"
export TASK1_FEWSHOT_SPLIT_SEED="${TASK1_FEWSHOT_SPLIT_SEED:-42}"
