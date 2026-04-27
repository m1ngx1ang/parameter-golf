# SP8192 Dual-Loop (3-4x1, 6-7x1) No-LoRA

This record is based on `2026-04-12_SP8192_RecurMLPLoRA/train_gpt.py` with LoRA fully removed.

## Architecture Change

The recurrent plan now uses two loop segments:

- segment `3-4` repeated `1` extra time
- segment `6-7` repeated `1` extra time

Default loop envs in this record:

- `NUM_LOOPS1=1`
- `LOOP1_START=3`
- `LOOP1_END=4`
- `NUM_LOOPS2=1`
- `LOOP2_START=6`
- `LOOP2_END=7`
- `ENABLE_LOOPING_AT=0.35`

## Training

Example local launch:

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Smoke run:

```bash
DATA_DIR=/home/max/parameter-golf/data \
RUN_ID=sp8192_dualloop_smoke \
SEED=42 \
ITERATIONS=200 \
WARMUP_STEPS=0 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_SEQ_LEN=256 \
EVAL_SEQ_LEN=256 \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=20 \
SLIDING_WINDOW_ENABLED=0 \
GPTQ_CALIBRATION_BATCHES=1 \
GPTQ_RESERVE_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Runpod

Dry run:

```bash
records/track_10min_16mb/2026-04-27_SP8192_DualLoop34x1_67x1_NoLoRA/launch_8xh100_runpod.sh --dry-run
```

Launch:

```bash
records/track_10min_16mb/2026-04-27_SP8192_DualLoop34x1_67x1_NoLoRA/launch_8xh100_runpod.sh
```

The launcher keeps SP8192 data setup unchanged:

- `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf`
- `DATA_TRAIN_SHARDS=128` by default

## Logs and Artifacts

- Train/eval logs: `logs/<RUN_ID>.txt`
- Export: `final_model.pt`
- Quantized export: `final_model.int6.ptz`
- Runpod collected outputs: `artifacts/`
