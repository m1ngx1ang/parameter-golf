# SP8192 Prelude2-Recur3-Post2 Loop LoRA

This record tests a stage-structured depth-recurrent architecture while keeping the existing SP8192 `Block` implementation.

## Architecture

Default physical layout is `2 + 3 + 2` layers:

- Prelude: layers `0,1` (run once)
- Recurrent core: layers `2,3,4` (run `NUM_RECUR_LOOPS + 1` passes when looping is enabled)
- Postlude: layers `5,6` (run once)

Key behaviors:

- U-net bridge between Prelude and Postlude is kept (`skip_weights` + `skip_gates`) for gradient flow.
- Recurrent update uses both previous hidden state and encoded prelude input:
  - `h_next = A*h_prev + B*e + core(h_prev, e)`
  - `A` is constrained to `(0,1)` via `recur_carry_log`.
  - `B` is learned via `recur_input_scale`.
- Block residual mixing uses baseline-style `resid_mix` (no `stable_resid_mix` branch).
- LoRA is applied to MLP input projection (`fc`) in recurrent blocks, specialized per `(loop_pass, recurrent_block)`.

## Default Knobs

- `PRELUDE_LAYERS=2`
- `RECURRENT_LAYERS=3`
- `POSTLUDE_LAYERS=2`
- `NUM_RECUR_LOOPS=2`
- `ENABLE_LOOPING_AT=0.35`
- `LOOP_SETTLE_FRAC=0.10`
- `RECUR_CARRY_INIT=0.99`
- `RECUR_INPUT_SCALE_INIT=0.0`
- `PRELUDE_MLP_MULT=4.0`
- `RECURRENT_MLP_MULT=6.0`
- `POSTLUDE_MLP_MULT=4.0`
- `MLP_LORA_RANK=8`
- `MLP_LORA_ALPHA=8.0`
- `MLP_LORA_LR=0.02`
- `MLP_LORA_WD=0.0`
- `MLP_LORA_WARMUP_STEPS=400`

`NUM_LAYERS` is derived internally as `PRELUDE_LAYERS + RECURRENT_LAYERS + POSTLUDE_LAYERS` and is no longer configured directly.
LoRA placement is fixed to recurrent physical blocks only.
`MLP_MULT` is removed in this variant; use stage-specific multipliers.

## Multiplier Budget Notes

With this 7-layer physical layout, we can afford a larger recurrent MLP while keeping pre/post conservative:

- `PRELUDE/POSTLUDE=4, RECURRENT=6`: recommended default.
- `PRELUDE/POSTLUDE=4, RECURRENT=8`: aggressive, budget-risky.

Rough model-size trend from this folder's architecture:

- `4/4/4`: ~24.6M params
- `4/6/4`: ~32.0M params
- `4/8/4`: ~39.4M params

Suggested sweep order under tight compute credit:

1. `RECURRENT_MLP_MULT=6` with defaults
2. `RECURRENT_MLP_MULT=8` only if step 1 remains comfortably under artifact/time budgets

## Training

Full run:

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Recommended run (default stage multipliers):

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Aggressive run (budget-risky):

```bash
SEED=42 RECURRENT_MLP_MULT=8 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Short smoke (single GPU):

```bash
DATA_DIR=/home/max/parameter-golf/data \
RUN_ID=sp8192_pre2_rec3_post2_smoke \
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

```bash
records/track_10min_16mb/2026-04-27_SP8192_Prelude2_Recur3_Post2_LoopLoRA/launch_8xh100_runpod.sh --dry-run
records/track_10min_16mb/2026-04-27_SP8192_Prelude2_Recur3_Post2_LoopLoRA/launch_8xh100_runpod.sh
```

Outputs:

- `logs/<RUN_ID>.txt`
- `final_model.pt`
- `final_model.int6.ptz`
- runpod-collected artifacts under `artifacts/`
