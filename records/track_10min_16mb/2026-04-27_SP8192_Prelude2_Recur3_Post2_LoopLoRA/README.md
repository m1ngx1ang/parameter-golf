# SP8192 Looped Transformer (Parcae-style)

Stage-structured looped transformer adapted from arXiv:2604.12946 (Parcae). Reuses the
existing SP8192 `Block` (pre-norm transformer with `resid_mix` anchor) — no new modules
introduced. The recurrent "block" is a chain of plain transformer blocks; depth is
randomized per microbatch.

## Architecture

Default physical layout `2 + 3 + 2`:

- Prelude: blocks `0..1` (run once, anchor `x0 = post-embed rms_norm`)
- `encoded = LN(prelude_output)` (RMSNorm — paper's `e = LN(P(s))` for stability)
- Recurrent block: blocks `2..4` chained (looped `T` times; T sampled per microbatch)
- Postlude: blocks `5..6` (run once with U-Net skips back to prelude)

Recurrent loop body, applied for `t = 0 ... T-1` (paper formula
`x_{t+1} = decay·x_t + inject·e + R̄(x_t,e)` realized as
`x_{t+1} = R̄(decay·x_t + inject·e)` since pre-norm blocks emit the `+R̄(·)` term
through their residual stream):

```
h ← carry · h + inject · encoded                        # adapter
x0 ← h                                                   # block anchor inside recurrent stack
for i in recurrent_indices:
    h ← Block_i(h, x0)                                  # plain transformer chain
```

- `carry = exp(-exp(recur_carry_log))` ∈ (0, 1) per channel — the diagonal A
- `inject = recur_input_scale` per channel — the B-projection scalar
- `h_0 = encoded` (default) or zeros, controlled by `RECUR_H0_FROM_ENCODED`

U-net skip bridges between Prelude and Postlude (`skip_weights` + `skip_gates`) are
preserved.

## Random recurrent depth

Depth `T` is sampled per microbatch from `RECUR_DEPTHS` (default `2,3,4`). The full
microbatch schedule for one optimizer step is sampled once on rank 0 using a dedicated,
seeded `torch.Generator`, broadcast as a `[grad_accum_steps]` tensor to every rank, and
materialized into a Python list with a single GPU→CPU sync. Depth then flows into the
model as an explicit `recur_depth` kwarg on `forward()` / `forward_logits()`, so
`torch.compile(dynamic=False, fullgraph=True)` specializes one cached graph per depth
value.

Warmup runs `WARMUP_STEPS` steps at each unique depth in `RECUR_DEPTHS ∪ {RECUR_DEPTH_EVAL}`,
so every depth is compiled before real training begins. Model and optimizer state are
restored from a CPU snapshot after warmup.

Eval (pre-quant, post-quant, sliding-window, TTT) uses a fixed `recur_depth = RECUR_DEPTH_EVAL`
(default 3).

## Quantization-aware calibration

`collect_hessians` cycles `recur_depth` through `RECUR_DEPTHS` round-robin during
calibration so the Hessians average activation distributions across the depths the
quantized model will see at eval.

## Default knobs

- `PRELUDE_LAYERS=2`, `RECURRENT_LAYERS=3`, `POSTLUDE_LAYERS=2`
- `PRELUDE_MLP_MULT=4.0`, `RECURRENT_MLP_MULT=4.0`, `POSTLUDE_MLP_MULT=4.0`
- `RECUR_DEPTHS=2,3,4`, `RECUR_DEPTH_EVAL=3`
- `RECUR_CARRY_INIT=0.99`, `RECUR_INPUT_SCALE_INIT=0.01`
- `RECUR_H0_FROM_ENCODED=1`, `RECUR_BPTT_TAIL=0` (full BPTT)
- `ENCODED_LN_ENABLED=1`
- `PARALLEL_RESIDUAL_START=5`

`MLP_MULT` is removed in this variant (use stage-specific multipliers). All LoRA
knobs (`MLP_LORA_*`) and the dual-loop knobs (`NUM_RECUR_LOOPS`, `ENABLE_LOOPING_AT`,
`LOOP_SETTLE_FRAC`) are gone.

## Training

Full run:

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Short smoke (single GPU):

```bash
DATA_DIR=/home/max/parameter-golf/data \
RUN_ID=sp8192_looped_smoke \
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
- `final_model.int${MATRIX_BITS}.ptz` (e.g. `final_model.int6.ptz` with `MATRIX_BITS=6`)
- runpod-collected artifacts under `artifacts/`
