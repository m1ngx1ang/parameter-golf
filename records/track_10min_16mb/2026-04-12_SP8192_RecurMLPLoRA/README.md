# SP8192 Recurrent MLP LoRA

Experimental SP8192 trainer built from the recurrence-based 10 minute / 16 MiB stack. The current implementation uses one consolidated model width (`MODEL_DIM`) and adds small repeated-pass LoRA adapters to recurrent MLP output projections.

## Current Implementation

The model defaults to an 11 layer, 512 dim, SP8192 transformer with tied token embeddings. `MODEL_DIM` is the single hidden and embedding width; `EMBEDDING_DIM` is no longer a separate supported model dimension. If a legacy `EMBEDDING_DIM` environment variable is set to a different value than `MODEL_DIM`, the script fails early with a clear error.

Recurrence is enabled by default with `NUM_LOOPS=2`, `LOOP_START=3`, and `LOOP_END=5`. The physical loop segment is reused in the encoder/decoder execution plan, while skip weights and optional skip gates bridge encoder outputs into decoder passes.

Repeated-pass MLP LoRA is attached only to repeated virtual occurrences of physical loop layers. The first use of a physical layer uses the shared base MLP. Later repeated uses add a small LoRA delta in parallel with the MLP output projection:

```text
hidden = leaky_relu(fc(x))^2
out = proj(hidden) + lora_up(lora_down(hidden)) * alpha / rank
```

The base MLP remains trainable; there is no current `FREEZE_REPEATED_MLP` path. LoRA parameters are optimized by a separate AdamW group and remain small enough to pass through export in fp16 rather than being GPTQ-quantized.

## Default Knobs

- `VOCAB_SIZE=8192`
- `MODEL_DIM=512`
- `NUM_LAYERS=11`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=4.0`
- `NUM_LOOPS=2`
- `LOOP_START=3`
- `LOOP_END=5`
- `ENABLE_LOOPING_AT=0.35`
- `MLP_LORA_RANK=4`
- `MLP_LORA_ALPHA=4.0`
- `MLP_LORA_LR=0.04`
- `MLP_LORA_WD=0.0`
- `ENABLE_MLP_LORA_AT=$ENABLE_LOOPING_AT` by default
- `PARALLEL_RESIDUAL_START=7`
- `STABLE_RESID_MIX=1`
- `QK_GAIN_INIT=5.25`
- `MATRIX_BITS=6`
- `EMBED_BITS=8`
- `COMPRESSOR=brotli`

If `MLP_LORA_LAYERS` is empty, adapters target the loop segment itself: `3,4,5`. With the default loop plan this creates 6 repeated-pass adapters. Requested LoRA rank is clipped when needed so each adapter tensor stays under the passthrough tensor threshold.

## Training

Full default run:

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Short local smoke:

```bash
DATA_DIR=/home/max/parameter-golf/data \
RUN_ID=sp8192_smoke \
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
MLP_LORA_RANK=8 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

No-LoRA comparison:

```bash
SEED=42 MLP_LORA_RANK=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Higher-rank adapter sweep:

```bash
SEED=42 MLP_LORA_RANK=8 MLP_LORA_ALPHA=8 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Runpod Launch

The experiment-local launcher targets this renamed directory directly:

```bash
records/track_10min_16mb/2026-04-12_SP8192_RecurMLPLoRA/launch_8xh100_runpod.sh --dry-run
records/track_10min_16mb/2026-04-12_SP8192_RecurMLPLoRA/launch_8xh100_runpod.sh
```

The launcher downloads SP8192 assets from `kevclark/parameter-golf`, runs the 8xH100 SOTA-aligned training/eval command, and collects:

- `records/track_10min_16mb/2026-04-12_SP8192_RecurMLPLoRA/logs`
- `records/track_10min_16mb/2026-04-12_SP8192_RecurMLPLoRA/artifacts`

## Export And Eval

The script writes:

- `final_model.pt`: full precision checkpoint
- `final_model.int6.ptz`: compressed mixed-precision submission artifact
- `logs/<RUN_ID>.txt`: training, eval, quantization, and environment log

Large MLP and attention matrices use GPTQ int6 by default. Tied token embeddings use int8. Small control tensors, skip tensors, and LoRA adapters pass through as fp16. If `brotli` is unavailable, compression falls back to `lzma` while preserving a codec header so immediate deserialize/eval still works.

Post-training eval flow is:

```text
pre-quantization post-ema eval
GPTQ collection and quantized serialization
quantized eval
optional quantized sliding-window eval
optional quantized TTT eval
```

## Notes

- `flash_attn_interface` is used when available; otherwise the script falls back to PyTorch SDPA.
- `SLIDING_WINDOW_ENABLED=1` is the default for full evaluation, but smoke configs disable it to keep runtime short.
- Historical logs in this directory may mention older fields such as `FREEZE_REPEATED_MLP`, `embedding_dim`, or `repeated_pass_mlp_loras`. Those logs predate the current simplified implementation.
- This directory is an experiment snapshot, not a finalized record claim.
