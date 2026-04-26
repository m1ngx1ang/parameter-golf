# SP4096 + 3-Layer Recurrence + Repeated-Pass MLP LoRA

This is an experimental trainer derived from the current recurrence-based SOTA stack, but with defaults changed to `SP4096` and a new **repeated-pass MLP LoRA** path for recurrent layers.

## Key Idea

Instead of fully tying the repeated recurrent passes, or fully untying them, this script adds a small LoRA refinement only to the **MLP down projection** on repeated passes:

- first use of a recurrent physical layer: shared base MLP
- repeated use of that same physical layer: shared base MLP + LoRA delta
- when the LoRA path is active, the repeated-pass base MLP weights are used with `detach()`, so specialization is pushed into the low-rank branch instead of back into the shared weights

The design is deliberately narrow:

- `SP4096` default to leave more artifact headroom than `SP8192`
- LoRA only on recurrent MLP passes
- no LoRA on attention
- delayed activation, with recurrence turning on first and LoRA turning on a bit later for a smoother transition

## Defaults

- `VOCAB_SIZE=4096`
- `NUM_LOOPS=2`
- `LOOP_START=3`
- `LOOP_END=5`
- `ENABLE_LOOPING_AT=0.35`
- `MLP_LORA_RANK=4`
- `MLP_LORA_ALPHA=4.0`
- `ENABLE_MLP_LORA_AT=0.50`
- `FREEZE_REPEATED_MLP=1`

By default, `MLP_LORA_LAYERS` is empty, which means the trainer targets the loop segment itself (`3,4,5`).
This means the repeated passes are active before the LoRA adapters start training.

## Main New Knobs

- `MLP_LORA_RANK`
- `MLP_LORA_ALPHA`
- `MLP_LORA_LR`
- `MLP_LORA_WD`
- `MLP_LORA_LAYERS`
- `ENABLE_MLP_LORA_AT`
- `FREEZE_REPEATED_MLP`

## Suggested First Runs

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Smaller first sweep:

```bash
SEED=42 MLP_LORA_RANK=2 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

More aggressive adapter sweep:

```bash
SEED=42 MLP_LORA_RANK=8 MLP_LORA_ALPHA=8 ENABLE_MLP_LORA_AT=0.5 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Practical Notes

- The LoRA tensors are intentionally small, so they are expected to stay in the passthrough path during export rather than join the large GPTQ matrices.
- The trainer uses a small `zero_proxy` dependency when LoRA adapters exist but are not yet active, which keeps DDP happy without changing the forward numerics.
- If `flash_attn_interface` is unavailable locally, the trainer falls back to PyTorch SDPA automatically. On H100/FA3 setups it will still use the FlashAttention 3 path.
- If `COMPRESSOR=brotli` is requested but the local environment does not have the `brotli` Python module, serialization now falls back automatically to `lzma` and records that choice in the artifact header so immediate deserialize/eval still works.
- This is an experiment scaffold, not a claimed record submission.
- The implementation is closest in spirit to the repo's earlier `repeat_untie_mlp` experiments, but trades full repeated-pass untie for a low-rank refinement.
