import collections
import copy
import glob
import io
import lzma
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

FLASH_ATTN_BACKEND = "flash_attn_3"
FLASH_ATTN_IMPORT_ERROR = None
PASSTHROUGH_TENSOR_MAX_NUMEL = 65536
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError as exc:
    FLASH_ATTN_BACKEND = "torch_sdpa_fallback"
    FLASH_ATTN_IMPORT_ERROR = exc

    def flash_attn_3_func(q, k, v, causal=True):
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=causal,
            enable_gqa=(k.size(1) != q.size(1)),
        )
        return y.transpose(1, 2)


class Hyperparameters:
    data_dir = os.environ.get("DATA_DIR", "./data/")
    seed = int(os.environ.get("SEED", 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 6e2))
    val_batch_tokens = int(os.environ.get("VAL_BATCH_TOKENS", 524288))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    sliding_window_enabled = bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "1")))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 4.0))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "1")))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 3e1))
    rope_base = float(os.environ.get("ROPE_BASE", 1e4))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", 2048))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    stable_resid_mix = bool(int(os.environ.get("STABLE_RESID_MIX", "1")))
    stable_resid_carry_init = float(os.environ.get("STABLE_RESID_CARRY_INIT", 0.99))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 5))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    mlp_lora_rank = int(os.environ.get("MLP_LORA_RANK", 4))
    mlp_lora_alpha = float(os.environ.get("MLP_LORA_ALPHA", 4.0))
    mlp_lora_lr = float(os.environ.get("MLP_LORA_LR", 0.04))
    mlp_lora_wd = float(os.environ.get("MLP_LORA_WD", 0.0))
    mlp_lora_layers = os.environ.get("MLP_LORA_LAYERS", "").strip()
    enable_mlp_lora_at = float(os.environ.get("ENABLE_MLP_LORA_AT", os.environ.get("ENABLE_LOOPING_AT", 0.35)))
    parallel_residual_start = int(os.environ.get("PARALLEL_RESIDUAL_START", 7))
    min_lr = float(os.environ.get("MIN_LR", 0.0))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_row_normalize = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-08))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    adam_wd = float(os.environ.get("ADAM_WD", 0.02))
    muon_wd = float(os.environ.get("MUON_WD", 0.095))
    embed_wd = float(os.environ.get("EMBED_WD", 0.085))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    compressor = os.environ.get("COMPRESSOR", "brotli")
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))
    matrix_bits = int(os.environ.get("MATRIX_BITS", 6))
    embed_bits = int(os.environ.get("EMBED_BITS", 8))
    matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", 2e1))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, "datasets", f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, "fineweb_train_*.bin")
    val_files = os.path.join(datasets_dir, "fineweb_val_*.bin")
    tokenizer_path = os.path.join(
        data_dir, "tokenizers", f"fineweb_{vocab_size}_bpe.model"
    )
    logfile = f"logs/{run_id}.txt"
    model_path = "final_model.pt"
    quantized_model_path = "final_model.int6.ptz"


_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h


def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)


def parse_layer_list(spec):
    return sorted({int(x) for x in spec.split(",") if x.strip()}) if spec else []


class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        sp_vocab = int(self.sp.vocab_size())
        if sp_vocab != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={sp_vocab}"
            )
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = (
            build_sentencepiece_luts(self.sp, h.vocab_size, device)
        )


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert (
        sp.piece_to_id("▁") != sp.unk_id()
    ), "Tokenizer must have '▁' (space) as its own token for correct BPB byte counting"
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    to_dev = lambda arr, dt: torch.tensor(arr, dtype=dt, device=device)
    return (
        to_dev(base_bytes_np, torch.int16),
        to_dev(has_leading_space_np, torch.bool),
        to_dev(is_boundary_token_np, torch.bool),
    )


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


_SHARD_NTOKENS_CACHE = {}
_MMAP_CACHE = {}


def _read_num_tokens(file):
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n


def _get_shard_memmap(file):
    key = str(file)
    mm = _MMAP_CACHE.get(key)
    if mm is not None:
        return mm
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=256 * np.dtype("<i4").itemsize, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm


class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank :: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(
            self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1)
        )
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array(
                    [len(s) for s in self.start_inds], dtype=np.float64
                )
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.from_numpy(
                mm[start_ind : start_ind + self.seq_len + 1].astype(np.int64)
            )
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (
            torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        needs_rebuild = (
            self._cos_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        )
        if needs_rebuild:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                new_base = self.base * (seq_len / self.train_seq_len) ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd
                )
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def _rotate(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        return torch.cat(
            (_rotate(x[..., :rope_dims], cos, sin), x[..., rope_dims:]), dim=-1
        )
    return _rotate(x, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = True

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.hidden_dim = hidden
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x, lora=None, lora_enabled=False):
        hidden = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
        out = self.proj(hidden)
        if lora is not None:
            out = out + lora(hidden, enabled=lora_enabled)
        return out


class LoRAAdapter(nn.Module):
    def __init__(self, hidden_dim, model_dim, rank, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.down = nn.Parameter(torch.empty(rank, hidden_dim))
        self.up = nn.Parameter(torch.zeros(model_dim, rank))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, hidden, enabled=True):
        if not enabled:
            # Keep parameters in the gradient graph for DDP without computing the full delta.
            return (self.down.sum() + self.up.sum()).to(hidden.dtype) * 0.0
        delta = F.linear(
            F.linear(hidden, self.down.to(hidden.dtype)), self.up.to(hidden.dtype)
        )
        return delta * self.scaling

    def zero_proxy(self):
        return (self.down.sum() + self.up.sum()) * 0.0


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        train_seq_len,
        layer_idx=0,
        ln_scale=False,
        stable_resid_mix=True,
        stable_resid_carry_init=0.99,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.stable_resid_mix = stable_resid_mix
        if stable_resid_mix:
            carry_init = min(max(stable_resid_carry_init, 1e-4), 1.0 - 1e-4)
            raw_carry_init = math.log(-math.log(carry_init))
            self.resid_carry_log = nn.Parameter(
                torch.full((dim,), raw_carry_init, dtype=torch.float32)
            )
            self.resid_input_scale = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        else:
            self.resid_mix = nn.Parameter(
                torch.stack((torch.ones(dim), torch.zeros(dim))).float()
            )
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0, mlp_lora=None, mlp_lora_enabled=False):
        if self.stable_resid_mix:
            carry = torch.exp(-torch.exp(self.resid_carry_log.clamp(-20, 20))).to(dtype=x.dtype)
            input_scale = self.resid_input_scale.to(dtype=x.dtype)
            x_in = carry[None, None, :] * x + input_scale[None, None, :] * x0
        else:
            mix = self.resid_mix.to(dtype=x.dtype)
            x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        attn_s = self.attn_scale.to(dtype=x_in.dtype)[None, None, :]
        mlp_s = self.mlp_scale.to(dtype=x_in.dtype)[None, None, :]
        mlp_kw = dict(lora=mlp_lora, lora_enabled=mlp_lora_enabled)
        if self.parallel:
            mlp_in = self.mlp_norm(x_in) * self.ln_scale_factor
            return x_in + attn_s * attn_out + mlp_s * self.mlp(mlp_in, **mlp_kw)
        x_out = x_in + attn_s * attn_out
        mlp_in = self.mlp_norm(x_out) * self.ln_scale_factor
        return x_out + mlp_s * self.mlp(mlp_in, **mlp_kw)


class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.qk_gain_init,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                    stable_resid_mix=h.stable_resid_mix,
                    stable_resid_carry_init=h.stable_resid_carry_init,
                )
                for i in range(h.num_layers)
            ]
        )
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = Rotary(
                    head_dim,
                    base=h.rope_base,
                    train_seq_len=h.train_seq_len,
                    rope_dims=h.rope_dims,
                )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None
            if h.tie_embeddings
            else CastedLinear(h.model_dim, h.vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.parallel_residual_start >= 0:
            for i in range(h.parallel_residual_start, h.num_layers):
                self.blocks[i].parallel = True
        self.looping_active = False
        if h.num_loops > 0:
            loop_seg = list(range(h.loop_start, h.loop_end + 1))
            all_indices = list(range(h.loop_start))
            for _ in range(h.num_loops + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.loop_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))
        self.num_skip_weights = min(
            len(self.encoder_indices), len(self.decoder_indices)
        )
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32)
        )
        self.skip_gates = (
            nn.Parameter(
                torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)
            )
            if h.skip_gates_enabled
            else None
        )
        self.active_encoder_plan = [
            (p, v) for (v, p) in enumerate(self.encoder_indices)
        ]
        self.active_decoder_plan = [
            (p, len(self.encoder_indices) + v)
            for (v, p) in enumerate(self.decoder_indices)
        ]
        self.plain_encoder_plan = [(p, None) for p in range(self.num_encoder_layers)]
        self.plain_decoder_plan = [
            (p, None) for p in range(self.num_encoder_layers, h.num_layers)
        ]
        requested_lora_layers = parse_layer_list(h.mlp_lora_layers)
        default_lora_layers = (
            list(range(h.loop_start, h.loop_end + 1)) if h.num_loops > 0 else []
        )
        self.mlp_lora_layers = sorted(set(requested_lora_layers or default_lora_layers))
        hidden_dim = int(h.mlp_mult * h.model_dim)
        max_sparse_lora_rank = min(
            PASSTHROUGH_TENSOR_MAX_NUMEL // hidden_dim,
            PASSTHROUGH_TENSOR_MAX_NUMEL // h.model_dim,
        )
        self.mlp_lora_rank = min(h.mlp_lora_rank, max_sparse_lora_rank)
        if h.mlp_lora_rank > self.mlp_lora_rank:
            log(
                f"mlp_lora:rank clipped from {h.mlp_lora_rank} to {self.mlp_lora_rank} "
                f"to keep adapter tensors <= {PASSTHROUGH_TENSOR_MAX_NUMEL} params"
            )
        self.recurrent_mlp_loras = nn.ModuleList()
        self.virtual_mlp_lora_indices = []
        occurrences = collections.defaultdict(int)
        for virtual_idx, physical_idx in enumerate(
            self.encoder_indices + self.decoder_indices
        ):
            is_repeated = occurrences[physical_idx] > 0
            occurrences[physical_idx] += 1
            if (
                h.num_loops > 0
                and is_repeated
                and self.mlp_lora_rank > 0
                and physical_idx in self.mlp_lora_layers
            ):
                self.virtual_mlp_lora_indices.append(len(self.recurrent_mlp_loras))
                self.recurrent_mlp_loras.append(
                    LoRAAdapter(
                        hidden_dim, h.model_dim, self.mlp_lora_rank, h.mlp_lora_alpha
                    )
                )
            else:
                self.virtual_mlp_lora_indices.append(-1)
        self.mlp_lora_active = False
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (
                    module.weight.ndim == 2
                    and module.weight.shape[0] >= 64
                    and module.weight.shape[1] >= 64
                ):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _run_block(self, i, x, x0, virtual_idx):
        lora_idx = (
            -1 if virtual_idx is None else self.virtual_mlp_lora_indices[virtual_idx]
        )
        if lora_idx < 0:
            return self.blocks[i](x, x0)
        return self.blocks[i](
            x,
            x0,
            mlp_lora=self.recurrent_mlp_loras[lora_idx],
            mlp_lora_enabled=self.mlp_lora_active,
        )

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        enc_plan = (
            self.active_encoder_plan if self.looping_active else self.plain_encoder_plan
        )
        dec_plan = (
            self.active_decoder_plan if self.looping_active else self.plain_decoder_plan
        )
        for i, virtual_idx in enc_plan:
            x = self._run_block(i, x, x0, virtual_idx)
            skips.append(x)
        for skip_idx, (i, virtual_idx) in enumerate(dec_plan):
            if skip_idx < self.num_skip_weights and skips:
                scaled_skip = (
                    self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :]
                    * skips.pop()
                )
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[
                        None, None, :
                    ]
                    x = torch.lerp(scaled_skip, x, g) * 2.0
                else:
                    x = x + scaled_skip
            x = self._run_block(i, x, x0, virtual_idx)
        if not self.looping_active and self.recurrent_mlp_loras:
            dummy = torch.zeros((), device=x.device, dtype=x.dtype)
            for adapter in self.recurrent_mlp_loras:
                dummy = dummy + adapter.zero_proxy().to(dtype=x.dtype, device=x.device)
            x = x + dummy
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            target_ids.reshape(-1),
            reduction="mean",
        )


def classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if name.startswith("recurrent_mlp_loras") or ".mlp." in name:
        return "mlp"
    if ".attn." in name or ".proj." in name:
        return "attn"
    return "other"


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum,
        backend_steps,
        nesterov=True,
        weight_decay=0.0,
        row_normalize=False,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        row_norms = (
                            g.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        )
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p
    for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,resid_carry_log,resid_input_scale,q_gain,skip_weight,skip_gates",
    ).split(",")
    if p
)


class Optimizers:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        is_control_name = lambda n: any(p in n for p in CONTROL_TENSOR_NAME_PATTERNS)
        matrix_params = [
            p for (n, p) in block_named_params if p.ndim == 2 and not is_control_name(n)
        ]
        scalar_params = [
            p for (n, p) in block_named_params if p.ndim < 2 or is_control_name(n)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None and base_model.skip_gates.numel() > 0:
            scalar_params.append(base_model.skip_gates)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        adam_kw = dict(betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
        self.optimizer_tok = torch.optim.AdamW(
            [
                {
                    "params": [base_model.tok_emb.weight],
                    "lr": token_lr,
                    "base_lr": token_lr,
                }
            ],
            weight_decay=h.embed_wd,
            **adam_kw,
        )
        self.optimizer_muon = Muon(
            matrix_params,
            lr=h.matrix_lr,
            momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps,
            weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            weight_decay=h.adam_wd,
            **adam_kw,
        )
        self.optimizers = [
            self.optimizer_tok,
            self.optimizer_muon,
            self.optimizer_scalar,
        ]
        lora_params = list(base_model.recurrent_mlp_loras.parameters())
        if lora_params:
            self.optimizer_lora = torch.optim.AdamW(
                [
                    {
                        "params": lora_params,
                        "lr": h.mlp_lora_lr,
                        "base_lr": h.mlp_lora_lr,
                    }
                ],
                weight_decay=h.mlp_lora_wd,
                **adam_kw,
            )
            self.optimizers.insert(2, self.optimizer_lora)
        else:
            self.optimizer_lora = None
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [
                    {
                        "params": [base_model.lm_head.weight],
                        "lr": h.head_lr,
                        "base_lr": h.head_lr,
                    }
                ],
                **adam_kw,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        return iter(self.optimizers)

    def zero_grad_all(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in self.optimizers:
            opt.step()
        self.zero_grad_all()


def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        is_control = param.ndim < 2 or any(
            p in name for p in CONTROL_TENSOR_NAME_PATTERNS
        )
        if (
            is_control or name.startswith("recurrent_mlp_loras")
        ) and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def _accumulate(name, x):
        x = x.detach().float()
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        if name not in hessians:
            hessians[name] = torch.zeros(
                x.shape[1], x.shape[1], dtype=torch.float32, device=device
            )
        hessians[name].addmm_(x.T, x)

    def make_hook(name):
        return lambda module, inp, out: _accumulate(name, inp[0])

    for name, module in model.named_modules():
        if (
            isinstance(module, CastedLinear)
            and module.weight.numel() > PASSTHROUGH_TENSOR_MAX_NUMEL
        ):
            if classify_param(name + ".weight") in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if model.tie_embeddings:
        hooks.append(
            model.final_norm.register_forward_hook(
                lambda module, inp, out: _accumulate("tok_emb.weight", out)
            )
        )
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def gptq_mixed_quantize(state_dict, hessians, h):
    result = {}
    meta = {}
    quantizable = [
        name
        for name, tensor in state_dict.items()
        if tensor.is_floating_point() and tensor.numel() > PASSTHROUGH_TENSOR_MAX_NUMEL
    ]
    quant_idx = 0
    if quantizable:
        log(f"GPTQ:quantizing {len(quantizable)} tensors on CPU")
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= PASSTHROUGH_TENSOR_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        is_embed = "tok_emb" in name
        cs = h.embed_clip_sigmas if is_embed else h.matrix_clip_sigmas
        bits = h.embed_bits if is_embed else h.matrix_bits
        quant_idx += 1
        t0 = time.perf_counter()
        log(
            f"GPTQ:quantizing {quant_idx}/{len(quantizable)} {name} shape={tuple(t.shape)} int{bits}"
        )
        q, s = gptq_quantize_weight(
            t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1
        )
        log(f"GPTQ:quantized {name} in {time.perf_counter()-t0:.1f}s")
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub("\\.\\d+$", "", re.sub("blocks\\.\\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        names = ", ".join(sorted(categories[cat]))
        log(f"  {cat}: {names}")
    return result, meta


def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            scale = s.float().view(q.shape[0], *[1] * (q.ndim - 1))
            out[name] = (q.float() * scale).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


_BSHF_MAGIC = b"BSHF"
_COMPRESS_MAGIC = b"PGC1"
_CODEC_BROTLI = b"b"
_CODEC_LZMA = b"l"


def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    out = np.concatenate([src[p::stride] for p in range(stride)])
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()


def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off : src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def _compress(data, compressor):
    data = _byte_shuffle(data)
    actual_compressor = compressor
    if compressor == "lzma":
        payload = lzma.compress(data, preset=6)
    elif compressor == "brotli":
        try:
            import brotli
        except ModuleNotFoundError:
            actual_compressor = "lzma"
            log("compressor:brotli unavailable, falling back to lzma")
            payload = lzma.compress(data, preset=6)
        else:
            payload = brotli.compress(data, quality=11)
    else:
        raise ValueError(f"Unknown compressor: {compressor!r}")
    codec = _CODEC_BROTLI if actual_compressor == "brotli" else _CODEC_LZMA
    return _COMPRESS_MAGIC + codec + payload


def _decompress(data, compressor):
    actual_compressor = compressor
    payload = data
    if len(data) >= 5 and data[:4] == _COMPRESS_MAGIC:
        codec = data[4:5]
        if codec == _CODEC_BROTLI:
            actual_compressor = "brotli"
        elif codec == _CODEC_LZMA:
            actual_compressor = "lzma"
        else:
            raise ValueError(f"Unknown compressor code: {codec!r}")
        payload = data[5:]
    if actual_compressor == "lzma":
        raw = lzma.decompress(payload)
    elif actual_compressor == "brotli":
        import brotli

        raw = brotli.decompress(payload)
    else:
        raise ValueError(f"Unknown compressor: {actual_compressor!r}")
    raw = _byte_unshuffle(raw)
    return raw


def serialize(h, base_model, code):
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
        log(f"Code size: {code_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for (k, v) in base_model.state_dict().items()}
    device = torch.device("cuda", h.local_rank)
    log("GPTQ:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, device)
    hessians = collect_hessians(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.gptq_calibration_batches,
    )
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = _compress(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    bytes_total = quant_file_bytes + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes")
        log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
    return bytes_total, quant_file_bytes


def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for (k, v) in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(_decompress(quant_blob_disk, h.compressor)), map_location="cpu"
    )
    deq_state = dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    eval_model.load_state_dict(deq_state, strict=True)
    return eval_model


def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb


def _allreduce_sum(*tensors):
    if dist.is_available() and dist.is_initialized():
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


def _zero_accumulators(device):
    z = lambda: torch.zeros((), device=device, dtype=torch.float64)
    return z(), z(), z()


def _count_bytes(val_data, tgt, prev, dtype=torch.float64):
    tb = val_data.base_bytes_lut[tgt].to(dtype)
    tb += (
        val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]
    ).to(dtype)
    return tb


def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_TOKENS must provide at least one sequence per rank; got VAL_BATCH_TOKENS={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum, val_token_count, val_byte_count = _zero_accumulators(device)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            val_byte_count += _count_bytes(val_data, y.reshape(-1), x.reshape(-1)).sum()
    _allreduce_sum(val_loss_sum, val_token_count, val_byte_count)
    model.train()
    return _loss_bpb(val_loss_sum, val_token_count, val_byte_count)


def _build_window_batch(val_data, batch_ws, seq_len, total_tokens, device):
    bsz = len(batch_ws)
    x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
    y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
    wlens = []
    for i, ws in enumerate(batch_ws):
        we = min(ws + seq_len, total_tokens)
        wlen = we - ws
        wlens.append(wlen)
        chunk = val_data.val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
        x_batch[i, :wlen] = chunk[:-1]
        y_batch[i, :wlen] = chunk[1:]
    return x_batch, y_batch, wlens


def _score_windowed(
    logits_fn,
    val_data,
    my_windows,
    seq_len,
    total_tokens,
    context_size,
    batch_seqs,
    device,
):
    loss_sum, token_count, byte_count = _zero_accumulators(device)
    for bi in range(0, len(my_windows), batch_seqs):
        batch_ws = my_windows[bi : bi + batch_seqs]
        bsz = len(batch_ws)
        x_batch, y_batch, wlens = _build_window_batch(
            val_data, batch_ws, seq_len, total_tokens, device
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = logits_fn(x_batch)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y_batch.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s = 0 if ws == 0 else context_size
            loss_sum += nll[i, s:wlen].to(torch.float64).sum()
            token_count += float(wlen - s)
            byte_count += _count_bytes(
                val_data, y_batch[i, s:wlen], x_batch[i, s:wlen]
            ).sum()
    return loss_sum, token_count, byte_count


def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [
        ws
        for ws in range(0, total_tokens, h.eval_stride)
        if ws + context_size < total_tokens
    ]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    with torch.inference_mode():
        loss_sum, token_count, byte_count = _score_windowed(
            logits_fn,
            val_data,
            window_starts[my_s:my_e],
            seq_len,
            total_tokens,
            context_size,
            batch_seqs,
            device,
        )
    _allreduce_sum(loss_sum, token_count, byte_count)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)


def eval_val_ttt(h, device, val_data, base_model, batch_seqs=32):
    rank, world_size = h.rank, h.world_size
    seq_len, stride = h.eval_seq_len, h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [
        ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        s = 0 if ws == 0 else context_size
        ci = min((ws + s) // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log(f"ttt:start chunks={num_chunks} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs}")
    compiled_logits = torch.compile(
        base_model.forward_logits, dynamic=False, fullgraph=True
    )
    loss_sum, token_count, byte_count = _zero_accumulators(device)
    ttt_params = list(base_model.parameters())
    for p in ttt_params:
        p.requires_grad_(True)
    optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        base_model.eval()
        with torch.no_grad():
            ls, tc, bc = _score_windowed(
                compiled_logits,
                val_data,
                windows[my_s:my_e],
                seq_len,
                total_tokens,
                context_size,
                batch_seqs,
                device,
            )
            loss_sum += ls
            token_count += tc
            byte_count += bc
        if ci == num_chunks - 1 or h.ttt_epochs <= 0:
            continue
        base_model.train()
        chunk_seqs = (chunk_end - chunk_start) // seq_len
        if chunk_seqs <= 0:
            continue
        cos_lr = (
            h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
        )
        for pg in optimizer.param_groups:
            pg["lr"] = cos_lr
        my_seq_s = chunk_seqs * rank // world_size
        my_seq_e = chunk_seqs * (rank + 1) // world_size
        my_chunk_seqs = my_seq_e - my_seq_s
        for _ in range(h.ttt_epochs):
            for bs in range(0, my_chunk_seqs, batch_seqs):
                be = min(bs + batch_seqs, my_chunk_seqs)
                start_tok = chunk_start + (my_seq_s + bs) * seq_len
                end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                if end_tok > val_data.val_tokens.numel():
                    continue
                local = val_data.val_tokens[start_tok:end_tok].to(
                    device=device, dtype=torch.int64
                )
                x = local[:-1].reshape(-1, seq_len)
                y = local[1:].reshape(-1, seq_len)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = base_model(x, y)
                loss.backward()
                if world_size > 1:
                    for p in ttt_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                optimizer.step()
    _allreduce_sum(loss_sum, token_count, byte_count)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return _loss_bpb(loss_sum, token_count, byte_count)


def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    log(
        f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms"
    )
    return val_loss, val_bpb


def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log(
        f"stable_resid_mix:{int(h.stable_resid_mix)} carry_init:{h.stable_resid_carry_init:.4f}"
    )
    log(
        f"repeated_pass_mlp_lora: rank={base_model.mlp_lora_rank} requested_rank={h.mlp_lora_rank} alpha={h.mlp_lora_alpha} layers={base_model.mlp_lora_layers} adapters={len(base_model.recurrent_mlp_loras)} enable_at={h.enable_mlp_lora_at:.2f}"
    )
    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)
    max_wallclock_ms = (
        1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    )
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1e3
        log(
            f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms"
        )

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None:
            return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-09)

    def lr_mul(frac):
        if h.warmdown_frac <= 0:
            return 1.0
        if frac >= 1.0 - h.warmdown_frac:
            return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = (
            min(step / h.muon_momentum_warmup_steps, 1.0)
            if h.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {
            n: t.detach().cpu().clone() for (n, t) in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]

        def _run_warmup(label):
            for w in range(h.warmup_steps):
                step_fn(w, 1.0)
                if w <= 5 or (w + 1) % 10 == 0 or w + 1 == h.warmup_steps:
                    log(f"{label}: {w+1}/{h.warmup_steps}")

        model.train()
        _run_warmup("warmup_step")
        if h.num_loops > 0:
            base_model.looping_active = True
            log(
                f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
            _run_warmup("loop_warmup_step")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)
    ema_state = {
        name: t.detach().float().clone()
        for (name, t) in base_model.state_dict().items()
    }
    ema_tensors = list(ema_state.values())
    ema_source_tensors = list(base_model.state_dict().values())
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = (
            step == h.iterations
            or stop_after_step is not None
            and step >= stop_after_step
        )
        should_validate = (
            last_step or h.val_loss_every > 0 and step % h.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(
                f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = training_frac(step, elapsed_ms)
        scale = lr_mul(frac)
        if (
            h.num_loops > 0
            and not base_model.looping_active
            and frac >= h.enable_looping_at
        ):
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        if (
            base_model.recurrent_mlp_loras
            and not base_model.mlp_lora_active
            and frac >= h.enable_mlp_lora_at
        ):
            base_model.mlp_lora_active = True
            log(
                f"mlp_lora:enabled step:{step} frac:{frac:.3f} adapters:{len(base_model.recurrent_mlp_loras)}"
            )
        train_loss = step_fn(step, scale)
        with torch.no_grad():
            current_fp32 = [t.detach().float() for t in ema_source_tensors]
            torch._foreach_mul_(ema_tensors, ema_decay)
            torch._foreach_add_(ema_tensors, current_fp32, alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
            step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {
        name: t.to(dtype=current_state[name].dtype) for (name, t) in ema_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def train_and_eval(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(
        f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin')))}"
    )
    log(f"val_tokens: {val_data.val_tokens.numel()-1}")
    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    timed_eval(
        "pre-quantization post-ema", eval_val, h, device, val_data, compiled_model
    )
    serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed:
        dist.barrier()

    def _activate_eval_features(m):
        if h.num_loops > 0:
            m.looping_active = True
        if m.recurrent_mlp_loras:
            m.mlp_lora_active = True

    eval_model = deserialize(h, device)
    _activate_eval_features(eval_model)
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        timed_eval(
            "quantized_sliding_window",
            eval_val_sliding,
            h,
            device,
            val_data,
            eval_model,
        )
    if h.ttt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        ttt_model = deserialize(h, device)
        _activate_eval_features(ttt_model)
        timed_eval("quantized_ttt", eval_val_ttt, h, device, val_data, ttt_model)
        del ttt_model


def main():
    h = Hyperparameters()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        "EMBEDDING_DIM" in os.environ
        and int(os.environ["EMBEDDING_DIM"]) != h.model_dim
    ):
        raise ValueError(
            "EMBEDDING_DIM is no longer a separate model width; set MODEL_DIM instead"
        )
    if h.world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {h.world_size}")
    if 8 % h.world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={h.world_size} must divide 8 so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", h.local_rank)
    torch.cuda.set_device(device)
    if h.distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Hyperparameters:", console=True)
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            ).stdout,
            console=False,
        )
        log(f"attention_backend: {FLASH_ATTN_BACKEND}", console=True)
        if FLASH_ATTN_IMPORT_ERROR is not None:
            log(
                f"attention_backend_note: falling back to torch SDPA because flash_attn_interface import failed: {FLASH_ATTN_IMPORT_ERROR}",
                console=True,
            )
        log("=" * 100, console=False)
    train_and_eval(h, device)
    if h.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
