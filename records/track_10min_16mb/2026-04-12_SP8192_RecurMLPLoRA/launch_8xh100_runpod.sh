#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${EXPERIMENT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

RECORD_DIR="${RECORD_DIR:-records/track_10min_16mb/2026-04-12_SP8192_RecurMLPLoRA}"
RUNPOD_GPU_COUNT="${RUNPOD_GPU_COUNT:-8}"
TORCH_NPROC_PER_NODE="${TORCH_NPROC_PER_NODE:-${RUNPOD_GPU_COUNT}}"
DATA_TRAIN_SHARDS="${DATA_TRAIN_SHARDS:-80}"
RUN_NAME="${RUN_NAME:-sp8192-recurmlplora-${RUNPOD_GPU_COUNT}xh100-$(date -u +%Y%m%d-%H%M%S)}"
REPO_REF="${REPO_REF:-$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)}"
RUN_CONFIG_PATH="${RUN_CONFIG_PATH:-${RECORD_DIR}/.runpod/runpod-${RUNPOD_GPU_COUNT}xh100-sp8192.env}"
RUNPOD_RUNNER="${RUNPOD_RUNNER:-tools/runpod/run_experiment.sh}"
RUNPOD_SSH_PRIVATE_KEY="${RUNPOD_SSH_PRIVATE_KEY:-/home/max/.ssh/id_ed25519_lambda}"
RUNPOD_SSH_PUBLIC_KEY="${RUNPOD_SSH_PUBLIC_KEY:-${RUNPOD_SSH_PRIVATE_KEY}.pub}"
WAIT_FOR_REMOTE_SECONDS="${WAIT_FOR_REMOTE_SECONDS:-300}"
WAIT_FOR_REMOTE_POLL_SECONDS="${WAIT_FOR_REMOTE_POLL_SECONDS:-5}"

upstream_ref="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
upstream_remote="${upstream_ref%%/*}"
if [[ -z "${upstream_ref}" || "${upstream_remote}" == "${upstream_ref}" ]]; then
  upstream_remote="origin"
fi
repo_remote="${REPO_REMOTE:-${upstream_remote}}"
remote_url="$(git config --get "remote.${repo_remote}.url" || true)"
if [[ -z "${remote_url}" ]]; then
  echo "No git remote.${repo_remote}.url found. Set REPO_URL=https://github.com/<you>/parameter-golf.git" >&2
  exit 1
fi
if [[ "${remote_url}" =~ ^git@github.com:(.*)\.git$ ]]; then
  remote_url="https://github.com/${BASH_REMATCH[1]}.git"
fi
REPO_URL="${REPO_URL:-${remote_url}}"

if [[ ! -d "${RECORD_DIR}" ]]; then
  echo "Record directory not found: ${RECORD_DIR}" >&2
  exit 1
fi
if [[ ! -x "${RUNPOD_RUNNER}" ]]; then
  echo "Runpod runner not found or not executable: ${RUNPOD_RUNNER}" >&2
  echo "Set RUNPOD_RUNNER=/path/to/run_experiment.sh if it lives elsewhere." >&2
  exit 1
fi
if [[ ! -f "${RUNPOD_SSH_PRIVATE_KEY}" ]]; then
  echo "SSH private key not found: ${RUNPOD_SSH_PRIVATE_KEY}" >&2
  exit 1
fi
if [[ ! -f "${RUNPOD_SSH_PUBLIC_KEY}" ]]; then
  echo "SSH public key not found: ${RUNPOD_SSH_PUBLIC_KEY}" >&2
  exit 1
fi

dirty_status="$(git status --porcelain -- . ":(exclude)${RECORD_DIR}/.runpod")"
if [[ "${ALLOW_DIRTY:-0}" != "1" ]] && [[ -n "${dirty_status}" ]]; then
  echo "Worktree is dirty. Commit and push first, or set ALLOW_DIRTY=1 if you really mean it." >&2
  echo "${dirty_status}" >&2
  exit 1
fi

local_head="$(git rev-parse HEAD)"
if [[ "${RUNPOD_SKIP_REMOTE_SYNC_CHECK:-0}" != "1" && "${REPO_REF}" != "${local_head}" ]]; then
  echo "Waiting for ${REPO_URL} ${REPO_REF} to expose local HEAD ${local_head}..."
  deadline=$((SECONDS + WAIT_FOR_REMOTE_SECONDS))
  while true; do
    remote_head="$(git ls-remote "${REPO_URL}" "refs/heads/${REPO_REF}" | awk '{print $1}' || true)"
    if [[ "${remote_head}" == "${local_head}" ]]; then
      echo "Remote branch is synced: ${REPO_REF} -> ${remote_head}"
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for remote branch ${REPO_REF} to match local HEAD." >&2
      echo "Remote saw: ${remote_head:-<missing>}" >&2
      echo "Push first, or set RUNPOD_SKIP_REMOTE_SYNC_CHECK=1 to bypass this guard." >&2
      exit 1
    fi
    sleep "${WAIT_FOR_REMOTE_POLL_SECONDS}"
  done
fi

mkdir -p "$(dirname "${RUN_CONFIG_PATH}")"

setup_command="python3 -m pip install --break-system-packages --upgrade pip && python3 -m pip install --break-system-packages -r requirements.txt && python3 -m pip install --break-system-packages -r records/track_10min_16mb/2026-04-12_SP8192_RecurMLPLoRA/runpod_requirements.txt && if ! python3 -c \"from flash_attn_interface import flash_attn_func\" >/dev/null 2>&1; then python3 -m pip install --break-system-packages flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ || (cd /tmp && rm -rf flash-attention && git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git && cd flash-attention/hopper && NVCC_THREADS=2 python3 setup.py install); fi && cd /workspace/parameter-golf && python3 -c \"from flash_attn_interface import flash_attn_func; import brotli; print('deps ok')\" && rm -f data/manifest.json && MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards ${DATA_TRAIN_SHARDS}"
job_command="cd ${RECORD_DIR} && DATA_DIR=/workspace/parameter-golf/data RUN_ID=${RUN_NAME} SEED=${SEED:-42} ITERATIONS=${ITERATIONS:-20000} MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600} VOCAB_SIZE=8192 MODEL_DIM=512 NUM_LAYERS=11 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4.0 QK_GAIN_INIT=5.25 NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.35} PARALLEL_RESIDUAL_START=7 WARMDOWN_FRAC=0.72 WARMUP_STEPS=${WARMUP_STEPS:-20} TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-786432} TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-2048} VAL_BATCH_TOKENS=${VAL_BATCH_TOKENS:-524288} EVAL_SEQ_LEN=${EVAL_SEQ_LEN:-2048} VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-4000} TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY:-500} SLIDING_WINDOW_ENABLED=${SLIDING_WINDOW_ENABLED:-1} EVAL_STRIDE=${EVAL_STRIDE:-64} MATRIX_LR=0.022 SCALAR_LR=0.02 MUON_MOMENTUM=0.99 MUON_BACKEND_STEPS=5 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_ROW_NORMALIZE=1 ADAM_WD=0.02 MUON_WD=0.095 EMBED_WD=0.085 EMA_DECAY=0.9965 MATRIX_BITS=6 EMBED_BITS=8 MATRIX_CLIP_SIGMAS=12.85 EMBED_CLIP_SIGMAS=20 COMPRESSOR=brotli GPTQ_CALIBRATION_BATCHES=${GPTQ_CALIBRATION_BATCHES:-64} GPTQ_RESERVE_SECONDS=${GPTQ_RESERVE_SECONDS:-12} TTT_ENABLED=${TTT_ENABLED:-1} TTT_LR=0.005 TTT_EPOCHS=${TTT_EPOCHS:-3} TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768 MLP_LORA_RANK=${MLP_LORA_RANK:-4} MLP_LORA_ALPHA=${MLP_LORA_ALPHA:-4.0} MLP_LORA_LR=${MLP_LORA_LR:-0.04} MLP_LORA_WD=${MLP_LORA_WD:-0.0} MLP_LORA_LAYERS=${MLP_LORA_LAYERS:-3,4,5} ENABLE_MLP_LORA_AT=${ENABLE_MLP_LORA_AT:-${ENABLE_LOOPING_AT:-0.35}} STABLE_RESID_MIX=${STABLE_RESID_MIX:-1} STABLE_RESID_CARRY_INIT=${STABLE_RESID_CARRY_INIT:-0.99} torchrun --standalone --nproc_per_node=${TORCH_NPROC_PER_NODE} train_gpt.py"
result_paths="${RECORD_DIR}/logs,${RECORD_DIR}/final_model.pt,${RECORD_DIR}/final_model.int6.ptz,${RECORD_DIR}/final_model.int8.ptz"

cat > "${RUN_CONFIG_PATH}" <<EOF
RUN_NAME=${RUN_NAME}
REPO_URL=${REPO_URL}
REPO_REF=${REPO_REF}

RUNPOD_API_BASE=https://rest.runpod.io/v1
RUNPOD_CLOUD_TYPE=SECURE
RUNPOD_TEMPLATE_ID=y5cejece4j
RUNPOD_GPU_TYPE_IDS=NVIDIA H100 80GB HBM3
RUNPOD_GPU_COUNT=${RUNPOD_GPU_COUNT}
RUNPOD_IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
RUNPOD_CONTAINER_DISK_GB=200
RUNPOD_VOLUME_GB=300
RUNPOD_VOLUME_MOUNT_PATH=/workspace
RUNPOD_SUPPORT_PUBLIC_IP=1
RUNPOD_PORTS=22/tcp
RUNPOD_MIN_VCPU_PER_GPU=8
RUNPOD_MIN_RAM_PER_GPU=32
RUNPOD_INTERRUPTIBLE=0
RUNPOD_POLL_SECONDS=20
RUNPOD_POD_READY_TIMEOUT_SECONDS=1800
RUNPOD_SSH_READY_TIMEOUT_SECONDS=1800
RUNPOD_CUSTOM_SSH_BOOTSTRAP=${RUNPOD_CUSTOM_SSH_BOOTSTRAP:-1}

RUNPOD_SSH_PRIVATE_KEY=${RUNPOD_SSH_PRIVATE_KEY}
RUNPOD_SSH_PUBLIC_KEY=${RUNPOD_SSH_PUBLIC_KEY}
RUNPOD_REMOTE_USER=root

AUTO_TERMINATE=1
EXTRACT_RESULTS_INTO_REPO=0
GIT_PUSH_AFTER_FETCH=0
KEEP_RUNPOD_DEBUG_FILES=${KEEP_RUNPOD_DEBUG_FILES:-0}
EXTRACT_RUNPOD_RESULTS=1
RUNPOD_RESULTS_DIR=${EXPERIMENT_DIR}/artifacts

REMOTE_REPO_DIR=/workspace/parameter-golf
REMOTE_RUNS_DIR=/workspace/runpod-runs

SETUP_COMMAND=${setup_command}
JOB_COMMAND=${job_command}
RESULT_PATHS=${result_paths}
EOF

echo "Wrote ${RUN_CONFIG_PATH}"
if [[ "${DRY_RUN}" == "1" ]]; then
  exec "${RUNPOD_RUNNER}" --config "${RUN_CONFIG_PATH}" --dry-run
fi

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "RUNPOD_API_KEY is not set." >&2
  exit 1
fi

exec "${RUNPOD_RUNNER}" --config "${RUN_CONFIG_PATH}"
