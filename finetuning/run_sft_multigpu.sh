#!/usr/bin/env bash
# Multi-GPU fine-tuning of Qwen3-TTS-12Hz Base with multiple speakers / new languages.
#
# Usage:
#   bash run_sft_multigpu.sh                      # use all visible GPUs
#   NUM_GPUS=4 bash run_sft_multigpu.sh           # use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run_sft_multigpu.sh
#   SKIP_PREPARE=1 bash run_sft_multigpu.sh       # reuse an existing TRAIN_JSONL
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override any of these from the environment) ----
TOKENIZER_MODEL_PATH="${TOKENIZER_MODEL_PATH:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"

# TSV with header: audio, text, ref_audio  [, speaker, language]
RAW_TSV="${RAW_TSV:-train_raw.tsv}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
REF_PATH="${REF_PATH:-}"               # optional: one reference audio for every row
SPEAKER_COLUMN="${SPEAKER_COLUMN:-}"   # optional: per-row speaker column
SPEAKER="${SPEAKER:-}"                 # optional: global speaker (also fills empty cells)
LANGUAGE_COLUMN="${LANGUAGE_COLUMN:-}" # optional: per-row language column
LANGUAGE="${LANGUAGE:-}"               # optional: global language (also fills empty cells)
TRAIN_JSONL="${TRAIN_JSONL:-train_with_codes.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

BATCH_SIZE="${BATCH_SIZE:-2}"          # per GPU
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
SPEAKER_NAME="${SPEAKER_NAME:-speaker_test}"   # fallback for rows without a `speaker` column

NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
MAIN_PORT="${MAIN_PORT:-29500}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

if [ "${NUM_GPUS}" -lt 1 ]; then
  echo "No CUDA devices found." >&2
  exit 1
fi

# ---- 1) extract audio codes (single GPU is enough) ----
if [ "${SKIP_PREPARE}" != "1" ]; then
  REF_ARGS=()
  if [ -n "${REF_PATH}" ]; then
    REF_ARGS=(--ref_path "${REF_PATH}")
  fi
  if [ -n "${SPEAKER_COLUMN}" ]; then REF_ARGS+=(--speaker_column "${SPEAKER_COLUMN}"); fi
  if [ -n "${SPEAKER}" ]; then REF_ARGS+=(--speaker "${SPEAKER}"); fi
  if [ -n "${LANGUAGE_COLUMN}" ]; then REF_ARGS+=(--language_column "${LANGUAGE_COLUMN}"); fi
  if [ -n "${LANGUAGE}" ]; then REF_ARGS+=(--language "${LANGUAGE}"); fi
  python prepare_data.py \
    --device cuda:0 \
    --tokenizer_model_path "${TOKENIZER_MODEL_PATH}" \
    --input_tsv "${RAW_TSV}" \
    --audio_column "${AUDIO_COLUMN}" \
    --text_column "${TEXT_COLUMN}" \
    ${REF_ARGS[@]+"${REF_ARGS[@]}"} \
    --output_jsonl "${TRAIN_JSONL}"
fi

# ---- 2) fine-tune on all GPUs ----
# Mixed precision (bf16) and gradient accumulation are configured inside sft_12hz.py.
MULTI_GPU_FLAG=""
if [ "${NUM_GPUS}" -gt 1 ]; then
  MULTI_GPU_FLAG="--multi_gpu"
fi

accelerate launch \
  ${MULTI_GPU_FLAG} \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PORT}" \
  sft_12hz.py \
    --init_model_path "${INIT_MODEL_PATH}" \
    --output_model_path "${OUTPUT_DIR}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num_epochs "${EPOCHS}" \
    --speaker_name "${SPEAKER_NAME}"
