#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/scratch4/home/akrik/base/bin/python" ]]; then
    PYTHON_BIN="/scratch4/home/akrik/base/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DATASET_GLOB="${DATASET_GLOB:-data/*/tool_embedding_dataset.jsonl}"
DATASET_PATHS="${DATASET_PATHS:-}"
OUTPUT_ROOT_NAME="${OUTPUT_ROOT_NAME:-output}"
if [[ -z "${DEVICE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-7}" ]]; then
    DEVICE="cuda:0"
  else
    DEVICE="auto"
  fi
fi
RUN_DATE="${RUN_DATE:-$(date +%F)}"
LOSSES="${LOSSES:-prototype_ce circle functional_margin}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-best.pt}"
LOG_DIR="${LOG_DIR:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_PARALLEL="${GPU_PARALLEL:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-ntilc}"
WANDB_ENTITY="${WANDB_ENTITY:-andykr1k}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-embedding-space,normal,all-datasets}"
WANDB_NOTES="${WANDB_NOTES:-normal embedding sweep over all datasets}"
WANDB_GROUP_VALUE="${WANDB_GROUP:-normal-all-datasets-${RUN_DATE}}"

if [[ -n "${DATASET_PATHS}" ]]; then
  read -r -a DATASET_FILES <<< "${DATASET_PATHS}"
else
  mapfile -t DATASET_FILES < <(compgen -G "${DATASET_GLOB}" | sort)
fi

read -r -a LOSS_ARRAY <<< "${LOSSES}"
read -r -a EXTRA_TRAIN_ARGS <<< "${TRAIN_ARGS}"
CLI_ARGS=("$@")

if [[ "${#DATASET_FILES[@]}" -eq 0 ]]; then
  echo "No datasets found for DATASET_GLOB=${DATASET_GLOB}" >&2
  exit 1
fi

if [[ "${#LOSS_ARRAY[@]}" -eq 0 ]]; then
  echo "No losses configured. Set LOSSES, e.g. LOSSES='prototype_ce circle functional_margin'." >&2
  exit 1
fi

if ! [[ "${JOBS_PER_GPU}" =~ ^[0-9]+$ ]] || [[ "${JOBS_PER_GPU}" -lt 1 ]]; then
  echo "JOBS_PER_GPU must be a positive integer." >&2
  exit 1
fi

GPU_ARRAY=()
if [[ -n "${GPU_IDS}" ]]; then
  normalized_gpu_ids="${GPU_IDS// /}"
  IFS=',' read -r -a RAW_GPU_ARRAY <<< "${normalized_gpu_ids}"
  for gpu_id in "${RAW_GPU_ARRAY[@]}"; do
    [[ -n "${gpu_id}" ]] && GPU_ARRAY+=("${gpu_id}")
  done
fi

GPU_SLOTS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
  for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
    GPU_SLOTS+=("${gpu_id}")
  done
done

PARALLEL_ENABLED=0
if [[ "${GPU_PARALLEL}" == "1" ]]; then
  PARALLEL_ENABLED=1
elif [[ "${GPU_PARALLEL}" == "auto" && "${#GPU_SLOTS[@]}" -gt 1 ]]; then
  PARALLEL_ENABLED=1
elif [[ "${GPU_PARALLEL}" != "0" && "${GPU_PARALLEL}" != "auto" ]]; then
  echo "GPU_PARALLEL must be auto, 1, or 0." >&2
  exit 1
fi

if [[ "${PARALLEL_ENABLED}" == "1" && "${#GPU_SLOTS[@]}" -eq 0 ]]; then
  echo "GPU_PARALLEL=1 requires GPU_IDS or CUDA_VISIBLE_DEVICES to contain at least one GPU." >&2
  exit 1
fi

DRY_RUN_SERIAL_PREVIEW=0
if [[ "${DRY_RUN}" == "1" && "${PARALLEL_ENABLED}" == "1" ]]; then
  DRY_RUN_SERIAL_PREVIEW=1
  PARALLEL_ENABLED=0
fi

WANDB_ARGS=()
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  WANDB_ARGS+=("--wandb" "--wandb-group" "${WANDB_GROUP_VALUE}")
  [[ -n "${WANDB_PROJECT}" ]] && WANDB_ARGS+=("--wandb-project" "${WANDB_PROJECT}")
  [[ -n "${WANDB_ENTITY}" ]] && WANDB_ARGS+=("--wandb-entity" "${WANDB_ENTITY}")
  [[ -n "${WANDB_TAGS}" ]] && WANDB_ARGS+=("--wandb-tags" "${WANDB_TAGS}")
  [[ -n "${WANDB_NOTES}" ]] && WANDB_ARGS+=("--wandb-notes" "${WANDB_NOTES}")
  [[ -n "${WANDB_MODE}" ]] && WANDB_ARGS+=("--wandb-mode" "${WANDB_MODE}")
fi

if [[ "${PARALLEL_ENABLED}" == "1" && "${DRY_RUN}" != "1" && -z "${LOG_DIR}" ]]; then
  LOG_DIR="logs/normal-all-datasets-${RUN_DATE}"
fi

if [[ -n "${LOG_DIR}" ]]; then
  mkdir -p "${LOG_DIR}"
fi

echo "Root directory: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Device: ${DEVICE}"
if [[ "${#GPU_ARRAY[@]}" -gt 0 ]]; then
  echo "GPU IDs: ${GPU_ARRAY[*]}"
  echo "GPU slots: ${#GPU_SLOTS[@]} (${JOBS_PER_GPU} job(s) per GPU)"
else
  echo "GPU IDs: none configured"
fi
echo "GPU parallel: ${PARALLEL_ENABLED}"
echo "Datasets: ${#DATASET_FILES[@]}"
echo "Losses: ${LOSS_ARRAY[*]}"
echo "Output root name: ${OUTPUT_ROOT_NAME}"
echo "Skip existing: ${SKIP_EXISTING}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run: enabled"
  if [[ "${DRY_RUN_SERIAL_PREVIEW}" == "1" ]]; then
    echo "Dry run scheduler: serial preview of GPU assignments"
  fi
fi
if [[ "${WANDB_ENABLED}" == "1" ]]; then
  echo "W&B group: ${WANDB_GROUP_VALUE}"
fi
if [[ -n "${LOG_DIR}" ]]; then
  echo "Log dir: ${LOG_DIR}"
fi

run_command() {
  local dataset_name="$1"
  local loss="$2"
  local assigned_gpu="$3"
  shift 3

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '
    if [[ -n "${assigned_gpu}" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${assigned_gpu}"
    fi
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi

  if [[ -n "${LOG_DIR}" ]]; then
    if [[ -n "${assigned_gpu}" ]]; then
      CUDA_VISIBLE_DEVICES="${assigned_gpu}" "$@" 2>&1 | tee "${LOG_DIR}/${dataset_name}-normal-${loss}.log"
    else
      "$@" 2>&1 | tee "${LOG_DIR}/${dataset_name}-normal-${loss}.log"
    fi
  else
    if [[ -n "${assigned_gpu}" ]]; then
      CUDA_VISIBLE_DEVICES="${assigned_gpu}" "$@"
    else
      "$@"
    fi
  fi
}

JOB_DATASETS=()
JOB_LOSSES=()

for dataset_path in "${DATASET_FILES[@]}"; do
  if [[ ! -f "${dataset_path}" ]]; then
    echo "Dataset not found, skipping: ${dataset_path}" >&2
    continue
  fi

  dataset_dir="$(dirname "${dataset_path}")"
  dataset_name="$(basename "${dataset_dir}")"
  train_dataset_path="${dataset_dir}/tool_embedding_dataset_train.jsonl"
  test_dataset_path="${dataset_dir}/tool_embedding_dataset_test.jsonl"
  tools_path="${dataset_dir}/tools.json"
  output_dir="${dataset_dir}/${OUTPUT_ROOT_NAME}"

  trainer_dataset_args=("--dataset-path" "${dataset_path}")
  if [[ -f "${train_dataset_path}" ]]; then
    trainer_dataset_args+=("--train-dataset-path" "${train_dataset_path}")
  fi
  if [[ -f "${test_dataset_path}" ]]; then
    trainer_dataset_args+=("--test-dataset-path" "${test_dataset_path}")
  fi
  if [[ -f "${tools_path}" ]]; then
    trainer_dataset_args+=("--tools-path" "${tools_path}")
  fi

  echo
  echo "=== Dataset: ${dataset_name} ==="
  echo "Dataset path: ${dataset_path}"
  [[ -f "${train_dataset_path}" ]] && echo "Train split: ${train_dataset_path}"
  [[ -f "${test_dataset_path}" ]] && echo "Test split: ${test_dataset_path}"
  [[ -f "${tools_path}" ]] && echo "Tools: ${tools_path}"
  echo "Output: ${output_dir}"

  for loss in "${LOSS_ARRAY[@]}"; do
    checkpoint_path="${output_dir}/normal/${loss}/${CHECKPOINT_FILENAME}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${checkpoint_path}" ]]; then
      echo "Skipping ${dataset_name}/${loss}; checkpoint exists: ${checkpoint_path}"
      continue
    fi

    JOB_DATASETS+=("${dataset_path}")
    JOB_LOSSES+=("${loss}")
  done
done

if [[ "${#JOB_DATASETS[@]}" -eq 0 ]]; then
  echo
  echo "No training jobs to run."
  exit 0
fi

echo
echo "Queued jobs: ${#JOB_DATASETS[@]}"

train_one_job() {
  local dataset_path="$1"
  local loss="$2"
  local assigned_gpu="${3:-}"

  local dataset_dir
  local dataset_name
  local train_dataset_path
  local test_dataset_path
  local tools_path
  local output_dir
  local job_device
  dataset_dir="$(dirname "${dataset_path}")"
  dataset_name="$(basename "${dataset_dir}")"
  train_dataset_path="${dataset_dir}/tool_embedding_dataset_train.jsonl"
  test_dataset_path="${dataset_dir}/tool_embedding_dataset_test.jsonl"
  tools_path="${dataset_dir}/tools.json"
  output_dir="${dataset_dir}/${OUTPUT_ROOT_NAME}"
  job_device="${DEVICE}"
  if [[ -n "${assigned_gpu}" ]]; then
    job_device="cuda:0"
  fi

  local trainer_dataset_args=("--dataset-path" "${dataset_path}")
  if [[ -f "${train_dataset_path}" ]]; then
    trainer_dataset_args+=("--train-dataset-path" "${train_dataset_path}")
  fi
  if [[ -f "${test_dataset_path}" ]]; then
    trainer_dataset_args+=("--test-dataset-path" "${test_dataset_path}")
  fi
  if [[ -f "${tools_path}" ]]; then
    trainer_dataset_args+=("--tools-path" "${tools_path}")
  fi

  echo
  if [[ -n "${assigned_gpu}" ]]; then
    echo "--- Training normal embedding space: dataset=${dataset_name} loss=${loss} gpu=${assigned_gpu} ---"
  else
    echo "--- Training normal embedding space: dataset=${dataset_name} loss=${loss} ---"
  fi

  local command=(
    "${PYTHON_BIN}" -m training.train_embedding_space
    "${trainer_dataset_args[@]}"
    --output-dir "${output_dir}"
    --loss-type "${loss}"
    --device "${job_device}"
    "${WANDB_ARGS[@]}"
    --wandb-run-name "${dataset_name}-normal-${loss}-${RUN_DATE}"
    "${EXTRA_TRAIN_ARGS[@]}"
    "${CLI_ARGS[@]}"
  )
  run_command "${dataset_name}" "${loss}" "${assigned_gpu}" "${command[@]}"
}

run_worker() {
  local worker_index="$1"
  local assigned_gpu="$2"
  local job_index
  for ((job_index = worker_index; job_index < ${#JOB_DATASETS[@]}; job_index += ${#GPU_SLOTS[@]})); do
    train_one_job "${JOB_DATASETS[job_index]}" "${JOB_LOSSES[job_index]}" "${assigned_gpu}"
  done
}

if [[ "${PARALLEL_ENABLED}" == "1" ]]; then
  echo "Launching ${#GPU_SLOTS[@]} GPU worker(s)."
  pids=()
  for worker_index in "${!GPU_SLOTS[@]}"; do
    run_worker "${worker_index}" "${GPU_SLOTS[worker_index]}" &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "One or more training workers failed." >&2
    exit 1
  fi
else
  for job_index in "${!JOB_DATASETS[@]}"; do
    assigned_gpu=""
    if [[ "${#GPU_SLOTS[@]}" -gt 0 ]]; then
      assigned_gpu="${GPU_SLOTS[job_index % ${#GPU_SLOTS[@]}]}"
    fi
    train_one_job "${JOB_DATASETS[job_index]}" "${JOB_LOSSES[job_index]}" "${assigned_gpu}"
  done
fi

echo
echo "Finished normal embedding training for configured datasets."
