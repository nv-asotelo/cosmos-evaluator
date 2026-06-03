#!/usr/bin/env bash
set -euo pipefail

NETWORK="${NETWORK:-cosmos-evaluator-net}"
RUNTIME_DIR="${RUNTIME_DIR:-$HOME/.cosmos_evaluator/runtime}"
DATA_DIR="${DATA_DIR:-$PWD/checks/sample_data/cosmos_public}"
EVALUATOR_ENV_FILE="${EVALUATOR_ENV_FILE:-$HOME/.cosmos_evaluator/evaluator.env}"
NIM_CREDENTIAL_FILE="${NIM_CREDENTIAL_FILE:-$HOME/.cosmos_evaluator/nim.env}"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"

NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0}"
NIM_MODEL_SIZE="${NIM_MODEL_SIZE:-super}"
if [ -z "${NIM_MODEL_PROFILE:-}" ] && [ "$NIM_MODEL_SIZE" = "nano" ]; then
  # Cosmos3 Reasoner 1.7.0 Nano FP8 profile for L40S-class 48 GB GPUs.
  NIM_MODEL_PROFILE="17fedc428e5a9220fae87b540fc9f324eb9e521d35d733de5fe87253db14e6e7"
fi
NIM_CONTAINER="${NIM_CONTAINER:-cosmos3-nim}"
NIM_PORT="${NIM_PORT:-8000}"

VLM_IMAGE="${VLM_IMAGE:-vlm:1.15.0}"
ATTRIBUTE_IMAGE="${ATTRIBUTE_IMAGE:-attribute-verification-checker:1.0.0}"
HALLUCINATION_IMAGE="${HALLUCINATION_IMAGE:-hallucination-checker:1.0.0}"
OBSTACLE_IMAGE="${OBSTACLE_IMAGE:-obstacle-correspondence:1.15.0}"

VLM_PORT="${VLM_PORT:-8083}"
CONTROL_PORT="${CONTROL_PORT:-8090}"
ATTRIBUTE_PORT="${ATTRIBUTE_PORT:-8086}"
HALLUCINATION_PORT="${HALLUCINATION_PORT:-8085}"
OBSTACLE_PORT="${OBSTACLE_PORT:-8082}"

require_file_mode() {
  local file="$1"
  if [ -f "$file" ]; then
    local mode
    mode="$(stat -c %a "$file" 2>/dev/null || stat -f %Lp "$file")"
    case "$mode" in
      600|400) ;;
      *) echo "ERROR: $file must be chmod 0600 or 0400" >&2; exit 1 ;;
    esac
  fi
}

source_if_present() {
  local file="$1"
  if [ -f "$file" ]; then
    require_file_mode "$file"
    set -a
    # shellcheck disable=SC1090
    . "$file"
    set +a
  fi
}

docker_run_replace() {
  local name="$1"
  shift
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" "$@"
}

mkdir -p "$RUNTIME_DIR" "$LOCAL_NIM_CACHE"
source_if_present "$NIM_CREDENTIAL_FILE"
source_if_present "$EVALUATOR_ENV_FILE"

if [ -z "${NGC_API_KEY:-}" ]; then
  echo "ERROR: NGC_API_KEY is required via env or chmod-0600 $NIM_CREDENTIAL_FILE" >&2
  exit 1
fi

docker network create "$NETWORK" >/dev/null 2>&1 || true

NIM_GPU_ARGS=(--gpus all)
if docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  NIM_GPU_ARGS=(--runtime=nvidia "${NIM_GPU_ARGS[@]}")
fi
NIM_ENV_ARGS=(
  -e NGC_API_KEY
  -e "NIM_MODEL_SIZE=$NIM_MODEL_SIZE"
)
for var_name in \
  NIM_MODEL_PROFILE \
  NIM_KVCACHE_PERCENT \
  NIM_MAX_NUM_BATCHED_TOKENS \
  NIM_MAX_NUM_SEQS \
  NIM_COMPILATION_CONFIG \
  PYTORCH_CUDA_ALLOC_CONF; do
  if [ -n "${!var_name:-}" ]; then
    NIM_ENV_ARGS+=(-e "$var_name=${!var_name}")
  fi
done

printf '%s\n' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin >/dev/null
docker pull "$NIM_IMAGE"

docker_run_replace "$NIM_CONTAINER" \
  --network "$NETWORK" \
  "${NIM_GPU_ARGS[@]}" \
  --shm-size=32GB \
  "${NIM_ENV_ARGS[@]}" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p "$NIM_PORT:8000" \
  "$NIM_IMAGE"

echo "Waiting for Cosmos3 NIM on http://localhost:${NIM_PORT}/v1/models ..."
for _ in $(seq 1 240); do
  if curl -fsS "http://localhost:${NIM_PORT}/v1/models" >/tmp/cosmos3_nim_models.json; then
    cat /tmp/cosmos3_nim_models.json
    break
  fi
  sleep 10
done
if ! curl -fsS "http://localhost:${NIM_PORT}/v1/models" >/dev/null; then
  docker logs --tail 200 "$NIM_CONTAINER" >&2 || true
  exit 2
fi

COMMON_SERVICE_ENV=(
  --env-file "$EVALUATOR_ENV_FILE"
  -e COSMOS_EVALUATOR_VLM_RUNTIME_CONFIG=/runtime/vlm_runtime.json
  -e COSMOS3_NIM_API_KEY="${COSMOS3_NIM_API_KEY:-not-used}"
  -v "$RUNTIME_DIR:/runtime"
)
LOCAL_STORAGE_ENV=(
  -e COSMOS_EVALUATOR_STORAGE_TYPE="${COSMOS_EVALUATOR_STORAGE_TYPE:-local}"
  -v "$DATA_DIR:/data:ro"
)

docker_run_replace cosmos-evaluator-vlm \
  --network "$NETWORK" \
  "${COMMON_SERVICE_ENV[@]}" \
  "${LOCAL_STORAGE_ENV[@]}" \
  -p "$VLM_PORT:8000" \
  -p "$CONTROL_PORT:8000" \
  "$VLM_IMAGE"

docker_run_replace cosmos-evaluator-attribute \
  --network "$NETWORK" \
  "${COMMON_SERVICE_ENV[@]}" \
  -p "$ATTRIBUTE_PORT:8080" \
  "$ATTRIBUTE_IMAGE"

docker_run_replace cosmos-evaluator-hallucination \
  --network "$NETWORK" \
  --env-file "$EVALUATOR_ENV_FILE" \
  -p "$HALLUCINATION_PORT:8080" \
  "$HALLUCINATION_IMAGE"

docker_run_replace cosmos-evaluator-obstacle \
  --network "$NETWORK" \
  --env-file "$EVALUATOR_ENV_FILE" \
  "${LOCAL_STORAGE_ENV[@]}" \
  -p "$OBSTACLE_PORT:8000" \
  "$OBSTACLE_IMAGE"

echo "Service URLs:"
echo "  NIM:           http://localhost:${NIM_PORT}/v1/models"
echo "  VLM preset:    http://localhost:${VLM_PORT}/health"
echo "  Switch API:    http://localhost:${CONTROL_PORT}/runtime/vlm"
echo "  Attribute:     http://localhost:${ATTRIBUTE_PORT}/health"
echo "  Hallucination: http://localhost:${HALLUCINATION_PORT}/health"
echo "  Obstacle:      http://localhost:${OBSTACLE_PORT}/health"
