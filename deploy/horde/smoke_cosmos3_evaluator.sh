#!/usr/bin/env bash
set -euo pipefail

VLM_PORT="${VLM_PORT:-8083}"
CONTROL_PORT="${CONTROL_PORT:-8090}"
ATTRIBUTE_PORT="${ATTRIBUTE_PORT:-8086}"
HALLUCINATION_PORT="${HALLUCINATION_PORT:-8085}"
OBSTACLE_PORT="${OBSTACLE_PORT:-8082}"
NIM_PORT="${NIM_PORT:-8000}"
VIDEO_PATH="${VIDEO_PATH:-/data/01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000_0_Morning.30fps.mp4}"

curl -fsS "http://localhost:${NIM_PORT}/v1/models"
curl -fsS "http://localhost:${VLM_PORT}/health"
curl -fsS "http://localhost:${CONTROL_PORT}/runtime/vlm"
curl -fsS "http://localhost:${ATTRIBUTE_PORT}/health"
curl -fsS "http://localhost:${HALLUCINATION_PORT}/health"
curl -fsS "http://localhost:${OBSTACLE_PORT}/health"

curl -fsS -X POST "http://localhost:${CONTROL_PORT}/runtime/vlm/switch" \
  -H "Content-Type: application/json" \
  -d '{"endpoint":"cosmos3-super-reasoner"}'

curl -fsS -X POST "http://localhost:${VLM_PORT}/process/preset" \
  -H "Content-Type: application/json" \
  -d "{
    \"augmented_video_url\": \"${VIDEO_PATH}\",
    \"preset_conditions\": {
      \"name\": \"environment\",
      \"weather\": \"Clear Sky\",
      \"time_of_day_illumination\": \"Morning\",
      \"region_geography\": \"Dense City Center\",
      \"road_surface_conditions\": \"Dry\"
    }
  }"
