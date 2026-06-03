#!/usr/bin/env bash
set -euo pipefail

if [ ! -f build/env_setup.sh ]; then
  echo "Run this script from the cosmos-evaluator repository root." >&2
  exit 1
fi

. build/env_setup.sh

dazel run //services/vlm:image_load
dazel run //services/attribute_verification:image_load
dazel run //services/hallucination:image_load

if [ ! -f checks/utils/citysemsegformer.onnx ]; then
  echo "Skipping obstacle image: checks/utils/citysemsegformer.onnx is missing." >&2
  echo "Download the CitySemsegFormer ONNX model from NGC, copy it there, then run:" >&2
  echo "  dazel run //services/obstacle_correspondence:image_load" >&2
  exit 0
fi

dazel run //services/obstacle_correspondence:image_load
