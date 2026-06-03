# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from checks.utils.config_manager import ConfigManager
from checks.utils.video import extract_keyframes
from checks.vlm import utils
from checks.vlm.client_manager import ClientManager
from checks.vlm.runtime_config import resolve_preset_check_config
from utils.bazel import get_runfiles_path


class PresetProcessor:
    """Run VLM-based preset checks against a video.

    This processor loads prompts from `checks/vlm/config/prompts.json` under
    the `preset_check` key, which must be an array of objects with fields:
      - name: str
      - user_prompt: str or [str]
      - system_prompt: str or [str] (optional)
      - json_inputs: [dict] (optional) - array of JSON objects to include as extra text parts

    At runtime it selects the prompt by `presets["name"]`, strictly replaces
    occurrences of `${var}` using values from `presets`, sends the prompt
    (and optional system prompt) along with json_inputs as extra text parts
    with sampled frames to the VLM, and parses JSON output.

    The `process` method returns a dict keyed by the preset name
    (lowercased), with scoring details, overall score (float or None), frames
    used, processing time, and the model used.
    """

    def __init__(
        self,
        endpoint_type: str,
        public_config_path: Optional[str] = get_runfiles_path("checks/vlm/config/endpoints.json"),
        prompt_template_path: Optional[str] = get_runfiles_path("checks/vlm/config/prompts.json"),
        prompt_key: str = "preset_check",
    ) -> None:
        """Initialize the processor and load prompts/endpoints.

        Args:
            endpoint_type: Key of the target endpoint in the endpoints config.
            public_config_path: Path to public endpoints JSON; private overrides
                are loaded from ~/.cosmos_evaluator/endpoints.json.
            prompt_template_path: Path to `prompts.json`.
            prompt_key: Top-level key in prompt file that contains preset prompts.
        """
        self.endpoint_type = endpoint_type
        self.public_config_path = public_config_path
        self.logger = logging.getLogger(__name__)
        self.prompt_key = prompt_key
        # Load prompts first so any errors surface early
        self.prompts_by_name = utils.load_prompts_from_file(prompt_template_path, prompt_key)
        # Create client
        self.client, self.model = ClientManager(public_config_path).create_client(endpoint_type)
        self.logger.info(
            "Initialized PresetProcessor | endpoint_type=%s model=%s | prompts=%s",
            self.endpoint_type,
            self.model,
            list(self.prompts_by_name.keys()),
        )

    def _parse_response(
        self, response_text: str, presets: Dict[str, str], preset_name: str
    ) -> Tuple[Optional[float], Dict]:
        """Parse and validate the VLM JSON response.

        For the "environment" preset, strictly requires four keys with
        `{score, explanation}`; scores must be in {0, 0.5, 1}. On any violation,
        returns `(None, {})` to indicate a failed parse.

        For other presets, currently returns `(None, {})`.

        Args:
            response_text: The response text
            presets: The presets
            preset_name: The preset name

        Returns:
            (overall_score, details_dict) where overall_score is None on failure
            and details_dict contains per-key entries including the original
            `preset` value used to score.

        Raises:
            ValueError: If the response text is not valid JSON
        """
        # Strip code fences and fix common JSON formatting issues
        text = utils.strip_code_fences(response_text)

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            extracted = utils.extract_first_json_object(text)
            if not extracted:
                raise ValueError(f"Model did not return valid JSON: {e}: {text[:200]}") from e
            try:
                obj = json.loads(extracted)
            except json.JSONDecodeError as extracted_error:
                raise ValueError(
                    f"Model did not return valid JSON: {extracted_error}: {extracted[:200]}"
                ) from extracted_error

        scores_for_overall: List[float] = []
        preset_name_l = (preset_name or "").strip().lower()
        if preset_name_l == "environment":
            expected = [
                "weather",
                "time_of_day_illumination",
                "region_geography",
                "road_surface_conditions",
            ]
            scoring_details: Dict[str, Dict] = {}
            for key in expected:
                entry = obj.get(key)
                # Strict validation: if missing or malformed, fail fast
                if not isinstance(entry, dict) or "score" not in entry or "explanation" not in entry:
                    return None, {}
                score = entry.get("score")
                try:
                    score_f = float(score)
                except Exception:
                    return None, {}
                if score_f not in (0.0, 0.5, 1.0):
                    return None, {}
                scores_for_overall.append(score_f)
                entry["score"] = score_f
                entry["preset"] = presets.get(key)
                scoring_details[key] = entry
        else:
            return None, {}

        overall_score: Optional[float] = None
        if scores_for_overall:
            overall_score = sum(scores_for_overall) / len(scores_for_overall)
        return overall_score, scoring_details

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.

        Returns:
            Configuration summary
        """
        return {
            "model": self.model,
            "endpoint_type": self.endpoint_type,
            "client": repr(self.client),
        }

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration for VLM preset processing.

        Returns:
            Default configuration for VLM preset processing loaded from config.yaml
        """
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config("config")
            return {"preset_check": config["av.vlm"]["preset_check"]}
        except Exception as e:
            logging.error("Error loading default configuration: {}".format(e))
            raise

    def process(
        self,
        video_path: str,
        presets: Dict[str, str],
        keyframe_interval_s: float = 2.0,
        keyframe_width: int = 640,
        max_frames: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Dict:
        """Run a preset check for the provided video and preset inputs.

        Args:
            video_path: Path to a local video file.
            presets: Preset dict with at least `name` (selects the prompt). For
                the environment preset, also expects keys:
                weather, time_of_day_illumination, region_geography,
                road_surface_conditions.
            keyframe_interval_s: Sampling interval in seconds.
            keyframe_width: keyframe width for frame resize.
            max_frames: Maximum number of sampled frames to send to the VLM.
            max_tokens: Maximum number of output tokens to request from the VLM.
            temperature: VLM sampling temperature.

        Returns:
            A dict keyed by the preset name (lowercased) with:
              - overall_score: float | None
              - scoring_details: dict
              - frames_used: int
              - processing_time_s: float
              - model: str

        Raises:
            ValueError: If the frames are not extracted from the video
            ValueError: If the preset name is invalid
        """
        start_time_s = time.time()
        self.logger.debug(
            "Extracting frames | interval=%.2fs width=%s max_frames=%s jpeg_quality=%d",
            keyframe_interval_s,
            str(keyframe_width),
            str(max_frames),
            utils.JPEG_QUALITY,
        )

        frames = extract_keyframes(
            video_path=video_path,
            interval_seconds=keyframe_interval_s,
            jpeg_quality=utils.JPEG_QUALITY,
            target_width=keyframe_width,
            max_frames=max_frames,
        )
        if not frames:
            raise ValueError("No frames extracted from video; adjust interval or check the input video")
        self.logger.debug("Extracted %d frames", len(frames))

        # Resolve prompt by preset name
        preset_name = str(presets.get("name"))
        preset_name = preset_name.strip().lower()
        if preset_name not in self.prompts_by_name:
            raise ValueError(f"Unknown preset name '{preset_name}'. Available: {list(self.prompts_by_name.keys())}")
        entry = self.prompts_by_name[preset_name]
        prompt_template = entry.get("user_prompt", "")
        system_prompt = entry.get("system_prompt")
        json_inputs = entry.get("json_inputs", []) or []

        # replace ${var} with values from presets
        prompt_text = utils.render_prompt(prompt_template, presets)

        # Build messages with json_inputs as extra text parts
        messages = utils.build_messages(
            prompt_text,
            extra_texts=json_inputs if json_inputs else None,
            system_text=system_prompt,
            jpeg_images=frames,
        )

        # Shared VLM call via base helper
        response_text = utils.call_lang_model(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            extra_params={"max_tokens": max_tokens},
        )
        self.logger.debug("Response text: %s", response_text)

        overall_score, scoring_details = self._parse_response(response_text, presets, preset_name)
        processing_time_s = round(time.time() - start_time_s, 2)
        self.logger.info("PresetProcessor processing_time_s: %.2f", processing_time_s)

        return {
            preset_name: {
                "overall_score": overall_score,
                "scoring_details": scoring_details,
                "frames_used": len(frames),
                "processing_time_s": processing_time_s,
                "model": self.model,
            }
        }


def process_preset(
    video_file_path: str,
    preset_conditions: Dict[str, Any],
    preset_check_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience API to run the preset check end-to-end.

    Args:
        video_file_path: path to the input video
        preset_conditions: the preset object with fields:
            weather, time_of_day_illumination, region_geography, road_surface_conditions
        preset_check_config: the 'preset_check' section from config.yaml

    Returns:
        A dict keyed by the preset name (lowercase) with:
          - overall_score: float | None
          - scoring_details: dict
          - frames_used: int
          - processing_time_s: float
          - model: str
    """

    # Extract model/config parameters
    config = resolve_preset_check_config(PresetProcessor.get_default_config()["preset_check"], preset_check_config)
    model_cfg = config.get("model", {})
    endpoint = model_cfg.get("endpoint")
    if not endpoint:
        raise ValueError("preset_check_config.model.endpoint is required")
    temperature = float(model_cfg.get("temperature", 0.0))

    keyframe_interval_s = float(config.get("keyframe_interval_s", 2.0))
    keyframe_width = int(config.get("keyframe_width", 640))
    max_frames_value = config.get("max_frames")
    max_frames = int(max_frames_value) if max_frames_value is not None else None
    max_tokens_value = model_cfg.get("max_tokens")
    max_tokens = int(max_tokens_value) if max_tokens_value is not None else None

    processor = PresetProcessor(endpoint_type=endpoint)
    return processor.process(
        video_path=video_file_path,
        presets=preset_conditions,
        keyframe_interval_s=keyframe_interval_s,
        keyframe_width=keyframe_width,
        max_frames=max_frames,
        max_tokens=max_tokens,
        temperature=temperature,
    )
