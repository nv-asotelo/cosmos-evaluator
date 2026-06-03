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
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional
import unittest
from unittest.mock import MagicMock, patch

from checks.vlm.preset_processor import PresetProcessor, process_preset


class TestPresetProcessor(unittest.TestCase):
    def _write_video_stub(self, tmp: Path) -> str:
        # Create a tiny 2-frame video using OpenCV
        import cv2
        import numpy as np

        path = tmp / "stub.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, 2.0, (64, 36))
        for i in range(4):
            frame = np.full((36, 64, 3), 255 if i % 2 == 0 else 0, dtype=np.uint8)
            out.write(frame)
        out.release()
        return str(path)

    def _write_prompt(self, tmp: Path, additional_prompts: Optional[Dict[str, Any]] = None) -> str:
        prompts = {
            "preset_check": [
                {
                    "name": "environment",
                    "prompt_type": "dynamic",
                    "user_prompt": [
                        "You are an impartial evaluator.",
                        "Conditions:",
                        "1) Weather = ${weather}",
                        "2) Time of day = ${time_of_day_illumination}",
                        "3) Region = ${region_geography}",
                        "4) Road = ${road_surface_conditions}",
                        "Output strict JSON with required keys.",
                    ],
                }
            ]
        }
        if additional_prompts:
            prompts["preset_check"].extend(additional_prompts)
        p = tmp / "prompts.json"
        p.write_text(json.dumps(prompts))
        return str(p)

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_process_happy_path(self, mock_create_client):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = self._write_video_stub(tmp)
            prompt_path = self._write_prompt(tmp)

            # Fake client result
            fake_client = MagicMock()
            fake_result = MagicMock()
            fake_choice = MagicMock()
            fake_choice.message.content = json.dumps(
                {
                    "weather": {"score": 1, "explanation": "Sunny."},
                    "time_of_day_illumination": {"score": 0.5, "explanation": "Low light."},
                    "region_geography": {"score": 0, "explanation": "Urban."},
                    "road_surface_conditions": {"score": 1, "explanation": "Dry."},
                }
            )
            fake_result.choices = [fake_choice]
            fake_client.chat.completions.create.return_value = fake_result
            mock_create_client.return_value = (fake_client, "fake-model")

            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)
            out = proc.process(
                video_path=video_path,
                presets={
                    "name": "environment",
                    "weather": "sunny",
                    "time_of_day_illumination": "daytime",
                    "region_geography": "urban",
                    "road_surface_conditions": "dry",
                },
                keyframe_interval_s=0.2,
                keyframe_width=32,
            )
            # Returns keyed by preset name
            self.assertIn("environment", out)
            env = out["environment"]
            self.assertIn("scoring_details", env)
            self.assertEqual(env["scoring_details"]["weather"]["score"], 1.0)
            self.assertEqual(env["scoring_details"]["weather"]["preset"], "sunny")
            self.assertGreaterEqual(env["frames_used"], 1)
            self.assertIn("processing_time_s", env)
            self.assertEqual(env["model"], "fake-model")
            self.assertEqual(env["overall_score"], 0.625)  # (1+0.5+0+1)/4

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_process_with_missing_preset_variables(self, mock_create_client):
        """Test that process method fails appropriately when prompt has placeholders without matching preset values."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = self._write_video_stub(tmp)
            prompt_path = self._write_prompt(tmp)

            # Fake client result
            fake_client = MagicMock()
            mock_create_client.return_value = (fake_client, "fake-model")

            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Try to process with missing preset variables
            with self.assertRaises(ValueError) as context:
                proc.process(
                    video_path=video_path,
                    presets={
                        "name": "environment",
                        "weather": "sunny",
                        # Missing other required variables
                    },
                    keyframe_interval_s=0.2,
                    keyframe_width=32,
                )

            self.assertIn("Missing variables for placeholders", str(context.exception))

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_load_prompts_invalid_structure(self, mock_create_client):
        """Test load_prompts_from_file with invalid prompt file structures."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # Test with non-array preset_check
            invalid_prompts = {"preset_check": {"not": "an array"}}
            p = tmp / "invalid.json"
            p.write_text(json.dumps(invalid_prompts))

            with self.assertRaises(TypeError):
                PresetProcessor(endpoint_type="azure_openai", prompt_template_path=str(p))

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_load_prompts_no_valid_prompts(self, mock_create_client):
        """Test load_prompts_from_file when no valid prompts are found."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # Test with empty array
            empty_prompts = {"preset_check": []}
            p = tmp / "empty.json"
            p.write_text(json.dumps(empty_prompts))

            with self.assertRaises(ValueError):
                PresetProcessor(endpoint_type="azure_openai", prompt_template_path=str(p))

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_load_prompts_various_formats(self, mock_create_client):
        """Test that PresetProcessor correctly loads and stores various prompt formats."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # Test with string prompt, list prompt, and static prompt
            additional_prompts = [
                {"name": "static_test", "prompt_type": "static", "user_prompt": "This is a static prompt"},
                {"name": "list_prompt", "user_prompt": ["Line 1", "Line 2", "Line 3"]},
                {
                    "name": "invalid_prompt",
                    "user_prompt": 123,  # Invalid type
                },
            ]

            prompt_path = self._write_prompt(tmp, additional_prompts)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Should have loaded valid prompts into prompts_by_name
            self.assertIn("environment", proc.prompts_by_name)
            self.assertIn("static_test", proc.prompts_by_name)
            self.assertIn("list_prompt", proc.prompts_by_name)
            self.assertNotIn("invalid_prompt", proc.prompts_by_name)  # Invalid type should be skipped

            # Verify list prompt was joined correctly
            list_entry = proc.prompts_by_name["list_prompt"]
            self.assertEqual(list_entry["user_prompt"], "Line 1\nLine 2\nLine 3")

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_valid_environment(self, mock_create_client):
        """Test _parse_response with valid environment preset response."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Test valid response
            response_json = {
                "weather": {"score": 1.0, "explanation": "Clear sunny day"},
                "time_of_day_illumination": {"score": 0.5, "explanation": "Dawn lighting"},
                "region_geography": {"score": 0.0, "explanation": "Desert region"},
                "road_surface_conditions": {"score": 1.0, "explanation": "Dry asphalt"},
            }

            presets = {
                "weather": "sunny",
                "time_of_day_illumination": "dawn",
                "region_geography": "desert",
                "road_surface_conditions": "dry",
            }

            overall_score, details = proc._parse_response(json.dumps(response_json), presets, "environment")

            self.assertEqual(overall_score, 0.625)  # (1+0.5+0+1)/4
            self.assertEqual(len(details), 4)
            self.assertEqual(details["weather"]["score"], 1.0)
            self.assertEqual(details["weather"]["preset"], "sunny")
            self.assertEqual(details["time_of_day_illumination"]["score"], 0.5)

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_invalid_scores(self, mock_create_client):
        """Test _parse_response with invalid scores."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Test invalid score value (not 0, 0.5, or 1)
            response_json = {
                "weather": {"score": 0.7, "explanation": "Partly cloudy"},  # Invalid score
                "time_of_day_illumination": {"score": 0.5, "explanation": "Dawn lighting"},
                "region_geography": {"score": 0.0, "explanation": "Desert region"},
                "road_surface_conditions": {"score": 1.0, "explanation": "Dry asphalt"},
            }

            overall_score, details = proc._parse_response(json.dumps(response_json), {}, "environment")

            self.assertIsNone(overall_score)
            self.assertEqual(details, {})

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_missing_fields(self, mock_create_client):
        """Test _parse_response with missing required fields."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Test missing field
            response_json = {
                "weather": {"score": 1.0, "explanation": "Clear sunny day"},
                "time_of_day_illumination": {"score": 0.5, "explanation": "Dawn lighting"},
                "region_geography": {"score": 0.0, "explanation": "Desert region"},
                # Missing road_surface_conditions
            }

            overall_score, details = proc._parse_response(json.dumps(response_json), {}, "environment")

            self.assertIsNone(overall_score)
            self.assertEqual(details, {})

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_code_fences(self, mock_create_client):
        """Test _parse_response with code fence formatting."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            response_json = {
                "weather": {"score": 1.0, "explanation": "Clear sunny day"},
                "time_of_day_illumination": {"score": 0.5, "explanation": "Dawn lighting"},
                "region_geography": {"score": 0.0, "explanation": "Desert region"},
                "road_surface_conditions": {"score": 1.0, "explanation": "Dry asphalt"},
            }

            # Test with code fences
            response_with_fences = f"```json\n{json.dumps(response_json)}\n```"

            overall_score, details = proc._parse_response(response_with_fences, {}, "environment")

            self.assertEqual(overall_score, 0.625)
            self.assertEqual(len(details), 4)

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_extracts_fenced_json_without_closing_fence(self, mock_create_client):
        """Test _parse_response with a markdown fence that is missing its closer."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            response_json = {
                "weather": {"score": 1.0, "explanation": "Clear sunny day"},
                "time_of_day_illumination": {"score": 0.5, "explanation": "Dawn lighting"},
                "region_geography": {"score": 0.0, "explanation": "Desert region"},
                "road_surface_conditions": {"score": 1.0, "explanation": "Dry asphalt"},
            }

            response_with_fence = "```json\n" + json.dumps(response_json)

            overall_score, details = proc._parse_response(response_with_fence, {}, "environment")

            self.assertEqual(overall_score, 0.625)
            self.assertEqual(len(details), 4)

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_invalid_json(self, mock_create_client):
        """Test _parse_response with invalid JSON."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            with self.assertRaises(ValueError) as context:
                proc._parse_response("invalid json", {}, "environment")

            self.assertIn("Model did not return valid JSON", str(context.exception))

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_parse_response_non_environment_preset(self, mock_create_client):
        """Test _parse_response with non-environment preset."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Non-environment presets should return None, {}
            overall_score, details = proc._parse_response('{"some": "response"}', {}, "other_preset")

            self.assertIsNone(overall_score)
            self.assertEqual(details, {})

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_get_config_summary(self, mock_create_client):
        """Test get_config_summary method."""
        fake_client = MagicMock()
        fake_client.__repr__ = MagicMock(return_value="MockClient(endpoint='test')")
        mock_create_client.return_value = (fake_client, "test-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            prompt_path = self._write_prompt(tmp)
            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            config = proc.get_config_summary()

            self.assertIn("model", config)
            self.assertIn("endpoint_type", config)
            self.assertIn("client", config)
            self.assertEqual(config["model"], "test-model")
            self.assertEqual(config["endpoint_type"], "azure_openai")

    @patch("checks.utils.config_manager.ConfigManager.load_config")
    def test_get_default_config_success(self, mock_load_config):
        """Test get_default_config static method success."""
        mock_config = {
            "av.vlm": {
                "preset_check": {
                    "model": {"endpoint": "azure_openai"},
                    "keyframe_interval_s": 2.0,
                    "keyframe_width": 640,
                    "max_frames": 24,
                }
            }
        }
        mock_load_config.return_value = mock_config

        result = PresetProcessor.get_default_config()

        self.assertIn("preset_check", result)
        self.assertEqual(result["preset_check"], mock_config["av.vlm"]["preset_check"])

    @patch("checks.utils.config_manager.ConfigManager.load_config")
    def test_get_default_config_failure(self, mock_load_config):
        """Test get_default_config static method failure."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")

        with self.assertRaises(FileNotFoundError):
            PresetProcessor.get_default_config()

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_process_unknown_preset_name(self, mock_create_client):
        """Test process method with unknown preset name."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = self._write_video_stub(tmp)
            prompt_path = self._write_prompt(tmp)

            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            with self.assertRaises(ValueError) as context:
                proc.process(
                    video_path=video_path,
                    presets={"name": "unknown_preset"},
                )

            self.assertIn("Unknown preset name", str(context.exception))

    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_process_missing_preset_name(self, mock_create_client):
        """Test process method when preset 'name' field is missing.

        When presets dict doesn't have a 'name' key, presets.get("name") returns None,
        which becomes "none" after str() and lower(). This should raise an error.
        The calling code (run.py) should always set the name field before calling process().
        """
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = self._write_video_stub(tmp)
            prompt_path = self._write_prompt(tmp)

            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            # Preset without 'name' field - this happens if caller forgets to set it
            with self.assertRaises(ValueError) as context:
                proc.process(
                    video_path=video_path,
                    presets={"weather": "sunny"},  # Missing 'name' field
                )

            # str(None).lower() = "none", which should trigger "Unknown preset name"
            self.assertIn("Unknown preset name", str(context.exception))
            self.assertIn("none", str(context.exception).lower())

    @patch("checks.vlm.preset_processor.extract_keyframes")
    @patch("checks.vlm.client_manager.ClientManager.create_client")
    def test_process_no_frames_extracted(self, mock_create_client, mock_extract_keyframes):
        """Test process method when no frames are extracted."""
        fake_client = MagicMock()
        mock_create_client.return_value = (fake_client, "fake-model")
        mock_extract_keyframes.return_value = []  # No frames extracted

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = self._write_video_stub(tmp)
            prompt_path = self._write_prompt(tmp)

            proc = PresetProcessor(endpoint_type="azure_openai", prompt_template_path=prompt_path)

            with self.assertRaises(ValueError) as context:
                proc.process(
                    video_path=video_path,
                    presets={
                        "name": "environment",
                        "weather": "sunny",
                        "time_of_day_illumination": "daytime",
                        "region_geography": "urban",
                        "road_surface_conditions": "dry",
                    },
                )

            self.assertIn("No frames extracted from video", str(context.exception))


class TestProcessPresetFunction(unittest.TestCase):
    """Test the process_preset convenience function."""

    def _write_video_stub(self, tmp: Path) -> str:
        # Create a tiny 2-frame video using OpenCV
        import cv2
        import numpy as np

        path = tmp / "stub.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, 2.0, (64, 36))
        for i in range(4):
            frame = np.full((36, 64, 3), 255 if i % 2 == 0 else 0, dtype=np.uint8)
            out.write(frame)
        out.release()
        return str(path)

    @patch("checks.vlm.preset_processor.PresetProcessor.get_default_config")
    @patch("checks.vlm.preset_processor.PresetProcessor")
    def test_process_preset_function_success(self, mock_processor_class, mock_get_default_config):
        """Test process_preset convenience function with valid config."""
        # Mock get_default_config to return default values
        mock_get_default_config.return_value = {
            "preset_check": {
                "model": {},
                "keyframe_interval_s": 2.0,
                "keyframe_width": 640,
            }
        }

        # Mock the processor instance
        mock_processor = MagicMock()
        mock_result = {
            "environment": {
                "overall_score": 0.75,
                "scoring_details": {"weather": {"score": 1.0, "explanation": "Sunny"}},
                "frames_used": 5,
                "processing_time_s": 2.5,
                "model": "gpt-4-vision",
            }
        }
        mock_processor.process.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video_path = self._write_video_stub(tmp)

            preset_conditions = {
                "name": "environment",
                "weather": "sunny",
                "time_of_day_illumination": "daytime",
                "region_geography": "urban",
                "road_surface_conditions": "dry",
            }

            preset_check_config = {
                "model": {"endpoint": "azure_openai", "temperature": 0.2},
                "keyframe_interval_s": 1.5,
                "keyframe_width": 800,
                "max_frames": 4,
            }

            result = process_preset(video_path, preset_conditions, preset_check_config)

            # Verify processor was created with correct endpoint
            mock_processor_class.assert_called_once_with(endpoint_type="azure_openai")

            # Verify process was called with correct parameters
            mock_processor.process.assert_called_once_with(
                video_path=video_path,
                presets=preset_conditions,
                keyframe_interval_s=1.5,
                keyframe_width=800,
                max_frames=4,
                temperature=0.2,
            )

            self.assertEqual(result, mock_result)

    def test_process_preset_function_missing_endpoint(self):
        """Test process_preset function with missing endpoint."""
        preset_conditions = {"name": "environment"}
        preset_check_config = {"model": {}}  # Missing endpoint

        with self.assertRaises(ValueError) as context:
            process_preset("/fake/video.mp4", preset_conditions, preset_check_config)

        self.assertIn("preset_check_config.model.endpoint is required", str(context.exception))

    @patch("checks.vlm.preset_processor.PresetProcessor.get_default_config")
    @patch("checks.vlm.preset_processor.PresetProcessor")
    def test_process_preset_function_missing_keyframe_width(self, mock_processor_class, mock_get_default_config):
        """Test process_preset function uses default when keyframe_width is missing."""
        # Mock get_default_config to return default values
        mock_get_default_config.return_value = {
            "preset_check": {
                "model": {},
                "keyframe_interval_s": 2.0,
                "keyframe_width": 640,
            }
        }

        mock_processor = MagicMock()
        mock_processor.process.return_value = {}
        mock_processor_class.return_value = mock_processor

        preset_conditions = {"name": "environment"}
        preset_check_config = {
            "model": {"endpoint": "azure_openai"}
            # Missing keyframe_width - should use default 640
        }

        process_preset("/fake/video.mp4", preset_conditions, preset_check_config)

        # Verify default keyframe_width was used
        mock_processor.process.assert_called_once()
        call_kwargs = mock_processor.process.call_args[1]
        self.assertEqual(call_kwargs.get("keyframe_width"), 640)
        self.assertIsNone(call_kwargs.get("max_frames"))

    @patch("checks.vlm.preset_processor.PresetProcessor.get_default_config")
    @patch("checks.vlm.preset_processor.PresetProcessor")
    def test_process_preset_function_defaults(self, mock_processor_class, mock_get_default_config):
        """Test process_preset function uses correct defaults."""
        # Mock get_default_config to return default values
        mock_get_default_config.return_value = {
            "preset_check": {
                "model": {},
                "keyframe_interval_s": 2.0,
                "keyframe_width": 640,
            }
        }

        mock_processor = MagicMock()
        mock_processor.process.return_value = {}
        mock_processor_class.return_value = mock_processor

        preset_conditions = {"name": "environment"}
        preset_check_config = {
            "model": {"endpoint": "azure_openai"},
            "keyframe_width": 640,
            # All other parameters should use defaults
        }

        process_preset("/fake/video.mp4", preset_conditions, preset_check_config)

        # Verify defaults were used
        mock_processor.process.assert_called_once_with(
            video_path="/fake/video.mp4",
            presets=preset_conditions,
            keyframe_interval_s=2.0,  # default
            keyframe_width=640,  # default
            max_frames=None,  # default
            temperature=0.0,  # default
        )

    @patch("checks.vlm.preset_processor.PresetProcessor.get_default_config")
    @patch("checks.vlm.preset_processor.PresetProcessor")
    def test_process_preset_function_custom_keyframe_width(self, mock_processor_class, mock_get_default_config):
        """Test process_preset function with custom keyframe_width."""
        # Mock get_default_config to return default values
        mock_get_default_config.return_value = {
            "preset_check": {
                "model": {},
                "keyframe_interval_s": 2.0,
                "keyframe_width": 640,
            }
        }

        mock_processor = MagicMock()
        mock_processor.process.return_value = {}
        mock_processor_class.return_value = mock_processor

        preset_conditions = {"name": "environment"}
        preset_check_config = {
            "model": {"endpoint": "azure_openai"},
            "keyframe_width": 1024,
            "keyframe_interval_s": 3.0,
        }

        process_preset("/fake/video.mp4", preset_conditions, preset_check_config)

        # Verify custom values were passed through
        mock_processor.process.assert_called_once_with(
            video_path="/fake/video.mp4",
            presets=preset_conditions,
            keyframe_interval_s=3.0,
            keyframe_width=1024,
            max_frames=None,
            temperature=0.0,
        )


if __name__ == "__main__":
    unittest.main()
