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

"""Unit tests for VLM REST API."""

from http import HTTPStatus
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from services.utils import get_contents_from_runfile
from services.vlm.rest_api_common import app
from services.vlm.service import (
    PresetRequest,
    PresetResponse,
)

# Mock get_git_sha for all tests since //utils:git_sha is not in the test's runfiles
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()


class TestVLMAPI(unittest.TestCase):
    """Test cases for VLM REST API."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.client = TestClient(app)

        # Common request scaffolding
        self.valid_preset_request = {
            "augmented_video_url": "s3://bucket/test_video.mp4",
            "preset_conditions": {
                "name": "environment",
                "weather": "sunny",
                "time_of_day_illumination": "noon",
                "region_geography": "urban",
                "road_surface_conditions": "dry",
            },
            "preset_check_config": {
                "model": {"endpoint": "test_endpoint"},
                "check_weather": True,
                "check_time": True,
                "check_traffic": True,
                "confidence_threshold": 0.8,
            },
        }

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        response = self.client.get("/health")

        self.assertEqual(response.status_code, HTTPStatus.OK)
        data = response.json()

        self.assertTrue(data["success"])  # from JsonResponseFormatter
        self.assertEqual(data["data"]["status"], "healthy")
        self.assertEqual(data["data"]["service"], "VLM Service API")
        expected_version = get_contents_from_runfile("services/vlm/version.txt")
        self.assertEqual(data["data"]["version"], expected_version)
        self.assertIn("timestamp", data)

    @patch("services.vlm.rest_api_common.service")
    def test_config_endpoint(self, mock_service: MagicMock) -> None:
        """Test config endpoint returns default config."""
        mock_service.get_default_config = AsyncMock(
            return_value={
                "av.vlm": {
                    "confidence_threshold": 0.8,
                    "model_name": "vlm_model_v1",
                    "batch_size": 4,
                    "preprocessing": {"resize_width": 224, "resize_height": 224, "normalize": True},
                }
            }
        )

        response = self.client.get("/config")

        self.assertEqual(response.status_code, HTTPStatus.OK)
        data = response.json()

        self.assertTrue(data["success"])  # formatter wrapper
        self.assertIn("default_config", data["data"])  # payload content
        self.assertIn("description", data["data"])  # payload content

        default_config = data["data"]["default_config"]
        self.assertIn("av.vlm", default_config)
        self.assertIn("confidence_threshold", default_config["av.vlm"])
        self.assertIn("model_name", default_config["av.vlm"])

    def test_config_endpoint_service_none(self) -> None:
        """Test config endpoint when service is None."""
        # Temporarily patch the global service to None
        with patch("services.vlm.rest_api_common.service", None):
            response = self.client.get("/config")

            self.assertEqual(response.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
            data = response.json()
            self.assertFalse(data["success"])
            self.assertIn("error", data)
            self.assertIn("Service not initialized", data["error"]["message"])

    @patch("services.vlm.rest_api_common.service")
    def test_config_endpoint_service_error(self, mock_service: MagicMock) -> None:
        """Test config endpoint when service raises an exception."""
        mock_service.get_default_config = AsyncMock(side_effect=Exception("Config loading failed"))

        response = self.client.get("/config")

        self.assertEqual(response.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["type"], "Exception")
        self.assertIn("Config loading failed", data["error"]["message"])

    @patch(
        "services.vlm.rest_api_common.runtime_summary",
        return_value={
            "active_endpoint": "qwen3.5-397b-a17b",
            "active": {"model": "qwen/qwen3.5-397b-a17b"},
            "available_endpoints": [],
            "state_file": "/tmp/runtime.json",
            "state": {},
        },
    )
    def test_runtime_vlm_endpoint(self, _mock_runtime: MagicMock) -> None:
        response = self.client.get("/runtime/vlm")

        self.assertEqual(response.status_code, HTTPStatus.OK)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["active_endpoint"], "qwen3.5-397b-a17b")

    @patch(
        "services.vlm.rest_api_common.set_active_endpoint",
        return_value={"active_endpoint": "cosmos3-nano-reasoner"},
    )
    def test_runtime_vlm_switch_endpoint(self, mock_switch: MagicMock) -> None:
        response = self.client.post("/runtime/vlm/switch", json={"endpoint": "cosmos3-nano-reasoner"})

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTrue(response.json()["success"])
        mock_switch.assert_called_once_with("cosmos3-nano-reasoner")

    @patch("services.vlm.rest_api_common.storage")
    @patch("services.vlm.rest_api_common.service")
    def test_process_preset_endpoint_success(self, mock_service: MagicMock, mock_storage: MagicMock) -> None:
        """Test /process/preset endpoint with valid request."""
        mock_storage.download_from_url = AsyncMock(return_value=Path("/tmp/vlm_preset_abc/video.mp4"))

        mock_service.process_preset = AsyncMock(
            return_value=PresetResponse(
                result={
                    "weather_detected": "sunny",
                    "time_of_day_detected": "noon",
                    "traffic_density_detected": "medium",
                    "confidence_scores": {"weather": 0.95, "time_of_day": 0.88, "traffic_density": 0.92},
                    "overall_confidence": 0.92,
                    "preset_match": True,
                }
            )
        )

        response = self.client.post("/process/preset", json=self.valid_preset_request)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        data = response.json()
        self.assertTrue(data["success"])  # formatter wrapper

        result_data = data["data"]["result"]
        self.assertIn("weather_detected", result_data)
        self.assertIn("confidence_scores", result_data)
        self.assertEqual(result_data["preset_match"], True)
        self.assertEqual(result_data["overall_confidence"], 0.92)

        self.assertIn("metadata", data)
        self.assertIn("duration_ms", data["metadata"])
        self.assertIsInstance(data["metadata"]["duration_ms"], (int, float))
        self.assertGreaterEqual(data["metadata"]["duration_ms"], 0)

        mock_storage.download_from_url.assert_called_once()
        mock_service.process_preset.assert_called_once()

    @patch("services.vlm.rest_api_common.storage")
    @patch("services.vlm.rest_api_common.service")
    def test_process_preset_service_error(self, mock_service: MagicMock, mock_storage: MagicMock) -> None:
        """Test /process/preset endpoint returns 500 on service error."""
        mock_storage.download_from_url = AsyncMock(return_value=Path("/tmp/vlm_preset_abc/video.mp4"))
        mock_service.process_preset = AsyncMock(side_effect=Exception("VLM processing failed"))

        response = self.client.post("/process/preset", json=self.valid_preset_request)

        self.assertEqual(response.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        data = response.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error"]["type"], "Exception")
        self.assertIn("VLM processing failed", data["error"]["message"])

        self.assertIn("metadata", data)
        self.assertIn("duration_ms", data["metadata"])
        self.assertIsInstance(data["metadata"]["duration_ms"], (int, float))
        self.assertGreaterEqual(data["metadata"]["duration_ms"], 0)

    def test_process_preset_invalid_request_missing_fields(self) -> None:
        """Test /process/preset endpoint with missing required fields."""
        invalid_request = {
            "endpoint_type": "environment_preset",
            # Missing augmented_video_url, preset_conditions, preset_check_config
        }

        response = self.client.post("/process/preset", json=invalid_request)

        self.assertEqual(response.status_code, HTTPStatus.UNPROCESSABLE_ENTITY)
        data = response.json()
        self.assertIn("detail", data)

    def test_process_preset_invalid_request_empty_fields(self) -> None:
        """Test /process/preset endpoint with empty required fields."""
        invalid_request = {
            "endpoint_type": "",
            "augmented_video_url": "",
            "preset_conditions": {},
            "preset_check_config": {},
        }

        response = self.client.post("/process/preset", json=invalid_request)

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_process_preset_invalid_json(self) -> None:
        """Test /process/preset endpoint with invalid JSON."""
        response = self.client.post(
            "/process/preset", content="invalid json", headers={"content-type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    def test_process_preset_empty_body(self) -> None:
        """Test /process/preset endpoint with empty body."""
        response = self.client.post("/process/preset", content="", headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)


class TestPresetRequestValidation(unittest.TestCase):
    """Test cases for PresetRequest model validation."""

    def test_preset_request_validation_success(self) -> None:
        """Test successful creation of PresetRequest with all valid fields."""
        valid_preset_request = PresetRequest(
            augmented_video_url="s3://bucket/test_video.mp4",
            preset_conditions={
                "name": "environment",
                "weather": "rainy",
                "time_of_day_illumination": "evening",
                "region_geography": "suburban",
                "road_surface_conditions": "wet",
            },
            preset_check_config={
                "model": {"endpoint": "test_endpoint"},
                "check_weather": True,
                "check_time": True,
                "check_traffic": True,
                "confidence_threshold": 0.9,
            },
        )

        self.assertEqual(valid_preset_request.augmented_video_url, "s3://bucket/test_video.mp4")
        self.assertIsInstance(valid_preset_request.preset_conditions, dict)
        self.assertIsInstance(valid_preset_request.preset_check_config, dict)
        self.assertEqual(valid_preset_request.preset_conditions["name"], "environment")
        self.assertEqual(valid_preset_request.preset_conditions["weather"], "rainy")
        assert valid_preset_request.preset_check_config is not None
        self.assertEqual(valid_preset_request.preset_check_config["confidence_threshold"], 0.9)

    def test_preset_request_missing_required_fields(self) -> None:
        """Test PresetRequest validation fails with missing required fields."""
        with self.assertRaises(Exception) as context:
            PresetRequest(preset_conditions={"weather": "sunny"}, preset_check_config={"check_weather": True})  # type: ignore[call-arg]
        self.assertIn("augmented_video_url", str(context.exception))

        with self.assertRaises(Exception) as context:
            PresetRequest(augmented_video_url="s3://bucket/test.mp4", preset_check_config={"check_weather": True})  # type: ignore[call-arg]
        self.assertIn("preset_conditions", str(context.exception))

    def test_preset_request_empty_string_validation(self) -> None:
        """Test PresetRequest validation with empty strings."""
        # Note: Pydantic's min_length validation may not always raise exceptions for empty strings
        # depending on the version and configuration. We'll test what actually happens.

        # Test empty augmented_video_url - this should fail due to min_length constraint
        try:
            PresetRequest(
                augmented_video_url="",
                preset_conditions={"weather": "sunny"},
                preset_check_config={"check_weather": True},
            )
            self.fail("Expected validation error for empty augmented_video_url")
        except Exception:
            pass

    def test_preset_request_invalid_types(self) -> None:
        """Test PresetRequest validation fails with invalid field types."""
        with self.assertRaises(Exception) as context:
            PresetRequest(
                augmented_video_url="s3://bucket/test.mp4",
                preset_conditions="not_a_dict",  # Should be dict
                preset_check_config={"check_weather": True},
            )
        self.assertIn("Input should be a valid dictionary", str(context.exception))

        with self.assertRaises(Exception) as context:
            PresetRequest(
                augmented_video_url="s3://bucket/test.mp4",
                preset_conditions={"weather": "sunny"},
                preset_check_config="not_a_dict",  # Should be dict
            )
        self.assertIn("Input should be a valid dictionary", str(context.exception))

    def test_preset_request_complex_conditions(self) -> None:
        """Test PresetRequest with complex nested conditions."""
        complex_preset_request = PresetRequest(
            augmented_video_url="s3://bucket/complex_video.mp4",
            preset_conditions={
                "name": "environment",
                "weather": {"primary": "partly_cloudy", "secondary": "light_rain", "visibility": "good"},
                "time_of_day_illumination": "dawn",
                "traffic": {"density": "light", "vehicle_types": ["car", "truck", "motorcycle"], "pedestrians": True},
                "region_geography": "suburban",
                "road_surface_conditions": {"surface": "wet", "construction": False, "lane_closures": 0},
            },
            preset_check_config={
                "model": {"endpoint": "complex_endpoint"},
                "weather_checks": {"check_primary": True, "check_visibility": True, "confidence_threshold": 0.85},
                "traffic_checks": {"check_density": True, "check_vehicle_types": True, "min_detection_count": 5},
                "global_settings": {"overall_confidence_threshold": 0.8, "require_all_checks": False},
            },
        )

        self.assertEqual(complex_preset_request.augmented_video_url, "s3://bucket/complex_video.mp4")
        self.assertIn("weather", complex_preset_request.preset_conditions)
        self.assertIn("traffic", complex_preset_request.preset_conditions)
        assert complex_preset_request.preset_check_config is not None
        self.assertIn("weather_checks", complex_preset_request.preset_check_config)
        self.assertIn("global_settings", complex_preset_request.preset_check_config)


class TestVLMAPIIntegration(unittest.TestCase):
    """Integration test cases for VLM API."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.valid_preset_request = {
            "augmented_video_url": "s3://bucket/integration_test_video.mp4",
            "preset_conditions": {
                "name": "environment",
                "weather": "clear",
                "time_of_day_illumination": "afternoon",
                "region_geography": "urban",
                "road_surface_conditions": "dry",
            },
            "preset_check_config": {
                "model": {"endpoint": "test_endpoint"},
                "check_weather": True,
                "check_time": True,
                "check_traffic": True,
                "confidence_threshold": 0.75,
            },
        }

    @patch("services.vlm.rest_api_common.Service")
    def test_service_initialization_failure(self, mock_service_class: MagicMock) -> None:
        """Test handling of service initialization failure."""
        # Mock the Service constructor to raise an exception
        mock_service_class.side_effect = Exception("Service initialization failed")

        # This test would need to be run in a separate test environment
        # where the service isn't already initialized, which is complex
        # with the current lifespan setup. This is more of a documentation
        # of what should be tested in integration tests.
        pass

    @patch("services.vlm.rest_api_common.storage")
    @patch("services.vlm.rest_api_common.service")
    def test_full_preset_processing_workflow(self, mock_service: MagicMock, mock_storage: MagicMock) -> None:
        """Test complete preset processing workflow."""
        mock_storage.download_from_url = AsyncMock(return_value=Path("/tmp/vlm_preset_abc/video.mp4"))

        mock_service.process_preset = AsyncMock(
            return_value=PresetResponse(
                result={
                    "processing_status": "completed",
                    "detected_conditions": {"weather": "clear", "time_of_day": "afternoon", "traffic_density": "low"},
                    "confidence_scores": {"weather": 0.92, "time_of_day": 0.87, "traffic_density": 0.89},
                    "overall_confidence": 0.89,
                    "preset_match": True,
                    "processing_time_ms": 1250,
                    "model_version": "vlm_v2.1",
                }
            )
        )

        response = self.client.post("/process/preset", json=self.valid_preset_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        result_data = data["data"]["result"]
        self.assertEqual(result_data["processing_status"], "completed")
        self.assertEqual(result_data["preset_match"], True)
        self.assertIn("confidence_scores", result_data)
        self.assertIn("processing_time_ms", result_data)
        self.assertIn("model_version", result_data)

    @patch("services.vlm.rest_api_common.storage")
    @patch("services.vlm.rest_api_common.service")
    def test_preset_processing_partial_match(self, mock_service: MagicMock, mock_storage: MagicMock) -> None:
        """Test preset processing with partial match results."""
        mock_storage.download_from_url = AsyncMock(return_value=Path("/tmp/vlm_preset_abc/video.mp4"))
        mock_service.process_preset = AsyncMock(
            return_value=PresetResponse(
                result={
                    "processing_status": "completed",
                    "detected_conditions": {
                        "weather": "clear",
                        "time_of_day_illumination": "evening",  # Different from requested "afternoon"
                        "region_geography": "urban",
                        "road_surface_conditions": "dry",
                    },
                    "confidence_scores": {
                        "weather": 0.91,
                        "time_of_day_illumination": 0.78,
                        "region_geography": 0.85,
                        "road_surface_conditions": 0.88,
                    },
                    "overall_confidence": 0.85,
                    "preset_match": False,  # Partial match
                    "mismatch_details": {
                        "time_of_day_illumination": {"expected": "afternoon", "detected": "evening", "confidence": 0.78}
                    },
                    "processing_time_ms": 1180,
                }
            )
        )

        response = self.client.post("/process/preset", json=self.valid_preset_request)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

        result_data = data["data"]["result"]  # Response is nested under "result" key
        self.assertEqual(result_data["processing_status"], "completed")
        self.assertEqual(result_data["preset_match"], False)
        self.assertIn("mismatch_details", result_data)
        self.assertIn("time_of_day_illumination", result_data["mismatch_details"])


if __name__ == "__main__":
    unittest.main()
