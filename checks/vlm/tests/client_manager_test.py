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
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from checks.vlm.client_manager import ClientManager


def write_json(tmp_dir: Path, rel_path: str, data: dict) -> str:
    target = tmp_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data))
    return str(target)


class TestClientManager(unittest.TestCase):
    def test_invalid_endpoint_raises(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public = {
                "azure_openai": {
                    "api_version": "2025-01-01-preview",
                    "base_url": "https://example",
                    "model": "gpt-4o-mini-20240718",
                    "env_var": "AZURE_OPENAI_API_KEY",
                    "timeout": 300,
                },
            }
            public_path = write_json(tmp, "checks/vlm/config/endpoints.json", public)

            manager = ClientManager(public_path)
            with self.assertRaises(ValueError) as ctx:
                manager.create_client("nonexistent")
            self.assertIn("Invalid endpoint_type", str(ctx.exception))

    def test_env_var_required(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public = {
                "azure_openai": {
                    "api_version": "2025-01-01-preview",
                    "base_url": "https://example",
                    "model": "gpt-4o-mini-20240718",
                    "env_var": "AZURE_OPENAI_API_KEY",
                },
            }
            public_path = write_json(tmp, "checks/vlm/config/endpoints.json", public)

            # Simulate missing/empty API key without assigning None (which isn't allowed in os.environ)
            with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": ""}, clear=False):
                manager = ClientManager(public_path)
                with self.assertRaises(ValueError) as ctx:
                    manager.create_client("azure_openai")
                self.assertIn("Environment variable AZURE_OPENAI_API_KEY is required", str(ctx.exception))

    def test_api_key_optional_uses_placeholder(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public = {
                "local_nim": {
                    "base_url": "http://localhost:8000/v1",
                    "model": "nvidia/cosmos3-super-reasoner",
                    "env_var": "LOCAL_NIM_API_KEY",
                    "api_key_optional": True,
                },
            }
            public_path = write_json(tmp, "checks/vlm/config/endpoints.json", public)

            with patch.dict(os.environ, {"LOCAL_NIM_API_KEY": ""}, clear=False):
                with patch("checks.vlm.client_manager.OpenAI") as mock_openai:
                    manager = ClientManager(public_path)
                    _, model = manager.create_client("local_nim")

            self.assertEqual(model, "nvidia/cosmos3-super-reasoner")
            mock_openai.assert_called_once()
            self.assertEqual(mock_openai.call_args.kwargs["api_key"], "not-used")

    def test_private_overrides_public(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            public = {
                "azure_openai": {
                    "api_version": "2025-01-01-preview",
                    "base_url": "https://public",
                    "model": "public-model",
                    "env_var": "AZURE_OPENAI_API_KEY",
                    "timeout": 300,
                },
            }
            private_cfg = {
                "azure_openai": {
                    "base_url": "https://private",
                    "model": "private-model",
                    "timeout": 120,
                },
                "custom_service": {
                    "base_url": "https://custom",
                    "model": "custom-model",
                    "env_var": "CUSTOM_API_KEY",
                },
            }

            public_path = write_json(tmp, "checks/vlm/config/endpoints.json", public)
            # Write private file to HOME/.cosmos_evaluator/endpoints.json
            home_auto = tmp / ".cosmos_evaluator"
            home_auto.mkdir(parents=True, exist_ok=True)
            private_path = write_json(tmp, ".cosmos_evaluator/endpoints.json", private_cfg)
            self.assertTrue(os.path.exists(private_path))
            # Ensure ClientManager resolves HOME from tmp during test
            with patch.dict(os.environ, {"HOME": str(tmp)}, clear=False):
                with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "dummy", "CUSTOM_API_KEY": "dummy"}, clear=False):
                    manager = ClientManager(public_path)

                # Ensure merged values are from private
                azure_cfg = manager.config["azure_openai"]
                self.assertEqual(azure_cfg["base_url"], "https://private")
                self.assertEqual(azure_cfg["model"], "private-model")
                self.assertEqual(azure_cfg["timeout"], 120)

                # New endpoint from private exists
                self.assertIn("custom_service", manager.config)


if __name__ == "__main__":
    unittest.main()
