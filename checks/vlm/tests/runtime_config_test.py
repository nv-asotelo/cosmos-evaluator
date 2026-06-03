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
import unittest

from checks.vlm.runtime_config import (
    apply_runtime_attribute_vlm_config,
    resolve_preset_check_config,
    runtime_summary,
    set_active_endpoint,
)


def write_endpoints(tmp: Path) -> str:
    endpoints = {
        "cosmos3-super-reasoner": {
            "base_url": "http://cosmos3-nim:8000/v1",
            "model": "nvidia/cosmos3-super-reasoner",
            "env_var": "COSMOS3_NIM_API_KEY",
            "api_key_optional": True,
        },
        "other": {
            "base_url": "https://example.test/v1",
            "model": "example/model",
            "env_var": "OTHER_API_KEY",
        },
    }
    path = tmp / "endpoints.json"
    path.write_text(json.dumps(endpoints), encoding="utf-8")
    return str(path)


class TestRuntimeConfig(unittest.TestCase):
    def test_switch_persists_and_summary_masks_credentials(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            endpoints_path = write_endpoints(tmp)
            state_path = str(tmp / "runtime.json")

            summary = set_active_endpoint("other", endpoints_path, state_path)

            self.assertEqual(summary["active_endpoint"], "other")
            self.assertEqual(summary["active"]["model"], "example/model")
            self.assertFalse(summary["active"]["credentials_present"])
            self.assertEqual(json.loads(Path(state_path).read_text())["endpoint"], "other")

    def test_resolve_preset_uses_runtime_when_request_omits_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            endpoints_path = write_endpoints(tmp)
            state_path = str(tmp / "runtime.json")
            set_active_endpoint("other", endpoints_path, state_path)

            resolved = resolve_preset_check_config(
                {"model": {"endpoint": "cosmos3-super-reasoner", "temperature": 0.0}},
                {"model": {"temperature": 0.2}},
                state_file=state_path,
            )

            self.assertEqual(resolved["model"]["endpoint"], "other")
            self.assertEqual(resolved["model"]["temperature"], 0.2)

    def test_resolve_preset_preserves_explicit_request_endpoint(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            endpoints_path = write_endpoints(tmp)
            state_path = str(tmp / "runtime.json")
            set_active_endpoint("other", endpoints_path, state_path)

            resolved = resolve_preset_check_config(
                {"model": {"endpoint": "cosmos3-super-reasoner"}},
                {"model": {"endpoint": "cosmos3-super-reasoner"}},
                state_file=state_path,
            )

            self.assertEqual(resolved["model"]["endpoint"], "cosmos3-super-reasoner")

    def test_attribute_runtime_config_only_replaces_vlm(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            endpoints_path = write_endpoints(tmp)
            state_path = str(tmp / "runtime.json")
            set_active_endpoint("other", endpoints_path, state_path)

            config = {
                "question_generation": {"llm": {"endpoint": "https://llm", "model": "llm-model"}},
                "vlm_verification": {"vlm": {"endpoint": "old", "model": "old-model"}},
            }
            resolved = apply_runtime_attribute_vlm_config(
                config,
                public_config_path=endpoints_path,
                state_file=state_path,
            )

            self.assertEqual(resolved["question_generation"]["llm"]["endpoint"], "https://llm")
            self.assertEqual(resolved["vlm_verification"]["vlm"]["endpoint"], "https://example.test/v1")
            self.assertEqual(resolved["vlm_verification"]["vlm"]["model"], "example/model")

    def test_attribute_explicit_request_vlm_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            endpoints_path = write_endpoints(tmp)
            state_path = str(tmp / "runtime.json")
            set_active_endpoint("other", endpoints_path, state_path)

            config = {"vlm_verification": {"vlm": {"endpoint": "request", "model": "request-model"}}}
            resolved = apply_runtime_attribute_vlm_config(
                config,
                {"vlm_verification": {"vlm": {"endpoint": "request", "model": "request-model"}}},
                endpoints_path,
                state_path,
            )

            self.assertEqual(resolved["vlm_verification"]["vlm"]["endpoint"], "request")
            self.assertEqual(resolved["vlm_verification"]["vlm"]["model"], "request-model")

    def test_summary_defaults_to_cosmos3(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            endpoints_path = write_endpoints(tmp)

            summary = runtime_summary(endpoints_path, str(tmp / "missing.json"))

            self.assertEqual(summary["active_endpoint"], "cosmos3-super-reasoner")


if __name__ == "__main__":
    unittest.main()
