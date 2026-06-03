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
import time
from typing import Any, Dict, Optional

from checks.vlm.client_manager import ClientManager
from utils.bazel import get_runfiles_path


DEFAULT_ENDPOINT = "cosmos3-super-reasoner"
RUNTIME_ENDPOINT_ENV = "COSMOS_EVALUATOR_VLM_ENDPOINT"
RUNTIME_STATE_ENV = "COSMOS_EVALUATOR_VLM_RUNTIME_CONFIG"
DEFAULT_RUNTIME_STATE = "~/.cosmos_evaluator/vlm_runtime.json"


def _state_path(path: Optional[str] = None) -> Path:
    return Path(os.path.expanduser(path or os.environ.get(RUNTIME_STATE_ENV, DEFAULT_RUNTIME_STATE)))


def deep_merge(base: Dict[str, Any], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(base or {})
    for key, value in (update or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_endpoint_config(public_config_path: Optional[str] = None) -> Dict[str, Any]:
    config_path = public_config_path or get_runfiles_path("checks/vlm/config/endpoints.json")
    return ClientManager(config_path).config


def read_runtime_state(state_file: Optional[str] = None) -> Dict[str, Any]:
    path = _state_path(state_file)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid runtime VLM state file '{path}': {exc}") from exc


def _configured_endpoint(default_endpoint: Optional[str] = None, state_file: Optional[str] = None) -> str:
    state_endpoint = read_runtime_state(state_file).get("endpoint")
    env_endpoint = os.environ.get(RUNTIME_ENDPOINT_ENV)
    return str(state_endpoint or env_endpoint or default_endpoint or DEFAULT_ENDPOINT)


def _request_has_preset_endpoint(request_config: Optional[Dict[str, Any]]) -> bool:
    model_cfg = (request_config or {}).get("model")
    return isinstance(model_cfg, dict) and bool(model_cfg.get("endpoint"))


def resolve_preset_check_config(
    default_config: Dict[str, Any],
    request_config: Optional[Dict[str, Any]] = None,
    state_file: Optional[str] = None,
) -> Dict[str, Any]:
    config = deep_merge(default_config, request_config)
    model_cfg = config.setdefault("model", {})
    if not _request_has_preset_endpoint(request_config):
        model_cfg["endpoint"] = _configured_endpoint(model_cfg.get("endpoint"), state_file)
    return config


def _endpoint_public_view(name: str, endpoint_config: Dict[str, Any]) -> Dict[str, Any]:
    env_var = endpoint_config.get("env_var")
    return {
        "endpoint": name,
        "base_url": endpoint_config.get("base_url"),
        "model": endpoint_config.get("model"),
        "env_var": env_var,
        "timeout": endpoint_config.get("timeout"),
        "api_key_optional": bool(endpoint_config.get("api_key_optional")),
        "credentials_present": bool(os.environ.get(env_var)) if env_var else True,
        "nim_image": endpoint_config.get("nim_image"),
        "nim_model_size": endpoint_config.get("nim_model_size"),
    }


def runtime_summary(
    public_config_path: Optional[str] = None,
    state_file: Optional[str] = None,
) -> Dict[str, Any]:
    endpoints = load_endpoint_config(public_config_path)
    active = _configured_endpoint(DEFAULT_ENDPOINT, state_file)
    active_config = endpoints.get(active, {})
    return {
        "active_endpoint": active,
        "active": _endpoint_public_view(active, active_config) if active_config else None,
        "available_endpoints": [
            _endpoint_public_view(name, cfg) for name, cfg in sorted(endpoints.items())
        ],
        "state_file": str(_state_path(state_file)),
        "state": read_runtime_state(state_file),
    }


def set_active_endpoint(
    endpoint: str,
    public_config_path: Optional[str] = None,
    state_file: Optional[str] = None,
) -> Dict[str, Any]:
    endpoints = load_endpoint_config(public_config_path)
    if endpoint not in endpoints:
        available = ", ".join(sorted(endpoints.keys()))
        raise ValueError(f"Unknown VLM endpoint '{endpoint}'. Available endpoints: {available}")

    cfg = endpoints[endpoint]
    state = {
        "endpoint": endpoint,
        "base_url": cfg.get("base_url"),
        "model": cfg.get("model"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _state_path(state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)
    return runtime_summary(public_config_path, state_file)


def _request_has_attribute_vlm(request_config: Optional[Dict[str, Any]]) -> bool:
    vlm_config = ((request_config or {}).get("vlm_verification") or {}).get("vlm")
    return isinstance(vlm_config, dict) and bool(vlm_config.get("endpoint") or vlm_config.get("model"))


def apply_runtime_attribute_vlm_config(
    config: Dict[str, Any],
    request_config: Optional[Dict[str, Any]] = None,
    public_config_path: Optional[str] = None,
    state_file: Optional[str] = None,
) -> Dict[str, Any]:
    if _request_has_attribute_vlm(request_config):
        return config

    endpoints = load_endpoint_config(public_config_path)
    active = _configured_endpoint(DEFAULT_ENDPOINT, state_file)
    endpoint_config = endpoints.get(active)
    if not endpoint_config:
        raise ValueError(f"Active VLM endpoint '{active}' is not configured")

    result = deep_merge(config, {})
    vlm_config = result.setdefault("vlm_verification", {}).setdefault("vlm", {})
    vlm_config["endpoint"] = endpoint_config["base_url"].rstrip("/")
    vlm_config["model"] = endpoint_config["model"]
    return result
