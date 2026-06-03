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

from openai import AzureOpenAI, OpenAI


class ClientManager:
    """Manager for VLM clients."""

    def __init__(self, config_file):
        """Initialize client manager with JSON file.

        Args:
            config_file: The path to the configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON file and optional private overrides from ~/.cosmos_evaluator/endpoints.json.

        Returns:
            The configuration

        Raises:
            FileNotFoundError: If the configuration file is not found
            ValueError: If the configuration file is invalid
            ValueError: If the configuration file is not a JSON object
            ValueError: If the private configuration file is invalid
            ValueError: If the private configuration file is not a JSON object
        """
        try:
            with open(self.config_file, "r") as f:
                public_config = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e

        if not isinstance(public_config, dict):
            raise ValueError("Top-level configuration must be a JSON object")

        # Attempt to load private overrides from fixed path: ~/.cosmos_evaluator/endpoints.json
        home_dir = os.path.expanduser("~")
        private_config_path = os.path.join(home_dir, ".cosmos_evaluator", "endpoints.json")

        if os.path.exists(private_config_path):
            try:
                with open(private_config_path, "r") as lf:
                    private_config = json.load(lf)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in private configuration file '{private_config_path}': {e}") from e

            if not isinstance(private_config, dict):
                raise ValueError("Top-level private configuration must be a JSON object")

            # Merge: private overrides public per endpoint key; shallow merge of dicts
            for endpoint_key, private_value in private_config.items():
                if (
                    endpoint_key in public_config
                    and isinstance(public_config[endpoint_key], dict)
                    and isinstance(private_value, dict)
                ):
                    merged = dict(public_config[endpoint_key])
                    merged.update(private_value)
                    public_config[endpoint_key] = merged
                else:
                    public_config[endpoint_key] = private_value

        return public_config

    def create_client(self, endpoint_type):
        """Create VLM client with given endpoint.

        Args:
            endpoint_type: The type of endpoint to create

        Returns:
            The VLM client and model

        Raises:
            ValueError: If the endpoint_type is invalid
            ValueError: If the environment variable is not set
        """
        if endpoint_type not in self.config.keys():
            available_endpoints = ", ".join(self.config.keys())
            raise ValueError(f"Invalid endpoint_type: {endpoint_type}.  Please, choose one of: {available_endpoints}")

        endpoint_config = self.config[endpoint_type]
        env_var = endpoint_config.get("env_var")
        api_key = os.environ.get(env_var, "") if env_var else ""
        if not api_key and endpoint_config.get("api_key_optional"):
            api_key = endpoint_config.get("api_key_placeholder", "not-used")
        if not api_key:
            raise ValueError(f"Environment variable {env_var} is required")

        default_timeout = 60  # seconds
        if endpoint_type == "azure_openai":
            return (
                AzureOpenAI(
                    api_version=endpoint_config["api_version"],
                    azure_endpoint=endpoint_config["base_url"],
                    api_key=api_key,
                    timeout=endpoint_config.get("timeout", default_timeout),
                ),
                endpoint_config["model"],
            )
        elif endpoint_type.endswith("_openai"):
            return (
                OpenAI(api_key=api_key, timeout=endpoint_config.get("timeout", default_timeout)),
                endpoint_config["model"],
            )
        else:
            return (
                OpenAI(
                    base_url=endpoint_config["base_url"],
                    api_key=api_key,
                    timeout=endpoint_config.get("timeout", default_timeout),
                ),
                endpoint_config["model"],
            )

    def get_endpoints(self):
        """Get all endpoints configuration."""
        return self.config.keys()
