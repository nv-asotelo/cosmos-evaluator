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

"""
AttributeVerificationService implementation for processing attribute verification.

This service wraps the AttributeVerificationProcessor and provides a standardized
service interface for deployment as a microservice.
"""

import logging
import os
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from checks.attribute_verification.processor import AttributeVerificationProcessor, AttributeVerificationResult
from checks.utils.multistorage import validate_uri
from checks.vlm.runtime_config import apply_runtime_attribute_vlm_config, deep_merge
from services.framework.service_base import ServiceBase


class AttributeVerificationRequest(BaseModel):
    clip_id: str
    augmented_video_path: str
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration dictionary for processing")
    verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")


class AttributeVerificationService(ServiceBase[AttributeVerificationRequest, AttributeVerificationResult]):
    def __init__(self, verbose: str = "INFO") -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, verbose.upper()))

    async def validate_input(self, request: AttributeVerificationRequest) -> bool:
        """
        Validate that the augmented video path exists in multistorage and can be accessed.

        Args:
            request: The attribute verification request containing the augmented video path

        Returns:
            True if the augmented video path is valid and accessible

        Raises:
            ValueError: If the augmented video path does not exist
        """
        if not validate_uri(request.augmented_video_path, is_file=True):
            raise ValueError(
                'The provided augmented video path (field "augmented_video_path") does not exist, or cannot be accessed'
            )

        return True

    @staticmethod
    async def get_default_config() -> Dict[str, Any]:
        """
        Get the default configuration for attribute verification processing.

        Returns:
            Default configuration for attribute verification processing
        """
        config_dir = os.getenv("CONFIG_DIR", None)
        return AttributeVerificationProcessor.get_default_config(config_dir).get(
            "metropolis.attribute_verification", {}
        )

    async def process(self, request: AttributeVerificationRequest) -> AttributeVerificationResult:
        config = await AttributeVerificationService.get_default_config()
        if request.config is not None:
            config = deep_merge(config, request.config)
        config = apply_runtime_attribute_vlm_config(config, request.config)
        config_dir = os.getenv("CONFIG_DIR", None)
        processor = AttributeVerificationProcessor(params=config, config_dir=config_dir, verbose=request.verbose)
        return await processor.process(request.clip_id, request.augmented_video_path)
