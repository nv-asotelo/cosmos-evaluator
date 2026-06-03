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
FastAPI application for the Environment Preset Service.

This application provides a REST API for the Environment Preset Service,
providing a synchronous endpoint for environment preset processing.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from checks.vlm.runtime_config import runtime_summary, set_active_endpoint
from services import utils
from services.framework.protocols.storage_provider import StorageProvider
from services.framework.response_formatters.json_response_formatter import JsonResponseFormatter
from services.framework.storage_providers.factory import build_storage_provider
from services.settings_base import Environment
from services.vlm.service import PresetRequest, PresetResponse, Service
from services.vlm.settings import get_settings

# Get settings instance
settings = get_settings()

# Create logger
logger = logging.getLogger(__name__)

# Global service instance
service: Optional[Service] = None
storage: Optional[StorageProvider] = None

# Initialize request handlers and response formatters
json_formatter = JsonResponseFormatter()


class VlmRuntimeSwitchRequest(BaseModel):
    endpoint: str = Field(..., description="Configured VLM endpoint key to make active", min_length=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan manager for service initialization and cleanup.

    Args:
        _: FastAPI application instance
    """
    global service, storage

    # Startup
    logger.info("Initializing Vision Language Model Service...")

    service = Service()
    storage = build_storage_provider(settings)

    logger.info("Service initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down service...")
    logger.info("Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="VLM Service API",
    description="REST API for Vision Language Model processing",
    version=utils.get_contents_from_runfile("services/vlm/version.txt"),
    lifespan=lifespan,
    debug=settings.env == Environment.LOCAL,  # Enable debug in local
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await json_formatter.format_success(
        {
            "service": app.title,
            "status": "healthy",
            "version": app.version,
            "git_sha": utils.get_git_sha(),
            "environment": settings.env.value,
        }
    )


@app.post("/process/preset", response_model=PresetResponse)
async def process_preset(request: PresetRequest) -> JSONResponse:
    """
    Process environment preset.

    This endpoint processes an environment preset and returns the result.
    Downloads the video from ``augmented_video_url`` to a local temp file
    before passing it to the checker, which requires a local filesystem path.

    Args:
        request: The request containing the video URL, preset conditions, and preset check configuration

    Returns:
        The environment preset result
    """
    global service, storage
    start_time = time.perf_counter()
    temp_dir: str | None = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="vlm_preset_")
        local_video_path = Path(temp_dir) / "video.mp4"
        await storage.download_from_url(request.augmented_video_url, local_video_path)

        response = await service.process_preset(request, str(local_video_path))
        duration_ms = (time.perf_counter() - start_time) * 1000
        return await json_formatter.format_success(
            response.model_dump(mode="json"), metadata={"duration_ms": duration_ms}
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Error processing preset: {e}")
        return await json_formatter.format_error(e, status_code=500, metadata={"duration_ms": duration_ms})
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/config")
async def get_default_config() -> JSONResponse:
    """Get the default configuration for VLM processing."""
    global service
    try:
        if service is None:
            raise RuntimeError("Service not initialized")
        default_config = await service.get_default_config()

        return await json_formatter.format_success(
            {"default_config": default_config, "description": "Default configuration for VLM processing"}
        )
    except Exception as e:
        logger.error(f"Error getting default config: {e}")
        return await json_formatter.format_error(e, status_code=500)


@app.get("/runtime/vlm")
async def get_vlm_runtime() -> JSONResponse:
    """Get active VLM endpoint and configured endpoint metadata."""
    try:
        return await json_formatter.format_success(runtime_summary())
    except Exception as e:
        logger.error(f"Error getting VLM runtime: {e}")
        return await json_formatter.format_error(e, status_code=500)


@app.get("/runtime/vlm/endpoints")
async def get_vlm_runtime_endpoints() -> JSONResponse:
    """List configured VLM endpoints."""
    try:
        summary = runtime_summary()
        return await json_formatter.format_success(
            {
                "active_endpoint": summary["active_endpoint"],
                "available_endpoints": summary["available_endpoints"],
            }
        )
    except Exception as e:
        logger.error(f"Error getting VLM endpoints: {e}")
        return await json_formatter.format_error(e, status_code=500)


@app.post("/runtime/vlm/switch")
async def switch_vlm_runtime(request: VlmRuntimeSwitchRequest) -> JSONResponse:
    """Persist the active VLM endpoint used when requests omit a model override."""
    try:
        return await json_formatter.format_success(set_active_endpoint(request.endpoint))
    except ValueError as e:
        return await json_formatter.format_error(e, status_code=400)
    except Exception as e:
        logger.error(f"Error switching VLM endpoint: {e}")
        return await json_formatter.format_error(e, status_code=500)
