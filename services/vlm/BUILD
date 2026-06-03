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

load("@pypi//:requirements.bzl", "requirement")
load("@rules_distroless//distroless:defs.bzl", "flatten")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load(":version.bzl", "VERSION")

py_library(
    name = "service",
    srcs = ["service.py"],
    tags = ["no-mypy"],
    visibility = ["//services/vlm/tests:__pkg__"],
    deps = [
        "//checks/vlm:preset_processor",
        "//services/framework:service_base",
        "//services/framework:url_security",
        requirement("pydantic"),
    ],
)

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//services/vlm/tests:__pkg__"],
)

py_library(
    name = "rest_api_common",
    srcs = ["rest_api_common.py"],
    data = [":generate_version_txt"],
    tags = ["no-mypy"],
    visibility = ["//services/vlm/tests:__pkg__"],
    deps = [
        ":service",
        ":settings",
        "//checks/vlm:runtime_config",
        "//services:settings_base",
        "//services:utils",
        "//services/framework:service_base",
        "//services/framework/protocols:storage_provider",
        "//services/framework/response_formatters:json_response_formatter",
        "//services/framework/storage_providers:factory",
        requirement("fastapi"),
        requirement("pydantic"),
        requirement("uvicorn"),
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = ["//utils:git_sha"],
    visibility = ["//services/vlm/tests:__pkg__"],
    deps = [":rest_api_common"],
)

# //utils:git_sha excluded from the deployed rest_api to avoid invalidating the cache for app_layer on every commit
py_binary(
    name = "rest_api_deployment",
    srcs = ["rest_api.py"],
    main = "rest_api.py",
    tags = ["manual"],
    deps = [":rest_api_common"],
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = [
        "//services/vlm:__pkg__",
        "//services/vlm/tests:__pkg__",
    ],
    deps = [
        "//services:settings_base",
        requirement("pydantic-settings"),
    ],
)

# Create a startup script that starts the app.
# System packages are pre-installed via the @vlm_packages deb layer.
genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
set -e

# Create non-root user for running the application
useradd --create-home --shell /bin/bash --uid 1001 appuser

# Ensure /tmp is world-writable (distroless base images may not set this)
chmod 1777 /tmp

# Switch to non-root user and run the application
cd /app
exec su appuser -c "./rest_api_deployment --host $${UVICORN_HOST:-0.0.0.0} --port $${UVICORN_PORT:-8000}"
EOF
""",
)

pkg_tar(
    name = "startup_layer",
    srcs = [":startup_script"],
    mode = "0755",
    package_dir = "/usr/local/bin",
    tags = ["manual"],
    visibility = ["//services/vlm:__pkg__"],
)

# Package the Python binary with all its runfiles.
# owner/group set to 1001 (appuser) at build time to avoid expensive chown -R at startup.
pkg_tar(
    name = "app_layer",
    srcs = [":rest_api_deployment"],
    include_runfiles = True,
    owner = "1001.1001",
    ownername = "appuser.appuser",
    package_dir = "/app",
    tags = ["manual"],
)

# Merge overlay package metadata into the base image's /var/lib/dpkg/status.
# This ensures dpkg -l and security scanners (e.g. nSpect) see all packages —
# both from the base image and from packages.yaml — in a single database.
genrule(
    name = "dpkg_status_merged",
    srcs = [
        "@ubuntu_22_04_image_linux_amd64",
        "@vlm_packages//:dpkg_status",
    ],
    outs = ["dpkg_status_merged.tar"],
    cmd = "$(execpath //services:merge_dpkg_status) $(location @ubuntu_22_04_image_linux_amd64//:ubuntu_22_04_image_linux_amd64) $(location @vlm_packages//:dpkg_status) $@",
    tags = ["manual"],
    tools = ["//services:merge_dpkg_status"],
)

pkg_tar(
    name = "third_party_attributions_layer",
    srcs = ["//services/vlm:THIRD-PARTY-NOTICES.txt"],
    package_dir = "/app",
    tags = ["manual"],
)

flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = [
        "@vlm_packages//:packages",
        ":dpkg_status_merged",
    ],
)

oci_image(
    name = "image",
    base = "@ubuntu_22_04_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    },
    exposed_ports = ["8000"],
    labels = "//utils:git_sha",
    tags = ["manual"],
    tars = [
        ":system_packages_layer",
        "//services:python3_symlink_layer",
        ":startup_layer",
        ":app_layer",
        ":third_party_attributions_layer",
        "//services:git_sha_layer",
    ],
    workdir = "/app",
)

oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["vlm:" + VERSION],
    tags = ["manual"],
    visibility = ["//services/vlm/tests:__pkg__"],
)
