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
load("@rules_pex//pex:defs.bzl", "py_pex_binary")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load(":version.bzl", "VERSION")

# ======================================
# Application packages
# ======================================

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//services/attribute_verification/tests:__pkg__"],
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = [
        "//services/attribute_verification:__pkg__",
        "//services/attribute_verification/tests:__pkg__",
    ],
    deps = [
        "//services:settings_base",
        requirement("pydantic-settings"),
    ],
)

py_library(
    name = "attribute_verification_service",
    srcs = ["attribute_verification_service.py"],
    visibility = ["//services/attribute_verification/tests:__pkg__"],
    deps = [
        "//checks/attribute_verification:processor",
        "//checks/utils:multistorage",
        "//checks/vlm:runtime_config",
        "//services/framework:service_base",
        requirement("pydantic"),
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = [
        ":generate_version_txt",
        "//checks/vlm/config:endpoints",
        "//utils:git_sha",
    ],
    main = "rest_api.py",
    visibility = ["//services/attribute_verification/tests:__pkg__"],
    deps = [
        ":attribute_verification_service",
        ":settings",
        "//checks/utils:multistorage",
        "//checks/vlm:runtime_config",
        "//services:settings_base",
        "//services:utils",
        "//services/framework/response_formatters:json_response_formatter",
        requirement("pydantic"),
        requirement("pydantic-settings"),
        requirement("fastapi"),
        requirement("uvicorn"),
    ],
)

# Package the Rest API code and its dependency tree into a single executable for deployment
py_pex_binary(
    name = "rest_api_pex",
    binary = ":rest_api",
)

# ======================================
# Container Image Build
# ======================================

# System packages layer adds the system packages created from packages.yaml to the container image
# Flatten deb package data into a single layer (excludes dpkg_status)
flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = ["@attribute_verification_packages//:packages"],
)

# License layer adds the third party notices and license files to the container image
pkg_tar(
    name = "license_layer",
    srcs = ["//services/attribute_verification:THIRD-PARTY-NOTICES.txt"],
    owner = "1000.1000",
    ownername = "nvs.nvs",
    package_dir = "/app",
    tags = ["manual"],
)

# App layer adds a python executable with all python code and dependencies to the container image
pkg_tar(
    name = "app_layer",
    srcs = [":rest_api_pex"],
    owner = "1000.1000",
    ownername = "nvs.nvs",
    package_dir = "/app",
    tags = ["manual"],
)

# Configuration layer adds the configuration files to the container image
pkg_tar(
    name = "config_layer",
    srcs = ["//checks:config"],
    owner = "1000.1000",
    ownername = "nvs.nvs",
    package_dir = "/app/config",
    tags = ["manual"],
)

oci_image(
    name = "image",
    base = "@nvidia_distroless_python_313_image",
    entrypoint = [
        "/usr/local/bin/python3.13",
        "/app/rest_api_pex.pex",
    ],
    env = {
        "CONFIG_DIR": "/app/config",
    },
    exposed_ports = ["8000"],
    labels = "//utils:git_sha",
    tags = ["manual"],
    tars = [
        ":system_packages_layer",
        ":license_layer",
        ":app_layer",
        ":config_layer",
    ],
    workdir = "/app",
)

# ======================================
# Container Image Publishing and Testing
# ======================================

# Load the container image for testing
oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["attribute-verification-checker:" + VERSION],
    tags = ["manual"],
)
