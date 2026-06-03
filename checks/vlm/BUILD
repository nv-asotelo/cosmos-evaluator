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
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "client_manager",
    srcs = ["client_manager.py"],
    data = ["//checks/vlm/config:endpoints"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/vlm/tests:__pkg__",
    ],
    deps = [
        "//checks/utils:video",
        requirement("openai"),
    ],
)

py_library(
    name = "preset_processor",
    srcs = ["preset_processor.py"],
    data = [
        "//checks:config",
        "//checks/vlm/config:prompts",
    ],
    tags = ["no-mypy"],
    visibility = [
        "//checks/vlm/tests:__pkg__",
        "//services/vlm:__pkg__",
    ],
    deps = [
        ":client_manager",
        ":runtime_config",
        ":utils",
        "//checks/utils:config_manager",
        "//checks/utils:video",
        "//utils:bazel",
    ],
)

py_library(
    name = "runtime_config",
    srcs = ["runtime_config.py"],
    data = ["//checks/vlm/config:endpoints"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/vlm/tests:__pkg__",
        "//services/attribute_verification:__pkg__",
        "//services/attribute_verification/tests:__pkg__",
        "//services/vlm:__pkg__",
        "//services/vlm/tests:__pkg__",
    ],
    deps = [
        ":client_manager",
        "//utils:bazel",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    tags = ["no-mypy"],
    visibility = [
        "//checks/vlm/tests:__pkg__",
    ],
    deps = ["//checks/utils:video"],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = ["//checks:config"],
    tags = ["no-mypy"],
    visibility = ["//checks/vlm/tests:__pkg__"],
    deps = [
        ":preset_processor",
        "//checks/utils:cli_common",
        "//checks/utils:config_manager",
    ],
)
