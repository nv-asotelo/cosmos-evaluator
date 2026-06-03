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
load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "client_manager_test",
    srcs = ["client_manager_test.py"],
    tags = ["no-mypy"],
    deps = ["//checks/vlm:client_manager"],
)

py_test(
    name = "preset_processor_test",
    srcs = ["preset_processor_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/vlm:preset_processor",
        requirement("opencv_python_headless"),
        requirement("numpy"),
    ],
)

py_test(
    name = "runtime_config_test",
    srcs = ["runtime_config_test.py"],
    tags = ["no-mypy"],
    deps = ["//checks/vlm:runtime_config"],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    tags = ["no-mypy"],
    deps = ["//checks/vlm:utils"],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    tags = ["no-mypy"],
    deps = [
        "//checks/utils:cli_common",
        "//checks/vlm:run",
    ],
)
