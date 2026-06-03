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

from checks.vlm import utils


class TestRenderPrompt(unittest.TestCase):
    """Test the render_prompt utility function."""

    def test_render_prompt_with_valid_variables(self):
        """Test render_prompt with all required variables provided."""
        template = "Weather: ${weather}, Time: ${time_of_day_illumination}"
        variables = {"weather": "sunny", "time_of_day_illumination": "daytime"}

        result = utils.render_prompt(template, variables)

        self.assertEqual(result, "Weather: sunny, Time: daytime")

    def test_render_prompt_missing_key_raises(self):
        """Test that render_prompt raises ValueError when a placeholder lacks a matching variable."""
        with self.assertRaises(ValueError) as context:
            utils.render_prompt("${weather}", {"time_of_day_illumination": "x"})

        self.assertIn("Missing variables for placeholders", str(context.exception))
        self.assertIn("weather", str(context.exception))

    def test_render_prompt_no_placeholders(self):
        """Test render_prompt with no placeholders (static prompt)."""
        template = "This is a static prompt"
        variables = {"weather": "sunny"}

        result = utils.render_prompt(template, variables)

        self.assertEqual(result, "This is a static prompt")

    def test_render_prompt_extra_variables_ignored(self):
        """Test that extra variables not in template are ignored."""
        template = "Weather: ${weather}"
        variables = {
            "weather": "sunny",
            "time_of_day_illumination": "daytime",  # Extra, should be ignored
            "region": "urban",  # Extra, should be ignored
        }

        result = utils.render_prompt(template, variables)

        self.assertEqual(result, "Weather: sunny")

    def test_render_prompt_none_value(self):
        """Test that None values are replaced with empty string."""
        template = "Weather: ${weather}, Time: ${time}"
        variables = {"weather": None, "time": "noon"}

        result = utils.render_prompt(template, variables)

        self.assertEqual(result, "Weather: , Time: noon")

    def test_render_prompt_empty_variables_raises(self):
        """Test that template with placeholders but empty variables dict raises."""
        with self.assertRaises(ValueError) as context:
            utils.render_prompt("${weather}", {})

        # Check for either error message format (the implementation uses a specific message for empty/None)
        error_msg = str(context.exception)
        self.assertTrue(
            "Missing variables for placeholders" in error_msg
            or "placeholders" in error_msg
            and "variables is None or empty" in error_msg
        )


class TestBuildMessages(unittest.TestCase):
    """Test the build_messages utility function."""

    def test_build_messages_text_only(self):
        """Test build_messages with only text content."""
        messages = utils.build_messages("Test prompt")

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["role"], "user")
        self.assertEqual(len(message["content"]), 1)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], "Test prompt")

    def test_build_messages_with_images(self):
        """Test build_messages with text and images."""
        fake_jpeg1 = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # Minimal JPEG header
        fake_jpeg2 = b"\xff\xd8\xff\xe0\x00\x10JFIF"

        messages = utils.build_messages("Test prompt", jpeg_images=[fake_jpeg1, fake_jpeg2])

        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message["role"], "user")
        self.assertIn("content", message)

        content = message["content"]
        self.assertEqual(len(content), 3)  # 1 text + 2 images
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "Test prompt")

        for i in range(1, 3):
            self.assertEqual(content[i]["type"], "image_url")
            self.assertIn("image_url", content[i])
            self.assertIn("url", content[i]["image_url"])
            self.assertTrue(content[i]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_build_messages_with_extra_texts(self):
        """Test build_messages with extra text parts."""
        extra_texts = ['{"key1": "value1"}', '{"key2": "value2"}']

        messages = utils.build_messages("Test prompt", extra_texts=extra_texts)

        self.assertEqual(len(messages), 1)
        message = messages[0]
        content = message["content"]
        self.assertEqual(len(content), 3)  # 1 main text + 2 extra texts
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "Test prompt")
        self.assertEqual(content[1]["type"], "text")
        self.assertEqual(content[1]["text"], '{"key1": "value1"}')
        self.assertEqual(content[2]["type"], "text")
        self.assertEqual(content[2]["text"], '{"key2": "value2"}')

    def test_build_messages_with_system_text(self):
        """Test build_messages with system message."""
        messages = utils.build_messages("Test prompt", system_text="You are a helpful assistant")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        self.assertEqual(messages[0]["content"][0]["text"], "You are a helpful assistant")
        self.assertEqual(messages[1]["role"], "user")

    def test_build_messages_complete(self):
        """Test build_messages with all parameters."""
        fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        extra_texts = ['{"extra": "data"}']

        messages = utils.build_messages(
            "Test prompt", extra_texts=extra_texts, system_text="System prompt", jpeg_images=[fake_jpeg]
        )

        self.assertEqual(len(messages), 2)  # system + user
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        user_content = messages[1]["content"]
        self.assertEqual(len(user_content), 3)  # prompt + extra text + 1 image
        self.assertEqual(user_content[0]["text"], "Test prompt")
        self.assertEqual(user_content[1]["text"], '{"extra": "data"}')
        self.assertEqual(user_content[2]["type"], "image_url")


class TestLoadPromptsFromFile(unittest.TestCase):
    """Test the load_prompts_from_file utility function."""

    def _write_prompts_file(self, tmp: Path, prompts_data: dict) -> str:
        """Helper to write a prompts JSON file."""
        p = tmp / "prompts.json"
        p.write_text(json.dumps(prompts_data))
        return str(p)

    def test_load_prompts_various_formats(self):
        """Test load_prompts_from_file with various prompt formats."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            prompts_data = {
                "test_key": [
                    {"name": "static_test", "prompt_type": "static", "user_prompt": "This is a static prompt"},
                    {"name": "list_prompt", "user_prompt": ["Line 1", "Line 2", "Line 3"]},
                    {"name": "with_system", "user_prompt": "User prompt", "system_prompt": "System prompt"},
                    {
                        "name": "with_json_inputs",
                        "user_prompt": "Prompt",
                        "json_inputs": [{"key": "value"}, {"key2": "value2"}],
                    },
                    {
                        "name": "invalid_prompt",
                        "user_prompt": 123,  # Invalid type
                    },
                ]
            }

            prompt_path = self._write_prompts_file(tmp, prompts_data)
            prompts = utils.load_prompts_from_file(prompt_path, "test_key")

            # Should have loaded valid prompts
            self.assertIn("static_test", prompts)
            self.assertIn("list_prompt", prompts)
            self.assertIn("with_system", prompts)
            self.assertIn("with_json_inputs", prompts)
            self.assertNotIn("invalid_prompt", prompts)  # Invalid type should be skipped

            # Test static prompt
            static_entry = prompts["static_test"]
            self.assertEqual(static_entry["user_prompt"], "This is a static prompt")

            # Test list prompt joining
            list_entry = prompts["list_prompt"]
            self.assertEqual(list_entry["user_prompt"], "Line 1\nLine 2\nLine 3")

            # Test system prompt
            system_entry = prompts["with_system"]
            self.assertEqual(system_entry["system_prompt"], "System prompt")

            # Test json_inputs
            json_entry = prompts["with_json_inputs"]
            self.assertEqual(len(json_entry["json_inputs"]), 2)
            self.assertEqual(json_entry["json_inputs"][0], '{"key": "value"}')

    def test_load_prompts_invalid_structure(self):
        """Test load_prompts_from_file with invalid prompt file structures."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # Test with non-array prompt_key
            invalid_prompts = {"test_key": {"not": "an array"}}
            p = self._write_prompts_file(tmp, invalid_prompts)

            with self.assertRaises(TypeError) as context:
                utils.load_prompts_from_file(p, "test_key")

            self.assertIn("must be an array", str(context.exception))

    def test_load_prompts_no_valid_prompts(self):
        """Test load_prompts_from_file when no valid prompts are found."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            # Test with empty array
            empty_prompts = {"test_key": []}
            p = self._write_prompts_file(tmp, empty_prompts)

            with self.assertRaises(ValueError) as context:
                utils.load_prompts_from_file(p, "test_key")

            self.assertIn("No valid prompts found", str(context.exception))

    def test_load_prompts_missing_required_fields(self):
        """Test that prompts missing required fields are skipped."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            prompts_data = {
                "test_key": [
                    {"name": "valid", "user_prompt": "Valid prompt"},
                    {"user_prompt": "No name"},  # Missing name
                    {"name": "no_prompt"},  # Missing user_prompt
                    {},  # Empty
                ]
            }

            prompt_path = self._write_prompts_file(tmp, prompts_data)
            prompts = utils.load_prompts_from_file(prompt_path, "test_key")

            # Only valid prompt should be loaded
            self.assertEqual(len(prompts), 1)
            self.assertIn("valid", prompts)


class TestStripCodeFences(unittest.TestCase):
    """Test the strip_code_fences utility function."""

    def test_strip_code_fences_with_json_marker(self):
        """Test stripping code fences with json marker."""
        text = '```json\n{"key": "value"}\n```'
        result = utils.strip_code_fences(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_strip_code_fences_plain(self):
        """Test stripping plain code fences."""
        text = '```\n{"key": "value"}\n```'
        result = utils.strip_code_fences(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_strip_code_fences_no_fences(self):
        """Test text without code fences."""
        text = '{"key": "value"}'
        result = utils.strip_code_fences(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_strip_code_fences_trailing_comma(self):
        """Test removal of trailing commas."""
        text = '{"key": "value",}'
        result = utils.strip_code_fences(text)
        self.assertEqual(result, '{"key": "value"}')


class TestJoinPromptLines(unittest.TestCase):
    """Test the join_prompt_lines utility function."""

    def test_join_prompt_lines_list(self):
        """Test joining list of strings."""
        lines = ["Line 1", "Line 2", "Line 3"]
        result = utils.join_prompt_lines(lines)
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_join_prompt_lines_string(self):
        """Test with single string (no-op)."""
        text = "Single line"
        result = utils.join_prompt_lines(text)
        self.assertEqual(result, "Single line")

    def test_join_prompt_lines_mixed_list(self):
        """Test list with non-string items."""
        lines = ["Line 1", 123, "Line 2", None, "Line 3"]
        result = utils.join_prompt_lines(lines)
        # Should only include strings
        self.assertEqual(result, "Line 1\nLine 2\nLine 3")

    def test_join_prompt_lines_invalid_type(self):
        """Test with invalid type."""
        result = utils.join_prompt_lines(123)
        self.assertIsNone(result)

    def test_join_prompt_lines_empty_list(self):
        """Test with empty list."""
        result = utils.join_prompt_lines([])
        self.assertEqual(result, "")

    def test_join_prompt_lines_empty_string(self):
        """Test with empty string."""
        result = utils.join_prompt_lines("")
        self.assertEqual(result, "")


class TestExtractFirstJsonObject(unittest.TestCase):
    """Test the extract_first_json_object utility function."""

    def test_extract_first_json_object_simple(self):
        """Test extraction of simple JSON object."""
        text = '{"key": "value"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_extract_first_json_object_with_prefix(self):
        """Test extraction when JSON is prefixed with text."""
        text = 'Some text before {"key": "value"} and after'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_extract_first_json_object_nested(self):
        """Test extraction of nested JSON object."""
        text = '{"outer": {"inner": "value"}}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"outer": {"inner": "value"}}')

    def test_extract_first_json_object_with_string_braces(self):
        """Test that braces inside strings don't affect extraction."""
        text = '{"key": "value with } brace", "other": "data"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"key": "value with } brace", "other": "data"}')

    def test_extract_first_json_object_with_escaped_quotes(self):
        """Test extraction with escaped quotes in strings."""
        text = '{"key": "value with \\"escaped\\" quotes"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"key": "value with \\"escaped\\" quotes"}')

    def test_extract_first_json_object_no_json(self):
        """Test with text containing no JSON object."""
        text = "No JSON here"
        result = utils.extract_first_json_object(text)
        self.assertIsNone(result)

    def test_extract_first_json_object_incomplete(self):
        """Test with incomplete JSON object (unclosed)."""
        text = '{"key": "value"'
        result = utils.extract_first_json_object(text)
        self.assertIsNone(result)

    def test_extract_first_json_object_multiline(self):
        """Test extraction of multiline JSON object."""
        # Test with JSON at the start (after strip)
        text = """{
            "key": "value",
            "nested": {
                "inner": "data"
            }
        }"""
        result = utils.extract_first_json_object(text)
        expected_dict = {"key": "value", "nested": {"inner": "data"}}
        self.assertEqual(result, json.dumps(expected_dict))

    def test_extract_first_json_object_multiple_objects(self):
        """Test that only the first JSON object is extracted."""
        text = '{"first": "object"} {"second": "object"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"first": "object"}')

    def test_extract_first_json_object_with_backslash(self):
        """Test extraction with backslashes in strings."""
        text = '{"path": "C:\\\\Users\\\\test"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"path": "C:\\\\Users\\\\test"}')

    def test_extract_first_json_object_with_whitespace_prefix(self):
        """Test that leading whitespace is stripped before parsing."""
        text = '   \n  {"key": "value"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_extract_first_json_object_returns_compact_json(self):
        """Test that the returned JSON is compact (no extra whitespace)."""
        text = """{
            "key":    "value",
            "number":   42
        }"""
        result = utils.extract_first_json_object(text)
        expected_dict = {"key": "value", "number": 42}
        self.assertEqual(result, json.dumps(expected_dict))

    def test_extract_first_json_object_from_unclosed_fence(self):
        """Test extraction from markdown-fenced JSON even if the fence is unclosed."""
        text = '```json\n{"key": "value"}'
        result = utils.extract_first_json_object(text)
        self.assertEqual(result, '{"key": "value"}')


class TestBuildTextContentParts(unittest.TestCase):
    """Test the _build_text_content_parts utility function."""

    def test_build_text_content_parts_text_only(self):
        """Test building content parts with only main text."""
        result = utils._build_text_content_parts("Main prompt")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[0]["text"], "Main prompt")

    def test_build_text_content_parts_with_extra_texts(self):
        """Test building content parts with extra texts."""
        result = utils._build_text_content_parts("Main prompt", extra_texts=["Extra 1", "Extra 2"])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["text"], "Main prompt")
        self.assertEqual(result[1]["text"], "Extra 1")
        self.assertEqual(result[2]["text"], "Extra 2")

    def test_build_text_content_parts_empty_extra_texts(self):
        """Test that empty strings in extra_texts are filtered out."""
        result = utils._build_text_content_parts("Main prompt", extra_texts=["", "Valid", ""])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Main prompt")
        self.assertEqual(result[1]["text"], "Valid")

    def test_build_text_content_parts_none_extra_texts(self):
        """Test with None extra_texts parameter."""
        result = utils._build_text_content_parts("Main prompt", extra_texts=None)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Main prompt")

    def test_build_text_content_parts_non_string_extra_texts(self):
        """Test that non-string items in extra_texts are filtered out."""
        result = utils._build_text_content_parts("Main prompt", extra_texts=["Valid", 123, None, "Also valid"])
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["text"], "Main prompt")
        self.assertEqual(result[1]["text"], "Valid")
        self.assertEqual(result[2]["text"], "Also valid")


class TestIsTemperatureSupported(unittest.TestCase):
    """Test the _is_temperature_supported utility function."""

    def test_is_temperature_supported_gpt4(self):
        """Test that GPT-4 models support temperature."""
        self.assertTrue(utils._is_temperature_supported("gpt-4"))
        self.assertTrue(utils._is_temperature_supported("gpt-4-turbo"))
        self.assertTrue(utils._is_temperature_supported("GPT-4"))

    def test_is_temperature_supported_gpt5(self):
        """Test that GPT-5 models don't support temperature."""
        self.assertFalse(utils._is_temperature_supported("gpt-5"))
        self.assertFalse(utils._is_temperature_supported("gpt-5-turbo"))
        self.assertFalse(utils._is_temperature_supported("GPT-5"))
        self.assertFalse(utils._is_temperature_supported("gpt-5-preview"))

    def test_is_temperature_supported_other_models(self):
        """Test that other models support temperature."""
        self.assertTrue(utils._is_temperature_supported("claude-3"))
        self.assertTrue(utils._is_temperature_supported("llama-2"))
        self.assertTrue(utils._is_temperature_supported("mistral"))

    def test_is_temperature_supported_whitespace(self):
        """Test with whitespace in model name."""
        self.assertFalse(utils._is_temperature_supported("  gpt-5  "))
        self.assertTrue(utils._is_temperature_supported("  gpt-4  "))


class TestCallLangModel(unittest.TestCase):
    """Test the call_lang_model utility function."""

    def test_call_lang_model_basic(self):
        """Test basic language model call."""

        # Mock client
        class MockChoice:
            def __init__(self):
                self.message = type("obj", (object,), {"content": "Test response"})

        class MockResult:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockCompletions:
            def create(self, **_kwargs):
                return MockResult()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client = MockClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]

        result = utils.call_lang_model(client, "gpt-4", messages, temperature=0.5)
        self.assertEqual(result, "Test response")

    def test_call_lang_model_gpt5_no_temperature(self):
        """Test that GPT-5 models don't receive temperature parameter."""
        call_kwargs = {}

        class MockChoice:
            def __init__(self):
                self.message = type("obj", (object,), {"content": "Response"})

        class MockResult:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockCompletions:
            def create(self, **kwargs):
                call_kwargs.update(kwargs)
                return MockResult()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client = MockClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]

        utils.call_lang_model(client, "gpt-5", messages, temperature=0.5)
        # Temperature should not be in kwargs for gpt-5
        self.assertNotIn("temperature", call_kwargs)
        self.assertEqual(call_kwargs["model"], "gpt-5")
        self.assertEqual(call_kwargs["messages"], messages)

    def test_call_lang_model_gpt4_with_temperature(self):
        """Test that GPT-4 models receive temperature parameter."""
        call_kwargs = {}

        class MockChoice:
            def __init__(self):
                self.message = type("obj", (object,), {"content": "Response"})

        class MockResult:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockCompletions:
            def create(self, **kwargs):
                call_kwargs.update(kwargs)
                return MockResult()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client = MockClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]

        utils.call_lang_model(client, "gpt-4", messages, temperature=0.7)
        # Temperature should be in kwargs for gpt-4
        self.assertIn("temperature", call_kwargs)
        self.assertEqual(call_kwargs["temperature"], 0.7)

    def test_call_lang_model_with_extra_params(self):
        """Test call_lang_model with extra parameters."""
        call_kwargs = {}

        class MockChoice:
            def __init__(self):
                self.message = type("obj", (object,), {"content": "Response"})

        class MockResult:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockCompletions:
            def create(self, **kwargs):
                call_kwargs.update(kwargs)
                return MockResult()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client = MockClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
        extra_params = {"max_tokens": 100, "top_p": 0.9}

        utils.call_lang_model(client, "gpt-4", messages, extra_params=extra_params)
        self.assertEqual(call_kwargs["max_tokens"], 100)
        self.assertEqual(call_kwargs["top_p"], 0.9)

    def test_call_lang_model_extra_params_none_values_filtered(self):
        """Test that None values in extra_params are not included."""
        call_kwargs = {}

        class MockChoice:
            def __init__(self):
                self.message = type("obj", (object,), {"content": "Response"})

        class MockResult:
            def __init__(self):
                self.choices = [MockChoice()]

        class MockCompletions:
            def create(self, **kwargs):
                call_kwargs.update(kwargs)
                return MockResult()

        class MockChat:
            def __init__(self):
                self.completions = MockCompletions()

        class MockClient:
            def __init__(self):
                self.chat = MockChat()

        client = MockClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
        extra_params = {"max_tokens": 100, "top_p": None}

        utils.call_lang_model(client, "gpt-4", messages, extra_params=extra_params)
        self.assertEqual(call_kwargs["max_tokens"], 100)
        self.assertNotIn("top_p", call_kwargs)


class TestStripCodeFencesAdditional(unittest.TestCase):
    """Additional tests for strip_code_fences to improve coverage."""

    def test_strip_code_fences_with_language_marker(self):
        """Test stripping code fences with different language markers."""
        test_cases = [
            ('```python\n{"key": "value"}\n```', '{"key": "value"}'),
            ('```javascript\n{"key": "value"}\n```', '{"key": "value"}'),
            ("```\ncode\n```", "code"),
        ]
        for input_text, expected in test_cases:
            result = utils.strip_code_fences(input_text)
            self.assertEqual(result, expected)

    def test_strip_code_fences_multiple_trailing_commas(self):
        """Test removal of trailing comma (regex only removes one at a time)."""
        text = '{"key": "value",}'
        result = utils.strip_code_fences(text)
        self.assertEqual(result, '{"key": "value"}')

    def test_strip_code_fences_trailing_comma_in_array(self):
        """Test removal of trailing comma before closing bracket."""
        text = '["item1", "item2",]'
        result = utils.strip_code_fences(text)
        self.assertEqual(result, '["item1", "item2"]')

    def test_strip_code_fences_trailing_comma_with_whitespace(self):
        """Test removal of trailing comma with various whitespace."""
        test_cases = [
            ('{"key": "value", }', '{"key": "value" }'),
            ('{"key": "value",  }', '{"key": "value"  }'),
            ('{"key": "value",\n}', '{"key": "value"\n}'),
        ]
        for input_text, expected in test_cases:
            result = utils.strip_code_fences(input_text)
            self.assertEqual(result, expected)

    def test_strip_code_fences_no_closing_fence(self):
        """Test with opening fence but no closing fence."""
        text = '```json\n{"key": "value"}'
        result = utils.strip_code_fences(text)
        # Should return as-is since no closing fence
        self.assertEqual(result, '```json\n{"key": "value"}')

    def test_strip_code_fences_empty_code_block(self):
        """Test with empty code block (requires at least 3 lines)."""
        text = "```\n\n```"
        result = utils.strip_code_fences(text)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
