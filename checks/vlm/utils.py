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
import logging
import re
from typing import Any, Dict, List, Optional, Union

from checks.utils.video import jpeg_bytes_to_data_url

logger = logging.getLogger(__name__)
# Default JPEG encode quality for images sent to VLM unless overridden
JPEG_QUALITY: int = 85


def load_prompts_from_file(template_json_path: str, prompt_key: str) -> Dict[str, Dict[str, Any]]:
    """Load prompts from a JSON file.

    Each prompt object should include:
        - name: str
        - user_prompt: str or [str]
        - system_prompt: str or [str] (optional)
        - json_inputs: [dict] (optional) - array of JSON objects to include as extra text parts

    Args:
        template_json_path: Path to the JSON file containing prompts
        prompt_key: Top-level key in the JSON file (e.g., "preset_check")

    Returns:
        Dict mapping lowercased prompt name -> {user_prompt, system_prompt, json_inputs}

    Raises:
        TypeError: If prompt_key is not an array or no valid prompts found
        ValueError: If no valid prompts found
    """
    with open(template_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node = data.get(prompt_key)
    if not isinstance(node, list):
        raise TypeError(f"'{prompt_key}' must be an array of prompt objects in {template_json_path}")

    prompts: Dict[str, Dict[str, Any]] = {}
    for item in node:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        system_prompt = item.get("system_prompt")
        user_prompt = item.get("user_prompt")
        json_inputs = item.get("json_inputs", [])

        if not name or user_prompt is None:
            continue

        # Join prompt lines if it's an array
        user_prompt_str = join_prompt_lines(user_prompt)
        system_prompt_str = join_prompt_lines(system_prompt) if system_prompt is not None else None

        if not isinstance(user_prompt_str, str):
            continue

        # Process json_inputs: convert each to a JSON string
        json_input_strings: List[str] = []
        if isinstance(json_inputs, list):
            for json_obj in json_inputs:
                if isinstance(json_obj, (dict, list)):
                    json_input_strings.append(json.dumps(json_obj, ensure_ascii=False))

        key = str(name).strip().lower()
        prompts[key] = {
            "user_prompt": user_prompt_str,
            "system_prompt": system_prompt_str,
            "json_inputs": json_input_strings,
        }

    if not prompts:
        raise ValueError(f"No valid prompts found under '{prompt_key}' in {template_json_path}")

    return prompts


def join_prompt_lines(prompt: Union[List[str], str]) -> str:
    """Join prompt lines if it's an array.

    Args:
        prompt: Prompt string or list of strings

    Returns:
        Joined prompt string or None if prompt is not a string or list
    """
    if isinstance(prompt, list):
        lines = [p for p in prompt if isinstance(p, str)]
        return "\n".join(lines)

    if isinstance(prompt, str):
        return prompt


def render_prompt(prompt_template: str, variables: Optional[Dict[str, str]]) -> str:
    """Render prompt by replacing placeholders with strict validation.

    Placeholder syntax: ${var_name}

    Behavior:
    - Finds all ${var_name} instances in template
    - Replaces each with corresponding value from variables dict
    - Raises ValueError if any ${var_name} lacks a matching key in variables
    - Extra variables not in template are ignored

    Args:
        prompt_template: Template string with ${placeholder} syntax
        variables: Dict of variable names to values (optional)

    Returns:
        Rendered string with all placeholders replaced

    Raises:
        ValueError: If any placeholder in template is missing from variables dict
    """
    # Find all ${var_name} placeholders
    placeholders = re.findall(r"\$\{(\w+)\}", prompt_template)

    if not placeholders:
        # No placeholders to replace
        return prompt_template

    # Check if all placeholders have corresponding variables
    if not variables:
        raise ValueError(f"Template contains placeholders {placeholders} but variables is None or empty")

    placeholder_set = set(placeholders)
    missing = placeholder_set - set(variables.keys())
    if missing:
        raise ValueError(f"Missing variables for placeholders: {sorted(missing)}")

    # Replace all placeholders (sort by length descending to avoid partial replacements)
    rendered = prompt_template
    for key in sorted(placeholder_set, key=len, reverse=True):
        placeholder = "${" + str(key) + "}"
        value = variables[key]
        # Convert None to empty string, otherwise convert to string
        rendered = rendered.replace(placeholder, "" if value is None else str(value))

    return rendered


def strip_code_fences(text: str) -> str:
    """Strip code fences and fix common JSON formatting issues from LLM output.

    Args:
        text: Response text that may contain markdown code fences

    Returns:
        Cleaned text with fences removed and trailing commas fixed
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            t = "\n".join(lines[1:-1])

    # Remove trailing commas before } or ] (common LLM JSON output error)
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return t


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object from a string, ignoring strings and comments.

    Args:
        text: String to extract JSON from

    Returns:
        First JSON object or None if no JSON found
    """
    s = text.strip()
    decoder = json.JSONDecoder()
    for start, char in enumerate(s):
        if char != "{":
            continue
        try:
            first_json_object, _ = decoder.raw_decode(s[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(first_json_object, dict):
            return json.dumps(first_json_object)

    logger.error("Failed to extract first JSON object from %s", s)
    return None


def _build_text_content_parts(prompt_text: str, extra_texts: Optional[List[str]] = None) -> List[Dict]:
    """Build text content parts for messages.

    Args:
        prompt_text: Main prompt text
        extra_texts: Additional text parts to include as separate content items

    Returns:
        List of text content parts
    """
    # Start with main prompt as first text part
    content = [{"type": "text", "text": prompt_text}]

    # Add each extra_text as a separate text part
    if extra_texts:
        for extra_text in extra_texts:
            if isinstance(extra_text, str) and extra_text:
                content.append({"type": "text", "text": extra_text})

    return content


def build_messages(
    prompt_text: str,
    extra_texts: Optional[List[str]] = None,
    system_text: Optional[str] = None,
    jpeg_images: Optional[List[bytes]] = None,
) -> List[Dict]:
    """Build messages for VLM with text and image content.

    Args:
        prompt_text: Main prompt text
        jpeg_images: List of JPEG image bytes
        extra_texts: Additional text parts to include as separate content items
        system_text: Optional system message

    Returns:
        Tuple of (messages list, original prompt_text)
    """
    # Build text content parts
    content = _build_text_content_parts(prompt_text, extra_texts)

    # Add image parts
    if jpeg_images:
        for j in jpeg_images:
            content.append({"type": "image_url", "image_url": {"url": jpeg_bytes_to_data_url(j)}})

    messages: List[Dict] = []
    if system_text:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    messages.append({"role": "user", "content": content})
    return messages


def call_lang_model(
    client: Any,
    model: str,
    messages: List[Dict],
    temperature: float = 0.0,
    extra_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Call the chat.completions API and return the text content.

    Centralizes shared VLM invocation so all processors use the same code path.
    """
    kwargs: Dict[str, Any] = {"model": model, "messages": messages}
    # Only include temperature if supported for the model
    if _is_temperature_supported(model):
        kwargs["temperature"] = float(temperature)
    if extra_params:
        for k, v in extra_params.items():
            if v is not None:
                kwargs[k] = v
    logger.debug("Calling VLM | model=%s temperature=%.2f", model, temperature)
    result = client.chat.completions.create(**kwargs)
    content = result.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text"))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        ]
        if parts:
            return "\n".join(parts)
    finish_reason = getattr(result.choices[0], "finish_reason", None)
    raise ValueError(f"Model returned no text content; finish_reason={finish_reason}")


def _is_temperature_supported(model: str) -> bool:
    """Return False for all models in the 'gpt-5' family; True otherwise.

    This provides a conservative guard so we don't send unsupported params.
    """
    m = model.strip().lower()
    return not m.startswith("gpt-5")
