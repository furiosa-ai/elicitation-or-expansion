"""Prompt helpers shared by AppWorld harnesses (kept here so candidate files need no file I/O).

Modified port of AppWorld's experiments/code/simplified/react_code_agent.py
(StonyBrookNLP/appworld, commit 42b5bcf3cd334fee33f0c37c02070a9f5807add5).
Prompt rendering, transcript parsing and code extraction are adapted to this harness interface.
The upstream component is licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Template

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
REACT_INSTRUCTIONS: str = (_PROMPTS_DIR / "react_instructions.txt").read_text().lstrip()

FULL_CODE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
PARTIAL_CODE_RE = re.compile(r".*```python\n(.*)", re.DOTALL)
ROLE_RE = re.compile(r"(USER|ASSISTANT|SYSTEM):\n", re.IGNORECASE)


def render_react_prompt(env: Any, template: str = REACT_INSTRUCTIONS) -> str:
    """Render the official ReAct instructions for one task (env = ctx.tools)."""
    app_descriptions = json.dumps([{"name": k, "description": v} for k, v in env.app_descriptions.items()], indent=1)
    return Template(template).render(instruction=env.instruction, main_user=env.supervisor, app_descriptions=app_descriptions)


def text_to_messages(text: str) -> list[dict[str, str]]:
    """Split a USER:/ASSISTANT:/SYSTEM: transcript into chat messages (official parser)."""
    messages: list[dict[str, Any]] = []
    last = 0
    for m in ROLE_RE.finditer(text):
        start = m.span()[0]
        if messages:
            messages[-1]["content"] = text[last:start]
        elif start != 0:
            raise ValueError("prompt must start with a role marker")
        messages.append({"role": m.group(1).lower(), "content": None})
        last = m.span()[1]
    messages[-1]["content"] = text[last:]
    return messages


def extract_code_and_fix_content(text: str, ignore_multiple_calls: bool = True) -> tuple[str, str]:
    """Return (code, content_to_append). Official behaviour: first ```python block wins; truncate the rest."""
    original = text
    code_out = ""
    match_end = 0
    for m in FULL_CODE_RE.finditer(original):
        code = m.group(1).strip()
        if ignore_multiple_calls:
            return code, original[: m.end()]
        code_out += code + "\n"
        match_end = m.end()
    partial = PARTIAL_CODE_RE.match(original[match_end:])
    if partial:
        code_out += partial.group(1).strip()
        if not text.endswith("\n"):
            text += "\n"
        text += "```"
    return (code_out, text) if code_out else ("", text)


def format_output_message(output: str) -> str:
    if not output.endswith("\n"):
        output += "\n"
    return "Output:\n```\n" + output + "```\n\n"
