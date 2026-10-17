"""Baseline: the official AppWorld ReAct code agent (appworld-agents `simplified_react_code_agent`).

Faithful port of experiments/code/simplified/react_code_agent.py:
  * prompt = official `react_code_agent/instructions.txt` (Jinja2) rendered with instruction, supervisor and app
    descriptions, split into USER:/ASSISTANT: turns -> chat messages;
  * each step: one LLM call -> take the FIRST ```python block (ignore_multiple_calls=True) -> execute -> append
    "Output:\n```\n...\n```" as the next user message; stop when apis.supervisor.complete_task was called;
  * max_steps (50, official simplified config) is enforced by ctx.tools; no history trimming (max_output_length=None).

Modified for this harness interface from StonyBrookNLP/appworld at commit
42b5bcf3cd334fee33f0c37c02070a9f5807add5. The upstream component is licensed
under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from typing import Any

from core.harness import BaseHarness, RunContext
from domains.appworld.prompting import (
    extract_code_and_fix_content,
    format_output_message,
    render_react_prompt,
    text_to_messages,
)


class Harness(BaseHarness):
    def run(self, item: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
        env = ctx.tools
        messages = text_to_messages(render_react_prompt(env) + "\n\n")
        steps = 0
        for step in range(env.max_steps):
            resp = ctx.llm.chat(messages, tag=f"step{step + 1}")
            code, fixed = extract_code_and_fix_content(resp.text or "")
            messages.append({"role": "assistant", "content": fixed + "\n\n"})
            output = env.execute(code)
            steps = step + 1
            messages.append({"role": "user", "content": format_output_message(output)})
            if env.task_completed():
                break
        return {"completed": env.task_completed(), "steps": steps}
