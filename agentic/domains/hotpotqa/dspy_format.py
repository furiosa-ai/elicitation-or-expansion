"""Plain-Python reproduction of DSPy's ChatAdapter prompt format (dspy 3.x).

This lets the zero-shot baseline be byte-for-byte the prompt a `dspy.ChainOfThought("question, summary_1 -> query")`
module would send, without depending on DSPy at runtime. Harness code may reuse `Module` for new signatures
or ignore it entirely and build prompts by hand.

Usage:
    m = Module(inputs=["question", "passages"], outputs=["summary"])   # ChainOfThought adds `reasoning` first
    out = m(ctx.llm, question=q, passages=[...])   -> {"reasoning": ..., "summary": ...}
"""

from __future__ import annotations

import re
from typing import Any

FIELD_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


def default_instructions(inputs: list[str], outputs: list[str]) -> str:
    ins = ", ".join(f"`{f}`" for f in inputs)
    outs = ", ".join(f"`{f}`" for f in outputs)
    return f"Given the fields {ins}, produce the fields {outs}."


def format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value):
            return "\n".join(f"[{i + 1}] «{v}»" for i, v in enumerate(value))
        return "\n".join(f"[{i + 1}] «{v}»" for i, v in enumerate(value))
    if value is None:
        return ""
    return str(value).strip()


def _field_desc_block(fields: list[str], descs: dict[str, str]) -> str:
    lines = []
    for i, name in enumerate(fields):
        desc = descs.get(name, "")
        lines.append(f"{i + 1}. `{name}` (str): {desc}")
    return "\n".join(lines)


def _structure_block(fields: list[str]) -> str:
    return "\n\n".join(f"[[ ## {f} ## ]]\n{{{f}}}" for f in fields)


class Module:
    """A DSPy-style predictor: signature (inputs -> outputs) rendered with the ChatAdapter format.

    `chain_of_thought=True` prepends a `reasoning` output field (dspy.ChainOfThought); False mimics dspy.Predict.
    `instructions` defaults to DSPy's "Given the fields ..., produce the fields ..." text.
    `demos` are few-shot examples as dicts containing both input and output field values.
    """

    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        instructions: str | None = None,
        descs: dict[str, str] | None = None,
        chain_of_thought: bool = True,
        demos: list[dict[str, Any]] | None = None,
    ):
        self.inputs = list(inputs)
        self.user_outputs = list(outputs)
        self.outputs = (["reasoning"] if chain_of_thought and "reasoning" not in outputs else []) + list(outputs)
        self.instructions = instructions if instructions is not None else default_instructions(self.inputs, self.user_outputs)
        self.descs = dict(descs or {})
        self.demos = list(demos or [])

    # -- rendering --------------------------------------------------------------------------
    def system_message(self) -> str:
        parts = [
            f"Your input fields are:\n{_field_desc_block(self.inputs, self.descs)}\nYour output fields are:\n{_field_desc_block(self.outputs, self.descs)}",
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
            + _structure_block(self.inputs) + "\n\n" + _structure_block(self.outputs) + "\n\n[[ ## completed ## ]]",
            "In adhering to this structure, your objective is: " + ("\n" + " " * 8).join([""] + self.instructions.splitlines()),
        ]
        return "\n\n".join(parts).strip()

    def user_message(self, values: dict[str, Any], main_request: bool = True) -> str:
        chunks = [f"[[ ## {k} ## ]]\n{format_value(values.get(k))}" for k in self.inputs if k in values]
        if main_request:
            req = "Respond with the corresponding output fields, starting with the field "
            req += ", then ".join(f"`[[ ## {f} ## ]]`" for f in self.outputs)
            req += ", and then ending with the marker for `[[ ## completed ## ]]`."
            chunks.append(req)
        return "\n\n".join(chunks).strip()

    def assistant_message(self, values: dict[str, Any]) -> str:
        body = "\n\n".join(f"[[ ## {k} ## ]]\n{format_value(values.get(k, ''))}" for k in self.outputs)
        return body + "\n\n[[ ## completed ## ]]\n"

    def messages(self, **values: Any) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_message()}]
        for d in self.demos:
            msgs.append({"role": "user", "content": self.user_message(d, main_request=False)})
            msgs.append({"role": "assistant", "content": self.assistant_message(d)})
        msgs.append({"role": "user", "content": self.user_message(values, main_request=True)})
        return msgs

    # -- parsing ----------------------------------------------------------------------------
    def parse(self, completion: str) -> dict[str, str]:
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        for line in (completion or "").splitlines():
            m = FIELD_HEADER.match(line.strip())
            if m:
                sections.append((m.group(1), []))
            else:
                sections[-1][1].append(line)
        out = {k: "\n".join(v).strip() for k, v in sections if k and k != "completed"}
        result = {}
        for f in self.outputs:
            if f in out:
                result[f] = out[f]
        # graceful fallback: if the model ignored the markers, put the raw text in the last field
        if self.user_outputs and self.user_outputs[-1] not in result:
            result[self.user_outputs[-1]] = (completion or "").strip()
        return result

    def __call__(self, llm, tag: str | None = None, **values: Any) -> dict[str, str]:
        resp = llm.chat(self.messages(**values), tag=tag)
        return self.parse(resp.text)
