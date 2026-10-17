import os
from typing import Any

from ..llm import LLMCallable
from .fewshot_memory import FewShotMemory, PROMPT_TEMPLATE, _seed_for_input
from ..memory_system import extract_json_field

RUN_SEED = os.environ.get("FEWSHOT_RUN_SEED")


class FewShotAllReroll(FewShotMemory):
    """Same as upstream FewShotAll, but mixes the run seed into the example-selection seed."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm, max_examples=9999)
        self._run_seed = int(RUN_SEED) if RUN_SEED is not None else None

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        seed = _seed_for_input(input)
        if self._run_seed is not None:
            # Mix so that each run yields a different permutation; per-item determinism is kept
            # (the same run and item always give the same seed).
            seed = (seed ^ (self._run_seed * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
        examples_section = self._format_examples_section(seed=seed)
        prompt = PROMPT_TEMPLATE.format(examples_section=examples_section, input=input)
        response = self.call_llm(prompt)
        answer = extract_json_field(response, "final_answer")
        return answer, {"full_response": response, "fewshot_run_seed": self._run_seed}
