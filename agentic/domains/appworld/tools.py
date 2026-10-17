"""`ctx.tools` for AppWorld: a traced, step-budgeted wrapper around one `appworld.AppWorld` instance."""

from __future__ import annotations

import threading
from typing import Any

from core.clock import wall
from core.llm import BudgetExceeded, Trace


class AppWorldEnv:
    """What a harness sees for one task.

    Attributes: task_id, instruction, supervisor (dict: first_name,last_name,email,phone_number), app_descriptions (dict)
    Methods:    execute(code) -> str      run python in the task's REPL; counts one step
                task_completed() -> bool  True once apis.supervisor.complete_task(...) was called
    """

    def __init__(self, world: Any, trace: Trace, max_steps: int, allow_evaluate: bool = False):
        self._world = world
        self._trace = trace
        self._allow_evaluate = allow_evaluate  # True only for TRAIN tasks opened during harness.prepare
        self.max_steps = int(max_steps)
        self.steps = 0
        self.task_id = world.task_id
        self.instruction: str = world.task.instruction
        sup = world.task.supervisor
        self.supervisor: dict[str, str] = {k: str(getattr(sup, k, sup.get(k) if isinstance(sup, dict) else "")) for k in ("first_name", "last_name", "email", "phone_number")}
        self.app_descriptions: dict[str, str] = dict(world.task.app_descriptions)
        self._lock = threading.Lock()

    def execute(self, code: str) -> str:
        with self._lock:
            if self.steps >= self.max_steps:
                raise BudgetExceeded(f"max_steps={self.max_steps} reached")
            self.steps += 1
            step = self.steps
        t0 = wall()
        code = code if isinstance(code, str) else str(code)
        output = self._world.execute(code)
        output = output if isinstance(output, str) else str(output)
        self._trace.add("env", step=step, code=code[:20000], output=output[:20000], elapsed=round(wall() - t0, 3))
        return output

    def task_completed(self) -> bool:
        return bool(self._world.task_completed())

    def evaluate(self) -> bool:
        """Ground-truth check. Only available for TRAIN tasks (during prepare); raises otherwise."""
        if not self._allow_evaluate:
            raise PermissionError("evaluate() is only available for train tasks during prepare()")
        tracker = self._world.evaluate()
        ok = bool(getattr(tracker, "success", False))
        self._trace.add("evaluate", success=ok)
        return ok

    # host-only (used by the domain scorer, not by harness code)
    def _evaluate(self) -> Any:
        return self._world.evaluate()
