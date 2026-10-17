"""OpenAI-compatible chat client with fixed experiment controls, per-item budgets and tracing.

Design rules:
- The framework constructs the client from the run config. Harness code only ever receives an
  `ItemLLM` and cannot change model, temperature, top_p or reasoning effort.
- Every call is counted against the item's `Budget`; exceeding it raises `BudgetExceeded`,
  which the evaluator records as a failed item. This is what stops a harness from turning
  pass@1 into an internal best-of-N.
- Every call is appended to the item's trace so the proposer can read raw prompts/responses.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from .clock import wall
from .credentials import DEFAULT_API_KEY_ENV, normalize_solver_config, resolve_api_key
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


@dataclass(frozen=True)
class LLMConfig:
    """Experiment controls. Frozen: harness code cannot mutate it."""

    model: str = "gpt-oss-120b"
    api_base: str = "http://localhost:8000/v1"
    api_key_env: str = DEFAULT_API_KEY_ENV
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None  # vLLM extension (sent via extra_body)
    max_tokens: int = 4096  # hard cap on completion tokens per call
    reasoning_effort: str | None = "medium"  # gpt-oss: injected as a "Reasoning: <effort>" system line
    timeout_s: float = 600.0
    max_retries: int = 6
    seed: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMConfig":
        allowed = {k: v for k, v in normalize_solver_config(d).items() if k in cls.__dataclass_fields__}
        return cls(**allowed)


@dataclass
class Budget:
    """Per-item resource limits. None = unlimited."""

    max_calls: int | None = None
    max_output_tokens: int | None = None
    max_input_tokens: int | None = None
    max_seconds: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Budget":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BudgetExceeded(RuntimeError):
    """Raised inside harness code when an item exceeds its budget. Recorded as a failure."""


@dataclass
class LLMResponse:
    text: str
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cached: bool = False

    def __str__(self) -> str:  # lets harness code do f"{resp}" or str(resp)
        return self.text


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in (408, 409, 425, 429, 500, 502, 503, 504)
    if isinstance(exc, httpx.HTTPError):
        return True
    return False


class LLMClient:
    """Shared, thread-safe client. One per process."""

    def __init__(self, cfg: LLMConfig, cache_dir: str | os.PathLike | None = None, cache_salt: str = ""):
        self.cfg = cfg
        self._client = OpenAI(
            base_url=cfg.api_base,
            api_key=resolve_api_key(cfg.api_key_env),
            timeout=httpx.Timeout(cfg.timeout_s, connect=30.0),
            max_retries=0,  # we do our own retries (tenacity) so they are logged/controlled
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_salt = cache_salt
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._is_gpt_oss = "gpt-oss" in cfg.model.lower()

    # -- prompt shaping -------------------------------------------------------------------
    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs = [dict(m) for m in messages]
        if self._is_gpt_oss and self.cfg.reasoning_effort:
            line = f"Reasoning: {self.cfg.reasoning_effort}"
            if msgs and msgs[0].get("role") == "system":
                if not str(msgs[0].get("content", "")).startswith("Reasoning:"):
                    msgs[0]["content"] = f"{line}\n\n{msgs[0].get('content', '')}"
            else:
                msgs.insert(0, {"role": "system", "content": line})
        return msgs

    def _cache_path(self, payload: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        blob = json.dumps({"salt": self.cache_salt, **payload}, sort_keys=True, default=str).encode()
        return self.cache_dir / (hashlib.sha256(blob).hexdigest() + ".json")

    # -- the call ----------------------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        stop: list[str] | str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        cfg = self.cfg
        max_tokens = min(int(max_tokens), cfg.max_tokens) if max_tokens else cfg.max_tokens
        msgs = self._prepare_messages(messages)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": msgs,
            "temperature": cfg.temperature,
            "max_tokens": max_tokens,
        }
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kwargs["extra_body"] = {"top_k": int(cfg.top_k)}
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        if stop:
            kwargs["stop"] = [stop] if isinstance(stop, str) else list(stop)

        cache_path = self._cache_path({"api_base": cfg.api_base, **kwargs})
        if cache_path and cache_path.exists():
            try:
                d = json.loads(cache_path.read_text())
                return LLMResponse(**{**d, "cached": True})
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        resp = self._call_with_retry(kwargs)
        if cache_path:
            tmp = cache_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(asdict(resp)))
                tmp.replace(cache_path)
            except OSError:
                pass
        with self._lock:
            self.total_calls += 1
            self.total_prompt_tokens += resp.prompt_tokens
            self.total_completion_tokens += resp.completion_tokens
        return resp

    def _call_with_retry(self, kwargs: dict[str, Any]) -> LLMResponse:
        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential_jitter(initial=2, max=60, jitter=5),
            stop=stop_after_attempt(self.cfg.max_retries),
            reraise=True,
        )
        def _do() -> LLMResponse:
            raw = self._client.chat.completions.create(**kwargs)
            choice = raw.choices[0]
            msg = choice.message
            text = msg.content if isinstance(msg.content, str) else ""
            extra = msg.model_dump() if hasattr(msg, "model_dump") else {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
            usage = raw.usage
            return LLMResponse(
                text=text or "",
                reasoning=reasoning if isinstance(reasoning, str) else None,
                finish_reason=choice.finish_reason,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )

        return _do()


class Trace:
    """Append-only per-item event log. Serialized to JSON by the evaluator."""

    def __init__(self, item_id: str):
        self.item_id = item_id
        self.events: list[dict[str, Any]] = []
        self.t0 = wall()
        self._last_messages: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def add(self, type_: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"type": type_, "t": round(wall() - self.t0, 3), **data})

    def add_llm(self, messages: list[dict[str, Any]], resp: LLMResponse, elapsed: float, **extra: Any) -> None:
        """Store messages delta-encoded against the previous call to keep multi-turn traces small."""
        prev = self._last_messages
        if prev is not None and len(messages) >= len(prev) and messages[: len(prev)] == prev:
            enc: dict[str, Any] = {"prefix_of_previous": len(prev), "new_messages": messages[len(prev):]}
        else:
            enc = {"messages": messages}
        self._last_messages = list(messages)
        self.add(
            "llm",
            **enc,
            response=resp.text,
            reasoning=(resp.reasoning[:4000] if resp.reasoning else None),
            finish_reason=resp.finish_reason,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached=resp.cached,
            elapsed=round(elapsed, 3),
            **extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "events": self.events}


class ItemLLM:
    """The only LLM handle harness code gets. Enforces the item budget and records the trace."""

    def __init__(self, client: LLMClient, budget: Budget, trace: Trace):
        self._client = client
        self._budget = budget
        self._trace = trace
        self._t0 = wall()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._client.cfg.model

    @property
    def max_tokens(self) -> int:
        return self._client.cfg.max_tokens

    def _check_budget(self) -> None:
        b = self._budget
        if b.max_calls is not None and self.calls >= b.max_calls:
            raise BudgetExceeded(f"max_calls={b.max_calls} reached")
        if b.max_output_tokens is not None and self.completion_tokens >= b.max_output_tokens:
            raise BudgetExceeded(f"max_output_tokens={b.max_output_tokens} reached")
        if b.max_input_tokens is not None and self.prompt_tokens >= b.max_input_tokens:
            raise BudgetExceeded(f"max_input_tokens={b.max_input_tokens} reached")
        if b.max_seconds is not None and (wall() - self._t0) >= b.max_seconds:
            raise BudgetExceeded(f"max_seconds={b.max_seconds} reached")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        stop: list[str] | str | None = None,
        max_tokens: int | None = None,
        tag: str | None = None,
    ) -> LLMResponse:
        """Chat-completions call. `messages` = [{"role": ..., "content": ...}, ...]."""
        with self._lock:
            self._check_budget()
            self.calls += 1
        t0 = wall()
        resp = self._client.complete(messages, stop=stop, max_tokens=max_tokens)
        with self._lock:
            self.prompt_tokens += resp.prompt_tokens
            self.completion_tokens += resp.completion_tokens
        self._trace.add_llm(messages, resp, wall() - t0, tag=tag)
        return resp

    def __call__(self, prompt: str, system: str | None = None, **kw: Any) -> str:
        """Single-prompt convenience: returns the response text."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kw).text

    def usage(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def make_stub_client(responses: list[str] | None = None) -> "StubClient":
    return StubClient(responses or ["stub"])


class StubClient(LLMClient):
    """Offline stand-in for tests: cycles through canned responses."""

    def __init__(self, responses: list[str]):
        self.cfg = LLMConfig(model="stub", api_base="http://stub")
        self.cache_dir = None
        self.cache_salt = ""
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._is_gpt_oss = False
        self._responses = list(responses)
        self._i = 0

    def complete(self, messages, *, stop=None, max_tokens=None) -> LLMResponse:  # type: ignore[override]
        with self._lock:
            text = self._responses[self._i % len(self._responses)]
            self._i += 1
            self.total_calls += 1
        n_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        return LLMResponse(text=text, reasoning=None, finish_reason="stop", prompt_tokens=n_in, completion_tokens=len(text) // 4)
