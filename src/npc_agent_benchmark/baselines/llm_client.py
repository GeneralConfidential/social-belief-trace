"""LLM clients for local Ollama and test mocks."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class LlmClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return raw model text for a single-turn prompt."""


@dataclass(frozen=True)
class MockLlmClient:
    """Deterministic client for unit tests."""

    response: str

    def complete(self, prompt: str) -> str:
        return self.response


@dataclass(frozen=True)
class OllamaClient:
    """Minimal Ollama /api/generate client (stdlib HTTP, no extra deps)."""

    base_url: str
    model: str
    temperature: float = 0.0
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> OllamaClient:
        return cls(
            base_url=os.environ.get("SBT_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            model=os.environ.get("SBT_OLLAMA_MODEL", "llama3.2:3b-instruct-q4_K_M"),
            temperature=float(os.environ.get("SBT_OLLAMA_TEMPERATURE", "0")),
            timeout_s=float(os.environ.get("SBT_OLLAMA_TIMEOUT_S", "120")),
        )

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        body = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except TimeoutError as exc:
            raise RuntimeError(f"Ollama request timed out ({url}): {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed ({url}): {exc}") from exc

        text = payload.get("response")
        if not isinstance(text, str):
            raise RuntimeError("Ollama response missing string 'response' field")
        return text


@dataclass(frozen=True)
class OpenAIClient:
    """Minimal OpenAI chat-completions client (stdlib HTTP)."""

    api_key: str
    model: str
    temperature: float | None = 0.0
    timeout_s: float = 120.0
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls) -> OpenAIClient:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        # Some models (e.g. gpt-6.1-sol) reject temperature=0; set SBT_OPENAI_TEMPERATURE=omit.
        temp_raw = os.environ.get("SBT_OPENAI_TEMPERATURE", "0").strip().lower()
        temperature: float | None
        if temp_raw in {"", "omit", "default", "none"}:
            temperature = None
        else:
            temperature = float(temp_raw)
        return cls(
            api_key=api_key,
            model=os.environ.get("SBT_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            timeout_s=float(os.environ.get("SBT_OPENAI_TIMEOUT_S", "120")),
            base_url=os.environ.get("SBT_OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    def complete(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed ({url}): {exc}") from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI response missing choices")
        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI response missing message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("OpenAI response missing string content")
        return content
