"""Constrained local LLM policy (Ollama) on the observe/act contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from npc_agent_benchmark.baselines.action_schema import (
    ActionValidationError,
    allowed_action_kinds,
    validate_action_dict,
)
from npc_agent_benchmark.baselines.agents import BaseAgent
from npc_agent_benchmark.baselines.llm_client import LlmClient, OllamaClient
from npc_agent_benchmark.baselines.llm_prompt import (
    PROMPT_VERSION,
    build_act_prompt,
    build_repair_prompt,
)
from npc_agent_benchmark.models import Action, AgentState, ExplanationTrace

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class LlmActStats:
    attempts: int = 0
    parse_failures: int = 0
    validation_failures: int = 0
    fallbacks: int = 0
    successes: int = 0
    repairs: int = 0

    def record_attempt(self) -> None:
        self.attempts += 1

    def to_dict(self) -> dict[str, object]:
        invalid = self.parse_failures + self.validation_failures
        rate = (invalid / self.attempts) if self.attempts else 0.0
        return {
            "llm_act_attempts": self.attempts,
            "llm_act_parse_failures": self.parse_failures,
            "llm_act_validation_failures": self.validation_failures,
            "llm_act_fallbacks": self.fallbacks,
            "llm_act_successes": self.successes,
            "llm_act_repairs": self.repairs,
            "llm_invalid_action_rate": rate,
        }


def extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    fence = _JSON_FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()

    # Prefer the last brace-balanced object (models often prepend prose).
    candidates: list[str] = []
    depth = 0
    start: int | None = None
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(stripped[start : i + 1])
                start = None

    if not candidates:
        start_i = stripped.find("{")
        end_i = stripped.rfind("}")
        if start_i == -1 or end_i == -1 or end_i <= start_i:
            raise json.JSONDecodeError("no JSON object found", stripped, 0)
        candidates = [stripped[start_i : end_i + 1]]

    last_err: json.JSONDecodeError | None = None
    for blob in reversed(candidates):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if isinstance(parsed, dict):
            return parsed
    if last_err is not None:
        raise last_err
    raise json.JSONDecodeError("top-level JSON must be an object", stripped, 0)


def normalize_action_dict(raw: dict[str, object], *, peer_ids: list[str]) -> dict[str, object]:
    """Fill missing fields small models often omit (esp. ``kind``)."""
    out = dict(raw)
    kind = out.get("kind")
    target = out.get("target")
    payload = out.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    out["payload"] = payload

    if isinstance(kind, str) and kind.strip():
        out["kind"] = kind.strip()
        if "target" not in out:
            out["target"] = None
        return out

    peer_set = set(peer_ids)
    subject = payload.get("subject") if isinstance(payload, dict) else None
    if isinstance(target, str) and target in peer_set and isinstance(subject, str) and subject.strip():
        out["kind"] = "gossip"
        return out
    if isinstance(target, str) and target in peer_set and not payload:
        out["kind"] = "help_stranger"
        return out
    if target in (None, "null") and not payload:
        out["kind"] = "rest"
        out["target"] = None
        return out

    # Still missing kind — leave as-is so validation reports clearly.
    return out


def _snippet(text: str, *, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class ConstrainedLlmAgent(BaseAgent):
    """Local LLM head: schema-valid actions only; fallback to rest on failure."""

    def __init__(
        self,
        agent_id: str,
        *,
        seed: int,
        disabled_action_kinds: set[str] | None = None,
        client: LlmClient | None = None,
        model_id: str | None = None,
        prompt_version: str = PROMPT_VERSION,
        temperature: float = 0.0,
        repair_on_failure: bool = True,
    ):
        super().__init__(agent_id, seed=seed)
        self._disabled_action_kinds = disabled_action_kinds or set()
        self._client = client if client is not None else OllamaClient.from_env()
        self._model_id = model_id or getattr(self._client, "model", "ollama")
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._repair_on_failure = repair_on_failure
        self._stats = LlmActStats()

    def llm_stats(self) -> dict[str, object]:
        out = self._stats.to_dict()
        out["llm_model_id"] = self._model_id
        out["llm_prompt_version"] = self._prompt_version
        out["llm_temperature"] = self._temperature
        return out

    def act(self, *, t: int, state: AgentState, observation: dict[str, object]) -> tuple[Action, ExplanationTrace]:
        self._stats.record_attempt()
        peer_ids_raw = observation.get("peer_ids", [])
        peer_ids = [p for p in peer_ids_raw if isinstance(p, str)] if isinstance(peer_ids_raw, list) else []
        allowed = sorted(allowed_action_kinds(disabled_action_kinds=self._disabled_action_kinds))

        prompt = build_act_prompt(
            agent_id=self.agent_id,
            t=t,
            state=state,
            observation=observation,
            disabled_action_kinds=self._disabled_action_kinds,
            prompt_version=self._prompt_version,
        )

        raw_text = ""
        try:
            raw_text = self._client.complete(prompt)
            action = self._parse_and_validate(raw_text, peer_ids=peer_ids)
        except json.JSONDecodeError as exc:
            repaired = self._try_repair(
                previous_text=raw_text,
                error=f"JSON parse error: {exc}",
                allowed=allowed,
                peer_ids=peer_ids,
            )
            if repaired is not None:
                return repaired
            self._stats.parse_failures += 1
            return self._fallback(notes=["LLM parse failure → rest fallback.", f"raw={_snippet(raw_text)}"])
        except ActionValidationError as exc:
            repaired = self._try_repair(
                previous_text=raw_text,
                error=str(exc),
                allowed=allowed,
                peer_ids=peer_ids,
            )
            if repaired is not None:
                return repaired
            self._stats.validation_failures += 1
            return self._fallback(
                notes=["LLM validation failure → rest fallback.", f"error={exc}", f"raw={_snippet(raw_text)}"]
            )
        except RuntimeError as exc:
            self._stats.parse_failures += 1
            return self._fallback(notes=[f"LLM client error → rest fallback: {exc}"])

        self._stats.successes += 1
        return (
            action,
            ExplanationTrace(
                factors={"llm_valid": 1.0},
                notes=[
                    f"ConstrainedLlmAgent model={self._model_id} prompt={self._prompt_version}",
                    f"Selected {action.kind}.",
                ],
            ),
        )

    def _parse_and_validate(self, raw_text: str, *, peer_ids: list[str]) -> Action:
        parsed = extract_json_object(raw_text)
        normalized = normalize_action_dict(parsed, peer_ids=peer_ids)
        return validate_action_dict(
            normalized,
            peer_ids=peer_ids,
            disabled_action_kinds=self._disabled_action_kinds,
        )

    def _try_repair(
        self,
        *,
        previous_text: str,
        error: str,
        allowed: list[str],
        peer_ids: list[str],
    ) -> tuple[Action, ExplanationTrace] | None:
        if not self._repair_on_failure or not previous_text:
            return None
        repair_prompt = build_repair_prompt(
            previous_text=previous_text,
            error=error,
            allowed_kinds=allowed,
            peer_ids=peer_ids,
        )
        try:
            repair_text = self._client.complete(repair_prompt)
            action = self._parse_and_validate(repair_text, peer_ids=peer_ids)
        except (json.JSONDecodeError, ActionValidationError, RuntimeError):
            return None
        self._stats.repairs += 1
        self._stats.successes += 1
        return (
            action,
            ExplanationTrace(
                factors={"llm_valid": 1.0, "llm_repaired": 1.0},
                notes=[
                    f"ConstrainedLlmAgent repaired model={self._model_id} prompt={self._prompt_version}",
                    f"Selected {action.kind} after repair.",
                ],
            ),
        )

    def _fallback(self, *, notes: list[str]) -> tuple[Action, ExplanationTrace]:
        self._stats.fallbacks += 1
        return (
            Action(kind="rest"),
            ExplanationTrace(
                factors={"llm_fallback": 1.0},
                notes=notes,
            ),
        )
