from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from npc_agent_benchmark.baselines.llm_client import OpenAIClient


def test_openai_client_complete_parses_response() -> None:
    payload = {
        "choices": [{"message": {"content": '{"kind":"eat","target":null,"payload":{}}'}}],
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")
    with patch("urllib.request.urlopen", return_value=mock_resp):
        text = client.complete("prompt")
    assert '"kind":"eat"' in text or '"kind": "eat"' in text


def test_openai_client_from_env_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIClient.from_env()
