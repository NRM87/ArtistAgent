import argparse

import pytest

from artist_agent.backends import HostedLLMBackend, OllamaLLMBackend
from artist_agent.constants import HostedCallError
from artist_agent.runtime import build_llm_backend


def _raise_runtime_error(*args, **kwargs):
    raise RuntimeError("forced failure")


def test_build_hosted_llm_without_key_raises_in_hybrid(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    args = argparse.Namespace(
        llm_backend="gemini",
        llm_model="gemini-2.5-pro",
        llm_api_key="",
        llm_temperature=0.2,
        run_policy="hybrid",
        trace_prompts=False,
    )
    with pytest.raises(HostedCallError):
        build_llm_backend(args)


def test_ollama_critique_fails_closed(monkeypatch):
    backend = OllamaLLMBackend(model="qwen2.5:3b")
    monkeypatch.setattr(backend, "_chat_json", _raise_runtime_error)
    with pytest.raises(HostedCallError):
        backend.critique("ignored.txt", "vision", 0)


def test_hosted_critique_fails_closed_even_when_allow_fallback(monkeypatch):
    backend = HostedLLMBackend(
        provider="gemini",
        model="gemini-2.5-pro",
        api_key="test-key",
        allow_fallback=True,
    )
    monkeypatch.setattr(backend, "_chat_json", _raise_runtime_error)
    with pytest.raises(HostedCallError):
        backend.critique("ignored.txt", "vision", 0)
