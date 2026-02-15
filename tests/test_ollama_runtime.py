import argparse
from pathlib import Path

from artist_agent.backends import OllamaLLMBackend, OllamaVisionBackend
from artist_agent.runtime import build_llm_backend, build_vision_backend


def test_build_ollama_llm_backend():
    args = argparse.Namespace(
        llm_backend="ollama",
        llm_model="qwen2.5:3b",
        llm_temperature=0.2,
        ollama_base_url="http://localhost:11434",
    )
    backend = build_llm_backend(args)
    assert isinstance(backend, OllamaLLMBackend)
    assert backend.model == "qwen2.5:3b"


def test_build_ollama_vision_backend():
    args = argparse.Namespace(
        vision_backend="ollama",
        vision_model="qwen2.5:3b",
        vision_temperature=0.4,
        ollama_base_url="http://localhost:11434",
    )
    backend = build_vision_backend(args)
    assert isinstance(backend, OllamaVisionBackend)
    assert backend.model == "qwen2.5:3b"
