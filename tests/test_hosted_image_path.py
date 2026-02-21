import base64
from pathlib import Path

import pytest

from artist_agent.backends import HostedImageBackend
from artist_agent.constants import HostedCallError


def _structured_prompt() -> str:
    return (
        'Run vision (fixed for this run): "My vision for this run is to create a crimson tower over flooded stone."\n'
        'Iteration image prompt: "Create a crimson tower with hard rim light and shallow water reflections."\n'
        "Create a coherent 2D composition using the iteration image prompt while staying faithful to the fixed run vision."
    )


def test_hosted_image_prompt_normalization_prefers_iteration_prompt():
    prompt = _structured_prompt()
    out = HostedImageBackend._normalize_model_prompt(prompt)
    assert "Iteration image prompt" not in out
    assert "Run vision (fixed for this run)" not in out
    assert "crimson tower with hard rim light" in out.lower()
    assert "stay faithful to this run vision" in out.lower()


def test_openai_hosted_image_writes_png(monkeypatch, tmp_path: Path):
    backend = HostedImageBackend(
        provider="openai",
        model="gpt-image-1",
        api_key="test-key",
        temp_dir=tmp_path,
        allow_fallback=False,
    )

    captured = {}
    expected = b"\x89PNG\r\n\x1a\nfake"

    def _fake_http(url, payload, headers):
        captured["url"] = url
        captured["payload"] = payload
        return {"data": [{"b64_json": base64.b64encode(expected).decode("utf-8")}]}

    monkeypatch.setattr(backend, "_http_json", _fake_http)
    out_path = Path(backend.generate(_structured_prompt(), iteration=0, creation_id=1))
    assert out_path.exists()
    assert out_path.read_bytes() == expected
    assert "Iteration image prompt" not in str(captured["payload"]["prompt"])
    assert "stay faithful to this run vision" in str(captured["payload"]["prompt"]).lower()


def test_gemini_hosted_image_accepts_inline_data_variants(monkeypatch, tmp_path: Path):
    backend = HostedImageBackend(
        provider="gemini",
        model="gemini-image-model",
        api_key="test-key",
        temp_dir=tmp_path,
        allow_fallback=False,
    )
    expected = b"\x89PNG\r\n\x1a\nfake-gemini"

    def _fake_http(url, payload, headers):
        return {
            "candidates": [
                {"content": {"parts": [{"inline_data": {"data": base64.b64encode(expected).decode("utf-8")}}]}}
            ]
        }

    monkeypatch.setattr(backend, "_http_json", _fake_http)
    out_path = Path(backend.generate(_structured_prompt(), iteration=0, creation_id=2))
    assert out_path.exists()
    assert out_path.read_bytes() == expected


def test_hosted_image_fallback_to_ascii_when_allowed(monkeypatch, tmp_path: Path):
    class DummyLLM:
        def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
            return "###\n###"

    backend = HostedImageBackend(
        provider="openai",
        model="gpt-image-1",
        api_key="test-key",
        temp_dir=tmp_path,
        allow_fallback=True,
        fallback_mode="ascii",
        llm_backend=DummyLLM(),
        ascii_size="40x20",
    )
    monkeypatch.setattr(backend, "_http_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
    out_path = Path(backend.generate(_structured_prompt(), iteration=0, creation_id=3))
    assert out_path.suffix == ".txt"
    assert "renderer: llm" in out_path.read_text(encoding="utf-8")


def test_hosted_image_fails_closed_without_fallback(monkeypatch, tmp_path: Path):
    backend = HostedImageBackend(
        provider="openai",
        model="gpt-image-1",
        api_key="test-key",
        temp_dir=tmp_path,
        allow_fallback=False,
    )
    monkeypatch.setattr(backend, "_http_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")))
    with pytest.raises(HostedCallError):
        backend.generate(_structured_prompt(), iteration=0, creation_id=4)
