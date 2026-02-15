import argparse
import json
from pathlib import Path

from artist_agent.runtime import _extract_model_rows, _infer_provider_for_listing


def test_extract_model_rows_gemini_filters_method_and_name():
    payload = {
        "models": [
            {
                "name": "models/gemini-2.5-pro",
                "displayName": "Gemini 2.5 Pro",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "models/imagen-4.0-generate-001",
                "displayName": "Imagen 4",
                "supportedGenerationMethods": ["predict"],
            },
        ]
    }
    rows = _extract_model_rows("gemini", payload, method_filter="generateContent", contains="gemini")
    assert len(rows) == 1
    assert rows[0]["id"] == "models/gemini-2.5-pro"


def test_extract_model_rows_openai_and_anthropic():
    openai_payload = {"data": [{"id": "gpt-4.1-mini"}, {"id": "gpt-5-mini"}]}
    anthropic_payload = {"data": [{"id": "claude-3-5-sonnet-latest", "display_name": "Claude 3.5 Sonnet"}]}

    o_rows = _extract_model_rows("openai", openai_payload, contains="gpt-5")
    a_rows = _extract_model_rows("anthropic", anthropic_payload)

    assert len(o_rows) == 1
    assert o_rows[0]["id"] == "gpt-5-mini"
    assert len(a_rows) == 1
    assert a_rows[0]["display"] == "Claude 3.5 Sonnet"


def test_extract_model_rows_ollama():
    payload = {"models": [{"name": "qwen2.5:3b"}, {"model": "llama3.2:3b"}]}
    rows = _extract_model_rows("ollama", payload, contains="qwen")
    assert len(rows) == 1
    assert rows[0]["id"] == "qwen2.5:3b"


def test_infer_provider_from_profile(tmp_path: Path):
    profiles_dir = tmp_path / "profiles"
    artists_dir = tmp_path / "artists"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    artists_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "default.json").write_text(
        json.dumps({"llm_backend": "anthropic"}, indent=2),
        encoding="utf-8",
    )
    args = argparse.Namespace(provider="", profile="", artist="default", profiles_dir=str(profiles_dir), artists_dir=str(artists_dir))
    assert _infer_provider_for_listing(args) == "anthropic"


def test_infer_provider_uses_artist_manifest_profile(tmp_path: Path):
    profiles_dir = tmp_path / "profiles"
    artists_dir = tmp_path / "artists"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    artists_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "default.json").write_text(json.dumps({"llm_backend": "gemini"}, indent=2), encoding="utf-8")
    (profiles_dir / "alt.json").write_text(json.dumps({"llm_backend": "ollama"}, indent=2), encoding="utf-8")
    artist_dir = artists_dir / "my_artist"
    artist_dir.mkdir(parents=True, exist_ok=True)
    (artist_dir / "artist.json").write_text(json.dumps({"profile": "alt"}, indent=2), encoding="utf-8")

    args = argparse.Namespace(
        provider="",
        profile="",
        artist="my_artist",
        profiles_dir=str(profiles_dir),
        artists_dir=str(artists_dir),
    )
    assert _infer_provider_for_listing(args) == "ollama"
