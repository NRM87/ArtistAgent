import argparse
import json
from pathlib import Path

from artist_agent.runtime import resolve_artist_runtime


def test_legacy_profile_inherits_new_defaults(tmp_path: Path):
    artists_dir = tmp_path / "artists"
    profiles_dir = tmp_path / "profiles"
    artists_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Simulate an older profile file that lacks recently added keys.
    legacy_profile = {
        "run_policy": "strict",
        "vision_backend": "gemini",
        "vision_model": "gemini-2.5-pro",
        "llm_backend": "gemini",
        "llm_model": "gemini-2.5-pro",
        "image_backend": "gemini",
        "image_model": "gemini-2.0-flash-exp-image-generation",
    }
    (profiles_dir / "default.json").write_text(json.dumps(legacy_profile, indent=2), encoding="utf-8")

    artist_dir = artists_dir / "legacy_artist"
    artist_dir.mkdir(parents=True, exist_ok=True)
    (artist_dir / "artist.json").write_text(
        json.dumps({"name": "Legacy", "profile": "default"}, indent=2),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        artists_dir=str(artists_dir),
        profiles_dir=str(profiles_dir),
        profile="",
        artist="legacy_artist",
        run_policy="strict",
        vision_backend="local",
        vision_model="",
        vision_temperature=0.4,
        llm_backend="mock",
        llm_model="",
        llm_temperature=0.2,
        image_backend="mock",
        image_model="",
        image_size="",
        image_fallback="defer",
        trace_prompts=False,
        _explicit_args=set(),
    )

    runtime = resolve_artist_runtime(args)
    assert runtime.profile_id == "default"
    # Missing key should be injected from DEFAULT_PROFILE_CONFIG.
    assert args.image_fallback == "ascii"
    assert args.trace_prompts is False
    # Existing values from legacy profile should remain.
    assert args.image_model == "gemini-2.0-flash-exp-image-generation"


def test_legacy_gemini_model_aliases_are_normalized(tmp_path: Path):
    artists_dir = tmp_path / "artists"
    profiles_dir = tmp_path / "profiles"
    artists_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    old_profile = {
        "run_policy": "strict",
        "vision_backend": "gemini",
        "vision_model": "gemini-1.5-pro",
        "llm_backend": "gemini",
        "llm_model": "gemini-1.5-pro",
        "image_backend": "gemini",
        "image_model": "gemini-2.0-flash-preview-image-generation",
    }
    (profiles_dir / "default.json").write_text(json.dumps(old_profile, indent=2), encoding="utf-8")

    artist_dir = artists_dir / "legacy_models"
    artist_dir.mkdir(parents=True, exist_ok=True)
    (artist_dir / "artist.json").write_text(
        json.dumps({"name": "LegacyModels", "profile": "default"}, indent=2),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        artists_dir=str(artists_dir),
        profiles_dir=str(profiles_dir),
        profile="",
        artist="legacy_models",
        run_policy="strict",
        vision_backend="local",
        vision_model="",
        vision_temperature=0.4,
        llm_backend="mock",
        llm_model="",
        llm_temperature=0.2,
        image_backend="mock",
        image_model="",
        image_size="",
        image_fallback="defer",
        trace_prompts=False,
        _explicit_args=set(),
    )

    resolve_artist_runtime(args)
    assert args.vision_model == "gemini-2.5-pro"
    assert args.llm_model == "gemini-2.5-pro"
    assert args.image_model == "gemini-2.0-flash-exp-image-generation"
