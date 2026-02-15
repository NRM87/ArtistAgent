import argparse

import pytest

from artist_agent.runtime import validate_backend_choices


def test_offline_policy_forces_local_backends():
    args = argparse.Namespace(
        run_policy="offline",
        vision_backend="gemini",
        llm_backend="gemini",
        image_backend="gemini",
    )
    validate_backend_choices(args)
    assert args.vision_backend == "ollama"
    assert args.llm_backend == "ollama"
    assert args.image_backend == "ascii"


def test_strict_policy_rejects_local_or_mock_backends():
    args = argparse.Namespace(
        run_policy="strict",
        vision_backend="local",
        llm_backend="mock",
        image_backend="mock",
    )
    with pytest.raises(ValueError):
        validate_backend_choices(args)


def test_strict_policy_accepts_hosted_backends():
    args = argparse.Namespace(
        run_policy="strict",
        vision_backend="gemini",
        llm_backend="gemini",
        image_backend="gemini",
    )
    validate_backend_choices(args)


def test_strict_policy_accepts_ollama_for_text_backends():
    args = argparse.Namespace(
        run_policy="strict",
        vision_backend="ollama",
        llm_backend="ollama",
        image_backend="gemini",
    )
    validate_backend_choices(args)


def test_mock_llm_backend_rejected_for_run():
    args = argparse.Namespace(
        run_policy="hybrid",
        vision_backend="gemini",
        llm_backend="mock",
        image_backend="gemini",
    )
    with pytest.raises(ValueError):
        validate_backend_choices(args)


def test_invalid_image_capability_rejected():
    args = argparse.Namespace(
        run_policy="hybrid",
        vision_backend="gemini",
        llm_backend="gemini",
        image_backend="anthropic",
    )
    with pytest.raises(ValueError):
        validate_backend_choices(args)


def test_mock_image_backend_rejected_for_run():
    args = argparse.Namespace(
        run_policy="hybrid",
        vision_backend="ollama",
        llm_backend="ollama",
        image_backend="mock",
        image_fallback="ascii",
    )
    with pytest.raises(ValueError):
        validate_backend_choices(args)


def test_mock_image_fallback_rejected():
    args = argparse.Namespace(
        run_policy="hybrid",
        vision_backend="ollama",
        llm_backend="ollama",
        image_backend="gemini",
        image_fallback="mock",
    )
    with pytest.raises(ValueError):
        validate_backend_choices(args)
