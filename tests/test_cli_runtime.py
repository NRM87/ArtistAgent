import argparse
from pathlib import Path

from artist_agent.backends import CliLLMBackend, CliVisionBackend, CodexImageBackend
from artist_agent.runtime import build_image_backend, build_llm_backend, build_vision_backend, validate_backend_choices


def test_build_cli_llm_backend():
    args = argparse.Namespace(
        llm_backend="cli",
        llm_cli="gemini",
        llm_model="gemini",
        llm_temperature=0.2,
        trace_prompts=False,
    )
    backend = build_llm_backend(args)
    assert isinstance(backend, CliLLMBackend)
    assert backend.cli == "gemini"


def test_build_cli_vision_backend():
    args = argparse.Namespace(
        vision_backend="cli",
        vision_cli="codex",
        vision_model="codex",
        vision_temperature=0.4,
        trace_prompts=False,
    )
    backend = build_vision_backend(args)
    assert isinstance(backend, CliVisionBackend)
    assert backend.cli == "codex"


def test_strict_policy_accepts_cli_text_backends():
    args = argparse.Namespace(
        run_policy="strict",
        run_mode="create",
        vision_backend="cli",
        llm_backend="cli",
        image_backend="ascii",
        image_fallback="ascii",
    )
    validate_backend_choices(args)


def test_build_codex_first_class_backends():
    vision_args = argparse.Namespace(
        vision_backend="codex",
        vision_model="gpt-5",
        vision_temperature=0.4,
        trace_prompts=False,
    )
    llm_args = argparse.Namespace(
        llm_backend="codex",
        llm_model="gpt-5",
        llm_temperature=0.2,
        trace_prompts=False,
    )
    image_args = argparse.Namespace(
        image_backend="codex",
        image_model="gpt-5",
        image_size="1024x1024",
        image_fallback="ascii",
        ascii_size="160x60",
        trace_prompts=False,
    )
    v = build_vision_backend(vision_args)
    l = build_llm_backend(llm_args)
    i = build_image_backend(image_args, Path(".tmp_tests"), llm_backend=l)
    assert isinstance(v, CliVisionBackend)
    assert isinstance(l, CliLLMBackend)
    assert isinstance(i, CodexImageBackend)


def test_strict_policy_accepts_codex_backends():
    args = argparse.Namespace(
        run_policy="strict",
        run_mode="create",
        vision_backend="codex",
        llm_backend="codex",
        image_backend="codex",
        image_fallback="ascii",
    )
    validate_backend_choices(args)
