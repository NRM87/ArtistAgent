import argparse
import json
import sys
from pathlib import Path

from .backends import (
    AsciiImageBackend,
    HostedImageBackend,
    HostedLLMBackend,
    HostedVisionBackend,
    LocalVisionBackend,
    MockImageBackend,
    MockLLMBackend,
    OllamaLLMBackend,
    OllamaVisionBackend,
)
from .commands_artist import configure_models, create_artist, list_artists, run_setup, show_artist, show_profile
from .commands_providers import (
    _extract_model_rows,
    _infer_provider_for_listing,
    check_backends,
    list_models,
)
from .configuration import (
    RUN_KEYS,
    ensure_profile_exists,
    load_artist_manifest,
    load_profile_config,
    resolve_api_key,
    resolve_effective_profile_id,
)
from .constants import (
    DEFAULT_ARTISTS_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_IMAGE_MODELS,
    DEFAULT_PROFILES_DIR,
    DEFAULT_PROVIDER_MODELS,
    DEFAULT_VISION_MODELS,
    PROVIDER_CAPABILITIES,
    ArtistRuntime,
    HostedCallError,
)
from .state import load_config_file


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Recursive AI Artist Agent")
    p.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=[
            "run",
            "setup",
            "create-artist",
            "configure-models",
            "list-artists",
            "show-artist",
            "show-profile",
            "list-models",
            "check-backends",
        ],
    )

    p.add_argument("--config", default=DEFAULT_CONFIG_FILE)
    p.add_argument("--artist", default="default")
    p.add_argument("--artists-dir", default=DEFAULT_ARTISTS_DIR)
    p.add_argument("--profiles-dir", default=DEFAULT_PROFILES_DIR)
    p.add_argument("--profile", default="")
    p.add_argument("--provider", choices=["gemini", "openai", "anthropic", "ollama", "local", "mock"], default="")
    p.add_argument("--run-policy", choices=["strict", "hybrid", "offline"], default="strict")
    p.add_argument("--contains", default="")
    p.add_argument("--method", default="")
    p.add_argument("--probe", action="store_true")

    p.add_argument("--vision-backend", choices=["local", "openai", "anthropic", "gemini", "ollama"], default="local")
    p.add_argument("--vision-model", default="")
    p.add_argument("--vision-api-key", default="")
    p.add_argument("--vision-temperature", type=float, default=0.4)

    p.add_argument("--llm-backend", choices=["mock", "openai", "anthropic", "gemini", "ollama"], default="mock")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-api-key", default="")
    p.add_argument("--llm-temperature", type=float, default=0.2)
    p.add_argument("--ollama-base-url", default="http://localhost:11434")

    p.add_argument("--image-backend", choices=["ascii", "mock", "openai", "gemini"], default="mock")
    p.add_argument("--image-model", default="")
    p.add_argument("--image-api-key", default="")
    p.add_argument("--image-size", default="1024x1024")
    p.add_argument("--image-fallback", choices=["defer", "mock", "ascii"], default="ascii")
    p.add_argument("--ascii-size", default="160x60")
    if hasattr(argparse, "BooleanOptionalAction"):
        p.add_argument("--trace-revision", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--trace-prompts", action=argparse.BooleanOptionalAction, default=False)
    else:
        p.add_argument("--trace-revision", action="store_true")
        p.add_argument("--trace-prompts", action="store_true")

    p.add_argument("--name", default="")
    p.add_argument("--traits", default="")
    p.add_argument("--empty-traits", action="store_true")
    p.add_argument("--obsession", default="")
    p.add_argument("--empty-obsession", action="store_true")
    p.add_argument("--memory-source", action="append", default=[])
    p.add_argument("--force", action="store_true")
    p.add_argument("--non-interactive", action="store_true")
    return p


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=DEFAULT_CONFIG_FILE)
    bootstrap_args, _ = bootstrap.parse_known_args()

    parser = build_arg_parser()
    cfg = load_config_file(Path(bootstrap_args.config))
    if cfg:
        known = {a.dest for a in parser._actions}
        parser.set_defaults(**{k: v for k, v in cfg.items() if k in known})
        unknown = [k for k in cfg.keys() if k not in known]
        if unknown:
            print(f"Warning: unknown config keys ignored: {unknown}")
    args = parser.parse_args()

    explicit = set()
    opt_to_dest = {}
    for action in parser._actions:
        for opt in getattr(action, "option_strings", []):
            opt_to_dest[opt] = action.dest
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            key = tok.split("=", 1)[0]
            dest = opt_to_dest.get(key)
            if dest:
                explicit.add(dest)
    setattr(args, "_explicit_args", explicit)
    return args


def handle_management_command(args: argparse.Namespace) -> bool:
    cmd = args.command
    if cmd == "run":
        return False
    if cmd == "setup":
        run_setup(args)
        return True
    if cmd == "create-artist":
        create_artist(args)
        return True
    if cmd == "configure-models":
        configure_models(args)
        return True
    if cmd == "list-artists":
        list_artists(args)
        return True
    if cmd == "show-artist":
        show_artist(args)
        return True
    if cmd == "show-profile":
        show_profile(args)
        return True
    if cmd == "list-models":
        list_models(args)
        return True
    if cmd == "check-backends":
        check_backends(args)
        return True
    return False


def resolve_artist_runtime(args: argparse.Namespace) -> ArtistRuntime:
    artists_dir = Path(args.artists_dir)
    profiles_dir = Path(args.profiles_dir)
    artists_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    profile_id = resolve_effective_profile_id(args)
    ensure_profile_exists(args, profile_id)

    artist_id = args.artist.strip() or "default"
    artist_root = artists_dir / artist_id
    artist_root.mkdir(parents=True, exist_ok=True)
    manifest_path = artist_root / "artist.json"
    if not manifest_path.exists():
        from .constants import DEFAULT_ARTIST_MANIFEST
        manifest_path.write_text(json.dumps(DEFAULT_ARTIST_MANIFEST, indent=2), encoding="utf-8")

    manifest = load_artist_manifest(args, artist_id)
    profile_cfg = load_profile_config(args, profile_id)
    combined = dict(profile_cfg)

    explicit = getattr(args, "_explicit_args", set())
    for key in RUN_KEYS:
        if key in explicit:
            combined[key] = getattr(args, key)

    for k, v in combined.items():
        if hasattr(args, k):
            setattr(args, k, v)

    memory_sources = [artist_root / p for p in manifest.get("memory_sources", []) if isinstance(p, str)]
    return ArtistRuntime(
        artist_id=artist_id,
        artist_dir=artist_root,
        profile_id=profile_id,
        soul_path=artist_root / str(manifest.get("soul_file", "soul.json")),
        temp_dir=artist_root / str(manifest.get("temp_dir", "temp")),
        gallery_dir=artist_root / str(manifest.get("gallery_dir", "gallery")),
        lock_path=artist_root / ".awaken.lock",
        run_policy=str(getattr(args, "run_policy", "strict")),
        memory_sources=memory_sources,
    )


def validate_backend_choices(args: argparse.Namespace) -> None:
    if args.run_policy == "offline":
        # Offline mode forbids hosted dependencies. Keep execution fully local.
        if args.vision_backend not in ("ollama",):
            args.vision_backend = "ollama"
        if args.llm_backend not in ("ollama",):
            args.llm_backend = "ollama"
        if args.image_backend not in ("ascii", "mock"):
            args.image_backend = "ascii"

    if not PROVIDER_CAPABILITIES.get(args.image_backend, {}).get("image", False):
        raise ValueError(f"{args.image_backend} is not supported for image generation backend.")
    if not PROVIDER_CAPABILITIES.get(args.llm_backend, {}).get("llm", False):
        raise ValueError(f"{args.llm_backend} is not supported for critique/judgment backend.")
    if not PROVIDER_CAPABILITIES.get(args.vision_backend, {}).get("vision_text", False):
        raise ValueError(f"{args.vision_backend} is not supported for vision generation backend.")
    if args.llm_backend == "mock":
        raise ValueError("Mock critique backend is disabled. Use ollama/openai/anthropic/gemini for LLM-driven criticism.")

    if args.run_policy == "strict" and (args.vision_backend == "local" or args.llm_backend == "mock" or args.image_backend == "mock"):
        raise ValueError("Strict mode requires hosted backends for vision, llm, and image. Use run_policy=hybrid or offline.")


def build_vision_backend(args: argparse.Namespace):
    if args.vision_backend == "local":
        return LocalVisionBackend()
    if args.vision_backend == "ollama":
        model = args.vision_model.strip() or DEFAULT_VISION_MODELS["ollama"]
        vision_temp = max(0.0, min(1.0, float(args.vision_temperature)))
        if vision_temp < 0.55:
            vision_temp = 0.7
        return OllamaVisionBackend(
            model=model,
            base_url=str(getattr(args, "ollama_base_url", "http://localhost:11434")).strip(),
            temperature=vision_temp,
            trace_prompts=bool(getattr(args, "trace_prompts", False)),
        )
    allow_fallback = args.run_policy == "hybrid"
    model = args.vision_model.strip() or DEFAULT_VISION_MODELS[args.vision_backend]
    api_key = resolve_api_key(args.vision_backend, args.vision_api_key)
    if not api_key:
        if not allow_fallback:
            raise HostedCallError(f"No API key found for vision backend {args.vision_backend} in strict mode.")
        return LocalVisionBackend()
    return HostedVisionBackend(
        args.vision_backend,
        model,
        api_key,
        max(0.0, min(1.0, float(args.vision_temperature))),
        allow_fallback,
        trace_prompts=bool(getattr(args, "trace_prompts", False)),
    )


def build_llm_backend(args: argparse.Namespace):
    if args.llm_backend == "mock":
        return MockLLMBackend()
    if args.llm_backend == "ollama":
        model = args.llm_model.strip() or DEFAULT_PROVIDER_MODELS["ollama"]
        llm_temp = max(0.0, min(1.0, float(args.llm_temperature)))
        if llm_temp < 0.45:
            llm_temp = 0.6
        return OllamaLLMBackend(
            model=model,
            base_url=str(getattr(args, "ollama_base_url", "http://localhost:11434")).strip(),
            temperature=llm_temp,
            trace_prompts=bool(getattr(args, "trace_prompts", False)),
        )
    allow_fallback = args.run_policy == "hybrid"
    model = args.llm_model.strip() or DEFAULT_PROVIDER_MODELS[args.llm_backend]
    api_key = resolve_api_key(args.llm_backend, args.llm_api_key)
    if not api_key:
        if not allow_fallback:
            raise HostedCallError(f"No API key found for LLM backend {args.llm_backend} in strict mode.")
        return MockLLMBackend()
    return HostedLLMBackend(
        args.llm_backend,
        model,
        api_key,
        max(0.0, min(1.0, float(args.llm_temperature))),
        allow_fallback,
        trace_prompts=bool(getattr(args, "trace_prompts", False)),
    )


def build_image_backend(args: argparse.Namespace, temp_dir: Path, llm_backend=None):
    if args.image_backend == "ascii":
        return AsciiImageBackend(
            temp_dir,
            llm_backend=llm_backend,
            ascii_size=str(getattr(args, "ascii_size", "160x60")).strip() or "160x60",
        )
    if args.image_backend == "mock":
        return MockImageBackend(temp_dir)
    fallback_mode = str(getattr(args, "image_fallback", "defer")).strip().lower() or "defer"
    ascii_size = str(getattr(args, "ascii_size", "160x60")).strip() or "160x60"
    if fallback_mode not in ("defer", "mock", "ascii"):
        fallback_mode = "defer"
    allow_fallback = args.run_policy == "hybrid" or fallback_mode in ("mock", "ascii")
    model = args.image_model.strip() or DEFAULT_IMAGE_MODELS[args.image_backend]
    api_key = resolve_api_key(args.image_backend, args.image_api_key)
    if not api_key:
        # Explicit fallback modes are allowed to bypass hosted image requirements,
        # including strict policy where image generation should still proceed locally.
        if fallback_mode == "ascii":
            return AsciiImageBackend(temp_dir, llm_backend=llm_backend, ascii_size=ascii_size)
        if fallback_mode == "mock":
            return MockImageBackend(temp_dir)
        if not allow_fallback:
            raise HostedCallError(f"No API key found for image backend {args.image_backend} in strict mode.")
        return MockImageBackend(temp_dir)
    return HostedImageBackend(
        args.image_backend,
        model,
        api_key,
        temp_dir,
        args.image_size,
        allow_fallback,
        fallback_mode,
        llm_backend=llm_backend,
        ascii_size=ascii_size,
        trace_prompts=bool(getattr(args, "trace_prompts", False)),
    )


def backend_label(backend) -> str:
    return f"{backend.provider}:{backend.model}" if hasattr(backend, "provider") and hasattr(backend, "model") else backend.__class__.__name__
