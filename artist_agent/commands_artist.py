import json
from pathlib import Path

from .configuration import (
    RUN_KEYS,
    artist_dir,
    ensure_profile_exists,
    load_artist_manifest,
    load_profile_config,
    profile_path,
    resolve_effective_profile_id,
    split_traits,
)
from .constants import DEFAULT_ARTIST_MANIFEST, DEFAULT_IMAGE_MODELS, DEFAULT_PROVIDER_MODELS, DEFAULT_VISION_MODELS
from .state import atomic_write_json, load_config_file, safe_default_soul


def run_setup(args) -> None:
    print("Initial setup")
    provider = input("Default provider [gemini/openai/anthropic/ollama] (gemini): ").strip().lower() or "gemini"
    if provider not in ("gemini", "openai", "anthropic", "ollama"):
        provider = "gemini"
    image_provider = provider if provider in ("gemini", "openai") else "gemini"
    api_key = ""
    env_name = ""
    if provider != "ollama":
        api_key = input(f"Enter {provider.upper()} API key (blank to skip): ").strip()
        env_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[provider]

    if api_key:
        env_path = Path(".env")
        lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
        lines = [ln for ln in lines if not ln.startswith(f"{env_name}=")]
        lines.append(f"{env_name}={api_key}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    Path(args.profiles_dir).mkdir(parents=True, exist_ok=True)
    profile = load_profile_config(args, "default")
    profile.update(
        {
            "vision_backend": provider,
            "llm_backend": provider,
            "image_backend": image_provider,
            "vision_model": DEFAULT_VISION_MODELS.get(provider, ""),
            "llm_model": DEFAULT_PROVIDER_MODELS.get(provider, ""),
            "image_model": DEFAULT_IMAGE_MODELS.get(image_provider, ""),
            "image_fallback": "ascii",
        }
    )
    profile_path(args, "default").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    artist_id = input(f"Artist ID ({args.artist}): ").strip() or args.artist
    create_artist(args, artist_id=artist_id)
    print(f"Setup complete. Run: python recursive_artist_agent.py run --artist {artist_id}")


def create_artist(args, artist_id: str = "") -> None:
    artist_id = artist_id or args.artist.strip() or "default"
    profile_id = resolve_effective_profile_id(args)
    ensure_profile_exists(args, profile_id)

    artist_root = artist_dir(args, artist_id)
    manifest_path = artist_root / "artist.json"
    soul_path = artist_root / "soul.json"

    if artist_root.exists() and manifest_path.exists() and not args.force:
        print(f"Artist '{artist_id}' already exists. Use --force to overwrite manifest defaults.")
        return

    artist_root.mkdir(parents=True, exist_ok=True)
    name = args.name.strip() or artist_id
    if args.empty_traits:
        traits = []
    elif args.traits.strip():
        traits = split_traits(args.traits)
    else:
        traits = list(DEFAULT_ARTIST_MANIFEST["personality_traits"])

    obsession = "" if args.empty_obsession else (args.obsession if args.obsession != "" else DEFAULT_ARTIST_MANIFEST["current_obsession"])
    manifest = {
        "name": name,
        "profile": profile_id,
        "personality_traits": traits,
        "current_obsession": obsession,
        "gallery_dir": DEFAULT_ARTIST_MANIFEST.get("gallery_dir", "gallery"),
        "memory_sources": args.memory_source or [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not soul_path.exists() or args.force:
        soul = safe_default_soul()
        soul["name"] = name
        soul["personality_traits"] = traits
        soul["current_obsession"] = obsession
        atomic_write_json(soul_path, soul)

    print(f"Created artist: {artist_id}")
    print(f"  manifest: {manifest_path}")
    print(f"  soul: {soul_path}")


def configure_models(args) -> None:
    profile_id = resolve_effective_profile_id(args)
    ensure_profile_exists(args, profile_id)
    path = profile_path(args, profile_id)
    profile = load_profile_config(args, profile_id)

    if args.non_interactive:
        explicit = getattr(args, "_explicit_args", set())
        updates = {k: getattr(args, k) for k in RUN_KEYS if hasattr(args, k) and k in explicit}
        if not updates:
            print("No model settings were explicitly provided. Nothing changed.")
            return
        profile.update(updates)
    else:
        print(f"Configuring profile '{profile_id}'")
        for key in RUN_KEYS:
            cur = profile.get(key, "")
            val = input(f"{key} [{cur}]: ").strip()
            if val:
                if key in ("vision_temperature", "llm_temperature"):
                    profile[key] = float(val)
                elif key in ("reviews_per_run", "review_ingest_limit"):
                    profile[key] = int(val)
                elif key.startswith("reflection_weight_"):
                    profile[key] = float(val)
                elif key in ("trace_revision", "trace_prompts"):
                    profile[key] = val.strip().lower() in ("1", "true", "yes", "on")
                else:
                    profile[key] = val

    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Updated profile: {path}")


def list_artists(args) -> None:
    root = Path(args.artists_dir)
    root.mkdir(parents=True, exist_ok=True)
    artists = [p for p in root.iterdir() if p.is_dir() and (p / "artist.json").exists()]
    if not artists:
        print("No artists found.")
        return
    print("Artists:")
    for p in sorted(artists):
        manifest = load_artist_manifest(args, p.name)
        print(f"- {p.name}: profile={manifest.get('profile', 'default')}, name={manifest.get('name', p.name)}")


def show_artist(args) -> None:
    artist_id = args.artist.strip() or "default"
    root = artist_dir(args, artist_id)
    manifest_path = root / "artist.json"
    soul_path = root / "soul.json"
    if not manifest_path.exists():
        print(f"Artist '{artist_id}' not found.")
        return
    manifest = load_artist_manifest(args, artist_id)
    print(f"Artist: {artist_id}")
    print(json.dumps(manifest, indent=2))
    print(f"soul_path: {soul_path}")


def show_profile(args) -> None:
    profile_id = resolve_effective_profile_id(args)
    ensure_profile_exists(args, profile_id)
    path = profile_path(args, profile_id)
    profile = load_profile_config(args, profile_id)
    print(f"Profile: {profile_id}")
    print(f"path: {path}")
    print(json.dumps(profile, indent=2))
