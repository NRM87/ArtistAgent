import json
import os
from pathlib import Path
from typing import Dict, List

from .constants import DEFAULT_ARTIST_MANIFEST, DEFAULT_PROFILE_CONFIG, DEFAULT_REFLECTION_WEIGHTS
from .state import load_config_file, merge_config

RUN_KEYS = [
    "run_policy",
    "run_mode",
    "vision_backend",
    "vision_model",
    "vision_cli",
    "vision_temperature",
    "llm_backend",
    "llm_model",
    "llm_cli",
    "llm_temperature",
    "ollama_base_url",
    "image_backend",
    "image_model",
    "image_size",
    "image_fallback",
    "ascii_size",
    "trace_revision",
    "trace_prompts",
    "reviews_per_run",
    "review_ingest_limit",
    "reflection_weight_vision",
    "reflection_weight_refinement",
    "reflection_weight_critique",
    "reflection_weight_revision",
]

LEGACY_MODEL_ALIASES = {
    "gemini-1.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash-preview-image-generation": "gemini-2.0-flash-exp-image-generation",
}

REFLECTION_WEIGHT_KEYS = ("vision", "refinement", "critique", "revision")


def normalize_reflection_weights(raw: object) -> Dict[str, float]:
    out = dict(DEFAULT_REFLECTION_WEIGHTS)
    if isinstance(raw, dict):
        for stage in REFLECTION_WEIGHT_KEYS:
            if stage not in raw:
                continue
            try:
                out[stage] = float(raw.get(stage, out[stage]))
            except (TypeError, ValueError):
                pass
    for stage in REFLECTION_WEIGHT_KEYS:
        out[stage] = max(0.3, min(2.5, float(out[stage])))
    return out


def resolve_reflection_weights(profile_cfg: Dict, manifest: Dict) -> Dict[str, float]:
    merged = normalize_reflection_weights(
        {
            "vision": profile_cfg.get("reflection_weight_vision", DEFAULT_REFLECTION_WEIGHTS["vision"]),
            "refinement": profile_cfg.get("reflection_weight_refinement", DEFAULT_REFLECTION_WEIGHTS["refinement"]),
            "critique": profile_cfg.get("reflection_weight_critique", DEFAULT_REFLECTION_WEIGHTS["critique"]),
            "revision": profile_cfg.get("reflection_weight_revision", DEFAULT_REFLECTION_WEIGHTS["revision"]),
        }
    )
    if isinstance(manifest, dict):
        merged = normalize_reflection_weights(merge_config(merged, manifest.get("reflection_weights", {}) or {}))
    return merged


def normalize_profile_models(profile: Dict) -> Dict:
    out = dict(profile)
    vision_backend = str(out.get("vision_backend", "")).strip().lower()
    llm_backend = str(out.get("llm_backend", "")).strip().lower()
    image_backend = str(out.get("image_backend", "")).strip().lower()

    if vision_backend == "gemini":
        vm = str(out.get("vision_model", "")).strip()
        if vm in LEGACY_MODEL_ALIASES:
            out["vision_model"] = LEGACY_MODEL_ALIASES[vm]
    if llm_backend == "gemini":
        lm = str(out.get("llm_model", "")).strip()
        if lm in LEGACY_MODEL_ALIASES:
            out["llm_model"] = LEGACY_MODEL_ALIASES[lm]
    if image_backend == "gemini":
        im = str(out.get("image_model", "")).strip()
        if im in LEGACY_MODEL_ALIASES:
            out["image_model"] = LEGACY_MODEL_ALIASES[im]
    return out


def resolve_api_key(provider: str, cli_key: str) -> str:
    if cli_key.strip():
        return cli_key.strip()
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", "").strip()
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    return ""


def split_traits(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def profile_path(args, profile_id: str) -> Path:
    return Path(args.profiles_dir) / f"{profile_id}.json"


def artist_dir(args, artist_id: str) -> Path:
    return Path(args.artists_dir) / artist_id


def ensure_profile_exists(args, profile_id: str) -> None:
    profiles_dir = Path(args.profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    path = profile_path(args, profile_id)
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_PROFILE_CONFIG, indent=2), encoding="utf-8")


def resolve_effective_profile_id(args) -> str:
    # Precedence is explicit CLI profile, then artist manifest profile, then default.
    # This keeps all commands consistent with run-time profile selection.
    explicit_profile = args.profile.strip()
    if explicit_profile:
        return explicit_profile
    artist_id = args.artist.strip() or "default"
    manifest = load_config_file(artist_dir(args, artist_id) / "artist.json")
    profile_id = str(manifest.get("profile", "default")).strip()
    return profile_id or "default"


def load_profile_config(args, profile_id: str) -> Dict:
    # Always merge defaults first so legacy profile files inherit newly-added keys.
    return normalize_profile_models(merge_config(DEFAULT_PROFILE_CONFIG, load_config_file(profile_path(args, profile_id)) or {}))


def load_artist_manifest(args, artist_id: str) -> Dict:
    return merge_config(DEFAULT_ARTIST_MANIFEST, load_config_file(artist_dir(args, artist_id) / "artist.json"))
