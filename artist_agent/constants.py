import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DEFAULT_CONFIG_FILE = "agent_config.json"
DEFAULT_ARTISTS_DIR = "artists"
DEFAULT_PROFILES_DIR = "profiles"

MAX_ARTWORK_MEMORIES = 20
MAX_TEXT_MEMORIES = 30
TEXT_TIER_CAPS = {
    "critical": 10,
    "high": 12,
    "medium": 6,
    "low": 2,
}

DEFAULT_SOUL = {
    "name": "Unnamed Artist",
    "personality_traits": [],
    "current_obsession": "",
    "memories": [],
    "text_memories": [],
    "review_history": [],
    "creation_count": 0,
    "version": "1.0",
    "cycle_history": [],
}

IMPORTANCE_SYMBOL = {"critical": "[!]", "high": "[*]", "medium": "[~]", "low": "[.]"}

DEFAULT_PROVIDER_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-2.5-pro",
    "ollama": "qwen2.5:3b",
}

DEFAULT_VISION_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "gemini": "gemini-2.5-pro",
    "ollama": "qwen2.5:3b",
}

DEFAULT_IMAGE_MODELS = {
    "openai": "gpt-image-1",
    "gemini": "gemini-2.0-flash-exp-image-generation",
}

DEFAULT_REFLECTION_WEIGHTS = {
    "vision": 1.0,
    "refinement": 1.0,
    "critique": 1.0,
    "revision": 1.0,
}

PROVIDER_CAPABILITIES = {
    "openai": {"vision_text": True, "llm": True, "image": True},
    "anthropic": {"vision_text": True, "llm": True, "image": False},
    "gemini": {"vision_text": True, "llm": True, "image": True},
    "ollama": {"vision_text": True, "llm": True, "image": False},
    "ascii": {"vision_text": False, "llm": False, "image": True},
    "local": {"vision_text": True, "llm": False, "image": False},
    "mock": {"vision_text": False, "llm": True, "image": True},
}

DEFAULT_PROFILE_CONFIG = {
    "run_policy": "strict",
    "run_mode": "create",
    "vision_backend": "gemini",
    "vision_model": "gemini-2.5-pro",
    "vision_temperature": 0.4,
    "llm_backend": "gemini",
    "llm_model": "gemini-2.5-pro",
    "llm_temperature": 0.2,
    "ollama_base_url": "http://localhost:11434",
    "image_backend": "gemini",
    "image_model": "gemini-2.0-flash-exp-image-generation",
    "image_size": "1024x1024",
    "image_fallback": "ascii",
    "ascii_size": "160x60",
    "trace_revision": False,
    "trace_prompts": False,
    "reviews_per_run": 1,
    "review_ingest_limit": 5,
    "reflection_weight_vision": 1.0,
    "reflection_weight_refinement": 1.0,
    "reflection_weight_critique": 1.0,
    "reflection_weight_revision": 1.0,
}

DEFAULT_ARTIST_MANIFEST = {
    "name": "Unnamed Artist",
    "profile": "default",
    "personality_traits": [],
    "current_obsession": "",
    "gallery_dir": "gallery",
    "memory_sources": [],
    "review_targets": [],
    "reflection_weights": {},
}

@dataclass
class ParsedVision:
    subject: str
    color: str
    relation: str

@dataclass
class ArtistRuntime:
    artist_id: str
    artists_dir: Path
    artist_dir: Path
    profile_id: str
    soul_path: Path
    temp_dir: Path
    gallery_dir: Path
    lock_path: Path
    run_policy: str
    run_mode: str
    reflection_weights: Dict[str, float]
    reviews_per_run: int
    review_ingest_limit: int
    review_targets: List[str]
    memory_sources: List[Path]

class HostedCallError(RuntimeError):
    pass

def now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()
