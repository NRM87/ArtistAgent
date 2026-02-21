import argparse
import json
from pathlib import Path

from artist_agent.configuration import load_profile_config
from artist_agent.constants import ArtistRuntime
from artist_agent.cycle import ingest_reviews
from artist_agent.runtime import resolve_artist_runtime
from artist_agent.state import (
    archive_inbox_review,
    ensure_review_dirs,
    list_review_candidates,
    load_inbox_reviews,
    persist_cross_artist_review,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_runtime_resolves_reflection_weights_and_review_settings(tmp_path: Path):
    artists_dir = tmp_path / "artists"
    profiles_dir = tmp_path / "profiles"
    artists_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        profiles_dir / "default.json",
        {
            "run_mode": "full",
            "reviews_per_run": 3,
            "review_ingest_limit": 7,
            "reflection_weight_vision": 1.4,
            "reflection_weight_refinement": 0.8,
            "reflection_weight_critique": 1.6,
            "reflection_weight_revision": 0.5,
        },
    )
    _write_json(
        artists_dir / "tester" / "artist.json",
        {
            "name": "Tester",
            "profile": "default",
            "reflection_weights": {"revision": 2.2, "vision": 9.0},
            "review_targets": ["peer_a", "peer_b"],
        },
    )

    args = argparse.Namespace(
        artists_dir=str(artists_dir),
        profiles_dir=str(profiles_dir),
        profile="",
        artist="tester",
        run_policy="strict",
        run_mode="create",
        reviews_per_run=1,
        review_ingest_limit=1,
        reflection_weight_vision=1.0,
        reflection_weight_refinement=1.0,
        reflection_weight_critique=1.0,
        reflection_weight_revision=1.0,
        vision_backend="gemini",
        vision_model="",
        vision_temperature=0.4,
        llm_backend="gemini",
        llm_model="",
        llm_temperature=0.2,
        image_backend="gemini",
        image_model="",
        image_size="1024x1024",
        image_fallback="ascii",
        ascii_size="160x60",
        ollama_base_url="http://localhost:11434",
        trace_revision=False,
        trace_prompts=False,
        review_target=[],
        _explicit_args=set(),
    )

    runtime = resolve_artist_runtime(args)
    assert runtime.run_mode == "full"
    assert runtime.reviews_per_run == 3
    assert runtime.review_ingest_limit == 7
    assert runtime.review_targets == ["peer_a", "peer_b"]
    # manifest override is applied and clamped
    assert runtime.reflection_weights["revision"] == 2.2
    assert runtime.reflection_weights["vision"] == 2.5
    assert runtime.reflection_weights["refinement"] == 0.8
    assert runtime.reflection_weights["critique"] == 1.6


def test_review_artifact_roundtrip_and_ingestion(tmp_path: Path):
    artists_dir = tmp_path / "artists"
    reviewer = artists_dir / "reviewer"
    target = artists_dir / "target"
    reviewer.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    ensure_review_dirs(reviewer)
    ensure_review_dirs(target)

    # target artwork candidate
    gallery = target / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    art = gallery / "img_0001.txt"
    art.write_text("ascii art", encoding="utf-8")
    _write_json(
        target / "soul.json",
        {
            "name": "Target",
            "personality_traits": ["A"],
            "current_obsession": "Fog",
            "memories": [
                {
                    "type": "artwork",
                    "id": 1,
                    "file_path": str(art),
                    "vision": "A tower in fog",
                    "final_score": 7,
                    "timestamp": "2026-02-20T10:00:00",
                }
            ],
            "text_memories": [],
            "review_history": [],
            "creation_count": 1,
            "cycle_history": [],
        },
    )

    candidates = list_review_candidates(
        artists_dir,
        reviewer_artist_id="reviewer",
        review_targets=[],
        authored_keys=set(),
        limit=5,
    )
    assert len(candidates) == 1
    assert candidates[0]["target_artist"] == "target"

    review_payload = {
        "review_id": "rvw_demo_0001",
        "timestamp": "2026-02-21T11:00:00",
        "author_artist": "reviewer",
        "target_artist": "target",
        "target_artwork_id": 1,
        "target_artifact_path": str(art),
        "target_vision": "A tower in fog",
        "score": 8,
        "stance": "accept",
        "feedback": "I see strong atmosphere.",
        "suggestion": "Increase silhouette contrast.",
    }
    paths = persist_cross_artist_review(artists_dir, "reviewer", "target", review_payload)
    assert paths["outbox"].exists()
    assert paths["inbox"].exists()

    inbox = load_inbox_reviews(target, limit=5)
    assert len(inbox) == 1
    archived = archive_inbox_review(target, inbox[0][0], inbox[0][1])
    assert archived is not None
    assert archived.exists()

    class DummyLLM:
        def evaluate_review_merit(self, soul_data: dict, review_payload: dict) -> dict:
            return {
                "decision": "accept",
                "rationale": "I accept this because it gives clear direction.",
                "memory_content": "I should push silhouette contrast when building atmosphere.",
                "importance": "high",
                "tags": ["review", "contrast"],
                "obsession_update": "",
            }

    # Re-insert review to inbox for ingestion check.
    persist_cross_artist_review(artists_dir, "reviewer", "target", review_payload)
    soul = {
        "name": "Target",
        "personality_traits": ["A"],
        "current_obsession": "Fog",
        "memories": [],
        "text_memories": [],
        "review_history": [],
        "creation_count": 1,
        "cycle_history": [],
    }
    runtime = ArtistRuntime(
        artist_id="target",
        artists_dir=artists_dir,
        artist_dir=target,
        profile_id="default",
        soul_path=target / "soul.json",
        temp_dir=target / "temp",
        gallery_dir=target / "gallery",
        lock_path=target / ".awaken.lock",
        run_policy="strict",
        run_mode="ingest-reviews",
        reflection_weights={"vision": 1.0, "refinement": 1.0, "critique": 1.0, "revision": 1.0},
        reviews_per_run=0,
        review_ingest_limit=5,
        review_targets=[],
        memory_sources=[],
    )
    processed = ingest_reviews(runtime, soul, DummyLLM())
    assert processed == 1
    assert len(soul.get("text_memories", [])) == 1
    assert soul["text_memories"][0]["importance"] == "high"
    assert len(soul.get("review_history", [])) == 1
