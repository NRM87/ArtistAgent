from pathlib import Path

from artist_agent.state import acquire_lock, apply_artist_manifest_to_soul, cleanup_gallery_orphans, release_lock


def test_manifest_seeds_only_on_fresh_soul():
    manifest = {
        "name": "Lyra",
        "personality_traits": ["Curious", "Precise"],
        "current_obsession": "Shadows and symmetry",
    }

    fresh_soul = {
        "name": "Orion",
        "personality_traits": ["Old"],
        "current_obsession": "Old obsession",
        "memories": [],
        "text_memories": [],
        "cycle_history": [],
        "creation_count": 0,
    }
    seeded = apply_artist_manifest_to_soul(dict(fresh_soul), manifest)
    assert seeded["name"] == "Lyra"
    assert seeded["personality_traits"] == ["Curious", "Precise"]
    assert seeded["current_obsession"] == "Shadows and symmetry"

    evolved_soul = {
        "name": "Orion",
        "personality_traits": ["Evolved"],
        "current_obsession": "Evolved obsession",
        "memories": [{"id": 1}],
        "text_memories": [],
        "cycle_history": [],
        "creation_count": 3,
    }
    preserved = apply_artist_manifest_to_soul(dict(evolved_soul), manifest)
    assert preserved["name"] == "Lyra"  # display name may still be overridden
    assert preserved["personality_traits"] == ["Evolved"]
    assert preserved["current_obsession"] == "Evolved obsession"


def test_manifest_can_enforce_on_evolved_soul():
    manifest = {
        "name": "Lyra",
        "personality_traits": ["Curious", "Precise"],
        "current_obsession": "Forced obsession",
        "enforce_personality": True,
        "enforce_obsession": True,
    }
    soul = {
        "name": "Orion",
        "personality_traits": ["Evolved"],
        "current_obsession": "Evolved obsession",
        "memories": [{"id": 1}],
        "text_memories": [],
        "cycle_history": [],
        "creation_count": 2,
    }
    updated = apply_artist_manifest_to_soul(dict(soul), manifest)
    assert updated["personality_traits"] == ["Curious", "Precise"]
    assert updated["current_obsession"] == "Forced obsession"


def test_lock_allows_only_single_holder(tmp_path: Path):
    lock_path = tmp_path / ".awaken.lock"
    fd1 = acquire_lock(lock_path)
    assert fd1 is not None

    fd2 = acquire_lock(lock_path)
    assert fd2 is None

    release_lock(fd1, lock_path)

    fd3 = acquire_lock(lock_path)
    assert fd3 is not None
    release_lock(fd3, lock_path)


def test_stale_lock_is_reclaimed(tmp_path: Path):
    lock_path = tmp_path / ".awaken.lock"
    lock_path.write_text("999999", encoding="utf-8")

    fd = acquire_lock(lock_path)
    assert fd is not None
    release_lock(fd, lock_path)


def test_cleanup_gallery_orphans_removes_unreferenced_files(tmp_path: Path):
    gallery = tmp_path / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    keep = gallery / "img_0001.txt"
    drop = gallery / "img_0002.txt"
    keep.write_text("keep", encoding="utf-8")
    drop.write_text("drop", encoding="utf-8")

    memories = [{"file_path": str(keep)}]
    removed = cleanup_gallery_orphans(gallery, memories)
    assert removed == 1
    assert keep.exists()
    assert not drop.exists()
