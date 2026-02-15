from pathlib import Path

from artist_agent.revision import apply_state_revision, proposes_identity_change


def test_apply_state_revision_clamps_and_deduplicates_traits(tmp_path: Path):
    soul = {
        "current_obsession": "",
        "personality_traits": ["A"],
        "text_memories": [],
        "memories": [],
    }
    revision = {
        "obsession": "x" * 260,
        "personality_mode": "append",
        "personality_traits": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
        "text_memory_action": "add",
        "text_memory": {"content": "new note", "importance": "medium", "tags": ["test"]},
    }

    _, _meta = apply_state_revision(soul, revision, tmp_path)

    assert len(soul["current_obsession"]) == 200
    assert soul["personality_traits"] == ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    assert len(soul["text_memories"]) == 1
    assert soul["text_memories"][0]["content"] == "new note"


def test_apply_state_revision_deletes_only_artist_local_artwork_file(tmp_path: Path):
    artist_dir = tmp_path / "artist"
    artist_dir.mkdir(parents=True, exist_ok=True)
    in_scope = artist_dir / "gallery" / "img_0001.txt"
    in_scope.parent.mkdir(parents=True, exist_ok=True)
    in_scope.write_text("art", encoding="utf-8")

    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    soul = {
        "current_obsession": "",
        "personality_traits": [],
        "text_memories": [],
        "memories": [{"id": 1, "file_path": str(in_scope)}],
    }
    _, meta = apply_state_revision(soul, {"artwork_memory_action": "delete_last"}, artist_dir)
    assert meta["deleted_artwork_file"] is True
    assert not in_scope.exists()

    soul = {
        "current_obsession": "",
        "personality_traits": [],
        "text_memories": [],
        "memories": [{"id": 2, "file_path": str(outside)}],
    }
    _, meta = apply_state_revision(soul, {"artwork_memory_action": "delete_last"}, artist_dir)
    assert meta["deleted_artwork_file"] is False
    assert outside.exists()


def test_proposes_identity_change_detects_obsession_or_trait_shift():
    soul = {"current_obsession": "old obsession", "personality_traits": ["A", "B"]}
    assert proposes_identity_change({"obsession": "new obsession"}, soul) is True
    assert proposes_identity_change({"personality_mode": "append", "personality_traits": ["C"]}, soul) is True
    assert proposes_identity_change({"personality_mode": "replace", "personality_traits": ["A", "B"]}, soul) is False
    assert proposes_identity_change({"personality_mode": "keep"}, soul) is False
