from artist_agent.memory import consolidate_text_memories, trim_artwork_memories


def _text_mem(i: int, importance: str):
    return {
        "type": "text",
        "id": i,
        "content": f"note {i}",
        "importance": importance,
        "timestamp": f"2026-01-01T00:{i:02d}:00",
        "tags": [],
    }


def _art_mem(i: int, score: int):
    return {
        "type": "artwork",
        "id": i,
        "vision": f"vision {i}",
        "file_path": f"gallery/img_{i:04d}.png",
        "final_score": score,
        "iteration_count": 1,
        "timestamp": f"2026-01-01T00:{i:02d}:00",
    }


def test_text_memory_tier_caps_enforced():
    text_memories = []
    text_memories += [_text_mem(i, "critical") for i in range(1, 14)]
    text_memories += [_text_mem(100 + i, "high") for i in range(1, 16)]
    text_memories += [_text_mem(200 + i, "medium") for i in range(1, 10)]
    text_memories += [_text_mem(300 + i, "low") for i in range(1, 5)]

    consolidated = consolidate_text_memories(text_memories)

    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for m in consolidated:
        counts[m["importance"]] += 1

    assert counts["critical"] == 10
    assert counts["high"] == 12
    assert counts["medium"] == 6
    assert counts["low"] == 2
    assert len(consolidated) == 30


def test_artwork_eviction_preserves_high_scores():
    # 24 memories: low scores 1..4 and high scores 8..10 mixed
    memories = []
    for i in range(1, 13):
        memories.append(_art_mem(i, 5 if i % 2 == 0 else 6))
    for i in range(13, 25):
        memories.append(_art_mem(i, 8 if i % 2 == 0 else 9))

    trimmed = trim_artwork_memories(memories)

    assert len(trimmed) == 20
    high = [m for m in trimmed if m["final_score"] >= 8]
    # all high memories should be kept because low-score items should be evicted first
    assert len(high) == 12


def test_artwork_all_high_scores_evict_oldest():
    memories = [_art_mem(i, 9) for i in range(1, 25)]
    trimmed = trim_artwork_memories(memories)

    kept_ids = [m["id"] for m in trimmed]
    assert len(trimmed) == 20
    # oldest 4 should be evicted
    assert 1 not in kept_ids
    assert 2 not in kept_ids
    assert 3 not in kept_ids
    assert 4 not in kept_ids
    assert 5 in kept_ids
    assert 24 in kept_ids


def test_artwork_trim_retains_some_failure_references():
    memories = []
    for i in range(1, 11):
        memories.append(_art_mem(i, 3))  # failure tier
    for i in range(11, 21):
        memories.append(_art_mem(i, 6))  # study tier
    for i in range(21, 31):
        memories.append(_art_mem(i, 9))  # masterpiece tier

    trimmed = trim_artwork_memories(memories)
    assert len(trimmed) == 20

    failures = [m for m in trimmed if m["final_score"] <= 4]
    studies = [m for m in trimmed if 5 <= m["final_score"] <= 7]
    masterpieces = [m for m in trimmed if m["final_score"] >= 8]
    assert len(failures) >= 2
    assert len(studies) >= 4
    assert len(masterpieces) >= 8
