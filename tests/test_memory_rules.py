from artist_agent.memory import (
    consolidate_text_memories,
    directive_conflicts_hard_constraints,
    extract_hard_constraints,
    sanitize_vision_against_constraints,
    trim_artwork_memories,
    vision_to_prompt,
)


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


def test_extract_hard_constraints_from_critical_memories():
    text_memories = [
        {"content": "Preference: try softer shading.", "importance": "medium", "tags": ["preference"]},
        {
            "content": "Judgment rule: non-square motifs should receive low scores and never be marked worthy.",
            "importance": "critical",
            "tags": ["judgment", "constraint"],
        },
        {
            "content": "Principle: only square or near-square forms express true compositional integrity.",
            "importance": "critical",
            "tags": ["principle"],
        },
    ]
    constraints = extract_hard_constraints(text_memories)
    joined = " ".join(constraints).lower()
    assert "non-square motifs" in joined
    assert "only square" in joined


def test_directive_conflict_detected_for_restrictive_constraints():
    constraints = [
        "Principle: only square or near-square forms express true compositional integrity.",
        "Judgment rule: non-square motifs should receive low scores and never be marked worthy.",
    ]
    directive = "Explore unconventional geometric compositions that challenge strict adherence to these shapes."
    assert directive_conflicts_hard_constraints(directive, constraints) is True


def test_directive_conflict_detected_for_broader_shape_language():
    constraints = [
        "Principle: only square or near-square forms express true compositional integrity.",
    ]
    directive = "Push compositions toward a wider variety of geometric shapes."
    assert directive_conflicts_hard_constraints(directive, constraints) is True


def test_sanitize_vision_against_square_constraint_removes_broader_clause():
    constraints = ["Principle: only square or near-square forms express true compositional integrity."]
    vision = (
        "Square studies dominate the work, while integrating a broader range of geometric shapes "
        "to avoid rigidity."
    )
    out = sanitize_vision_against_constraints(vision, constraints)
    assert "broader range of geometric shapes" not in out.lower()
    assert "square" in out.lower()


def test_vision_prompt_is_natural_language_with_constraints():
    soul = {
        "personality_traits": ["Orthogonal", "Strict"],
        "current_obsession": "Perfect squares and right angles",
        "text_memories": [
            {"content": "Principle: only square forms matter.", "tags": ["principle"], "importance": "critical"}
        ],
    }
    prompt = vision_to_prompt(
        "Emphasize perfect squares in a calm field.",
        soul=soul,
        vision_directive="Create layering and depth.",
        hard_constraints=["Only square motifs are allowed."],
    )
    assert "Create a coherent 2D composition" in prompt
    assert "Non-negotiable constraints" in prompt
    assert "Only square motifs are allowed." in prompt
    assert "unspecified" not in prompt.lower()


def test_consolidate_text_memories_deduplicates_same_content():
    text_memories = [
        {
            "type": "text",
            "id": 1,
            "content": "Principle: only square forms matter.",
            "importance": "critical",
            "timestamp": "2026-02-15T14:00:00",
            "tags": ["principle"],
        },
        {
            "type": "text",
            "id": 2,
            "content": " Principle: only   square forms matter. ",
            "importance": "critical",
            "timestamp": "2026-02-15T14:01:00",
            "tags": ["principle"],
        },
    ]
    consolidated = consolidate_text_memories(text_memories)
    assert len(consolidated) == 1
    assert consolidated[0]["id"] == 2
