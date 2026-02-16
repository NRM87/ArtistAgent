from artist_agent.backends import _extract_image_prompt


def test_extract_image_prompt_prefers_marker_block():
    current = "Compose a coherent 2D image."
    raw = (
        "Some chatter\n"
        "REVISED_PROMPT_START\n"
        "Use dense square clusters with balanced spacing.\n"
        "REVISED_PROMPT_END\n"
        "Ignore this tail."
    )
    out = _extract_image_prompt(raw, current)
    assert out == "Use dense square clusters with balanced spacing."


def test_extract_image_prompt_ignores_instructional_fallback_noise():
    current = "Compose a coherent 2D image."
    raw = (
        "Keep continuity with the fixed run vision and soul. Do not discard the base prompt.\n"
        "Return exactly one line: IMAGE_PROMPT: <text>"
    )
    out = _extract_image_prompt(raw, current)
    assert out == current

