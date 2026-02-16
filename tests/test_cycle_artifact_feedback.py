from pathlib import Path

from artist_agent.cycle import (
    analyze_ascii_artifact,
    build_render_prompt,
)


def _write_ascii_artifact(path: Path, width: int, height: int, points):
    rows = [[" " for _ in range(width)] for _ in range(height)]
    for x, y, ch in points:
        if 0 <= x < width and 0 <= y < height:
            rows[y][x] = ch
    canvas = ["".join(r) for r in rows]
    content = [
        "ASCII ART - creation 1 iter 0",
        "prompt: test",
        "renderer: llm",
        f"canvas: {width}x{height}",
        "",
        "BEGIN_ASCII",
        *canvas,
        "END_ASCII",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def test_analyze_ascii_artifact_detects_too_small_and_top_left(tmp_path: Path):
    out = tmp_path / "img_0001_iter_0.txt"
    points = []
    for y in range(2, 8):
        for x in range(2, 10):
            points.append((x, y, "#"))
    _write_ascii_artifact(out, 120, 50, points)

    diag = analyze_ascii_artifact(out)
    assert "too_small" in diag["flags"]
    assert "top_left_collapse" in diag["flags"]
    assert "bbox_ratio" in diag["summary"]


def test_analyze_ascii_artifact_detects_readable_text(tmp_path: Path):
    out = tmp_path / "img_0002_iter_0.txt"
    points = [(i + 1, 1, ch) for i, ch in enumerate("ThisIsReadableText")]
    _write_ascii_artifact(out, 80, 20, points)
    diag = analyze_ascii_artifact(out)
    assert "readable_text" in diag["flags"]


def test_build_render_prompt_prioritizes_run_vision():
    prompt = build_render_prompt(
        "A square temple in mist.",
        "Use layered depth.",
    )
    assert "Run vision (fixed for this run)" in prompt
    assert "Iteration image prompt" in prompt
    assert "staying faithful to the fixed run vision" in prompt
