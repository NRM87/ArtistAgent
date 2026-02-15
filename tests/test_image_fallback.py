import argparse
from pathlib import Path

from artist_agent.backends import AsciiImageBackend
from artist_agent.constants import HostedCallError
from artist_agent.runtime import build_image_backend
from artist_agent.state import move_to_gallery


def test_ascii_image_fallback_without_api_key(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    args = argparse.Namespace(
        image_backend="gemini",
        run_policy="strict",
        image_model="gemini-2.0-flash-exp-image-generation",
        image_api_key="",
        image_size="1024x1024",
        image_fallback="ascii",
    )
    backend = build_image_backend(args, tmp_path)
    assert isinstance(backend, AsciiImageBackend)


def test_move_to_gallery_preserves_non_png_extension(tmp_path: Path):
    temp_file = tmp_path / "img_0001_iter_0.txt"
    temp_file.write_text("ascii art", encoding="utf-8")
    out_dir = tmp_path / "gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    moved = move_to_gallery(temp_file, 1, out_dir)
    assert moved.suffix == ".txt"
    assert moved.exists()


def test_ascii_backend_prefers_llm_rendering(tmp_path: Path):
    class DummyLLM:
        def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
            return "LLM_LINE_1\nLLM_LINE_2"

    backend = AsciiImageBackend(tmp_path, llm_backend=DummyLLM(), ascii_size="40x20")
    out = Path(backend.generate("A fox in snow", 0, 1))
    content = out.read_text(encoding="utf-8")
    assert "renderer: llm" in content
    assert "LLM_LINE_1" in content
    assert "canvas: 40x20" in content


def test_ascii_backend_enforces_exact_canvas_dimensions(tmp_path: Path):
    class TinyLLM:
        def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
            return "abc\n12345"

    backend = AsciiImageBackend(tmp_path, llm_backend=TinyLLM(), ascii_size="50x25")
    out = Path(backend.generate("test", 0, 2))
    lines = out.read_text(encoding="utf-8").splitlines()
    begin = lines.index("BEGIN_ASCII")
    end = lines.index("END_ASCII")
    canvas = lines[begin + 1 : end]
    assert len(canvas) == 25
    assert all(len(row) == 50 for row in canvas)


def test_ascii_backend_preserves_unicode_text_art(tmp_path: Path):
    class UnicodeLLM:
        def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
            return "┌───┐\n│█░│\n└───┘"

    backend = AsciiImageBackend(tmp_path, llm_backend=UnicodeLLM(), ascii_size="20x8")
    out = Path(backend.generate("unicode", 0, 3))
    text = out.read_text(encoding="utf-8")
    assert "┌───┐" in text
    assert "│█░│" in text


def test_ascii_backend_rejects_readable_note_text(tmp_path: Path):
    class ChattyLLM:
        def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
            return (
                "┌──────────────┐\n"
                "│██████████████│\n"
                "This is a note about the composition and should not remain.\n"
                "│██████████████│\n"
                "└──────────────┘"
            )

    backend = AsciiImageBackend(tmp_path, llm_backend=ChattyLLM(), ascii_size="60x20")
    try:
        backend.generate("chatty", 0, 4)
        assert False, "Expected HostedCallError for readable text in ASCII output."
    except HostedCallError:
        pass


def test_ascii_backend_retries_and_accepts_non_text_variant(tmp_path: Path):
    class FlakyLLM:
        def __init__(self):
            self.calls = 0

        def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
            self.calls += 1
            if self.calls == 1:
                return "This is a note line that should trigger retry.\nAnother text line."
            return "┌───┐\n│███│\n└───┘"

    llm = FlakyLLM()
    backend = AsciiImageBackend(tmp_path, llm_backend=llm, ascii_size="40x20")
    out = Path(backend.generate("retry", 0, 5))
    text = out.read_text(encoding="utf-8")
    assert "renderer: llm" in text
    assert llm.calls >= 2
