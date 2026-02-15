import base64
import hashlib
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import HostedCallError
from .memory import (
    artwork_tier_from_score,
    assign_importance,
    infer_guidance,
    memory_collision,
    next_text_memory_id,
    parse_vision,
    safe_text_memory,
)

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

TIER_GUIDANCE_TEXT = (
    "Use artwork tiers intentionally: masterpieces reinforce strengths, studies suggest experiments, "
    "failures indicate pitfalls to avoid repeating.\n"
)
VISION_WEIGHT_GUIDANCE = "Weighting guidance: personality 30%, obsession 35%, text memories 20%, artwork/history 15%.\n"
INTENT_WEIGHT_GUIDANCE = "Use this weighting: personality 25%, obsession 30%, text memories 25%, artwork + history 20%.\n"
REVISION_WEIGHT_GUIDANCE = "Weighting guidance for revision: obsession 35%, personality 25%, text memories 20%, artwork/history 20%.\n"


def _post_json_with_retry(url: str, payload: Dict, headers: Dict, timeout: int, attempts: int = 5) -> Dict:
    data = json.dumps(payload).encode("utf-8")
    last_exc = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            out = json.loads(body)
            if not isinstance(out, dict):
                raise ValueError("Provider response was not a JSON object.")
            return out
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after and retry_after.isdigit():
                    sleep_s = max(1.0, min(60.0, float(retry_after)))
                else:
                    sleep_s = min(30.0, 1.5 * (2 ** attempt))
                time.sleep(sleep_s)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(min(20.0, 1.2 * (2 ** attempt)))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown request failure")


def parse_json_object(text: str) -> Dict:
    raw = text.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    raise ValueError("Model response did not contain a valid JSON object.")


def _ollama_generate_text(base_url: str, model: str, prompt: str, temperature: float, timeout: int = 60) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": max(0.0, min(1.0, float(temperature))),
        },
    }
    out = _post_json_with_retry(url=url, payload=payload, headers={"Content-Type": "application/json"}, timeout=timeout, attempts=2)
    text = str(out.get("response", "")).strip()
    if not text:
        raise ValueError("Ollama response missing text.")
    return text


def _summarize_text_memories(text_memories: List[Dict], limit: int = 12) -> List[Dict]:
    out: List[Dict] = []
    for mem in text_memories[-limit:]:
        out.append(
            {
                "id": mem.get("id"),
                "importance": str(mem.get("importance", "medium")),
                "tags": list(mem.get("tags", []) or [])[:4],
                "content": str(mem.get("content", "")).strip()[:220],
            }
        )
    return out


def _summarize_artwork_memories(memories: List[Dict], limit: int = 8) -> List[Dict]:
    out: List[Dict] = []
    for mem in memories[-limit:]:
        score = int(mem.get("final_score", 0))
        tier = str(mem.get("tier", artwork_tier_from_score(score))).strip().lower()
        out.append(
            {
                "id": mem.get("id"),
                "vision": str(mem.get("vision", "")).strip()[:180],
                "score": score,
                "tier": tier,
                "note": str(mem.get("self_note", "")).strip()[:180],
            }
        )
    return out


def _history_summary(history: List[Dict], limit: int = 12) -> Dict:
    recent = history[-limit:]
    scores = [int(h.get("score", 0)) for h in recent]
    worthy = sum(1 for h in recent if bool(h.get("worthy", False)))
    avg = round(sum(scores) / len(scores), 2) if scores else 0.0
    recent_visions = [str(h.get("vision", "")).strip()[:140] for h in recent[-6:]]
    return {"avg_score": avg, "worthy_count": worthy, "recent_visions": recent_visions}


def build_soul_packet(soul: Dict) -> Dict:
    text_memories = list(soul.get("text_memories", []) or [])
    memories = list(soul.get("memories", []) or [])
    history = list(soul.get("cycle_history", []) or [])
    tier_counts = {"masterpiece": 0, "study": 0, "failure": 0}
    for mem in memories:
        score = int(mem.get("final_score", 0))
        tier = str(mem.get("tier", artwork_tier_from_score(score))).strip().lower()
        if tier in tier_counts:
            tier_counts[tier] += 1
    return {
        "name": str(soul.get("name", "")).strip(),
        "personality_traits": [str(t).strip() for t in list(soul.get("personality_traits", []) or [])[:10] if str(t).strip()],
        "current_obsession": str(soul.get("current_obsession", "")).strip(),
        "text_memories": _summarize_text_memories(text_memories, 14),
        "artwork_memories": _summarize_artwork_memories(memories, 10),
        "artwork_tier_counts": tier_counts,
        "history": _history_summary(history, 14),
    }


def _trace_prompt(enabled: bool, stage: str, provider: str, model: str, system_prompt: str, user_prompt: str) -> None:
    if not enabled:
        return
    sys_compact = " ".join(str(system_prompt).split())
    usr_compact = " ".join(str(user_prompt).split())
    if len(sys_compact) > 280:
        sys_compact = sys_compact[:280] + "...(truncated)"
    if len(usr_compact) > 900:
        usr_compact = usr_compact[:900] + "...(truncated)"
    print(f"[trace-prompts] {stage} ({provider}:{model})")
    print(f"  system: {sys_compact}")
    print(f"  user: {usr_compact}")


def _print_vision_context_summary(
    soul: Dict,
    preferences: List[str],
    principles: List[str],
    instructions: List[str],
    memories: List[Dict],
) -> None:
    history = list(soul.get("cycle_history", []) or [])
    h = _history_summary(history, 10)
    traits = [str(t).strip() for t in list(soul.get("personality_traits", []) or []) if str(t).strip()]
    obsession = str(soul.get("current_obsession", "")).strip()
    text_count = len(list(soul.get("text_memories", []) or []))
    other_count = max(0, text_count - len(preferences) - len(principles) - len(instructions))

    print("\nConceiving new vision...\n")
    print("  Soul context summary:")
    print(f"  - Personality traits: {len(traits)}")
    if traits:
        print(f"    {', '.join(traits[-4:])}")
    print(f"  - Current obsession: {obsession if obsession else '(none)'}")
    print(
        f"  - Text memories: {text_count} "
        f"(preferences={len(preferences)}, principles={len(principles)}, meta={len(instructions)}, other={other_count})"
    )
    print(f"  - Artwork memories considered: {len(memories)}")
    print(f"  - Recent cycle avg score: {h.get('avg_score', 0.0)} over {len(history[-10:])} runs")
    print(f"  - Novelty pressure: avoid repeating recent motifs ({len(h.get('recent_visions', []))} recent)")
    print("  - Note: this is a summary; full prompting uses personality, obsession, memories, and history.")

    anchors: List[str] = []
    anchors.extend([f"Preference: {p}" for p in preferences[-2:]])
    anchors.extend([f"Principle: {p}" for p in principles[-2:]])
    anchors.extend([f"Instruction: {i}" for i in instructions[-1:]])
    if anchors:
        print("\n  High-signal anchors:")
        for a in anchors[-4:]:
            print(f"  + {a}")


def save_rgb_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start : start + stride])

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = b"".join([b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", ihdr), chunk(b"IDAT", zlib.compress(bytes(raw), 9)), chunk(b"IEND", b"")])
    path.write_bytes(data)


def _set_px(buf: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        buf[idx], buf[idx + 1], buf[idx + 2] = color


def _draw_line(buf: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_px(buf, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def generate_png_without_pillow(prompt: str, iteration: int, creation_id: int, output_dir: Path) -> str:
    import math

    width, height = 512, 512
    buf = bytearray(width * height * 3)
    for y in range(height):
        base = int(8 + (y / height) * 30)
        for x in range(width):
            idx = (y * width + x) * 3
            buf[idx], buf[idx + 1], buf[idx + 2] = base, base, min(255, base + 6)

    digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    shape_color = (40 + int(digest[0:2], 16) % 180, 40 + int(digest[2:4], 16) % 180, 40 + int(digest[4:6], 16) % 180)
    shape = ["sphere", "cube", "spiral", "diamond"][int(digest[6:8], 16) % 4]
    cx, cy = width // 2, height // 2
    if int(digest[8:10], 16) % 3 == 0:
        cx, cy = int(width * 0.62), int(height * 0.42)

    complexity = max(1, min(5, iteration + 1))
    size = 80 + complexity * 22

    if shape in ("sphere", "orb"):
        r2 = size * size
        for y in range(cy - size, cy + size + 1):
            for x in range(cx - size, cx + size + 1):
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                    _set_px(buf, width, height, x, y, shape_color)
    elif shape in ("cube", "lattice"):
        for y in range(cy - size, cy + size + 1):
            for x in range(cx - size, cx + size + 1):
                _set_px(buf, width, height, x, y, shape_color)
        if complexity >= 3:
            step = max(8, size // 5)
            for x in range(cx - size, cx + size + 1, step):
                _draw_line(buf, width, height, x, cy - size, x, cy + size, (20, 20, 20))
            for y in range(cy - size, cy + size + 1, step):
                _draw_line(buf, width, height, cx - size, y, cx + size, y, (20, 20, 20))
    elif shape == "spiral":
        last = None
        for t in range((5 + complexity) * 65):
            angle = t / 12.0
            radius = (t / ((5 + complexity) * 65)) * size
            x = cx + int(radius * math.cos(angle))
            y = cy + int(radius * math.sin(angle))
            if last is not None:
                _draw_line(buf, width, height, last[0], last[1], x, y, shape_color)
            last = (x, y)
    elif shape == "diamond":
        for y in range(cy - size, cy + size + 1):
            span = size - abs(y - cy)
            for x in range(cx - span, cx + span + 1):
                _set_px(buf, width, height, x, y, shape_color)
    else:
        for y in range(cy - size, cy + size + 1):
            for x in range(cx - size, cx + size + 1):
                _set_px(buf, width, height, x, y, shape_color)

    path = output_dir / f"img_{creation_id:04d}_iter_{iteration}.png"
    save_rgb_png(path, width, height, buf)
    return str(path)


class MockImageGen:
    @staticmethod
    def generate(prompt: str, iteration: int, creation_id: int, output_dir: Path) -> str:
        if Image is None or ImageDraw is None:
            return generate_png_without_pillow(prompt, iteration, creation_id, output_dir)
        width, height = 512, 512
        image = Image.new("RGB", (width, height), (8, 8, 12))
        draw = ImageDraw.Draw(image)
        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        color = (40 + int(digest[0:2], 16) % 180, 40 + int(digest[2:4], 16) % 180, 40 + int(digest[4:6], 16) % 180)
        shape = ["sphere", "cube", "spiral", "diamond"][int(digest[6:8], 16) % 4]
        complexity = max(1, min(5, iteration + 1))
        cx, cy = width // 2, height // 2
        if int(digest[8:10], 16) % 3 == 0:
            cx, cy = int(width * 0.62), int(height * 0.42)
        size = 80 + complexity * 22

        if complexity >= 2:
            for y in range(height):
                shade = int(8 + (y / height) * 35)
                draw.line([(0, y), (width, y)], fill=(shade, shade, shade + 5))

        if shape in ("sphere", "orb"):
            draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=color)
        elif shape in ("cube", "lattice"):
            draw.rectangle((cx - size, cy - size, cx + size, cy + size), fill=color)
            if complexity >= 3:
                step = max(8, size // 5)
                for x in range(cx - size, cx + size, step):
                    draw.line((x, cy - size, x, cy + size), fill=(20, 20, 20))
                for y in range(cy - size, cy + size, step):
                    draw.line((cx - size, y, cx + size, y), fill=(20, 20, 20))
        elif shape == "spiral":
            import math

            points = []
            turns = 5 + complexity
            for t in range(turns * 40):
                angle = t / 18.0
                radius = (t / (turns * 40)) * size
                x = cx + int(radius * math.cos(angle))
                y = cy + int(radius * math.sin(angle))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3 + complexity // 2)
        elif shape == "diamond":
            draw.polygon([(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)], fill=color)
        else:
            draw.rectangle((cx - size, cy - size, cx + size, cy + size), fill=color)

        path = output_dir / f"img_{creation_id:04d}_iter_{iteration}.png"
        image.save(path)
        return str(path)


class ImageBackend:
    def generate(self, prompt: str, iteration: int, creation_id: int) -> str:
        raise NotImplementedError


class AsciiImageBackend(ImageBackend):
    def __init__(self, temp_dir: Path, llm_backend: Optional[object] = None, ascii_size: str = "160x60"):
        self.temp_dir = temp_dir
        self.llm_backend = llm_backend
        self.ascii_size = ascii_size

    @staticmethod
    def _sanitize_ascii(raw: str) -> str:
        text = str(raw).replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
        lines = []
        for ln in text.split("\n"):
            if ln.strip().startswith("```"):
                continue
            cleaned = "".join(ch if ch >= " " else " " for ch in ln)
            lines.append(cleaned)
        # Keep sparse outputs; exact canvas is enforced separately.
        return "\n".join(lines).strip("\n")

    @staticmethod
    def _line_has_readable_text(line: str) -> bool:
        # Detect likely explanatory/narrative text lines, not geometric glyph lines.
        token_count = len(re.findall(r"[A-Za-z]{3,}", line))
        alpha_count = sum(1 for ch in line if ch.isalpha())
        return token_count >= 2 or alpha_count >= 12

    @staticmethod
    def _parse_ascii_size(raw: str) -> Tuple[int, int]:
        value = str(raw).strip().lower()
        if "x" not in value:
            return 160, 60
        left, right = value.split("x", 1)
        try:
            w = int(left.strip())
            h = int(right.strip())
        except ValueError:
            return 160, 60
        # Clamp for practicality and terminal readability.
        w = max(40, min(300, w))
        h = max(20, min(120, h))
        return w, h

    @staticmethod
    def _enforce_canvas(text: str, width: int, height: int) -> str:
        # Enforce exact dimensions so ASCII outputs are stable and testable.
        # This mirrors fixed-size guarantees used for pixel images.
        src = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        out_lines = []
        for ln in src:
            cut = ln[:width]
            if len(cut) < width:
                cut = cut + (" " * (width - len(cut)))
            out_lines.append(cut)
            if len(out_lines) >= height:
                break
        while len(out_lines) < height:
            out_lines.append(" " * width)
        return "\n".join(out_lines)

    @staticmethod
    def _ink_ratio(canvas: str) -> float:
        if not canvas:
            return 0.0
        total = len(canvas.replace("\n", ""))
        if total <= 0:
            return 0.0
        non_space = sum(1 for ch in canvas if ch not in (" ", "\n"))
        return non_space / total

    @classmethod
    def _contains_readable_text(cls, canvas: str) -> bool:
        return any(cls._line_has_readable_text(ln) for ln in canvas.split("\n"))

    def generate(self, prompt: str, iteration: int, creation_id: int) -> str:
        width, height = self._parse_ascii_size(self.ascii_size)
        if self.llm_backend is None or not hasattr(self.llm_backend, "generate_ascii_art"):
            raise HostedCallError("ASCII image backend requires an LLM backend with generate_ascii_art().")

        lines = []
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                # Retry with stronger anti-collapse hints for small local models.
                variant_prompt = prompt
                if attempt == 1:
                    variant_prompt += ". Avoid tiny icon output; use broad strokes over most of the canvas. Do not include readable text or notes."
                elif attempt == 2:
                    variant_prompt += ". Avoid repeating recent motifs; commit to a distinct composition. Absolutely no readable words, captions, labels, or notes."
                llm_ascii = self.llm_backend.generate_ascii_art(variant_prompt, iteration, creation_id, width, height)
                canvas = self._enforce_canvas(self._sanitize_ascii(llm_ascii), width, height)
                if self._contains_readable_text(canvas):
                    raise ValueError("ASCII output contained readable text.")
                if self._ink_ratio(canvas) < 0.006:
                    raise ValueError("ASCII output too sparse.")
                lines = [
                    f"ASCII ART - creation {creation_id} iter {iteration}",
                    f"prompt: {prompt}",
                    "renderer: llm",
                    f"canvas: {width}x{height}",
                    "",
                    "BEGIN_ASCII",
                    canvas,
                    "END_ASCII",
                    "",
                ]
                break
            except Exception as exc:
                last_exc = exc
        if not lines:
            raise HostedCallError(f"LLM ASCII rendering failed after retries ({last_exc})")

        path = self.temp_dir / f"img_{creation_id:04d}_iter_{iteration}.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)


class MockImageBackend(ImageBackend):
    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir

    def generate(self, prompt: str, iteration: int, creation_id: int) -> str:
        return MockImageGen.generate(prompt, iteration, creation_id, self.temp_dir)


class HostedImageBackend(ImageBackend):
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        temp_dir: Path,
        size: str = "1024x1024",
        allow_fallback: bool = False,
        fallback_mode: str = "defer",
        llm_backend: Optional[object] = None,
        ascii_size: str = "160x60",
        trace_prompts: bool = False,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temp_dir = temp_dir
        self.size = size
        self.allow_fallback = allow_fallback
        self.fallback_mode = fallback_mode
        self.trace_prompts = trace_prompts
        self._fallback_backend = AsciiImageBackend(temp_dir, llm_backend=llm_backend, ascii_size=ascii_size) if fallback_mode == "ascii" else MockImageBackend(temp_dir)

    def _http_json(self, url: str, payload: Dict, headers: Dict) -> Dict:
        return _post_json_with_retry(url=url, payload=payload, headers=headers, timeout=70)

    def _write_image_bytes(self, creation_id: int, iteration: int, data: bytes) -> str:
        path = self.temp_dir / f"img_{creation_id:04d}_iter_{iteration}.png"
        path.write_bytes(data)
        return str(path)

    def generate(self, prompt: str, iteration: int, creation_id: int) -> str:
        try:
            if self.provider == "openai":
                _trace_prompt(
                    self.trace_prompts,
                    "image.generate",
                    self.provider,
                    self.model,
                    "Generate an image from prompt.",
                    f"prompt:{prompt}\nsize:{self.size}",
                )
                out = self._http_json(
                    "https://api.openai.com/v1/images/generations",
                    {"model": self.model, "prompt": prompt, "size": self.size, "response_format": "b64_json"},
                    {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                b64 = out.get("data", [{}])[0].get("b64_json", "")
                if not b64:
                    raise ValueError("OpenAI response missing image data.")
                return self._write_image_bytes(creation_id, iteration, base64.b64decode(b64))

            if self.provider == "gemini":
                _trace_prompt(
                    self.trace_prompts,
                    "image.generate",
                    self.provider,
                    self.model,
                    "Create an image from prompt.",
                    f"prompt:{prompt}\nsize:{self.size}",
                )
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
                out = self._http_json(url, {"contents": [{"parts": [{"text": f"Create an image: {prompt}"}]}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}, {"Content-Type": "application/json"})
                for cand in out.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        b64 = part.get("inlineData", {}).get("data")
                        if b64:
                            return self._write_image_bytes(creation_id, iteration, base64.b64decode(b64))
                raise ValueError("Gemini did not return inline image data.")

            raise ValueError(f"Unsupported image provider: {self.provider}")
        except Exception as exc:
            if not self.allow_fallback:
                raise HostedCallError(f"Hosted image generation failed: {exc}") from exc
            print(f"Warning: hosted image generation failed, falling back to {self.fallback_mode} ({exc})")
            return self._fallback_backend.generate(prompt, iteration, creation_id)


class LLMBackend:
    def critique(self, image_path: str, vision: str, iteration: int, critique_frame: str = "") -> Dict:
        raise NotImplementedError

    def judge_worthiness(self, image_path: str, score: int, vision: str, critique_frame: str = "") -> bool:
        raise NotImplementedError

    def generate_text_memory(self, soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        raise NotImplementedError

    def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
        raise NotImplementedError

    def generate_identity(self, current_name: str) -> Dict:
        raise NotImplementedError

    def generate_vision_fallback(self, soul: Dict) -> str:
        raise NotImplementedError

    def generate_run_intent(self, soul_data: Dict) -> Dict:
        raise NotImplementedError

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        raise NotImplementedError


class MockLLM:
    @staticmethod
    def critique(image_path: str, vision: str, iteration: int) -> Dict:
        low_high = {0: (4, 5), 1: (5, 6), 2: (7, 8)}.get(iteration, (8, 9))
        low, high = low_high
        digest = hashlib.md5(f"{image_path}|{vision}|{iteration}".encode("utf-8")).hexdigest()
        score = low + (int(digest[:2], 16) % (high - low + 1))
        feedback = (
            "Form is unclear and emotional intent is weak." if score <= 5 else
            "Composition improves, but depth and contrast need refinement." if score <= 6 else
            "Strong structure. Push atmosphere and precision further." if score <= 7 else
            "Now the concept resonates with confidence and clarity."
        )
        return {"score": int(score), "feedback": feedback}

    @staticmethod
    def judge_worthiness(image_path: str, score: int, vision: str) -> bool:
        return score >= 7

    @staticmethod
    def generate_text_memory(soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        parsed = parse_vision(creation_result.get("vision", ""))
        if trigger_reason == "self_correction":
            content = "IGNORE memory - recent outcomes prove it leads to derivative work."
            tags = ["meta", "instruction", "correction"]
        elif trigger_reason == "breakthrough":
            content = f"I discovered that {parsed.color} {parsed.subject} forms can achieve rare clarity."
            tags = ["learning", "pattern", "success_pattern"]
        elif trigger_reason == "repeated_failure":
            content = "Three weak outcomes confirm this strategy is hollow. Prioritize depth and contrast."
            tags = ["learning", "correction", "pattern"]
        else:
            content = f"Pattern confirmed: {parsed.subject} compositions align with my temperament."
            tags = ["principle", "pattern", "composition"]
        return {"content": content, "importance": assign_importance(content), "tags": tags}


class MockLLMBackend(LLMBackend):
    def critique(self, image_path: str, vision: str, iteration: int, critique_frame: str = "") -> Dict:
        return MockLLM.critique(image_path, vision, iteration)

    def judge_worthiness(self, image_path: str, score: int, vision: str, critique_frame: str = "") -> bool:
        return MockLLM.judge_worthiness(image_path, score, vision)

    def generate_text_memory(self, soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        return safe_text_memory(MockLLM.generate_text_memory(soul_data, creation_result, trigger_reason), soul_data)

    def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
        p = prompt.lower()
        if "fox" in p:
            return " ░▄░\n▐█▌\n ▀░"
        if "mountain" in p:
            return "   ▲\n  ▲▲\n ▲██▲\n██████"
        return "┌──────┐\n│ text │\n└──────┘"

    def generate_identity(self, current_name: str) -> Dict:
        raise HostedCallError("Identity generation requires an LLM backend, but current backend is mock.")

    def generate_vision_fallback(self, soul: Dict) -> str:
        raise HostedCallError("Vision fallback requires an LLM backend, but current backend is mock.")

    def generate_run_intent(self, soul_data: Dict) -> Dict:
        raise HostedCallError("Run intent requires an LLM backend, but current backend is mock.")

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        raise HostedCallError("State revision requires an LLM backend, but current backend is mock.")


class OllamaLLMBackend(LLMBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", temperature: float = 0.2, trace_prompts: bool = False):
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.trace_prompts = trace_prompts

    def _chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[str] = None,
        max_chars: int = 2200,
        timeout: int = 70,
        temperature_override: Optional[float] = None,
    ) -> str:
        artifact_note = ""
        if image_path:
            p = Path(image_path)
            if p.suffix.lower() == ".txt":
                snippet = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
                artifact_note = f"\n\nASCII_ARTIFACT:\n{snippet}"
            elif p.exists():
                artifact_note = f"\n\nIMAGE_PATH: {str(p)}"
        prompt = f"{system_prompt}\n\n{user_prompt}{artifact_note}"
        temp = self.temperature if temperature_override is None else temperature_override
        _trace_prompt(self.trace_prompts, "ollama.chat", self.provider, self.model, system_prompt, user_prompt)
        return _ollama_generate_text(self.base_url, self.model, prompt, temp, timeout=timeout)

    def _chat_json(self, system_prompt: str, user_prompt: str, image_path: Optional[str] = None) -> Dict:
        raw = self._chat_text(system_prompt, user_prompt, image_path=image_path)
        return parse_json_object(raw)

    def critique(self, image_path: str, vision: str, iteration: int, critique_frame: str = "") -> Dict:
        try:
            out = self._chat_json(
                "Return strict JSON only: {\"score\": 1-10 integer, \"feedback\": string}.",
                f"vision:{vision}\niteration:{iteration}\ncritique_frame:{critique_frame}",
                image_path=image_path,
            )
            return {"score": max(1, min(10, int(out.get("score", 0)))), "feedback": str(out.get("feedback", "Needs refinement."))}
        except Exception as exc:
            raise HostedCallError(f"Ollama critique failed: {exc}") from exc

    def judge_worthiness(self, image_path: str, score: int, vision: str, critique_frame: str = "") -> bool:
        try:
            out = self._chat_json(
                "Return strict JSON only: {\"worthy\": true|false}.",
                f"vision:{vision}\nscore:{score}\ncritique_frame:{critique_frame}",
                image_path=image_path,
            )
            return bool(out.get("worthy", score >= 7))
        except Exception as exc:
            raise HostedCallError(f"Ollama judgment failed: {exc}") from exc

    def generate_text_memory(self, soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        try:
            out = self._chat_json(
                "Return strict JSON only: {\"content\": string, \"importance\": \"critical|high|medium|low\", \"tags\": [string]}.",
                f"trigger:{trigger_reason}\nresult:{creation_result}",
            )
            return safe_text_memory(out, soul_data)
        except Exception as exc:
            raise HostedCallError(f"Ollama text-memory generation failed: {exc}") from exc

    def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
        try:
            w = width if width > 0 else 160
            h = height if height > 0 else 60
            return self._chat_text(
                "Return only text-art (no markdown fences, no explanation). "
                "You may use any visible characters available to you, including Unicode line/box/block glyphs. "
                f"Target canvas {w}x{h}. Aim for {h} lines and around {w} columns per line; exact width is not required here. "
                "Avoid writing readable words or labels. Blank background is allowed. "
                "Use multi-scale structure (foreground, midground, background) and keep composition visually substantial.",
                f"Create text-art for this prompt:\n{prompt}\niteration:{iteration}\ncreation_id:{creation_id}\ncanvas:{w}x{h}\n"
                "Avoid repeating your previous motifs in this run.",
                timeout=180,
                temperature_override=max(0.55, min(1.0, self.temperature + 0.25)),
            )
        except Exception as exc:
            raise HostedCallError(f"Ollama ASCII generation failed: {exc}") from exc

    def generate_identity(self, current_name: str) -> Dict:
        try:
            out = self._chat_json(
                "Return strict JSON only: {\"name\": string, \"personality_traits\": [3-7 strings], \"current_obsession\": string}.",
                f"Create a distinctive artistic identity. Current name hint: {current_name}",
            )
            name = str(out.get("name", "")).strip() or (current_name.strip() if current_name.strip() else "Unnamed Artist")
            traits = out.get("personality_traits", [])
            if not isinstance(traits, list):
                traits = []
            clean_traits = [str(t).strip() for t in traits if str(t).strip()][:7]
            if len(clean_traits) < 3:
                raise HostedCallError("LLM returned insufficient personality traits.")
            obsession = str(out.get("current_obsession", "")).strip()
            if not obsession:
                raise HostedCallError("LLM returned empty obsession.")
            return {"name": name, "personality_traits": clean_traits, "current_obsession": obsession}
        except Exception as exc:
            raise HostedCallError(f"Ollama identity generation failed: {exc}") from exc

    def generate_vision_fallback(self, soul: Dict) -> str:
        try:
            text_memories = soul.get("text_memories", []) or []
            prefs, principles, instructions, _ = infer_guidance(text_memories)
            memories = soul.get("memories", []) or []
            recent = [m.get("vision", "") for m in memories[-8:]]
            packet = build_soul_packet(soul)
            text = self._chat_text(
                "Return exactly one concise art vision sentence. No JSON. No bullet points.",
                (
                    VISION_WEIGHT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + f"soul_packet:{packet}\npreferences:{prefs[-8:]}\nprinciples:{principles[-8:]}\ninstructions:{instructions[-8:]}\nrecent:{recent}\n"
                    + "Prioritize novelty versus recent visions."
                ),
            )
            vision = text.strip().strip('"').splitlines()[0].strip()
            if not vision:
                raise HostedCallError("LLM vision fallback returned empty vision.")
            return vision
        except Exception as exc:
            raise HostedCallError(f"Ollama vision fallback failed: {exc}") from exc

    def generate_run_intent(self, soul_data: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            out = self._chat_json(
                "Return strict JSON only: "
                "{\"vision_directive\": string, \"critique_directive\": string, \"revision_directive\": string}.",
                (
                    INTENT_WEIGHT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + "vision_directive should push distinct composition/motif from recent works.\n"
                    + "critique_directive should evaluate alignment with soul, not generic quality alone.\n"
                    + "revision_directive should explain how identity should evolve from outcome.\n"
                    + f"soul_packet:{packet}"
                ),
            )
            return {
                "vision_directive": str(out.get("vision_directive", "")).strip(),
                "critique_directive": str(out.get("critique_directive", "")).strip(),
                "revision_directive": str(out.get("revision_directive", "")).strip(),
            }
        except Exception as exc:
            raise HostedCallError(f"Ollama run-intent generation failed: {exc}") from exc

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            out = self._chat_json(
                "Return strict JSON only with keys: "
                "{\"obsession\": string, "
                "\"personality_mode\": \"keep|append|replace\", "
                "\"personality_traits\": [string], "
                "\"text_memory_action\": \"none|add|edit_last|delete_last\", "
                "\"text_memory\": {\"content\": string, \"importance\": \"critical|high|medium|low\", \"tags\": [string]}, "
                "\"artwork_memory_action\": \"none|annotate_last|delete_last\", "
                "\"artwork_note\": string}.",
                (
                    REVISION_WEIGHT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + "If creation_result.score <= 7 or creation_result.worthy is false, identity drift is required: "
                    + "change obsession and/or change personality traits materially (not just restating current values).\n"
                    + f"soul_packet:{packet}\ncreation_result:{creation_result}"
                ),
            )
            return out if isinstance(out, dict) else {}
        except Exception as exc:
            raise HostedCallError(f"Ollama state revision failed: {exc}") from exc


class HostedLLMBackend(LLMBackend):
    def __init__(self, provider: str, model: str, api_key: str, temperature: float = 0.2, allow_fallback: bool = False, trace_prompts: bool = False):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.allow_fallback = allow_fallback
        self.trace_prompts = trace_prompts

    def _http_json(self, url: str, payload: Dict, headers: Dict) -> Dict:
        return _post_json_with_retry(url=url, payload=payload, headers=headers, timeout=45)

    def _chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 400, image_path: Optional[str] = None) -> str:
        _trace_prompt(self.trace_prompts, "hosted.chat", self.provider, self.model, system_prompt, user_prompt)
        image_b64 = None
        artifact_note = ""
        if image_path:
            p = Path(image_path)
            if p.suffix.lower() == ".txt":
                # ASCII fallback artifacts are text-first; include them as prompt context.
                snippet = p.read_text(encoding="utf-8", errors="replace")[:2000]
                artifact_note = f"\n\nASCII_ARTIFACT:\n{snippet}"
            else:
                image_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")

        if self.provider == "openai":
            user_content = [{"type": "text", "text": user_prompt + artifact_note}]
            if image_b64:
                user_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"})
            out = self._http_json("https://api.openai.com/v1/responses", {"model": self.model, "input": [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, {"role": "user", "content": user_content}], "temperature": self.temperature, "max_output_tokens": max_tokens}, {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            if isinstance(out.get("output_text"), str) and out.get("output_text").strip():
                return out["output_text"]
            for item in out.get("output", []):
                for c in item.get("content", []):
                    if isinstance(c.get("text"), str) and c["text"].strip():
                        return c["text"]

        if self.provider == "anthropic":
            content = [{"type": "text", "text": user_prompt + artifact_note}]
            if image_b64:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}})
            out = self._http_json("https://api.anthropic.com/v1/messages", {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature, "system": system_prompt, "messages": [{"role": "user", "content": content}]}, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"})
            for part in out.get("content", []):
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"]

        if self.provider == "gemini":
            parts = [{"text": f"{system_prompt}\n\n{user_prompt}{artifact_note}"}]
            if image_b64:
                parts.append({"inline_data": {"mime_type": "image/png", "data": image_b64}})
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
            out = self._http_json(url, {"contents": [{"parts": parts}], "generationConfig": {"temperature": self.temperature, "maxOutputTokens": max_tokens}}, {"Content-Type": "application/json"})
            for cand in out.get("candidates", []):
                for p in cand.get("content", {}).get("parts", []):
                    if isinstance(p.get("text"), str) and p["text"].strip():
                        return p["text"]

        raise ValueError(f"No text from provider: {self.provider}")

    def _chat_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 400, image_path: Optional[str] = None) -> Dict:
        return parse_json_object(self._chat_text(system_prompt, user_prompt, max_tokens, image_path))

    def critique(self, image_path: str, vision: str, iteration: int, critique_frame: str = "") -> Dict:
        try:
            out = self._chat_json("Return strict JSON: score (1-10), feedback.", f"vision:{vision}\niteration:{iteration}\ncritique_frame:{critique_frame}", 180, image_path)
            return {"score": max(1, min(10, int(out.get("score", 0)))), "feedback": str(out.get("feedback", "Needs refinement."))}
        except Exception as exc:
            raise HostedCallError(f"Hosted critique failed: {exc}") from exc

    def judge_worthiness(self, image_path: str, score: int, vision: str, critique_frame: str = "") -> bool:
        try:
            out = self._chat_json("Return strict JSON: worthy(boolean).", f"vision:{vision}\nscore:{score}\ncritique_frame:{critique_frame}", 120, image_path)
            return bool(out.get("worthy", score >= 7))
        except Exception as exc:
            raise HostedCallError(f"Hosted judgment failed: {exc}") from exc

    def generate_text_memory(self, soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        try:
            out = self._chat_json("Return strict JSON: content, importance, tags.", f"trigger:{trigger_reason}\nresult:{creation_result}", 220)
            return safe_text_memory(out, soul_data)
        except Exception as exc:
            raise HostedCallError(f"Hosted memory generation failed: {exc}") from exc

    def generate_ascii_art(self, prompt: str, iteration: int, creation_id: int, width: int = 0, height: int = 0) -> str:
        try:
            w = width if width > 0 else 160
            h = height if height > 0 else 60
            return self._chat_text(
                "Return only text-art (no markdown, no explanation). "
                "You may use any visible characters available to you, including Unicode line/box/block glyphs. "
                f"Target canvas {w}x{h}. Aim for {h} lines and around {w} columns per line; exact width is not required here. "
                "Avoid readable words or labels. Blank background is allowed. "
                "Make the drawing composition occupy most of the canvas with meaningful structure.",
                f"Prompt:{prompt}\niteration:{iteration}\ncreation_id:{creation_id}\ncanvas:{w}x{h}\n"
                "Avoid reusing recently repeated motifs.",
                550,
            )
        except Exception as exc:
            raise HostedCallError(f"Hosted ASCII generation failed: {exc}") from exc

    def generate_identity(self, current_name: str) -> Dict:
        try:
            out = self._chat_json(
                "Return strict JSON: name, personality_traits(array of 3-7), current_obsession.",
                f"Create a distinctive artistic identity. Current name hint: {current_name}",
                260,
            )
            name = str(out.get("name", "")).strip() or (current_name.strip() if current_name.strip() else "Unnamed Artist")
            traits = out.get("personality_traits", [])
            if not isinstance(traits, list):
                traits = []
            clean_traits = [str(t).strip() for t in traits if str(t).strip()][:7]
            if len(clean_traits) < 3:
                raise ValueError("insufficient traits")
            obsession = str(out.get("current_obsession", "")).strip()
            if not obsession:
                raise ValueError("empty obsession")
            return {"name": name, "personality_traits": clean_traits, "current_obsession": obsession}
        except Exception as exc:
            raise HostedCallError(f"Hosted identity generation failed: {exc}") from exc

    def generate_vision_fallback(self, soul: Dict) -> str:
        try:
            text_memories = soul.get("text_memories", []) or []
            prefs, principles, instructions, _ = infer_guidance(text_memories)
            memories = soul.get("memories", []) or []
            recent = [m.get("vision", "") for m in memories[-8:]]
            packet = build_soul_packet(soul)
            text = self._chat_text(
                "Return exactly one concise art vision sentence. No JSON. No bullet points.",
                (
                    VISION_WEIGHT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + f"soul_packet:{packet}\npreferences:{prefs[-8:]}\nprinciples:{principles[-8:]}\ninstructions:{instructions[-8:]}\nrecent:{recent}\n"
                    + "Prioritize novelty against recent visions."
                ),
                90,
            )
            vision = text.strip().strip('"').splitlines()[0].strip()
            if not vision:
                raise ValueError("empty vision")
            return vision
        except Exception as exc:
            raise HostedCallError(f"Hosted vision fallback failed: {exc}") from exc

    def generate_run_intent(self, soul_data: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            out = self._chat_json(
                "Return strict JSON: vision_directive, critique_directive, revision_directive.",
                (
                    INTENT_WEIGHT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + "vision_directive should push distinct composition/motif from recent works.\n"
                    + "critique_directive should evaluate alignment with soul, not generic quality alone.\n"
                    + "revision_directive should explain how identity should evolve from outcome.\n"
                    + f"soul_packet:{packet}"
                ),
                260,
            )
            return {
                "vision_directive": str(out.get("vision_directive", "")).strip(),
                "critique_directive": str(out.get("critique_directive", "")).strip(),
                "revision_directive": str(out.get("revision_directive", "")).strip(),
            }
        except Exception as exc:
            raise HostedCallError(f"Hosted run intent generation failed: {exc}") from exc

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            out = self._chat_json(
                "Return strict JSON with keys: obsession, personality_mode(keep|append|replace), personality_traits(array), "
                "text_memory_action(none|add|edit_last|delete_last), text_memory(object), artwork_memory_action(none|annotate_last|delete_last), artwork_note.",
                (
                    REVISION_WEIGHT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + "If creation_result.score <= 7 or creation_result.worthy is false, identity drift is required: "
                    + "change obsession and/or change personality traits materially (not just restating current values).\n"
                    + f"soul_packet:{packet}\ncreation_result:{creation_result}"
                ),
                420,
            )
            return out if isinstance(out, dict) else {}
        except Exception as exc:
            raise HostedCallError(f"Hosted state revision failed: {exc}") from exc


class VisionBackend:
    def generate_vision(self, soul: Dict, ignored_ids: set) -> str:
        raise NotImplementedError


class LocalVisionBackend(VisionBackend):
    def generate_vision(self, soul: Dict, ignored_ids: set) -> str:
        raise HostedCallError("Deterministic local vision is disabled. Use an LLM-backed vision path.")


class OllamaVisionBackend(VisionBackend):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", temperature: float = 0.4, trace_prompts: bool = False):
        self.provider = "ollama"
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.trace_prompts = trace_prompts

    def generate_vision(self, soul: Dict, ignored_ids: set) -> str:
        text_memories = soul.get("text_memories", [])
        preferences, principles, instructions, _ = infer_guidance(text_memories)
        packet = build_soul_packet(soul)
        memories = [m for m in soul.get("memories", []) if m.get("id") not in ignored_ids]
        _print_vision_context_summary(soul, preferences, principles, instructions, memories)
        recent_visions = [m.get("vision", "") for m in memories[-8:]]
        prompt = (
            "Generate one concise art vision sentence. "
            "Do not output JSON. Keep it under 18 words.\n\n"
            "Weighting guidance: obsession 35%, personality 30%, text memories 20%, artwork/history 15%.\n"
            + TIER_GUIDANCE_TEXT
            + f"soul_packet:{packet}\n"
            + f"preferences:{preferences[-8:]}\n"
            + f"principles:{principles[-8:]}\n"
            + f"instructions:{instructions[-8:]}\n"
            + f"recent:{recent_visions}\n"
            + "Avoid repeating subject/composition from recent visions."
        )
        try:
            _trace_prompt(self.trace_prompts, "vision.generate", self.provider, self.model, "Generate one concise art vision sentence.", prompt)
            candidate = _ollama_generate_text(self.base_url, self.model, prompt, self.temperature, timeout=60).strip().strip('"').replace("\n", " ")
            _, _, _, prioritized = infer_guidance(text_memories)
            if memory_collision(parse_vision(candidate), memories[-5:], prioritized):
                raise HostedCallError("Ollama vision collided with recent pattern.")
            print(f"\n  New Vision: \"{candidate}\"")
            return candidate
        except Exception as exc:
            raise HostedCallError(f"Ollama vision generation failed: {exc}") from exc


class HostedVisionBackend(VisionBackend):
    def __init__(self, provider: str, model: str, api_key: str, temperature: float = 0.4, allow_fallback: bool = False, trace_prompts: bool = False):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.allow_fallback = allow_fallback
        self.trace_prompts = trace_prompts

    def _http_json(self, url: str, payload: Dict, headers: Dict) -> Dict:
        return _post_json_with_retry(url=url, payload=payload, headers=headers, timeout=50)

    def _chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        _trace_prompt(self.trace_prompts, "vision.generate", self.provider, self.model, system_prompt, user_prompt)
        if self.provider == "openai":
            out = self._http_json("https://api.openai.com/v1/responses", {"model": self.model, "input": [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, {"role": "user", "content": [{"type": "text", "text": user_prompt}]}], "temperature": self.temperature, "max_output_tokens": max_tokens}, {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            if isinstance(out.get("output_text"), str) and out["output_text"].strip():
                return out["output_text"]
        if self.provider == "anthropic":
            out = self._http_json("https://api.anthropic.com/v1/messages", {"model": self.model, "max_tokens": max_tokens, "temperature": self.temperature, "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}]}, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"})
            for part in out.get("content", []):
                if isinstance(part.get("text"), str) and part["text"].strip():
                    return part["text"]
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
            out = self._http_json(url, {"contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}], "generationConfig": {"temperature": self.temperature, "maxOutputTokens": max_tokens}}, {"Content-Type": "application/json"})
            for cand in out.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if isinstance(part.get("text"), str) and part["text"].strip():
                        return part["text"]
        raise ValueError(f"Unsupported vision provider: {self.provider}")

    def generate_vision(self, soul: Dict, ignored_ids: set) -> str:
        text_memories = soul.get("text_memories", [])
        preferences, principles, instructions, _ = infer_guidance(text_memories)
        packet = build_soul_packet(soul)
        memories = [m for m in soul.get("memories", []) if m.get("id") not in ignored_ids]
        _print_vision_context_summary(soul, preferences, principles, instructions, memories)
        recent_visions = [m.get("vision", "") for m in memories[-8:]]
        try:
            candidate = self._chat_text(
                "Generate one concise art vision sentence.",
                "Weighting guidance: obsession 35%, personality 30%, text memories 20%, artwork/history 15%.\n"
                + TIER_GUIDANCE_TEXT
                + f"soul_packet:{packet}\npreferences:{preferences[-8:]}\nprinciples:{principles[-8:]}\ninstructions:{instructions[-8:]}\nrecent:{recent_visions}\n"
                + "Avoid repeating subject/composition from recent visions.",
                90,
            ).strip().strip('"').replace("\n", " ")
            _, _, _, prioritized = infer_guidance(text_memories)
            if memory_collision(parse_vision(candidate), memories[-5:], prioritized):
                raise HostedCallError("Hosted vision collided with recent pattern.")
            print(f"\n  New Vision: \"{candidate}\"")
            return candidate
        except Exception as exc:
            raise HostedCallError(f"Hosted vision generation failed: {exc}") from exc



