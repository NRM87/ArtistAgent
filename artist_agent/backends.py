import base64
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
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
SOUL_CONTEXT_GUIDANCE = (
    "Ground decisions in your personality traits, obsession, text memories, artwork memories, and recent history.\n"
)
FIRST_PERSON_HINT = (
    "Use first-person voice (I/me/my) when expressing artistic intent, critique, and reflection. "
    "Never refer to yourself by your artist name.\n"
)
REVISION_ACTION_HINT = "Prefer concrete commitments over abstract commentary when proposing soul revisions.\n"
ACTION_VISION_CONTRACT = (
    "Return exactly one line in this format:\n"
    "RUN_VISION: My vision for this run is to create <one concrete image goal>.\n"
    "Keep it visual and specific (subject, composition, mood). "
    "Do not output policy statements, context summaries, or instruction echoes."
)
ACTION_VERB_PATTERN = (
    r"(?i)^(create|compose|render|paint|draw|depict|illustrate|craft|focus|judge|score|identify|compare|"
    r"revise|update|record|preserve|avoid|tighten|strengthen|reduce|increase|add|remove|keep|test|check|"
    r"emphasize|clarify|balance|shift|improve|use|try|set)\b"
)


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


def _resolve_cli_executable(name: str) -> str:
    candidates = [name]
    if os.name == "nt":
        candidates = [f"{name}.cmd", f"{name}.exe", name]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError(f"CLI executable not found for '{name}'.")


def _split_cli_model_spec(cli_default: str, model_spec: str) -> Tuple[str, str]:
    cli = str(cli_default).strip().lower() or "gemini"
    spec = str(model_spec).strip()
    if not spec:
        return cli, ""
    low = spec.lower()
    if low in ("gemini", "codex"):
        return low, ""
    for prefix in ("gemini:", "codex:"):
        if low.startswith(prefix):
            return prefix[:-1], spec.split(":", 1)[1].strip()
    return cli, spec


def _extract_text_from_json_tree(value: object) -> str:
    if isinstance(value, str):
        v = value.strip()
        if v:
            return v
        return ""
    if isinstance(value, list):
        for item in value:
            found = _extract_text_from_json_tree(item)
            if found:
                return found
        return ""
    if isinstance(value, dict):
        for key in ("text", "output_text", "response", "message", "content", "result"):
            if key in value:
                found = _extract_text_from_json_tree(value.get(key))
                if found:
                    return found
        for child in value.values():
            found = _extract_text_from_json_tree(child)
            if found:
                return found
    return ""


def _run_cli_text(cli: str, model: str, prompt: str, timeout: int = 180) -> str:
    selected_cli, selected_model = _split_cli_model_spec(cli, model)
    prompt_text = str(prompt).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not prompt_text:
        raise ValueError("CLI prompt was empty.")

    if selected_cli == "codex":
        out_file = tempfile.NamedTemporaryFile(prefix="codex_last_", suffix=".txt", delete=False)
        out_file.close()
        proc = _run_codex_exec(
            prompt_text,
            model=selected_model,
            output_last_message=out_file.name,
            sandbox_mode="read-only",
            timeout=timeout,
        )
        response = ""
        try:
            response = Path(out_file.name).read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            response = ""
        finally:
            try:
                Path(out_file.name).unlink(missing_ok=True)
            except Exception:
                pass
        if not response:
            response = str(proc.stdout).strip()
        if proc.returncode != 0 and not response:
            stderr = str(proc.stderr).strip()
            raise RuntimeError(stderr or f"codex exited with code {proc.returncode}")
        return response

    if selected_cli == "gemini":
        exe = _resolve_cli_executable("gemini")
        args = [exe, "-p", prompt_text, "--output-format", "json"]
        if selected_model:
            args.extend(["-m", selected_model])
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, shell=False)
        stdout = str(proc.stdout).strip()
        stderr = str(proc.stderr).strip()
        parsed_text = ""
        if stdout:
            try:
                payload = json.loads(stdout)
                parsed_text = _extract_text_from_json_tree(payload)
            except Exception:
                parsed_text = ""
        if not parsed_text:
            parsed_text = _first_nonempty_line(stdout)
        if proc.returncode != 0 and not parsed_text:
            raise RuntimeError(stderr or f"gemini exited with code {proc.returncode}")
        if not parsed_text:
            raise RuntimeError("gemini returned empty output.")
        return parsed_text

    raise ValueError(f"Unsupported CLI adapter '{selected_cli}'. Use 'gemini' or 'codex'.")


def _run_codex_exec(
    prompt: str,
    model: str = "",
    output_last_message: str = "",
    sandbox_mode: str = "read-only",
    timeout: int = 240,
) -> subprocess.CompletedProcess:
    exe = _resolve_cli_executable("codex")
    args = [
        exe,
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox_mode,
        "-a",
        "never",
        "--ephemeral",
        "--color",
        "never",
    ]
    if output_last_message:
        args.extend(["--output-last-message", str(output_last_message)])
    if model:
        args.extend(["-m", model])
    args.append("-")
    return subprocess.run(args, input=str(prompt), capture_output=True, text=True, timeout=timeout, shell=False)


def _first_nonempty_line(text: str) -> str:
    for ln in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        v = ln.strip()
        if v:
            return v
    return ""


def _extract_labeled_value(text: str, label: str) -> str:
    pattern = rf"(?im)^\s*{re.escape(label)}\s*[:=-]\s*(.+?)\s*$"
    match = re.search(pattern, str(text))
    if match:
        return str(match.group(1)).strip()
    return ""


def _extract_int_in_range(text: str, low: int, high: int) -> Optional[int]:
    labeled = _extract_labeled_value(text, "score")
    candidates = [labeled, str(text)]
    for src in candidates:
        for m in re.finditer(r"-?\d+", str(src)):
            try:
                v = int(m.group(0))
            except Exception:
                continue
            if low <= v <= high:
                return v
    return None


def _extract_yes_no(text: str) -> Optional[bool]:
    labeled = _extract_labeled_value(text, "worthy")
    probe = f"{labeled}\n{text}".lower()
    if re.search(r"\b(true|yes|y|worthy)\b", probe):
        return True
    if re.search(r"\b(false|no|n|unworthy)\b", probe):
        return False
    return None


def _normalize_choice(text: str, choices: List[str], default: str) -> str:
    probe = str(text).strip().lower()
    line = _first_nonempty_line(probe)
    if ":" in line:
        line = line.split(":", 1)[1].strip()
    token = re.split(r"[\s,.;:!?\)\(]+", line)[0].strip()
    for c in choices:
        if token == c.lower():
            return c
    found = [c for c in choices if re.search(rf"\b{re.escape(c.lower())}\b", probe)]
    if len(found) == 1:
        return found[0]
    return default


def _split_list_text(text: str, max_items: int = 10) -> List[str]:
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.replace("|", ",").replace(";", ",")
    parts = []
    for line in raw.split("\n"):
        line = re.sub(r"^\s*[-*0-9\.\)\(]+\s*", "", line).strip()
        if not line:
            continue
        for token in line.split(","):
            clean = str(token).strip()
            if clean and clean not in parts:
                parts.append(clean)
            if len(parts) >= max_items:
                return parts
    return parts


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = raw.find(start_marker)
    if s < 0:
        return ""
    s += len(start_marker)
    e = raw.find(end_marker, s)
    if e < 0:
        return ""
    return raw[s:e].strip()


def _normalize_prompt_text(text: str) -> str:
    value = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"(?im)^\s*image_prompt\s*[:=-]\s*", "", value).strip()
    # Strip wrapper echo lines if the model repeats scaffold text.
    cleaned_lines: List[str] = []
    for line in value.split("\n"):
        ln = line.strip()
        if re.match(r'(?i)^run vision\s*\(fixed for this run\)\s*[:=-]\s*"', ln):
            continue
        if re.match(r'(?i)^iteration image prompt\s*[:=-]\s*"', ln):
            continue
        if "create a coherent 2d composition using the iteration image prompt" in ln.lower():
            continue
        cleaned_lines.append(line)
    value = "\n".join(cleaned_lines).strip()
    value = value.replace("RUN_VISION", "").replace("run_vision", "").strip()
    value = re.sub(r"(?i)\bmy vision for this run is to\b", "", value).strip()
    # Keep multiline prompts readable but collapse redundant blank runs.
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"\s+", " ", value).strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value


def _contains_first_person(text: str) -> bool:
    return bool(re.search(r"\b(i|me|my|mine|myself)\b", str(text), flags=re.I))


def _self_name_patterns(self_name: str) -> List[Tuple[str, str]]:
    name = str(self_name).strip()
    if not name:
        return []
    esc = re.escape(name)
    return [
        (rf"(?i)\b{esc}[\'\u2019]s\b", "my"),
        (rf"(?i)\b{esc}\s+should\b", "I should"),
        (rf"(?i)\b{esc}\s+will\b", "I will"),
        (rf"(?i)\b{esc}\s+wants?\b", "I want"),
        (rf"(?i)\b{esc}\s+needs?\b", "I need"),
        (rf"(?i)\b{esc}\s+prefers?\b", "I prefer"),
    ]


def _normalize_self_reference(text: str, self_name: str = "") -> str:
    value = str(text)
    patterns: List[Tuple[str, str]] = [
        (r"(?i)\bthe artist[\'\u2019]s\b", "my"),
        (r"(?i)\bthis artist[\'\u2019]s\b", "my"),
    ] + _self_name_patterns(self_name)
    for pattern, replacement in patterns:
        value = re.sub(pattern, replacement, value)
    return re.sub(r"\s+", " ", value).strip()


def _extract_artist_name_from_frame(frame: str) -> str:
    return _extract_labeled_value(str(frame), "artist_name")


def _extract_image_prompt(raw: str, current_prompt: str, self_name: str = "") -> str:
    text = str(raw)
    for start, end in (
        ("IMAGE_PROMPT_START", "IMAGE_PROMPT_END"),
        ("REVISED_PROMPT_START", "REVISED_PROMPT_END"),
        ("VISION_REFINED_START", "VISION_REFINED_END"),
    ):
        block = _extract_block(text, start, end)
        if block:
            return _normalize_self_reference(_normalize_prompt_text(block), self_name)

    for label in ("image_prompt", "revised_prompt", "vision_refined"):
        value = _extract_labeled_value(text, label)
        if value:
            return _normalize_self_reference(_normalize_prompt_text(value), self_name)

    # Last-resort fallback: accept a single clean line only.
    first = _first_nonempty_line(text)
    if first and "\n" not in first and len(first) <= 500:
        lowered = first.lower()
        if not any(
            token in lowered
            for token in (
                "return only",
                "base_prompt",
                "revised_prompt_start",
                "image_prompt_start",
                "vision_refined_start",
                "keep continuity",
            )
        ):
            return _normalize_self_reference(_normalize_prompt_text(first), self_name)
    return current_prompt


def _looks_meta_vision(text: str) -> bool:
    probe = str(text).strip().lower()
    if not probe:
        return True
    blocked = (
        "artwork tiers",
        "guide decisions",
        "soul packet",
        "soul context",
        "personality_traits:",
        "text_memories:",
        "artwork_memories:",
        "history:",
        "preferences:",
        "principles:",
        "instructions:",
        "respond using",
        "return exactly",
        "run_vision",
        "vision_directive",
        "critique_directive",
        "revision_directive",
        "<",
        ">",
    )
    return any(token in probe for token in blocked)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(text)))


def _meaningful_feedback(text: str) -> bool:
    v = str(text).strip()
    if len(v) < 28 or _word_count(v) < 6:
        return False
    lower = v.lower()
    blocked = (
        "my vision for",
        "run vision",
        "image prompt",
        "iteration image prompt",
        "create a coherent 2d composition",
    )
    if any(token in lower for token in blocked):
        return False
    # Require evaluative language so pure prompt restatements do not pass as critique.
    return bool(re.search(r"\b(should|need|needs|must|improv|lacks?|missing|too|more|less|because|but|however)\b", lower))


def _meaningful_next_action(text: str) -> bool:
    v = str(text).strip()
    if len(v) < 18 or _word_count(v) < 4:
        return False
    lower = v.lower()
    if "iteration image prompt" in lower or "run vision (fixed for this run)" in lower:
        return False
    if "create a coherent 2d composition using the iteration image prompt" in lower:
        return False
    if re.match(r"(?i)^(create|draw|paint|render|focus|improve|emphasize)\.?$", v):
        return False
    return True


def _prompt_keywords(text: str) -> List[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "while", "about", "across", "under",
        "over", "your", "my", "vision", "image", "create", "run", "fixed", "using", "stay", "faithful", "next",
    }
    out: List[str] = []
    for tok in re.findall(r"[a-zA-Z]{4,}", str(text).lower()):
        if tok in stop:
            continue
        if tok not in out:
            out.append(tok)
    return out


def _is_usable_image_prompt(prompt: str) -> bool:
    p = _normalize_prompt_text(prompt)
    if len(p) < 28 or _word_count(p) < 6:
        return False
    blocked = (
        "run vision (fixed for this run)",
        "iteration image prompt",
        "respond with exactly",
        "image_prompt:",
    )
    if any(token in p.lower() for token in blocked):
        return False
    return True


def _is_prompt_aligned_with_vision(prompt: str, vision: str) -> bool:
    p = _normalize_prompt_text(prompt).lower()
    v = str(vision).strip().lower()
    if not p or not v:
        return False
    p_tokens = set(_prompt_keywords(p))
    v_tokens = set(_prompt_keywords(v))
    if not p_tokens or not v_tokens:
        return True
    overlap = p_tokens.intersection(v_tokens)
    return len(overlap) >= 1


def _normalize_action_command(raw: str, self_name: str = "") -> str:
    value = " ".join(str(raw).replace("\r\n", "\n").replace("\r", "\n").split()).strip()
    value = value.strip('"').strip("'")
    value = re.sub(r"(?i)^next_action\s*[:=-]\s*", "", value).strip()
    value = re.sub(
        r"(?i)^(i\s+(will|want to|intend to|need to|should|am going to)\s+|my\s+next\s+action\s+is\s+to\s+)",
        "",
        value,
    ).strip()
    value = _normalize_self_reference(value, self_name)
    value = re.sub(r"(?i)^to\s+", "", value).strip()
    if "iteration image prompt" in value.lower() or "run vision (fixed for this run)" in value.lower():
        return ""
    if "create a coherent 2d composition using the iteration image prompt" in value.lower():
        return ""
    value = value.strip(" .,:;-")
    if not value or _looks_meta_vision(value):
        return ""
    if not re.match(ACTION_VERB_PATTERN, value):
        value = f"Emphasize {value[0].lower() + value[1:]}" if len(value) > 1 else ""
    value = value[:180].strip()
    if not value:
        return ""
    if not value.endswith((".", "!", "?")):
        value += "."
    return value


def _normalize_action_vision(raw: str, soul: Optional[Dict] = None) -> str:
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n")
    candidate = (
        _extract_labeled_value(text, "run_vision")
        or _extract_labeled_value(text, "vision")
        or _extract_labeled_value(text, "my_vision")
        or _first_nonempty_line(text)
    )
    candidate = " ".join(candidate.split()).strip().strip('"').strip("'")
    if not candidate:
        candidate = ""

    body = _normalize_self_reference(candidate, str((soul or {}).get("name", "")).strip())
    body = re.sub(r"(?i)\brun[_\s-]*vision\b", "", body).strip()
    body = re.sub(r"(?i)\b(vision|my_vision)\s*[:=-]\s*", "", body).strip()
    lower = body.lower()
    if lower.startswith("my vision for this run is to "):
        body = body[len("my vision for this run is to ") :].strip()
    elif lower.startswith("my vision for this run is "):
        body = body[len("my vision for this run is ") :].strip()
    body = re.sub(
        r"(?i)^(i\s+(will|want to|intend to|need to|should|am going to)\s+|my\s+(goal|focus|plan)\s+is\s+to\s+)",
        "",
        body,
    ).strip()
    body = re.sub(r"(?i)^to\s+", "", body).strip(" .,:;-")
    if re.fullmatch(r"(?i)(my vision for this run is to )?", body):
        body = ""

    if _looks_meta_vision(body):
        body = ""
    if body and not re.match(ACTION_VERB_PATTERN, body):
        body = f"Create an image that {body[0].lower() + body[1:]}" if len(body) > 1 else ""
    if not body:
        obsession = str((soul or {}).get("current_obsession", "")).strip()
        if obsession:
            body = f"create an image that explores {obsession.lower()} with a clear focal subject and composition"
        else:
            body = "create a concrete image with a clear focal subject, composition, and mood"

    body = body[:220].strip()
    body = re.sub(r"(?i)^to\s+", "", body).strip()
    if not body.endswith((".", "!", "?")):
        body += "."
    return f"My vision for this run is to {body[0].lower() + body[1:] if body and body[0].isupper() else body}"


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


def _normalize_reflection_weights(raw: object) -> Dict[str, float]:
    out = {"vision": 1.0, "refinement": 1.0, "critique": 1.0, "revision": 1.0}
    if isinstance(raw, dict):
        for stage in out.keys():
            if stage not in raw:
                continue
            try:
                out[stage] = float(raw.get(stage, out[stage]))
            except (TypeError, ValueError):
                pass
    for stage in out.keys():
        out[stage] = max(0.3, min(2.5, float(out[stage])))
    return out


def _stage_weight_guidance(packet: Dict, stage: str) -> str:
    weights = _normalize_reflection_weights(packet.get("reflection_weights", {}))
    w = weights.get(stage, 1.0)
    return f"reflection_weights:{weights} | stage:{stage} | stage_weight:{w:.2f}"


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
        "reflection_weights": _normalize_reflection_weights(soul.get("reflection_weights", {})),
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
        best_canvas: Optional[str] = None
        best_warnings: List[str] = []
        best_quality = -1.0
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
                warnings: List[str] = []
                if self._contains_readable_text(canvas):
                    warnings.append("readable_text")
                ink_ratio = self._ink_ratio(canvas)
                if ink_ratio < 0.006:
                    warnings.append("sparse_output")
                # Prefer cleaner outputs, but do not hard-fail on style quality checks.
                quality = ink_ratio
                if "readable_text" not in warnings:
                    quality += 1.0
                if "sparse_output" not in warnings:
                    quality += 0.7
                if quality > best_quality:
                    best_quality = quality
                    best_canvas = canvas
                    best_warnings = warnings
                if not warnings:
                    break
            except Exception as exc:
                last_exc = exc

        if best_canvas is None:
            raise HostedCallError(f"LLM ASCII rendering failed after retries ({last_exc})")
        warning_text = ",".join(best_warnings) if best_warnings else "none"
        lines = [
            f"ASCII ART - creation {creation_id} iter {iteration}",
            f"prompt: {prompt}",
            "renderer: llm",
            f"canvas: {width}x{height}",
            f"quality_warnings: {warning_text}",
            "",
            "BEGIN_ASCII",
            best_canvas,
            "END_ASCII",
            "",
        ]

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

    @staticmethod
    def _strip_wrapping_quotes(text: str) -> str:
        v = str(text).strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1].strip()
        return v

    @classmethod
    def _normalize_model_prompt(cls, prompt: str, max_chars: int = 1800) -> str:
        raw = str(prompt).replace("\r\n", "\n").replace("\r", "\n")
        run_vision = cls._strip_wrapping_quotes(_extract_labeled_value(raw, "Run vision (fixed for this run)"))
        iter_prompt = cls._strip_wrapping_quotes(_extract_labeled_value(raw, "Iteration image prompt"))
        if iter_prompt:
            model_prompt = iter_prompt
            if run_vision:
                model_prompt += f"\n\nStay faithful to this run vision: {run_vision}"
        else:
            model_prompt = " ".join(raw.split())
        model_prompt = re.sub(r"\s+", " ", model_prompt).strip()
        return model_prompt[:max_chars]

    @staticmethod
    def _download_bytes(url: str) -> bytes:
        req = urllib.request.Request(url=url, method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if not data:
            raise ValueError("Downloaded image URL returned empty payload.")
        return data

    @classmethod
    def _extract_openai_image_bytes(cls, payload: Dict) -> bytes:
        rows = payload.get("data", []) or []
        for row in rows:
            b64 = str(row.get("b64_json", "")).strip()
            if b64:
                return base64.b64decode(b64)
            url = str(row.get("url", "")).strip()
            if url:
                return cls._download_bytes(url)
        raise ValueError("OpenAI response missing image data.")

    @staticmethod
    def _extract_gemini_inline_b64(payload: Dict) -> str:
        for cand in payload.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []) or []:
                inline = part.get("inlineData") or part.get("inline_data") or {}
                b64 = str(inline.get("data", "")).strip()
                if b64:
                    return b64
        return ""

    def _write_image_bytes(self, creation_id: int, iteration: int, data: bytes) -> str:
        path = self.temp_dir / f"img_{creation_id:04d}_iter_{iteration}.png"
        path.write_bytes(data)
        return str(path)

    def generate(self, prompt: str, iteration: int, creation_id: int) -> str:
        try:
            model_prompt = self._normalize_model_prompt(prompt)
            if self.provider == "openai":
                _trace_prompt(
                    self.trace_prompts,
                    "image.generate",
                    self.provider,
                    self.model,
                    "Generate an image from prompt.",
                    f"prompt:{model_prompt}\nsize:{self.size}",
                )
                out = self._http_json(
                    "https://api.openai.com/v1/images/generations",
                    {"model": self.model, "prompt": model_prompt, "size": self.size, "response_format": "b64_json"},
                    {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                )
                return self._write_image_bytes(creation_id, iteration, self._extract_openai_image_bytes(out))

            if self.provider == "gemini":
                _trace_prompt(
                    self.trace_prompts,
                    "image.generate",
                    self.provider,
                    self.model,
                    "Create an image from prompt.",
                    f"prompt:{model_prompt}\nsize:{self.size}",
                )
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
                payload = {
                    "contents": [{"parts": [{"text": f"Create a single image only. Target size {self.size}. Prompt: {model_prompt}"}]}],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                }
                out = self._http_json(url, payload, {"Content-Type": "application/json"})
                b64 = self._extract_gemini_inline_b64(out)
                if not b64:
                    retry_payload = {
                        "contents": [{"parts": [{"text": f"Return IMAGE modality output only. Target size {self.size}. Prompt: {model_prompt}"}]}],
                        "generationConfig": {"responseModalities": ["IMAGE"]},
                    }
                    out = self._http_json(url, retry_payload, {"Content-Type": "application/json"})
                    b64 = self._extract_gemini_inline_b64(out)
                if not b64:
                    raise ValueError("Gemini did not return inline image data.")
                return self._write_image_bytes(creation_id, iteration, base64.b64decode(b64))

            raise ValueError(f"Unsupported image provider: {self.provider}")
        except Exception as exc:
            if not self.allow_fallback:
                raise HostedCallError(f"Hosted image generation failed: {exc}") from exc
            print(f"Warning: hosted image generation failed, falling back to {self.fallback_mode} ({exc})")
            return self._fallback_backend.generate(prompt, iteration, creation_id)


class CodexImageBackend(ImageBackend):
    def __init__(
        self,
        model: str,
        temp_dir: Path,
        image_size: str = "1024x1024",
        fallback_mode: str = "ascii",
        llm_backend: Optional[object] = None,
        ascii_size: str = "160x60",
        trace_prompts: bool = False,
    ):
        self.provider = "codex"
        self.model = model
        self.temp_dir = temp_dir
        self.image_size = image_size
        self.fallback_mode = fallback_mode if fallback_mode in ("ascii", "defer") else "ascii"
        self.trace_prompts = trace_prompts
        self._fallback_backend = AsciiImageBackend(temp_dir, llm_backend=llm_backend, ascii_size=ascii_size)

    def _build_codex_image_task(self, prompt: str, output_path: Path) -> str:
        return (
            "Generate exactly one PNG image and save it to output_path.\n"
            "Use Codex image generation capabilities. Do not output markdown.\n"
            "Do not create any file other than the requested output image.\n"
            f"output_path:{str(output_path)}\n"
            f"target_size:{self.image_size}\n"
            f"image_prompt:{str(prompt).strip()}"
        )

    def generate(self, prompt: str, iteration: int, creation_id: int) -> str:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.temp_dir / f"codex_img_{creation_id:04d}_iter_{iteration}.png"
        task = self._build_codex_image_task(prompt, out_path)
        try:
            _trace_prompt(
                self.trace_prompts,
                "image.generate.codex",
                self.provider,
                self.model or "(default)",
                "Generate one PNG image and write it to output_path.",
                task,
            )
            proc = _run_codex_exec(
                task,
                model=self.model,
                sandbox_mode="workspace-write",
                timeout=300,
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                return str(out_path)
            detail = _first_nonempty_line(proc.stderr) or _first_nonempty_line(proc.stdout) or "Codex image output missing."
            raise RuntimeError(detail)
        except Exception as exc:
            if self.fallback_mode == "ascii":
                print(f"Warning: codex image generation failed, falling back to ascii ({exc})")
                return self._fallback_backend.generate(prompt, iteration, creation_id)
            raise HostedCallError(f"Codex image generation failed: {exc}") from exc


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

    def generate_initial_render_prompt(self, soul_data: Dict, vision: str, run_intent: Optional[Dict] = None) -> str:
        raise NotImplementedError

    def refine_render_prompt(
        self,
        current_prompt: str,
        vision: str,
        critique_feedback: str,
        score: int,
        soul_data: Dict,
        run_intent: Optional[Dict] = None,
    ) -> str:
        raise NotImplementedError

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        raise NotImplementedError

    def evaluate_review_merit(self, soul_data: Dict, review_payload: Dict) -> Dict:
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
        next_action = (
            "Increase focal contrast and simplify the scene hierarchy." if score <= 5 else
            "Strengthen depth cues and clarify the primary subject." if score <= 7 else
            "Preserve the composition and tighten detail precision."
        )
        return {"score": int(score), "feedback": feedback, "next_action": next_action}

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

    def generate_initial_render_prompt(self, soul_data: Dict, vision: str, run_intent: Optional[Dict] = None) -> str:
        directive = str((run_intent or {}).get("vision_directive", "")).strip()
        base = str(vision).strip() or "Create a coherent 2D composition."
        if directive:
            return f"{base} {directive}"
        return base

    def refine_render_prompt(
        self,
        current_prompt: str,
        vision: str,
        critique_feedback: str,
        score: int,
        soul_data: Dict,
        run_intent: Optional[Dict] = None,
    ) -> str:
        raise HostedCallError("Prompt refinement requires an LLM backend, but current backend is mock.")

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        raise HostedCallError("State revision requires an LLM backend, but current backend is mock.")

    def evaluate_review_merit(self, soul_data: Dict, review_payload: Dict) -> Dict:
        score = int(review_payload.get("score", 0))
        decision = "accept" if score >= 8 else ("partial" if score >= 5 else "reject")
        memory = ""
        if decision in ("accept", "partial"):
            memory = str(review_payload.get("suggestion", "")).strip() or str(review_payload.get("feedback", "")).strip()
        return {
            "decision": decision,
            "rationale": "I accept this review because it gives concrete direction." if decision == "accept" else (
                "I will partially use this review because it has mixed value." if decision == "partial" else
                "I reject this review because it is not aligned with my intent."
            ),
            "memory_content": memory,
            "importance": "high" if decision == "accept" else ("medium" if decision == "partial" else "low"),
            "tags": ["review", "external_feedback", decision],
            "obsession_update": "",
        }


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

    def critique(self, image_path: str, vision: str, iteration: int, critique_frame: str = "") -> Dict:
        try:
            artist_name = _extract_artist_name_from_frame(critique_frame)
            raw = self._chat_text(
                "Evaluate this artwork and respond using exactly three lines:\n"
                "SCORE: <integer 1-10>\n"
                "FEEDBACK: <one concise critique sentence in first person>\n"
                "NEXT_ACTION: <one concrete command for the next image attempt>",
                f"vision:{vision}\niteration:{iteration}\ncritique_frame:{critique_frame}",
                image_path=image_path,
            )
            parsed_score = _extract_int_in_range(raw, 1, 10)
            if parsed_score is None:
                score_raw = self._chat_text(
                    "Respond with one integer from 1 to 10 only.",
                    f"vision:{vision}\niteration:{iteration}",
                    image_path=image_path,
                )
                parsed_score = _extract_int_in_range(score_raw, 1, 10)
            if parsed_score is None:
                raise ValueError("Could not parse critique score.")
            feedback = _extract_labeled_value(raw, "feedback") or _first_nonempty_line(raw)
            if not feedback or feedback.isdigit() or not _meaningful_feedback(feedback):
                feedback_raw = self._chat_text(
                    "Respond with exactly one critique sentence in first person. "
                    "It must name at least one concrete visual issue and one concrete improvement.",
                    f"vision:{vision}\niteration:{iteration}\nscore:{parsed_score}\ncritique_frame:{critique_frame}",
                    image_path=image_path,
                )
                feedback = _first_nonempty_line(feedback_raw)
            feedback = _normalize_self_reference(feedback, artist_name)
            if feedback and (not _contains_first_person(feedback) or not _meaningful_feedback(feedback)):
                fp_raw = self._chat_text(
                    "Rewrite in first person only. Return one concrete sentence with visual detail.",
                    f"feedback:{feedback}",
                    image_path=image_path,
                )
                feedback = _first_nonempty_line(fp_raw) or feedback
            feedback = _normalize_self_reference(feedback, artist_name)
            next_action = _normalize_action_command(
                _extract_labeled_value(raw, "next_action")
                or _extract_labeled_value(raw, "action")
                or "",
                artist_name,
            )
            if not next_action or not _meaningful_next_action(next_action):
                action_raw = self._chat_text(
                    "Respond with exactly one line: NEXT_ACTION: <specific command with subject/composition/lighting/color detail>.",
                    f"vision:{vision}\niteration:{iteration}\nscore:{parsed_score}\nfeedback:{feedback}",
                    image_path=image_path,
                )
                next_action = _normalize_action_command(action_raw, artist_name)
            if not feedback or not _meaningful_feedback(feedback):
                raise ValueError("Could not parse critique feedback.")
            if next_action and not _meaningful_next_action(next_action):
                next_action = ""
            return {"score": parsed_score, "feedback": feedback, "next_action": next_action}
        except Exception as exc:
            raise HostedCallError(f"Ollama critique failed: {exc}") from exc

    def judge_worthiness(self, image_path: str, score: int, vision: str, critique_frame: str = "") -> bool:
        try:
            raw = self._chat_text(
                "Decide if this artwork is worthy. Respond with one token only: YES or NO.",
                f"vision:{vision}\nscore:{score}\ncritique_frame:{critique_frame}",
                image_path=image_path,
            )
            worthy = _extract_yes_no(raw)
            if worthy is None:
                raise ValueError("Could not parse worthy decision.")
            return worthy
        except Exception as exc:
            raise HostedCallError(f"Ollama judgment failed: {exc}") from exc

    def generate_text_memory(self, soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        try:
            raw = self._chat_text(
                "Generate one text memory and respond using lines:\n"
                "CONTENT: <first-person text>\n"
                "IMPORTANCE: <critical|high|medium|low>\n"
                "TAGS: <comma separated tags>",
                f"trigger:{trigger_reason}\nresult:{creation_result}",
            )
            content = _extract_labeled_value(raw, "content") or _first_nonempty_line(raw)
            if content and not _contains_first_person(content):
                fp_raw = self._chat_text("Rewrite this memory in first person only.", f"content:{content}")
                content = _first_nonempty_line(fp_raw) or content
            importance = _normalize_choice(_extract_labeled_value(raw, "importance"), ["critical", "high", "medium", "low"], "medium")
            tags_raw = _extract_labeled_value(raw, "tags")
            tags = _split_list_text(tags_raw, max_items=6)
            return safe_text_memory({"content": content, "importance": importance, "tags": tags}, soul_data)
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
            seed = current_name.strip() if current_name.strip() else "Unnamed Artist"
            name_raw = self._chat_text(
                "Return exactly one artist name line only.",
                f"Current name hint: {seed}",
            )
            obsession_raw = self._chat_text(
                "Return exactly one current obsession line only.",
                f"Artist name: {seed}",
            )
            traits_raw = self._chat_text(
                "Return 3 to 7 personality traits, comma-separated. No numbering.",
                f"Artist name: {seed}\nCurrent obsession hint: {_first_nonempty_line(obsession_raw)}",
            )
            name = _first_nonempty_line(name_raw) or seed
            clean_traits = _split_list_text(traits_raw, max_items=7)
            if len(clean_traits) < 3:
                raise HostedCallError("LLM returned insufficient personality traits.")
            obsession = _first_nonempty_line(obsession_raw)
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
                ACTION_VISION_CONTRACT,
                (
                    SOUL_CONTEXT_GUIDANCE
                    + FIRST_PERSON_HINT
                    + TIER_GUIDANCE_TEXT
                    + f"{_stage_weight_guidance(packet, 'vision')}\n"
                    + f"soul_packet:{packet}\npreferences:{prefs[-8:]}\nprinciples:{principles[-8:]}\ninstructions:{instructions[-8:]}\nrecent:{recent}\n"
                    + "Prefer meaningful variation in composition while preserving continuity with the artist's soul."
                ),
            )
            vision = _normalize_action_vision(text, soul)
            if not vision:
                raise HostedCallError("LLM vision fallback returned empty vision.")
            return vision
        except Exception as exc:
            raise HostedCallError(f"Ollama vision fallback failed: {exc}") from exc

    def generate_run_intent(self, soul_data: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            context = (
                FIRST_PERSON_HINT
                + SOUL_CONTEXT_GUIDANCE
                + TIER_GUIDANCE_TEXT
                + f"{_stage_weight_guidance(packet, 'vision')}\n"
                + f"soul_packet:{packet}"
            )
            out = self._chat_text(
                "Respond using three lines:\n"
                "VISION_DIRECTIVE: <imperative command for the next image prompt>\n"
                "CRITIQUE_DIRECTIVE: <imperative command for how to judge/score>\n"
                "REVISION_DIRECTIVE: <imperative command for what to revise in soul>\n"
                "Use first-person phrasing and never use the artist name.\n"
                "Do not output policy summaries.",
                context,
            )
            vd = _extract_labeled_value(out, "vision_directive")
            cd = _extract_labeled_value(out, "critique_directive")
            rd = _extract_labeled_value(out, "revision_directive")
            return {"vision_directive": vd, "critique_directive": cd, "revision_directive": rd}
        except Exception as exc:
            raise HostedCallError(f"Ollama run-intent generation failed: {exc}") from exc

    def generate_initial_render_prompt(self, soul_data: Dict, vision: str, run_intent: Optional[Dict] = None) -> str:
        try:
            packet = build_soul_packet(soul_data)
            directive = str((run_intent or {}).get("vision_directive", "")).strip()
            out = self._chat_text(
                "Create the initial image-generation prompt for this run.\n"
                "The run vision is fixed and must be executed directly.\n"
                "Return exactly one line in this format:\n"
                "IMAGE_PROMPT: <prompt text>\n"
                "Do not use the artist name; use first-person framing only if needed.",
                (
                    f"fixed_run_vision:{vision}\n"
                    f"vision_directive:{directive}\n"
                    f"{_stage_weight_guidance(packet, 'vision')}\n"
                    f"soul_packet:{packet}\n"
                    "Write a concrete visual prompt (subject, composition, medium/style, mood, lighting, palette)."
                ),
            )
            prompt = _extract_image_prompt(out, "", str(soul_data.get("name", "")).strip())
            if not prompt or not _is_usable_image_prompt(prompt) or not _is_prompt_aligned_with_vision(prompt, vision):
                retry = self._chat_text(
                    "Respond with exactly one line: IMAGE_PROMPT: <text>.\n"
                    "The prompt must concretely depict the run vision and include at least subject, composition, and mood.",
                    f"fixed_run_vision:{vision}\nvision_directive:{directive}\n",
                )
                prompt = _extract_image_prompt(retry, "", str(soul_data.get("name", "")).strip())
            if not prompt or not _is_usable_image_prompt(prompt) or not _is_prompt_aligned_with_vision(prompt, vision):
                raise ValueError("empty initial image prompt")
            return prompt
        except Exception as exc:
            raise HostedCallError(f"Ollama initial prompt generation failed: {exc}") from exc

    def refine_render_prompt(
        self,
        current_prompt: str,
        vision: str,
        critique_feedback: str,
        score: int,
        soul_data: Dict,
        run_intent: Optional[Dict] = None,
    ) -> str:
        try:
            packet = build_soul_packet(soul_data)
            directive = str((run_intent or {}).get("vision_directive", "")).strip()
            out = self._chat_text(
                "You are revising only the iteration image prompt for the next attempt.\n"
                "The run vision is fixed and must not be rewritten.\n"
                "Apply the critique feedback as concrete visual editing commands.\n"
                "Make minimal, targeted edits to CURRENT_IMAGE_PROMPT.\n"
                "Return exactly one line in this format:\n"
                "IMAGE_PROMPT: <revised prompt text>",
                (
                    f"fixed_run_vision:{vision}\n"
                    f"current_image_prompt:{current_prompt}\n"
                    f"critique_feedback:{critique_feedback}\n"
                    f"score:{int(score)}\n"
                    f"{_stage_weight_guidance(packet, 'refinement')}\n"
                    f"vision_directive:{directive}\n"
                    f"soul_packet:{packet}\n"
                    "Keep continuity with the fixed run vision while improving the next image attempt."
                )
            )
            next_prompt = _extract_image_prompt(out, current_prompt, str(soul_data.get("name", "")).strip())
            if (
                next_prompt == current_prompt
                or not _is_usable_image_prompt(next_prompt)
                or not _is_prompt_aligned_with_vision(next_prompt, vision)
            ):
                retry = self._chat_text(
                    "Respond with exactly one line: IMAGE_PROMPT: <text>",
                    (
                        f"fixed_run_vision:{vision}\n"
                        f"current_image_prompt:{current_prompt}\n"
                        f"critique_feedback:{critique_feedback}\n"
                        f"score:{int(score)}\n"
                    ),
                )
                next_prompt = _extract_image_prompt(retry, current_prompt, str(soul_data.get("name", "")).strip())
            if not _is_usable_image_prompt(next_prompt) or not _is_prompt_aligned_with_vision(next_prompt, vision):
                return current_prompt
            return next_prompt or current_prompt
        except Exception as exc:
            raise HostedCallError(f"Ollama prompt refinement failed: {exc}") from exc

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            context = (
                SOUL_CONTEXT_GUIDANCE
                + TIER_GUIDANCE_TEXT
                + REVISION_ACTION_HINT
                + f"{_stage_weight_guidance(packet, 'revision')}\n"
                + f"soul_packet:{packet}\ncreation_result:{creation_result}"
            )
            revision: Dict = {}

            obsession_decision = self._chat_text(
                "Current obsession update.\nRespond with one line: KEEP or SET: <new obsession>",
                context,
            )
            obsession_line = _first_nonempty_line(obsession_decision)
            if obsession_line.lower().startswith("set:"):
                revision["obsession"] = obsession_line.split(":", 1)[1].strip()

            personality_mode_raw = self._chat_text(
                "Choose personality mode. Respond with one token only: keep, append, or replace.",
                context,
            )
            personality_mode = _normalize_choice(personality_mode_raw, ["keep", "append", "replace"], "keep")
            revision["personality_mode"] = personality_mode
            if personality_mode in ("append", "replace"):
                traits_raw = self._chat_text(
                    "List personality traits for this mode as comma-separated values.",
                    context,
                )
                revision["personality_traits"] = _split_list_text(traits_raw, max_items=10)

            text_action_raw = self._chat_text(
                "Choose text memory action. Respond with one token only: none, add, edit_last, or delete_last.",
                context,
            )
            text_action = _normalize_choice(text_action_raw, ["none", "add", "edit_last", "delete_last"], "none")
            revision["text_memory_action"] = text_action
            if text_action in ("add", "edit_last"):
                text_mem_raw = self._chat_text(
                    "Respond with lines:\nCONTENT: <first-person text>\nIMPORTANCE: <critical|high|medium|low>\nTAGS: <comma separated tags>",
                    context,
                )
                revision["text_memory"] = {
                    "content": _extract_labeled_value(text_mem_raw, "content") or _first_nonempty_line(text_mem_raw),
                    "importance": _normalize_choice(_extract_labeled_value(text_mem_raw, "importance"), ["critical", "high", "medium", "low"], "medium"),
                    "tags": _split_list_text(_extract_labeled_value(text_mem_raw, "tags"), max_items=8),
                }

            artwork_action_raw = self._chat_text(
                "Choose artwork memory action. Respond with one token only: none, annotate_last, or delete_last.",
                context,
            )
            artwork_action = _normalize_choice(artwork_action_raw, ["none", "annotate_last", "delete_last"], "none")
            revision["artwork_memory_action"] = artwork_action
            if artwork_action == "annotate_last":
                note_raw = self._chat_text(
                    "Provide one concise first-person artwork note line.",
                    context,
                )
                revision["artwork_note"] = _first_nonempty_line(note_raw)

            return revision
        except Exception as exc:
            raise HostedCallError(f"Ollama state revision failed: {exc}") from exc

    def evaluate_review_merit(self, soul_data: Dict, review_payload: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            context = (
                FIRST_PERSON_HINT
                + SOUL_CONTEXT_GUIDANCE
                + f"{_stage_weight_guidance(packet, 'revision')}\n"
                + f"soul_packet:{packet}\nreview_payload:{review_payload}"
            )
            raw = self._chat_text(
                "Evaluate this external review and respond using lines:\n"
                "DECISION: <accept|partial|reject>\n"
                "RATIONALE: <one first-person sentence>\n"
                "MEMORY_CONTENT: <one actionable first-person note or blank>\n"
                "IMPORTANCE: <critical|high|medium|low>\n"
                "TAGS: <comma separated tags>\n"
                "OBSESSION_UPDATE: <blank or new obsession text>",
                context,
            )
            decision = _normalize_choice(_extract_labeled_value(raw, "decision"), ["accept", "partial", "reject"], "reject")
            rationale = _extract_labeled_value(raw, "rationale") or _first_nonempty_line(raw)
            if rationale and not _contains_first_person(rationale):
                rewrite = self._chat_text("Rewrite rationale in first person. One sentence only.", f"rationale:{rationale}")
                rationale = _first_nonempty_line(rewrite) or rationale
            memory_content = _extract_labeled_value(raw, "memory_content")
            importance = _normalize_choice(_extract_labeled_value(raw, "importance"), ["critical", "high", "medium", "low"], "medium")
            tags = _split_list_text(_extract_labeled_value(raw, "tags"), max_items=8)
            obsession_update = _extract_labeled_value(raw, "obsession_update")
            return {
                "decision": decision,
                "rationale": rationale,
                "memory_content": memory_content,
                "importance": importance,
                "tags": tags,
                "obsession_update": obsession_update,
            }
        except Exception as exc:
            raise HostedCallError(f"Ollama review-merit evaluation failed: {exc}") from exc


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

    def critique(self, image_path: str, vision: str, iteration: int, critique_frame: str = "") -> Dict:
        try:
            artist_name = _extract_artist_name_from_frame(critique_frame)
            raw = self._chat_text(
                "Evaluate this artwork and respond using exactly three lines:\n"
                "SCORE: <integer 1-10>\n"
                "FEEDBACK: <one concise critique sentence in first person>\n"
                "NEXT_ACTION: <one concrete command for the next image attempt>",
                f"vision:{vision}\niteration:{iteration}\ncritique_frame:{critique_frame}",
                220,
                image_path,
            )
            parsed_score = _extract_int_in_range(raw, 1, 10)
            if parsed_score is None:
                score_raw = self._chat_text(
                    "Respond with one integer from 1 to 10 only.",
                    f"vision:{vision}\niteration:{iteration}",
                    40,
                    image_path,
                )
                parsed_score = _extract_int_in_range(score_raw, 1, 10)
            if parsed_score is None:
                raise ValueError("Could not parse critique score.")
            feedback = _extract_labeled_value(raw, "feedback") or _first_nonempty_line(raw)
            if not feedback or feedback.isdigit() or not _meaningful_feedback(feedback):
                feedback_raw = self._chat_text(
                    "Respond with exactly one critique sentence in first person. "
                    "It must name at least one concrete visual issue and one concrete improvement.",
                    f"vision:{vision}\niteration:{iteration}\nscore:{parsed_score}\ncritique_frame:{critique_frame}",
                    160,
                    image_path,
                )
                feedback = _first_nonempty_line(feedback_raw)
            feedback = _normalize_self_reference(feedback, artist_name)
            if feedback and (not _contains_first_person(feedback) or not _meaningful_feedback(feedback)):
                fp_raw = self._chat_text(
                    "Rewrite in first person only. Return one concrete sentence with visual detail.",
                    f"feedback:{feedback}",
                    140,
                    image_path,
                )
                feedback = _first_nonempty_line(fp_raw) or feedback
            feedback = _normalize_self_reference(feedback, artist_name)
            next_action = _normalize_action_command(
                _extract_labeled_value(raw, "next_action")
                or _extract_labeled_value(raw, "action")
                or "",
                artist_name,
            )
            if not next_action or not _meaningful_next_action(next_action):
                action_raw = self._chat_text(
                    "Respond with exactly one line: NEXT_ACTION: <specific command with subject/composition/lighting/color detail>.",
                    f"vision:{vision}\niteration:{iteration}\nscore:{parsed_score}\nfeedback:{feedback}",
                    120,
                    image_path,
                )
                next_action = _normalize_action_command(action_raw, artist_name)
            if not feedback or not _meaningful_feedback(feedback):
                raise ValueError("Could not parse critique feedback.")
            if next_action and not _meaningful_next_action(next_action):
                next_action = ""
            return {"score": parsed_score, "feedback": feedback, "next_action": next_action}
        except Exception as exc:
            raise HostedCallError(f"Hosted critique failed: {exc}") from exc

    def judge_worthiness(self, image_path: str, score: int, vision: str, critique_frame: str = "") -> bool:
        try:
            raw = self._chat_text(
                "Decide if this artwork is worthy. Respond with one token only: YES or NO.",
                f"vision:{vision}\nscore:{score}\ncritique_frame:{critique_frame}",
                60,
                image_path,
            )
            worthy = _extract_yes_no(raw)
            if worthy is None:
                raise ValueError("Could not parse worthy decision.")
            return worthy
        except Exception as exc:
            raise HostedCallError(f"Hosted judgment failed: {exc}") from exc

    def generate_text_memory(self, soul_data: Dict, creation_result: Dict, trigger_reason: str) -> Dict:
        try:
            raw = self._chat_text(
                "Generate one text memory and respond using lines:\n"
                "CONTENT: <first-person text>\n"
                "IMPORTANCE: <critical|high|medium|low>\n"
                "TAGS: <comma separated tags>",
                f"trigger:{trigger_reason}\nresult:{creation_result}",
                240,
            )
            content = _extract_labeled_value(raw, "content") or _first_nonempty_line(raw)
            if content and not _contains_first_person(content):
                fp_raw = self._chat_text("Rewrite this memory in first person only.", f"content:{content}", 140)
                content = _first_nonempty_line(fp_raw) or content
            importance = _normalize_choice(_extract_labeled_value(raw, "importance"), ["critical", "high", "medium", "low"], "medium")
            tags_raw = _extract_labeled_value(raw, "tags")
            tags = _split_list_text(tags_raw, max_items=6)
            return safe_text_memory({"content": content, "importance": importance, "tags": tags}, soul_data)
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
            seed = current_name.strip() if current_name.strip() else "Unnamed Artist"
            name_raw = self._chat_text(
                "Return exactly one artist name line only.",
                f"Current name hint: {seed}",
                80,
            )
            obsession_raw = self._chat_text(
                "Return exactly one current obsession line only.",
                f"Artist name: {seed}",
                120,
            )
            traits_raw = self._chat_text(
                "Return 3 to 7 personality traits, comma-separated. No numbering.",
                f"Artist name: {seed}\nCurrent obsession hint: {_first_nonempty_line(obsession_raw)}",
                180,
            )
            name = _first_nonempty_line(name_raw) or seed
            clean_traits = _split_list_text(traits_raw, max_items=7)
            if len(clean_traits) < 3:
                raise ValueError("insufficient traits")
            obsession = _first_nonempty_line(obsession_raw)
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
                ACTION_VISION_CONTRACT,
                (
                    SOUL_CONTEXT_GUIDANCE
                    + FIRST_PERSON_HINT
                    + TIER_GUIDANCE_TEXT
                    + f"{_stage_weight_guidance(packet, 'vision')}\n"
                    + f"soul_packet:{packet}\npreferences:{prefs[-8:]}\nprinciples:{principles[-8:]}\ninstructions:{instructions[-8:]}\nrecent:{recent}\n"
                    + "Prefer meaningful variation in composition while preserving continuity with the artist's soul."
                ),
                90,
            )
            vision = _normalize_action_vision(text, soul)
            if not vision:
                raise ValueError("empty vision")
            return vision
        except Exception as exc:
            raise HostedCallError(f"Hosted vision fallback failed: {exc}") from exc

    def generate_run_intent(self, soul_data: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            out = self._chat_text(
                "Respond using three lines:\n"
                "VISION_DIRECTIVE: <imperative command for the next image prompt>\n"
                "CRITIQUE_DIRECTIVE: <imperative command for how to judge/score>\n"
                "REVISION_DIRECTIVE: <imperative command for what to revise in soul>\n"
                "Use first-person phrasing and never use the artist name.\n"
                "Do not output policy summaries.",
                (
                    FIRST_PERSON_HINT
                    + SOUL_CONTEXT_GUIDANCE
                    + TIER_GUIDANCE_TEXT
                    + f"{_stage_weight_guidance(packet, 'vision')}\n"
                    + f"soul_packet:{packet}"
                ),
                320,
            )
            vd = _extract_labeled_value(out, "vision_directive")
            cd = _extract_labeled_value(out, "critique_directive")
            rd = _extract_labeled_value(out, "revision_directive")
            return {"vision_directive": vd, "critique_directive": cd, "revision_directive": rd}
        except Exception as exc:
            raise HostedCallError(f"Hosted run intent generation failed: {exc}") from exc

    def generate_initial_render_prompt(self, soul_data: Dict, vision: str, run_intent: Optional[Dict] = None) -> str:
        try:
            packet = build_soul_packet(soul_data)
            directive = str((run_intent or {}).get("vision_directive", "")).strip()
            out = self._chat_text(
                "Create the initial image-generation prompt for this run.\n"
                "The run vision is fixed and must be executed directly.\n"
                "Return exactly one line in this format:\n"
                "IMAGE_PROMPT: <prompt text>\n"
                "Do not use the artist name; use first-person framing only if needed.",
                (
                    f"fixed_run_vision:{vision}\n"
                    f"vision_directive:{directive}\n"
                    f"{_stage_weight_guidance(packet, 'vision')}\n"
                    f"soul_packet:{packet}\n"
                    "Write a concrete visual prompt (subject, composition, medium/style, mood, lighting, palette)."
                ),
                280,
            )
            prompt = _extract_image_prompt(out, "", str(soul_data.get("name", "")).strip())
            if not prompt or not _is_usable_image_prompt(prompt) or not _is_prompt_aligned_with_vision(prompt, vision):
                retry = self._chat_text(
                    "Respond with exactly one line: IMAGE_PROMPT: <text>.\n"
                    "The prompt must concretely depict the run vision and include at least subject, composition, and mood.",
                    f"fixed_run_vision:{vision}\nvision_directive:{directive}\n",
                    120,
                )
                prompt = _extract_image_prompt(retry, "", str(soul_data.get("name", "")).strip())
            if not prompt or not _is_usable_image_prompt(prompt) or not _is_prompt_aligned_with_vision(prompt, vision):
                raise ValueError("empty initial image prompt")
            return prompt
        except Exception as exc:
            raise HostedCallError(f"Hosted initial prompt generation failed: {exc}") from exc

    def refine_render_prompt(
        self,
        current_prompt: str,
        vision: str,
        critique_feedback: str,
        score: int,
        soul_data: Dict,
        run_intent: Optional[Dict] = None,
    ) -> str:
        try:
            packet = build_soul_packet(soul_data)
            directive = str((run_intent or {}).get("vision_directive", "")).strip()
            out = self._chat_text(
                "You are revising only the iteration image prompt for the next attempt.\n"
                "The run vision is fixed and must not be rewritten.\n"
                "Apply the critique feedback as concrete visual editing commands.\n"
                "Make minimal, targeted edits to CURRENT_IMAGE_PROMPT.\n"
                "Return exactly one line in this format:\n"
                "IMAGE_PROMPT: <revised prompt text>",
                (
                    f"fixed_run_vision:{vision}\n"
                    f"current_image_prompt:{current_prompt}\n"
                    f"critique_feedback:{critique_feedback}\n"
                    f"score:{int(score)}\n"
                    f"{_stage_weight_guidance(packet, 'refinement')}\n"
                    f"vision_directive:{directive}\n"
                    f"soul_packet:{packet}\n"
                    "Keep continuity with the fixed run vision while improving the next image attempt."
                ),
                300,
            )
            next_prompt = _extract_image_prompt(out, current_prompt, str(soul_data.get("name", "")).strip())
            if (
                next_prompt == current_prompt
                or not _is_usable_image_prompt(next_prompt)
                or not _is_prompt_aligned_with_vision(next_prompt, vision)
            ):
                retry = self._chat_text(
                    "Respond with exactly one line: IMAGE_PROMPT: <text>",
                    (
                        f"fixed_run_vision:{vision}\n"
                        f"current_image_prompt:{current_prompt}\n"
                        f"critique_feedback:{critique_feedback}\n"
                        f"score:{int(score)}\n"
                    ),
                    120,
                )
                next_prompt = _extract_image_prompt(retry, current_prompt, str(soul_data.get("name", "")).strip())
            if not _is_usable_image_prompt(next_prompt) or not _is_prompt_aligned_with_vision(next_prompt, vision):
                return current_prompt
            return next_prompt or current_prompt
        except Exception as exc:
            raise HostedCallError(f"Hosted prompt refinement failed: {exc}") from exc

    def propose_state_revision(self, soul_data: Dict, creation_result: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            context = (
                SOUL_CONTEXT_GUIDANCE
                + TIER_GUIDANCE_TEXT
                + REVISION_ACTION_HINT
                + f"{_stage_weight_guidance(packet, 'revision')}\n"
                + f"soul_packet:{packet}\ncreation_result:{creation_result}"
            )
            revision: Dict = {}

            obsession_decision = self._chat_text(
                "Current obsession update.\nRespond with one line: KEEP or SET: <new obsession>",
                context,
                140,
            )
            obsession_line = _first_nonempty_line(obsession_decision)
            if obsession_line.lower().startswith("set:"):
                revision["obsession"] = obsession_line.split(":", 1)[1].strip()

            personality_mode_raw = self._chat_text(
                "Choose personality mode. Respond with one token only: keep, append, or replace.",
                context,
                60,
            )
            personality_mode = _normalize_choice(personality_mode_raw, ["keep", "append", "replace"], "keep")
            revision["personality_mode"] = personality_mode
            if personality_mode in ("append", "replace"):
                traits_raw = self._chat_text(
                    "List personality traits for this mode as comma-separated values.",
                    context,
                    220,
                )
                revision["personality_traits"] = _split_list_text(traits_raw, max_items=10)

            text_action_raw = self._chat_text(
                "Choose text memory action. Respond with one token only: none, add, edit_last, or delete_last.",
                context,
                80,
            )
            text_action = _normalize_choice(text_action_raw, ["none", "add", "edit_last", "delete_last"], "none")
            revision["text_memory_action"] = text_action
            if text_action in ("add", "edit_last"):
                text_mem_raw = self._chat_text(
                    "Respond with lines:\nCONTENT: <first-person text>\nIMPORTANCE: <critical|high|medium|low>\nTAGS: <comma separated tags>",
                    context,
                    260,
                )
                revision["text_memory"] = {
                    "content": _extract_labeled_value(text_mem_raw, "content") or _first_nonempty_line(text_mem_raw),
                    "importance": _normalize_choice(_extract_labeled_value(text_mem_raw, "importance"), ["critical", "high", "medium", "low"], "medium"),
                    "tags": _split_list_text(_extract_labeled_value(text_mem_raw, "tags"), max_items=8),
                }

            artwork_action_raw = self._chat_text(
                "Choose artwork memory action. Respond with one token only: none, annotate_last, or delete_last.",
                context,
                80,
            )
            artwork_action = _normalize_choice(artwork_action_raw, ["none", "annotate_last", "delete_last"], "none")
            revision["artwork_memory_action"] = artwork_action
            if artwork_action == "annotate_last":
                note_raw = self._chat_text(
                    "Provide one concise first-person artwork note line.",
                    context,
                    180,
                )
                revision["artwork_note"] = _first_nonempty_line(note_raw)

            return revision
        except Exception as exc:
            raise HostedCallError(f"Hosted state revision failed: {exc}") from exc

    def evaluate_review_merit(self, soul_data: Dict, review_payload: Dict) -> Dict:
        try:
            packet = build_soul_packet(soul_data)
            raw = self._chat_text(
                "Evaluate this external review and respond using lines:\n"
                "DECISION: <accept|partial|reject>\n"
                "RATIONALE: <one first-person sentence>\n"
                "MEMORY_CONTENT: <one actionable first-person note or blank>\n"
                "IMPORTANCE: <critical|high|medium|low>\n"
                "TAGS: <comma separated tags>\n"
                "OBSESSION_UPDATE: <blank or new obsession text>",
                (
                    FIRST_PERSON_HINT
                    + SOUL_CONTEXT_GUIDANCE
                    + f"{_stage_weight_guidance(packet, 'revision')}\n"
                    + f"soul_packet:{packet}\nreview_payload:{review_payload}"
                ),
                260,
            )
            decision = _normalize_choice(_extract_labeled_value(raw, "decision"), ["accept", "partial", "reject"], "reject")
            rationale = _extract_labeled_value(raw, "rationale") or _first_nonempty_line(raw)
            if rationale and not _contains_first_person(rationale):
                rewrite = self._chat_text("Rewrite rationale in first person. One sentence only.", f"rationale:{rationale}", 120)
                rationale = _first_nonempty_line(rewrite) or rationale
            memory_content = _extract_labeled_value(raw, "memory_content")
            importance = _normalize_choice(_extract_labeled_value(raw, "importance"), ["critical", "high", "medium", "low"], "medium")
            tags = _split_list_text(_extract_labeled_value(raw, "tags"), max_items=8)
            obsession_update = _extract_labeled_value(raw, "obsession_update")
            return {
                "decision": decision,
                "rationale": rationale,
                "memory_content": memory_content,
                "importance": importance,
                "tags": tags,
                "obsession_update": obsession_update,
            }
        except Exception as exc:
            raise HostedCallError(f"Hosted review-merit evaluation failed: {exc}") from exc


class CliLLMBackend(HostedLLMBackend):
    def __init__(self, cli: str, model: str = "", temperature: float = 0.2, trace_prompts: bool = False):
        cli_name, cli_model = _split_cli_model_spec(cli, model)
        super().__init__(
            provider=f"cli-{cli_name}",
            model=cli_model,
            api_key="",
            temperature=temperature,
            allow_fallback=False,
            trace_prompts=trace_prompts,
        )
        self.cli = cli_name

    def _chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 400, image_path: Optional[str] = None) -> str:
        _trace_prompt(self.trace_prompts, "cli.chat", self.provider, self.model or "(default)", system_prompt, user_prompt)
        artifact_note = ""
        if image_path:
            p = Path(image_path)
            if p.suffix.lower() == ".txt":
                snippet = p.read_text(encoding="utf-8", errors="replace")[:2000]
                artifact_note = f"\n\nASCII_ARTIFACT:\n{snippet}"
            elif p.exists():
                artifact_note = f"\n\nIMAGE_PATH: {str(p)}"
        prompt = f"{system_prompt}\n\n{user_prompt}{artifact_note}"
        try:
            return _run_cli_text(self.cli, self.model, prompt, timeout=240)
        except Exception as exc:
            raise HostedCallError(f"CLI LLM call failed ({self.cli}): {exc}") from exc


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
            ACTION_VISION_CONTRACT
            + "\n\n"
            + SOUL_CONTEXT_GUIDANCE
            + FIRST_PERSON_HINT
            + TIER_GUIDANCE_TEXT
            + f"{_stage_weight_guidance(packet, 'vision')}\n"
            + f"soul_packet:{packet}\n"
            + f"preferences:{preferences[-8:]}\n"
            + f"principles:{principles[-8:]}\n"
            + f"instructions:{instructions[-8:]}\n"
            + f"recent:{recent_visions}\n"
            + "Prefer subtle variation while preserving continuity with the artist's soul."
        )
        try:
            _trace_prompt(self.trace_prompts, "vision.generate", self.provider, self.model, ACTION_VISION_CONTRACT, prompt)
            raw = _ollama_generate_text(self.base_url, self.model, prompt, self.temperature, timeout=60)
            candidate = _normalize_action_vision(raw, soul)
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
            raw = self._chat_text(
                ACTION_VISION_CONTRACT,
                FIRST_PERSON_HINT
                + SOUL_CONTEXT_GUIDANCE
                + TIER_GUIDANCE_TEXT
                + f"{_stage_weight_guidance(packet, 'vision')}\n"
                + f"soul_packet:{packet}\npreferences:{preferences[-8:]}\nprinciples:{principles[-8:]}\ninstructions:{instructions[-8:]}\nrecent:{recent_visions}\n"
                + "Prefer subtle variation while preserving continuity with the artist's soul.",
                90,
            )
            candidate = _normalize_action_vision(raw, soul)
            _, _, _, prioritized = infer_guidance(text_memories)
            if memory_collision(parse_vision(candidate), memories[-5:], prioritized):
                raise HostedCallError("Hosted vision collided with recent pattern.")
            print(f"\n  New Vision: \"{candidate}\"")
            return candidate
        except Exception as exc:
            raise HostedCallError(f"Hosted vision generation failed: {exc}") from exc


class CliVisionBackend(HostedVisionBackend):
    def __init__(self, cli: str, model: str = "", temperature: float = 0.4, trace_prompts: bool = False):
        cli_name, cli_model = _split_cli_model_spec(cli, model)
        super().__init__(
            provider=f"cli-{cli_name}",
            model=cli_model,
            api_key="",
            temperature=temperature,
            allow_fallback=False,
            trace_prompts=trace_prompts,
        )
        self.cli = cli_name

    def _chat_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 200) -> str:
        _trace_prompt(self.trace_prompts, "vision.generate.cli", self.provider, self.model or "(default)", system_prompt, user_prompt)
        prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            return _run_cli_text(self.cli, self.model, prompt, timeout=240)
        except Exception as exc:
            raise HostedCallError(f"CLI vision call failed ({self.cli}): {exc}") from exc



