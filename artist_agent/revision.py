import json
from pathlib import Path
from typing import Dict, List, Tuple

from .memory import consolidate_text_memories, safe_text_memory

MAX_OBSESSION_CHARS = 200
MAX_TRAIT_CHARS = 80
MAX_REVISION_NOTE_CHARS = 300
MAX_DIRECTIVE_CHARS = 240
MAX_TRAITS = 10


def normalize_run_intent(intent: Dict) -> Dict:
    if not isinstance(intent, dict):
        return {"vision_directive": "", "critique_directive": "", "revision_directive": ""}

    return {
        "vision_directive": str(intent.get("vision_directive", "")).strip()[:MAX_DIRECTIVE_CHARS],
        "critique_directive": str(intent.get("critique_directive", "")).strip()[:MAX_DIRECTIVE_CHARS],
        "revision_directive": str(intent.get("revision_directive", "")).strip()[:MAX_DIRECTIVE_CHARS],
    }


def proposes_identity_change(revision: Dict, soul: Dict) -> bool:
    if not isinstance(revision, dict):
        return False
    current_obsession = str(soul.get("current_obsession", "")).strip()
    new_obsession = str(revision.get("obsession", "")).strip()
    if new_obsession and new_obsession != current_obsession:
        return True

    mode = str(revision.get("personality_mode", "keep")).strip().lower()
    raw = revision.get("personality_traits", [])
    if not isinstance(raw, list):
        return False
    new_traits = [str(t).strip() for t in raw if str(t).strip()]
    current_traits = [str(t).strip() for t in list(soul.get("personality_traits", []) or []) if str(t).strip()]
    if mode == "replace" and new_traits and new_traits != current_traits:
        return True
    if mode == "append":
        for trait in new_traits:
            if trait not in current_traits:
                return True
    return False


def format_compact_json(value: Dict) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    except Exception:
        return "{}"


def revision_summary_lines(revision: Dict) -> List[str]:
    if not isinstance(revision, dict) or not revision:
        return ["Self-Revision Applied:", "  (no changes)"]
    lines = ["Self-Revision Applied:"]
    obsession = str(revision.get("obsession", "")).strip()
    if obsession:
        lines.append(f"  obsession -> {obsession}")
    lines.append(f"  personality_mode -> {revision.get('personality_mode', 'keep')}")
    lines.append(f"  text_memory_action -> {revision.get('text_memory_action', 'none')}")
    lines.append(f"  artwork_memory_action -> {revision.get('artwork_memory_action', 'none')}")
    return lines


def _clean_traits(raw_traits: object) -> List[str]:
    if not isinstance(raw_traits, list):
        return []
    out = []
    for value in raw_traits:
        trait = str(value).strip()
        if not trait:
            continue
        trait = trait[:MAX_TRAIT_CHARS]
        if trait not in out:
            out.append(trait)
        if len(out) >= MAX_TRAITS:
            break
    return out


def _safe_delete_artwork_file(artwork: Dict, artist_dir: Path) -> bool:
    file_path = str(artwork.get("file_path", "")).strip()
    if not file_path:
        return False

    artist_root = artist_dir.resolve()
    candidates: List[Path] = []
    p = Path(file_path)
    if p.is_absolute():
        candidates.append(p.resolve())
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((artist_root / p).resolve())

    for candidate in candidates:
        if candidate.exists() and (candidate == artist_root or artist_root in candidate.parents):
            try:
                candidate.unlink(missing_ok=True)
                return True
            except Exception:
                return False
    return False


def apply_state_revision(soul: Dict, revision: Dict, artist_dir: Path) -> Tuple[Dict, Dict]:
    if not isinstance(revision, dict):
        return {}, {"deleted_artwork_file": False, "identity_changed": False}

    old_obsession = str(soul.get("current_obsession", "")).strip()
    old_traits = [str(t).strip() for t in list(soul.get("personality_traits", []) or []) if str(t).strip()]

    obsession = str(revision.get("obsession", "")).strip()
    if obsession:
        soul["current_obsession"] = obsession[:MAX_OBSESSION_CHARS]

    mode = str(revision.get("personality_mode", "keep")).strip().lower()
    traits = _clean_traits(revision.get("personality_traits", []))
    if mode == "replace" and traits:
        soul["personality_traits"] = traits
    elif mode == "append" and traits:
        current = _clean_traits(list(soul.get("personality_traits", [])))
        for trait in traits:
            if trait not in current:
                current.append(trait)
            if len(current) >= MAX_TRAITS:
                break
        soul["personality_traits"] = current

    text_action = str(revision.get("text_memory_action", "none")).strip().lower()
    if text_action == "delete_last" and soul.get("text_memories"):
        soul["text_memories"] = list(soul.get("text_memories", []))[:-1]
    elif text_action in ("add", "edit_last"):
        candidate = revision.get("text_memory", {})
        if isinstance(candidate, dict):
            safe = safe_text_memory(candidate, soul)
            if text_action == "edit_last" and soul.get("text_memories"):
                soul["text_memories"][-1] = safe
            else:
                soul.setdefault("text_memories", []).append(safe)
            soul["text_memories"] = consolidate_text_memories(soul.get("text_memories", []))

    deleted_file = False
    artwork_action = str(revision.get("artwork_memory_action", "none")).strip().lower()
    if artwork_action == "delete_last" and soul.get("memories"):
        last = soul["memories"][-1]
        if isinstance(last, dict):
            deleted_file = _safe_delete_artwork_file(last, artist_dir)
        soul["memories"] = list(soul.get("memories", []))[:-1]
    elif artwork_action == "annotate_last" and soul.get("memories"):
        note = str(revision.get("artwork_note", "")).strip()
        if note:
            soul["memories"][-1]["self_note"] = note[:MAX_REVISION_NOTE_CHARS]

    new_obsession = str(soul.get("current_obsession", "")).strip()
    new_traits = [str(t).strip() for t in list(soul.get("personality_traits", []) or []) if str(t).strip()]
    identity_changed = (new_obsession != old_obsession) or (new_traits != old_traits)
    return revision, {"deleted_artwork_file": deleted_file, "identity_changed": identity_changed}
