import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import (
    IMPORTANCE_SYMBOL,
    MAX_ARTWORK_MEMORIES,
    MAX_TEXT_MEMORIES,
    TEXT_TIER_CAPS,
    ParsedVision,
    now_iso,
)


def parse_ignore_ids(text_memories: List[Dict]) -> set:
    ignored = set()
    for m in text_memories:
        content = str(m.get("content", ""))
        tags = m.get("tags", []) or []
        if "meta" in tags and "IGNORE" in content.upper():
            for found in re.findall(r"#(\d+)", content):
                ignored.add(int(found))
    return ignored


def artwork_tier_from_score(score: int) -> str:
    s = int(score)
    if s >= 8:
        return "masterpiece"
    if s >= 5:
        return "study"
    return "failure"


def parse_vision(vision: str) -> ParsedVision:
    text = vision.lower()
    words = [w for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 1]
    stop = {"the", "and", "with", "from", "into", "over", "under", "through", "that", "this", "these", "those", "for", "your", "their"}
    content = [w for w in words if w not in stop]
    subject = content[0] if content else "scene"
    color_words = {"red", "orange", "yellow", "green", "blue", "purple", "violet", "indigo", "white", "black", "gray", "grey", "amber", "gold", "silver", "crimson", "azure", "emerald", "teal", "rose"}
    color = next((w for w in content if w in color_words), "unspecified")
    relation = text.strip() if text.strip() else "untitled scene"
    return ParsedVision(subject=subject, color=color, relation=relation)


def summarize_mood(traits: List[str]) -> str:
    return "Uncertain" if not traits else ", ".join(traits[-3:])


def print_reflection(soul: Dict, ignored_ids: set) -> None:
    awakening = soul["creation_count"] + 1
    print("=" * 58)
    print(f"AWAKENING #{awakening}")
    print("=" * 58)
    print(f"\n{str(soul['name'])} awakens once more.\n")

    artworks = [m for m in soul["memories"] if m.get("id") not in ignored_ids]
    tier_counts = {"masterpiece": 0, "study": 0, "failure": 0}
    for m in artworks:
        tier = str(m.get("tier", artwork_tier_from_score(int(m.get("final_score", 0))))).strip().lower()
        if tier in tier_counts:
            tier_counts[tier] += 1
    print("Reviewing memories...")
    print(
        f"  Artwork memories: {len(artworks)} preserved "
        f"(masterpieces={tier_counts['masterpiece']}, studies={tier_counts['study']}, failures={tier_counts['failure']})"
    )
    for m in artworks[-2:]:
        file_name = Path(m.get("file_path", "unknown")).name
        print(f"  - {file_name}: \"{m.get('vision', 'Unknown')}\" ({m.get('final_score', '?')}/10)")

    tmem = soul.get("text_memories", [])
    print(f"  Text memories: {len(tmem)} notes")
    for m in tmem[-5:]:
        importance = m.get("importance", "low")
        symbol = IMPORTANCE_SYMBOL.get(importance, "[.]")
        print(f"  {symbol} \"{m.get('content', '')}\"")

    print(f"\nCurrent obsession: {soul.get('current_obsession', '')}")
    print(f"Dominant mood: {summarize_mood(soul.get('personality_traits', []))}")


def infer_guidance(text_memories: List[Dict]) -> Tuple[List[str], List[str], List[str], set]:
    preferences, principles, instructions, prioritized_subjects = [], [], [], set()
    for m in text_memories:
        tags = m.get("tags", []) or []
        content = str(m.get("content", ""))
        lc = content.lower()

        if "preference" in tags:
            preferences.append(content)
            words = [w for w in re.findall(r"[a-zA-Z']+", lc) if len(w) > 3]
            prioritized_subjects.update(words[:3])
        if "principle" in tags or "learning" in tags:
            principles.append(content)
        if "meta" in tags and "IGNORE" in content.upper():
            instructions.append(content)
    return preferences, principles, instructions, prioritized_subjects


def extract_hard_constraints(text_memories: List[Dict], limit: int = 6) -> List[str]:
    constraints: List[str] = []
    seen = set()
    for mem in reversed(text_memories):
        content = str(mem.get("content", "")).strip()
        if not content:
            continue
        tags = [str(t).lower().strip() for t in (mem.get("tags", []) or []) if str(t).strip()]
        importance = str(mem.get("importance", "medium")).lower().strip()
        lc = content.lower()
        is_restrictive = bool(
            re.search(r"\b(only|never|must|always|do not|don't|should not|cannot|can't)\b", lc)
        )
        if importance != "critical" and not is_restrictive:
            continue
        if not (is_restrictive or any(t in tags for t in ("constraint", "judgment", "principle", "rule"))):
            continue
        norm = re.sub(r"\s+", " ", content).strip()
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        constraints.append(norm)
        if len(constraints) >= limit:
            break
    constraints.reverse()
    return constraints


def directive_conflicts_hard_constraints(directive: str, hard_constraints: List[str]) -> bool:
    d = str(directive).strip().lower()
    if not d or not hard_constraints:
        return False
    joined = " ".join(hard_constraints).lower()
    restrictive = bool(re.search(r"\b(only|never|must|always|do not|don't|should not|cannot|can't)\b", joined))
    if not restrictive:
        return False
    if re.search(r"\b(challenge|break|subvert|ignore|abandon|deviate|reject|violate|contradict|discard)\b", d):
        return True
    if re.search(
        r"\b(wider variety|wider range|broader range|broader|diverse|more shapes|other geometric|broaden|expand scope|expanding scope|integrat)\b",
        d,
    ):
        return True
    if "non-square" in d and "square" in joined:
        return True
    if "square" in joined and re.search(r"\b(triangle|circle|organic curve|curves)\b", d):
        return True
    return False


def sanitize_vision_against_constraints(vision: str, hard_constraints: List[str]) -> str:
    v = str(vision).strip()
    if not v:
        return v
    constraints = [str(c).strip().lower() for c in hard_constraints if str(c).strip()]
    joined = " ".join(constraints)
    restrictive = bool(re.search(r"\b(only|never|must|always|do not|don't|should not|cannot|can't)\b", joined))
    if not restrictive or "square" not in joined:
        return v

    out = v
    out = re.sub(
        r"\b(despite|while|although|but|however)\b[^.]*\b(broader|wider|diverse|variety|other geometric|non-square|integrat)\b[^.]*\.?",
        "",
        out,
        flags=re.I,
    )
    out = re.sub(
        r"\b(broader range of geometric shapes|wider range of geometric shapes|wider variety of geometric shapes|broader geometric forms)\b",
        "subtle variation in square and near-square arrangements",
        out,
        flags=re.I,
    )
    if re.search(r"\b(wider|broader|diverse|variety|other geometric|integrat)\b", out, flags=re.I):
        out = re.sub(r"[,:;]\s*\b(wider|broader|diverse|variety|other geometric|integrat)\b.*", "", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip(" ,.;")
    if "square" not in out.lower():
        out = f"{out}. Focus on square and near-square forms."
    if not out.endswith("."):
        out += "."
    return out


def memory_collision(new_parsed: ParsedVision, last_memories: List[Dict], prioritized_subjects: set) -> bool:
    new_tokens = set(re.findall(r"[a-zA-Z']+", new_parsed.relation.lower()))
    for m in last_memories:
        parsed = parse_vision(str(m.get("vision", "")))
        old_tokens = set(re.findall(r"[a-zA-Z']+", parsed.relation.lower()))
        overlap = len(new_tokens & old_tokens)
        same_subject = parsed.subject == new_parsed.subject and parsed.subject != "scene"
        if same_subject and overlap >= 4:
            return True
        if new_parsed.subject not in prioritized_subjects and same_subject and overlap >= 2:
            return True
    return False


def vision_to_prompt(
    vision: str,
    soul: Optional[Dict] = None,
    vision_directive: str = "",
    hard_constraints: Optional[List[str]] = None,
) -> str:
    v = str(vision).strip() or "Create a distinctive visual composition."
    traits: List[str] = []
    obsession = ""
    anchors: List[str] = []
    if soul is not None:
        traits = [str(t).strip() for t in (soul.get("personality_traits", []) or []) if str(t).strip()][:6]
        obsession = str(soul.get("current_obsession", "")).strip()
        text_memories = list(soul.get("text_memories", []) or [])
        for mem in reversed(text_memories):
            content = str(mem.get("content", "")).strip()
            if not content:
                continue
            tags = [str(t).lower().strip() for t in (mem.get("tags", []) or []) if str(t).strip()]
            if any(t in tags for t in ("principle", "preference", "learning", "judgment")):
                anchors.append(content)
            if len(anchors) >= 3:
                break

    lines: List[str] = [f'Create a coherent 2D composition from this vision: "{v}"']
    if traits:
        lines.append(f"Artist traits: {', '.join(traits)}.")
    if obsession:
        lines.append(f"Current obsession: {obsession}.")
    if anchors:
        lines.append("Memory anchors:")
        for a in reversed(anchors):
            lines.append(f"- {a}")
    hc = [str(x).strip() for x in (hard_constraints or []) if str(x).strip()]
    if hc:
        lines.append("Non-negotiable constraints (override exploratory directives):")
        for rule in hc:
            lines.append(f"- {rule}")
    vd = str(vision_directive).strip()
    if vd:
        lines.append(f"Exploration directive (must respect constraints): {vd}")
    lines.append("Prioritize clear composition and visual intent over decorative noise.")
    return "\n".join(lines)


def next_text_memory_id(soul: Dict) -> int:
    existing = [m.get("id", 0) for m in soul.get("text_memories", []) if isinstance(m.get("id", None), int)]
    return (max(existing) if existing else 0) + 1


def assign_importance(content: str) -> str:
    c = content.lower()
    if any(k in c for k in ["ignore", "never", "always", "contradicts"]):
        return "critical"
    if any(k in c for k in ["learned", "discovered", "principle", "pattern", "confirm"]):
        return "high"
    if any(k in c for k in ["like", "prefer", "enjoy", "feels"]):
        return "medium"
    if any(k in c for k in ["wonder", "perhaps", "maybe", "question"]):
        return "low"
    return "medium"


def safe_text_memory(memory: Dict, soul_data: Dict) -> Dict:
    if not isinstance(memory, dict):
        memory = {}
    content = str(memory.get("content", "")).strip() or "I need to reflect further before drawing conclusions."
    importance = str(memory.get("importance", "medium")).lower()
    if importance not in TEXT_TIER_CAPS:
        importance = assign_importance(content)
    tags = memory.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).lower().strip() for t in tags if str(t).strip()]
    return {
        "type": "text",
        "id": next_text_memory_id(soul_data),
        "content": content,
        "importance": importance,
        "timestamp": now_iso(),
        "tags": tags[:8],
    }


def consolidate_text_memories(text_memories: List[Dict]) -> List[Dict]:
    # Drop duplicate memory statements (latest wins) before tier caps.
    # This reduces drift where repeated revision edits clone the same note.
    latest_by_content = {}
    for m in sorted(text_memories, key=lambda x: x.get("timestamp", "")):
        key = re.sub(r"\s+", " ", str(m.get("content", "")).strip().lower())
        if not key:
            continue
        latest_by_content[key] = m
    deduped = list(latest_by_content.values())

    by_tier = {k: [] for k in TEXT_TIER_CAPS}
    other = []
    for m in sorted(deduped, key=lambda x: x.get("timestamp", "")):
        tier = m.get("importance", "low")
        (by_tier[tier] if tier in by_tier else other).append(m)

    consolidated = []
    for tier in ["critical", "high", "medium", "low"]:
        consolidated.extend(by_tier[tier][-TEXT_TIER_CAPS[tier]:])
    consolidated.extend(other[-max(0, MAX_TEXT_MEMORIES - len(consolidated)):])
    consolidated = sorted(consolidated, key=lambda x: x.get("timestamp", ""))
    return consolidated[-MAX_TEXT_MEMORIES:] if len(consolidated) > MAX_TEXT_MEMORIES else consolidated


def trim_artwork_memories(memories: List[Dict]) -> List[Dict]:
    if len(memories) <= MAX_ARTWORK_MEMORIES:
        return memories
    items = sorted(memories, key=lambda x: x.get("timestamp", ""))
    tier_targets = {"masterpiece": 10, "study": 6, "failure": 4}

    def _tier(mem: Dict) -> str:
        return str(mem.get("tier", artwork_tier_from_score(int(mem.get("final_score", 0))))).strip().lower()

    while len(items) > MAX_ARTWORK_MEMORIES:
        buckets = {"masterpiece": [], "study": [], "failure": []}
        for idx, mem in enumerate(items):
            t = _tier(mem)
            if t not in buckets:
                t = artwork_tier_from_score(int(mem.get("final_score", 0)))
            buckets[t].append((idx, mem))

        excess = {t: len(buckets[t]) - tier_targets[t] for t in buckets}
        removable_tier = max(excess, key=lambda t: excess[t])
        if excess[removable_tier] <= 0:
            removable_tier = max(buckets.keys(), key=lambda t: len(buckets[t]))

        candidates = buckets[removable_tier]
        if removable_tier == "masterpiece":
            # When trimming masterpieces, evict oldest first.
            drop_idx = sorted(candidates, key=lambda p: p[1].get("timestamp", ""))[0][0]
        else:
            # For study/failure tiers, evict the weakest oldest examples first.
            drop_idx = sorted(
                candidates,
                key=lambda p: (int(p[1].get("final_score", 0)), p[1].get("timestamp", "")),
            )[0][0]
        items.pop(drop_idx)

    return sorted(items, key=lambda x: x.get("timestamp", ""))
