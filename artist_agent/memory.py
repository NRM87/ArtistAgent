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


def vision_to_prompt(vision: str, soul: Optional[Dict] = None) -> str:
    parsed = parse_vision(vision)
    composition = "asymmetric composition" if "off-center" in parsed.relation or "diagonal" in parsed.relation else "balanced composition"

    style = "expressive composition"
    if soul is not None:
        text_memories = soul.get("text_memories", []) or []
        style_mem = next((m for m in reversed(text_memories) if "style" in (m.get("tags", []) or [])), None)
        if style_mem and style_mem.get("content"):
            style = str(style_mem["content"])
        elif any("melancholic" in str(t).lower() for t in soul.get("personality_traits", [])):
            style = "moody cinematic atmosphere"

    return f"{style}: {parsed.color} {parsed.subject}, {parsed.relation}, {composition}"


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
    by_tier = {k: [] for k in TEXT_TIER_CAPS}
    other = []
    for m in sorted(text_memories, key=lambda x: x.get("timestamp", "")):
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
