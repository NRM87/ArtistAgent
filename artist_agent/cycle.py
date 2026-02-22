import json
from pathlib import Path
from typing import Dict, List, Optional
import re

from .constants import HostedCallError, now_iso
from .memory import (
    artwork_tier_from_score,
    consolidate_text_memories,
    parse_ignore_ids,
    parse_vision,
    print_reflection,
    safe_text_memory,
    trim_artwork_memories,
)
from .runtime import (
    backend_label,
    build_image_backend,
    build_llm_backend,
    build_vision_backend,
    handle_management_command,
    parse_args,
    resolve_artist_runtime,
    validate_backend_choices,
)
from .revision import (
    apply_state_revision,
    format_compact_json,
    normalize_run_intent,
    revision_summary_lines,
)
from .state import (
    acquire_lock,
    apply_artist_manifest_to_soul,
    atomic_write_json,
    clear_temp,
    cleanup_gallery_orphans,
    ensure_dirs,
    ensure_review_dirs,
    archive_inbox_review,
    list_review_candidates,
    load_config_file,
    load_inbox_reviews,
    load_dotenv,
    load_memory_sources,
    load_soul,
    move_to_gallery,
    persist_cross_artist_review,
    release_lock,
)


def build_render_prompt(vision: str, image_prompt: str) -> str:
    v = str(vision).strip() or "Create a coherent 2D composition."
    ip = str(image_prompt).strip() or v
    lines = [
        f'Run vision (fixed for this run): "{v}"',
        f'Iteration image prompt: "{ip}"',
        "Create a coherent 2D composition using the iteration image prompt while staying faithful to the fixed run vision.",
    ]
    return "\n".join(lines)


def record_deferred_cycle(soul: Dict, creation_id: int, reason: str) -> None:
    soul["creation_count"] = creation_id
    soul["cycle_history"].append({
        "timestamp": now_iso(),
        "score": 0,
        "worthy": False,
        "subject": "deferred",
        "color": "none",
        "relation": "none",
        "vision": f"DEFERRED: {reason}",
    })
    soul["cycle_history"] = soul["cycle_history"][-40:]


def defer_and_persist(soul: Dict, creation_id: int, reason: str, soul_path: Path, temp_dir: Optional[Path] = None) -> None:
    # Single defer path keeps failure persistence consistent across backend stages.
    print(f"Deferred: {reason}")
    record_deferred_cycle(soul, creation_id, reason)
    atomic_write_json(soul_path, soul)
    if temp_dir is not None:
        clear_temp(temp_dir)


def ensure_llm_identity(soul: Dict, llm_backend) -> None:
    has_traits = isinstance(soul.get("personality_traits"), list) and len(soul.get("personality_traits")) > 0
    has_obsession = bool(str(soul.get("current_obsession", "")).strip())
    if has_traits and has_obsession:
        return
    identity = llm_backend.generate_identity(str(soul.get("name", "")).strip())
    soul["name"] = str(identity.get("name", soul.get("name", "Unnamed Artist"))).strip() or "Unnamed Artist"
    soul["personality_traits"] = [str(t).strip() for t in identity.get("personality_traits", []) if str(t).strip()]
    soul["current_obsession"] = str(identity.get("current_obsession", "")).strip()
    print("Generated artist identity from LLM.")


def build_critique_frame(soul: Dict, run_intent: Dict) -> str:
    artist_name = str(soul.get("name", "")).strip()
    traits = [str(t).strip() for t in soul.get("personality_traits", []) if str(t).strip()][:10]
    obsession = str(soul.get("current_obsession", "")).strip()
    text_mem = soul.get("text_memories", [])[-10:]
    art_mem = soul.get("memories", [])[-8:]
    history = soul.get("cycle_history", [])[-8:]
    reflection_weights = soul.get("reflection_weights", {})
    return (
        f"artist_name:{artist_name}\n"
        f"personality_traits:{traits}\n"
        f"current_obsession:{obsession}\n"
        f"text_memories:{text_mem}\n"
        f"artwork_memories:{art_mem}\n"
        f"recent_history:{history}\n"
        f"reflection_weights:{reflection_weights}\n"
        f"critique_directive:{str(run_intent.get('critique_directive', '')).strip()}\n"
        "Use this context for subjective evaluation according to the artist's own values. "
        "Respond in first person and include one concrete next-action command."
    )


def analyze_ascii_artifact(path: Path) -> Dict:
    result = {"flags": [], "summary": ""}
    if path.suffix.lower() != ".txt" or not path.exists():
        return result
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        begin = lines.index("BEGIN_ASCII")
        end = lines.index("END_ASCII")
        canvas = lines[begin + 1 : end]
    except Exception:
        return result
    if not canvas:
        return result

    def _line_has_readable_text(line: str) -> bool:
        token_count = len(re.findall(r"[A-Za-z]{3,}", line))
        alpha_count = sum(1 for ch in line if ch.isalpha())
        return token_count >= 2 or alpha_count >= 12

    width = max(len(row) for row in canvas) if canvas else 0
    height = len(canvas)
    if width <= 0 or height <= 0:
        return result
    coords = []
    for y, row in enumerate(canvas):
        for x, ch in enumerate(row):
            if ch != " ":
                coords.append((x, y))
    if not coords:
        result["flags"] = ["empty_canvas"]
        result["summary"] = f"canvas={width}x{height}, ink_ratio=0.0, bbox_ratio=0.0"
        return result

    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    total = max(1, width * height)
    ink_ratio = len(coords) / total
    bbox_ratio = (bbox_w * bbox_h) / total
    cx = ((x0 + x1) / 2.0 + 1) / width
    cy = ((y0 + y1) / 2.0 + 1) / height

    flags = []
    if any(_line_has_readable_text(row) for row in canvas):
        flags.append("readable_text")
    if ink_ratio < 0.006:
        flags.append("sparse_output")
    if bbox_ratio < 0.11 and ink_ratio < 0.08:
        flags.append("too_small")
    width_coverage = bbox_w / max(1, width)
    if cx < 0.34 and width_coverage < 0.55:
        flags.append("top_left_collapse")

    result["flags"] = flags
    result["summary"] = (
        f"canvas={width}x{height}, ink_ratio={ink_ratio:.3f}, bbox_ratio={bbox_ratio:.3f}, "
        f"bbox={bbox_w}x{bbox_h}@({x0},{y0}), center=({cx:.2f},{cy:.2f})"
    )
    return result


def _weighted_soul_context(soul: Dict, reflection_weights: Dict[str, float]) -> Dict:
    out = json.loads(json.dumps(soul))
    out["reflection_weights"] = dict(reflection_weights or {})
    return out


def _review_key(target_artist: str, target_artwork_id: int) -> str:
    return f"{str(target_artist).strip()}:{int(target_artwork_id)}"


def _authored_review_keys(soul: Dict) -> set:
    out = set()
    for row in list(soul.get("review_history", []) or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("kind", "")).strip().lower() != "authored":
            continue
        key = str(row.get("target_key", "")).strip()
        if key:
            out.add(key)
    return out


def _build_review_critique_frame(soul: Dict, runtime, candidate: Dict) -> str:
    traits = [str(t).strip() for t in soul.get("personality_traits", []) if str(t).strip()][:10]
    obsession = str(soul.get("current_obsession", "")).strip()
    weights = dict(runtime.reflection_weights or {})
    return (
        f"reviewer_traits:{traits}\n"
        f"reviewer_obsession:{obsession}\n"
        f"review_target_artist:{candidate.get('target_artist', '')}\n"
        f"review_target_artwork_id:{candidate.get('target_artwork_id', 0)}\n"
        f"review_target_vision:{candidate.get('target_vision', '')}\n"
        f"reflection_weights:{weights}\n"
        "Write critique as first-person review feedback with one actionable next step."
    )


def perform_cross_artist_reviews(runtime, soul: Dict, llm_backend) -> int:
    if runtime.reviews_per_run <= 0:
        return 0
    ensure_review_dirs(runtime.artist_dir)
    authored = _authored_review_keys(soul)
    candidates = list_review_candidates(
        runtime.artists_dir,
        reviewer_artist_id=runtime.artist_id,
        review_targets=runtime.review_targets,
        authored_keys=authored,
        limit=max(1, runtime.reviews_per_run * 5),
    )
    if not candidates:
        print("Cross-review: no eligible external artworks found.")
        return 0

    completed = 0
    for candidate in candidates:
        if completed >= runtime.reviews_per_run:
            break
        target_key = str(candidate.get("review_key", "")).strip()
        if not target_key or target_key in authored:
            continue
        image_path = str(candidate.get("target_artifact_path", "")).strip()
        if not image_path:
            continue
        frame = _build_review_critique_frame(soul, runtime, candidate)
        try:
            critique = llm_backend.critique(image_path, str(candidate.get("target_vision", "")).strip(), 0, frame)
        except HostedCallError as exc:
            print(f"Cross-review skipped for {target_key}: {exc}")
            continue

        score = int(critique.get("score", 0))
        feedback = str(critique.get("feedback", "")).strip()
        suggestion = str(critique.get("next_action", "")).strip()
        stance = "accept" if score >= 8 else ("partial" if score >= 5 else "reject")
        payload = {
            "review_id": f"rvw_{runtime.artist_id}_{candidate.get('target_artist', 'unknown')}_{int(candidate.get('target_artwork_id', 0)):04d}_{now_iso().replace(':', '').replace('-', '')}",
            "timestamp": now_iso(),
            "author_artist": runtime.artist_id,
            "target_artist": str(candidate.get("target_artist", "")).strip(),
            "target_artwork_id": int(candidate.get("target_artwork_id", 0)),
            "target_artifact_path": image_path,
            "target_vision": str(candidate.get("target_vision", "")).strip(),
            "score": score,
            "stance": stance,
            "feedback": feedback,
            "suggestion": suggestion,
        }
        try:
            persist_cross_artist_review(runtime.artists_dir, runtime.artist_id, payload["target_artist"], payload)
            soul.setdefault("review_history", []).append(
                {
                    "kind": "authored",
                    "timestamp": now_iso(),
                    "target_key": target_key,
                    "review_id": payload["review_id"],
                    "target_artist": payload["target_artist"],
                    "target_artwork_id": payload["target_artwork_id"],
                    "score": score,
                    "stance": stance,
                }
            )
            authored.add(target_key)
            completed += 1
            print(f"Cross-review authored: {payload['target_artist']}#{payload['target_artwork_id']} ({score}/10, stance={stance})")
        except Exception as exc:
            print(f"Cross-review persistence failed for {target_key}: {exc}")
    soul["review_history"] = list(soul.get("review_history", []))[-120:]
    return completed


def ingest_reviews(runtime, soul: Dict, llm_backend) -> int:
    if runtime.review_ingest_limit <= 0:
        return 0
    ensure_review_dirs(runtime.artist_dir)
    inbox_items = load_inbox_reviews(runtime.artist_dir, limit=runtime.review_ingest_limit)
    if not inbox_items:
        print("Review ingestion: no incoming reviews.")
        return 0

    processed = 0
    for inbox_path, review in inbox_items:
        decision_payload = {
            "decision": "reject",
            "rationale": "I reject this review because I cannot evaluate it right now.",
            "memory_content": "",
            "importance": "low",
            "tags": ["review", "reject"],
            "obsession_update": "",
        }
        try:
            decision_payload = llm_backend.evaluate_review_merit(_weighted_soul_context(soul, runtime.reflection_weights), review)
        except HostedCallError as exc:
            print(f"Review merit evaluation failed for {inbox_path.name}: {exc}")

        decision = str(decision_payload.get("decision", "reject")).strip().lower()
        if decision not in ("accept", "partial", "reject"):
            decision = "reject"
        rationale = str(decision_payload.get("rationale", "")).strip()
        memory_content = str(decision_payload.get("memory_content", "")).strip()
        importance = str(decision_payload.get("importance", "medium")).strip().lower()
        tags = [str(t).strip().lower() for t in (decision_payload.get("tags", []) or []) if str(t).strip()]
        obsession_update = str(decision_payload.get("obsession_update", "")).strip()

        if decision in ("accept", "partial"):
            if not memory_content:
                memory_content = str(review.get("suggestion", "")).strip() or str(review.get("feedback", "")).strip()
            if memory_content:
                safe = safe_text_memory(
                    {
                        "content": memory_content,
                        "importance": importance,
                        "tags": list(dict.fromkeys(tags + ["review", "external_feedback", decision]))[:8],
                    },
                    soul,
                )
                soul.setdefault("text_memories", []).append(safe)
                soul["text_memories"] = consolidate_text_memories(soul.get("text_memories", []))
            if obsession_update:
                soul["current_obsession"] = obsession_update[:200]

        review_with_decision = dict(review)
        review_with_decision["processed_at"] = now_iso()
        review_with_decision["review_merit"] = {
            "decision": decision,
            "rationale": rationale,
            "memory_content": memory_content,
            "importance": importance,
            "tags": tags,
            "obsession_update": obsession_update,
        }
        archive_inbox_review(runtime.artist_dir, inbox_path, review_with_decision)
        soul.setdefault("review_history", []).append(
            {
                "kind": "received",
                "timestamp": now_iso(),
                "review_id": str(review.get("review_id", "")).strip(),
                "from_artist": str(review.get("author_artist", "")).strip(),
                "target_artwork_id": int(review.get("target_artwork_id", 0)),
                "decision": decision,
            }
        )
        processed += 1
        print(
            "Review ingested: "
            f"from={review.get('author_artist', '?')} "
            f"artwork={review.get('target_artwork_id', '?')} "
            f"decision={decision}"
        )
    soul["review_history"] = list(soul.get("review_history", []))[-120:]
    return processed


def run() -> None:
    args = parse_args()
    load_dotenv()
    if handle_management_command(args):
        return

    runtime = resolve_artist_runtime(args)
    try:
        validate_backend_choices(args)
    except Exception as exc:
        print(f"Configuration error: {exc}")
        return

    lock_fd = acquire_lock(runtime.lock_path)
    if lock_fd is None:
        print(f"Another awakening is already running for artist '{runtime.artist_id}'.")
        return

    try:
        vision_backend = None
        image_backend = None
        try:
            llm_backend = build_llm_backend(args)
            if runtime.run_mode in ("create", "full"):
                vision_backend = build_vision_backend(args)
                image_backend = build_image_backend(args, runtime.temp_dir, llm_backend=llm_backend)
        except HostedCallError as exc:
            print(f"Initialization deferred: {exc}")
            return

        ensure_dirs(runtime.temp_dir, runtime.gallery_dir)
        clear_temp(runtime.temp_dir)

        soul = load_soul(runtime.soul_path)
        manifest = load_config_file(runtime.artist_dir / "artist.json")
        soul = apply_artist_manifest_to_soul(soul, manifest)
        try:
            ensure_llm_identity(soul, llm_backend)
        except Exception as exc:
            print(f"Initialization deferred: identity generation failed ({exc})")
            return

        extra = load_memory_sources(runtime.memory_sources)
        soul_for_guidance = json.loads(json.dumps(soul))
        soul_for_guidance["text_memories"] = soul_for_guidance.get("text_memories", []) + extra.get("text_memories", [])
        soul_for_guidance["memories"] = soul_for_guidance.get("memories", []) + extra.get("memories", [])
        soul_for_guidance = _weighted_soul_context(soul_for_guidance, runtime.reflection_weights)
        ignored_ids = parse_ignore_ids(soul_for_guidance.get("text_memories", []))

        print_reflection(soul_for_guidance, ignored_ids)
        print(f"\nArtist: {runtime.artist_id} (profile={runtime.profile_id}, policy={runtime.run_policy}, mode={runtime.run_mode})")
        print(f"Reflection weights: {runtime.reflection_weights}")
        if vision_backend is not None:
            print(f"\nVision backend: {backend_label(vision_backend)}")
        print(f"\nLLM backend: {backend_label(llm_backend)}")
        if image_backend is not None:
            print(f"\nImage backend: {backend_label(image_backend)}")

        if runtime.run_mode == "ingest-reviews":
            ingested = ingest_reviews(runtime, soul, llm_backend)
            print(f"\nReview ingestion complete: {ingested} processed")
            atomic_write_json(runtime.soul_path, soul)
            clear_temp(runtime.temp_dir)
            print("\nReturning to slumber.")
            return

        creation_id = soul.get("creation_count", 0) + 1
        best_score = -1
        best_path: Optional[Path] = None
        best_iteration = -1
        best_feedback = ""

        if vision_backend is None or image_backend is None:
            print("Configuration error: run mode requires vision and image backends.")
            return

        try:
            vision = vision_backend.generate_vision(soul_for_guidance, ignored_ids)
        except HostedCallError as exc:
            try:
                vision = llm_backend.generate_vision_fallback(soul_for_guidance)
                print(f"Vision backend failed; used LLM fallback ({exc})")
            except Exception as fallback_exc:
                defer_and_persist(soul, creation_id, f"{exc} | fallback failed: {fallback_exc}", runtime.soul_path, runtime.temp_dir)
                return

        run_intent = normalize_run_intent({})
        try:
            run_intent = normalize_run_intent(
                llm_backend.generate_run_intent(soul_for_guidance),
                self_name=str(soul_for_guidance.get("name", "")).strip(),
            )
        except Exception as exc:
            print(f"Run intent generation skipped: {exc}")
        if getattr(args, "trace_revision", False):
            print(f"Run Intent Trace: {format_compact_json(run_intent)}")

        vision_directive = str(run_intent.get("vision_directive", "")).strip()
        critique_directive = build_critique_frame(soul_for_guidance, run_intent)
        revision_directive = str(run_intent.get("revision_directive", "")).strip()
        try:
            image_prompt = llm_backend.generate_initial_render_prompt(
                soul_for_guidance,
                vision,
                run_intent=run_intent,
            )
            prompt = build_render_prompt(vision, image_prompt)
        except HostedCallError as exc:
            defer_and_persist(soul, creation_id, str(exc), runtime.soul_path, runtime.temp_dir)
            return
        print(f"\nRun directive (vision): {vision_directive}")
        print(f"Run directive (critique): {str(run_intent.get('critique_directive', '')).strip()}")
        print(f"Run directive (revision): {revision_directive}")
        print("\n" + "-" * 58)

        for i in range(5):
            print("")
            print(f"Iteration {i + 1}: {'Creating' if i == 0 else 'Refining'}...")
            print(f"  Prompt: \"{prompt}\"")
            try:
                image_path = image_backend.generate(prompt, i, creation_id)
            except HostedCallError as exc:
                defer_and_persist(soul, creation_id, str(exc), runtime.soul_path, runtime.temp_dir)
                return
            artifact_diag = analyze_ascii_artifact(Path(image_path))
            if artifact_diag.get("summary"):
                print(f"  Artifact analysis: {artifact_diag['summary']}")
                if artifact_diag.get("flags"):
                    print(f"  Artifact flags: {', '.join(artifact_diag['flags'])}")

            try:
                critique = llm_backend.critique(image_path, vision, i, critique_directive)
            except HostedCallError as exc:
                defer_and_persist(soul, creation_id, str(exc), runtime.soul_path, runtime.temp_dir)
                return

            score = int(critique["score"])
            feedback = str(critique["feedback"])
            next_action = str(critique.get("next_action", "")).strip()
            print(f"  Score: {score}/10")
            print(f"  Critique: \"{feedback}\"")
            if next_action:
                print(f"  Next action: \"{next_action}\"")

            feedback_for_revision = feedback if not next_action else f"{feedback} Next action: {next_action}"
            if score > best_score:
                best_score, best_path, best_iteration, best_feedback = score, Path(image_path), i + 1, feedback_for_revision
            if score >= 8:
                break
            try:
                image_prompt = llm_backend.refine_render_prompt(
                    image_prompt,
                    vision,
                    feedback_for_revision,
                    score,
                    soul_for_guidance,
                    run_intent=run_intent,
                )
                prompt = build_render_prompt(vision, image_prompt)
            except HostedCallError as exc:
                defer_and_persist(soul, creation_id, str(exc), runtime.soul_path, runtime.temp_dir)
                return

        print("-" * 58)

        if best_path is None or not best_path.exists():
            soul["creation_count"] = creation_id
            soul["cycle_history"].append({
                "timestamp": now_iso(),
                "score": 0,
                "worthy": False,
                "subject": parse_vision(vision).subject,
                "color": parse_vision(vision).color,
                "relation": parse_vision(vision).relation,
                "vision": vision,
            })
            soul["cycle_history"] = soul["cycle_history"][-40:]
            atomic_write_json(runtime.soul_path, soul)
            return

        try:
            worthy = llm_backend.judge_worthiness(str(best_path), best_score, vision, critique_directive)
        except HostedCallError as exc:
            defer_and_persist(soul, creation_id, str(exc), runtime.soul_path, runtime.temp_dir)
            return

        parsed = parse_vision(vision)
        soul["creation_count"] = creation_id
        soul["cycle_history"].append({
            "timestamp": now_iso(),
            "score": best_score,
            "worthy": worthy,
            "subject": parsed.subject,
            "color": parsed.color,
            "relation": parsed.relation,
            "vision": vision,
        })
        soul["cycle_history"] = soul["cycle_history"][-40:]

        if worthy:
            print("WORTHY: This creation shall be remembered.")
        else:
            print("UNWORTHY: This creation is flawed, but retained as a reference.")
        tier = "masterpiece" if worthy and int(best_score) >= 8 else artwork_tier_from_score(best_score)
        try:
            final_path = move_to_gallery(best_path, creation_id, runtime.gallery_dir)
        except Exception as exc:
            print(f"Failed to preserve artwork ({exc}). Keeping in temp.")
            final_path = best_path

        soul["memories"].append({
            "type": "artwork",
            "id": creation_id,
            "file_path": str(final_path).replace("\\", "/"),
            "vision": vision,
            "final_score": best_score,
            "iteration_count": best_iteration,
            "worthy": worthy,
            "tier": tier,
            "timestamp": now_iso(),
        })
        soul["memories"] = trim_artwork_memories(soul["memories"])
        print(f"Artwork memory tier: {tier}")

        creation_result = {
            "vision": vision,
            "score": best_score,
            "tier": tier,
            "feedback": best_feedback,
            "worthy": worthy,
            "revision_directive": revision_directive,
            "critique_directive": critique_directive,
        }
        try:
            revision = llm_backend.propose_state_revision(_weighted_soul_context(soul, runtime.reflection_weights), creation_result)
            if getattr(args, "trace_revision", False):
                print(f"Revision Proposal Trace: {format_compact_json(revision if isinstance(revision, dict) else {})}")
            revision, revision_meta = apply_state_revision(soul, revision, runtime.artist_dir)
            print("")
            for line in revision_summary_lines(revision):
                print(line)
            if revision_meta.get("deleted_artwork_file", False):
                print("  artwork_file_deleted -> true")
            if revision_meta.get("identity_changed", False):
                print("  identity_changed -> true")
        except HostedCallError as exc:
            print(f"State revision skipped (strict hosted failure): {exc}")

        manifest_enforced = False
        if bool(manifest.get("enforce_personality", False)) or bool(manifest.get("enforce_obsession", False)):
            before_obsession = str(soul.get("current_obsession", "")).strip()
            before_traits = [str(t).strip() for t in list(soul.get("personality_traits", []) or []) if str(t).strip()]
            soul = apply_artist_manifest_to_soul(soul, manifest)
            after_obsession = str(soul.get("current_obsession", "")).strip()
            after_traits = [str(t).strip() for t in list(soul.get("personality_traits", []) or []) if str(t).strip()]
            manifest_enforced = (before_obsession != after_obsession) or (before_traits != after_traits)
        if manifest_enforced:
            print("  manifest_identity_enforced -> true")

        reviews_authored = 0
        reviews_ingested = 0
        if runtime.run_mode == "full":
            reviews_authored = perform_cross_artist_reviews(runtime, soul, llm_backend)
            reviews_ingested = ingest_reviews(runtime, soul, llm_backend)
            print(f"  cross_reviews_authored -> {reviews_authored}")
            print(f"  reviews_ingested -> {reviews_ingested}")

        removed_orphans = cleanup_gallery_orphans(runtime.gallery_dir, soul.get("memories", []))
        if removed_orphans > 0:
            print(f"  gallery_orphans_removed -> {removed_orphans}")

        print("\nSoul Updated:")
        print(f"  New obsession: {soul.get('current_obsession', '')}")
        print(f"  Artwork memories: {len(soul.get('memories', []))}")
        print(f"  Text memories: {len(soul.get('text_memories', []))}")
        print(f"  Review history: {len(soul.get('review_history', []))}")
        print(f"  Total creations: {soul.get('creation_count', 0)}")

        atomic_write_json(runtime.soul_path, soul)

        for p in runtime.temp_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
        print("\nReturning to slumber.")
    finally:
        release_lock(lock_fd, runtime.lock_path)
