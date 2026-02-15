import json
from pathlib import Path
from typing import Dict, Optional

from .constants import HostedCallError, now_iso
from .memory import (
    artwork_tier_from_score,
    parse_ignore_ids,
    parse_vision,
    print_reflection,
    trim_artwork_memories,
    vision_to_prompt,
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
    proposes_identity_change,
    revision_summary_lines,
)
from .state import (
    acquire_lock,
    apply_artist_manifest_to_soul,
    atomic_write_json,
    clear_temp,
    cleanup_gallery_orphans,
    ensure_dirs,
    load_config_file,
    load_dotenv,
    load_memory_sources,
    load_soul,
    move_to_gallery,
    release_lock,
)


def apply_prompt_refinement(prompt: str, feedback: str, iteration: int) -> str:
    p = prompt
    f = feedback.lower()
    if "depth" in f or "contrast" in f:
        p += ", enhanced chiaroscuro lighting"
    if "unclear" in f or "weak" in f:
        p += ", sharper silhouette"
    if "atmosphere" in f or "precision" in f:
        p += ", subtle ambient haze"
    if iteration >= 2:
        p += ", richer layering"
    return p


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


def defer_and_persist(soul: Dict, creation_id: int, reason: str, soul_path: Path) -> None:
    # Single defer path keeps failure persistence consistent across backend stages.
    print(f"Deferred: {reason}")
    record_deferred_cycle(soul, creation_id, reason)
    atomic_write_json(soul_path, soul)


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
    traits = [str(t).strip() for t in soul.get("personality_traits", []) if str(t).strip()][:10]
    obsession = str(soul.get("current_obsession", "")).strip()
    text_mem = soul.get("text_memories", [])[-10:]
    art_mem = soul.get("memories", [])[-8:]
    history = soul.get("cycle_history", [])[-8:]
    return (
        f"personality_traits:{traits}\n"
        f"current_obsession:{obsession}\n"
        f"text_memories:{text_mem}\n"
        f"artwork_memories:{art_mem}\n"
        f"recent_history:{history}\n"
        f"critique_directive:{str(run_intent.get('critique_directive', '')).strip()}\n"
        "Weighting guidance for critique/judgment: obsession 30%, personality 25%, memories 30%, technical execution 15%.\n"
        "Tier guidance: masterpieces reflect strengths to preserve, studies suggest experiments to continue, failures indicate pitfalls to avoid repeating."
    )


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
        try:
            vision_backend = build_vision_backend(args)
            llm_backend = build_llm_backend(args)
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
        ignored_ids = parse_ignore_ids(soul_for_guidance.get("text_memories", []))

        print_reflection(soul_for_guidance, ignored_ids)
        print(f"\nArtist: {runtime.artist_id} (profile={runtime.profile_id}, policy={runtime.run_policy})")
        print(f"\nVision backend: {backend_label(vision_backend)}")
        print(f"\nLLM backend: {backend_label(llm_backend)}")
        print(f"\nImage backend: {backend_label(image_backend)}")

        creation_id = soul.get("creation_count", 0) + 1
        best_score = -1
        best_path: Optional[Path] = None
        best_iteration = -1
        best_feedback = ""

        try:
            vision = vision_backend.generate_vision(soul_for_guidance, ignored_ids)
        except HostedCallError as exc:
            try:
                vision = llm_backend.generate_vision_fallback(soul_for_guidance)
                print(f"Vision backend failed; used LLM fallback ({exc})")
            except Exception as fallback_exc:
                defer_and_persist(soul, creation_id, f"{exc} | fallback failed: {fallback_exc}", runtime.soul_path)
                return

        run_intent = {"vision_directive": "", "critique_directive": "", "revision_directive": ""}
        try:
            run_intent = normalize_run_intent(llm_backend.generate_run_intent(soul_for_guidance))
        except Exception as exc:
            print(f"Run intent generation skipped: {exc}")
        if getattr(args, "trace_revision", False):
            print(f"Run Intent Trace: {format_compact_json(run_intent)}")

        prompt = vision_to_prompt(vision, soul_for_guidance)
        vision_directive = str(run_intent.get("vision_directive", "")).strip()
        critique_directive = build_critique_frame(soul_for_guidance, run_intent)
        revision_directive = str(run_intent.get("revision_directive", "")).strip()
        if vision_directive:
            prompt = f"{prompt}. {vision_directive}"
        print("\n" + "-" * 58)

        for i in range(5):
            print(f"Iteration {i + 1}: {'Creating' if i == 0 else 'Refining'}...")
            print(f"  Prompt: \"{prompt}\"")
            try:
                image_path = image_backend.generate(prompt, i, creation_id)
            except HostedCallError as exc:
                defer_and_persist(soul, creation_id, str(exc), runtime.soul_path)
                return

            try:
                critique = llm_backend.critique(image_path, vision, i, critique_directive)
            except HostedCallError as exc:
                defer_and_persist(soul, creation_id, str(exc), runtime.soul_path)
                return

            score = int(critique.get("score", 0))
            feedback = str(critique.get("feedback", ""))
            print(f"  Score: {score}/10")
            print(f"  Critique: \"{feedback}\"")

            if score > best_score:
                best_score, best_path, best_iteration, best_feedback = score, Path(image_path), i + 1, feedback
            if score >= 8:
                break
            prompt = apply_prompt_refinement(prompt, feedback, i)

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
            defer_and_persist(soul, creation_id, str(exc), runtime.soul_path)
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
            revision = llm_backend.propose_state_revision(soul, creation_result)
            weak_outcome = (not worthy) or int(best_score) <= 7
            if weak_outcome and not proposes_identity_change(revision, soul):
                retry_payload = dict(creation_result)
                retry_payload["revision_requirement"] = (
                    "Identity drift required for this weak outcome. "
                    "You must change current obsession and/or personality traits in a meaningful way."
                )
                retry_payload["current_identity"] = {
                    "personality_traits": soul.get("personality_traits", []),
                    "current_obsession": soul.get("current_obsession", ""),
                }
                retry_revision = llm_backend.propose_state_revision(soul, retry_payload)
                if proposes_identity_change(retry_revision, soul):
                    revision = retry_revision
                    print("Revision retry: enforced identity drift for weak outcome.")
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

        removed_orphans = cleanup_gallery_orphans(runtime.gallery_dir, soul.get("memories", []))
        if removed_orphans > 0:
            print(f"  gallery_orphans_removed -> {removed_orphans}")

        print("\nSoul Updated:")
        print(f"  New obsession: {soul.get('current_obsession', '')}")
        print(f"  Artwork memories: {len(soul.get('memories', []))}")
        print(f"  Text memories: {len(soul.get('text_memories', []))}")
        print(f"  Total creations: {soul.get('creation_count', 0)}")

        atomic_write_json(runtime.soul_path, soul)

        for p in runtime.temp_dir.iterdir():
            if p.is_file() or p.is_symlink():
                p.unlink(missing_ok=True)
        print("\nReturning to slumber.")
    finally:
        release_lock(lock_fd, runtime.lock_path)
