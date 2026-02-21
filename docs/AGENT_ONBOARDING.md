# Agent Onboarding Guide

This document is for future coding agents/maintainers working on this repository.

## Project Intent (Operational)

- Maintain a persistent, evolving artist identity.
- Support multiple artists with independent config/state.
- Keep backend architecture provider-agnostic and easy to extend.
- Preserve a reliable local development path (Ollama + ASCII fallback).
- Prioritize CLI ergonomics for repeated workflows.

## Execution Flow

Entry:
- `recursive_artist_agent.py` -> `artist_agent.cycle.run()`

Run sequence in `artist_agent/cycle.py`:
1. Parse CLI args (`runtime.parse_args`)
2. Load env (`state.load_dotenv`)
3. Handle management command (`runtime.handle_management_command`)
4. Resolve runtime context (`runtime.resolve_artist_runtime`)
5. Validate backend policy/capabilities
6. Acquire per-artist lock
7. Build backends (vision, llm, image)
8. Load and apply artist state + manifest
9. Generate fixed run vision from soul context
10. Iterate creation loop: image prompt -> image generation -> critique -> image prompt refinement
11. Judge worthiness, update memories/personality/obsession
12. Persist soul and cleanup temp
13. Release lock

## Module Responsibilities

`artist_agent/runtime.py`
- CLI parser, command dispatcher, runtime context resolution, backend builders.
- Should remain orchestration-focused.

`artist_agent/configuration.py`
- Shared config/profile helpers.
- Single source for profile normalization and effective profile resolution.

`artist_agent/commands_artist.py`
- Artist/profile lifecycle commands:
  - `setup`
  - `create-artist`
  - `configure-models`
  - `list-artists`
  - `show-artist`
  - `show-profile`

`artist_agent/commands_providers.py`
- Provider introspection/health:
  - `list-models`
  - `check-backends`
- Contains provider response parsing and probe helpers.

`artist_agent/backends.py`
- Backend classes and provider HTTP implementations.
- Supports:
  - vision: local, hosted, ollama
  - llm: hosted, ollama
  - image: hosted, ascii
  - fallback: LLM-generated ASCII with deterministic canvas enforcement
- Structured text-field prompting/parsing for weak-model robustness (loop-critical actions avoid strict JSON contracts).
- First-person voice guidance for self-reflection pathways (vision/critique/memory/revision text).

`artist_agent/memory.py`
- Vision parsing/generation rules, memory consolidation, evolution heuristics.

`artist_agent/state.py`
- Persistence, lock lifecycle, filesystem helpers, manifest application.

`artist_agent/constants.py`
- Defaults, capability matrix, shared dataclasses.

## Data Model

Per artist:
- `artists/<id>/artist.json`
  - stable intent/wiring (profile, initial/enforced identity, memory sources)
- `artists/<id>/soul.json`
  - mutable runtime state (memories/history/evolved identity)

Profiles:
- `profiles/<profile>.json`
  - backend/model settings and runtime policy

Important distinction:
- Manifest is configuration intent.
- Soul is evolving state.

## Invariants to Preserve

1. CLI explicit override contract
- Explicit CLI flags override profile values.
- Non-explicit fields come from resolved profile.

2. Multi-artist isolation
- No cross-artist state mutation unless explicitly configured via memory sources.

3. Locking correctness
- Only one run per artist at a time.
- Stale lock reclamation must remain safe.

4. Policy semantics
- `offline` must force local Ollama text reasoning and ASCII image mode.
- `strict` must reject local/mock core backends as configured.
- Runtime should fail closed on LLM failures (no hidden deterministic critique fallback).

5. Run vision contract
- The run vision is generated once at run start and remains fixed through iterations.
- Vision text should be actionable and concrete (target format: `My vision for this run is to ...`).
- Iterative refinement modifies only the per-iteration image prompt.
- Do not reintroduce iteration-time vision rewriting.

6. Fallback consistency
- ASCII fallback should remain model-aware and canvas-enforced.
- Do not reintroduce template/deterministic ASCII rendering.

7. Voice contract
- Artist-facing reflective language should remain first-person where feasible (vision/critique/memory notes).
- Avoid third-person self-reference in generated self-reflection text.
- Critique should include a concrete next-step command (for prompt refinement), not only abstract commentary.

## Extension Playbook

Adding a new provider:
1. Add capabilities in `constants.PROVIDER_CAPABILITIES`.
2. Add defaults in `DEFAULT_PROVIDER_MODELS` and/or `DEFAULT_VISION_MODELS` / `DEFAULT_IMAGE_MODELS`.
3. Extend CLI choices in `runtime.build_arg_parser`.
4. Implement backend logic in `backends.py`.
5. Add provider handling in:
   - `commands_providers.list_models`
   - `commands_providers.check_backends` probe path
6. Add tests covering parser/builder behavior and model-row extraction.

Adding a new command:
1. Add command name to parser choices in `runtime.build_arg_parser`.
2. Implement in:
   - `commands_artist.py` or `commands_providers.py`
3. Wire dispatch in `runtime.handle_management_command`.
4. Add at least one command-level smoke/logic test.

## Known Hotspots / Risk Areas

1. `memory.py` heuristics
- High impact on creative behavior and bias.
- Keep changes intentional and test with blank artists + seeded personalities.

2. `backends.py` parsing
- Provider response formats drift; parsing must be defensive.
- Prefer labeled-field prompts with tolerant parsing and bounded retries.
- Keep fail-closed behavior for critique/judgment/revision paths.

3. Profile drift
- Defaults evolve over time.
- Keep legacy normalization logic updated to avoid breaking old profiles.

4. PowerShell UX
- Users copy commands directly.
- Favor one-line examples for reliability.

## Test Strategy

Current tests emphasize:
- runtime policy enforcement
- profile normalization/legacy aliasing
- locking correctness
- memory consolidation rules
- fallback behavior (including ASCII path)
- provider model row extraction/inference

When adding functionality:
- prioritize narrow, deterministic unit tests in `tests/`.
- avoid coupling tests to live network/provider availability.

## Suggested Future Refactors

1. Separate backend HTTP clients from backend orchestration classes.
2. Add a lightweight typed schema layer for profile/manifest/soul validation.
3. Add command snapshot tests for stable CLI output contracts.
4. Add bounded retries/circuit-breaker config to provider probes.

## Fast Recovery Checklist

If runtime seems broken:
1. `python recursive_artist_agent.py show-profile --artist <id>`
2. `python recursive_artist_agent.py check-backends --artist <id> --probe`
3. Ensure lock file is not stale:
   - `artists/<id>/.awaken.lock`
4. Run with local fallback profile to isolate provider issues.
