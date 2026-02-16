# Recursive Artist Agent

A multi-artist CLI system where each artist recursively:
- generates a vision
- runs iterative attempts with a mutable image prompt
- creates an artifact (image or ASCII fallback)
- critiques and judges output in first-person voice
- updates memories, personality, and obsession over time

The project supports hosted providers (Gemini/OpenAI/Anthropic), local Ollama models, and LLM-driven ASCII fallback behavior.

## Key Features

- Multi-artist architecture (`artists/<artist_id>/...`)
- Profile-based model/runtime configuration (`profiles/<profile>.json`)
- CLI commands for setup, creation, configuration, running, and health checks
- Local Ollama support for vision + LLM reasoning
- LLM-driven ASCII fallback with enforced canvas size
- Fixed run vision with iterative image-prompt refinement per run
- Tolerant labeled-field parsing for weak local models (no strict JSON dependency in loop decisions)
- First-person artist voice for vision, critique, and reflection
- Stale lock recovery for interrupted runs
- Artwork gallery storage under each artist (`gallery/`)

## Directory Layout

```text
recursive_artist_agent.py      # thin entrypoint
artist_agent/
  cycle.py                     # main run loop
  runtime.py                   # CLI parser + dispatch + backend builders
  configuration.py             # shared profile/config resolution helpers
  commands_artist.py           # setup/artist/profile commands
  commands_providers.py        # model listing + backend health checks
  backends.py                  # vision/llm/image backend implementations
  memory.py                    # vision parsing, evolution, memory logic
  state.py                     # filesystem, locking, persistence helpers
  constants.py                 # defaults, capability matrix, dataclasses
artists/
profiles/
tests/
```

## Quick Start (PowerShell)

Install dependencies:

```powershell
python -m pip install -U pillow
```

Create an artist:

```powershell
python recursive_artist_agent.py create-artist --artist demo --profile default
```

Run:

```powershell
python recursive_artist_agent.py run --artist demo
```

## Common Commands

List artists:

```powershell
python recursive_artist_agent.py list-artists
```

Show artist manifest:

```powershell
python recursive_artist_agent.py show-artist --artist qwen_artist
```

Show effective profile:

```powershell
python recursive_artist_agent.py show-profile --artist qwen_artist
```

Configure profile non-interactively (only explicit flags are changed):

```powershell
python recursive_artist_agent.py configure-models --profile local_ollama --non-interactive --run-policy offline --vision-backend ollama --vision-model qwen2.5:3b --llm-backend ollama --llm-model qwen2.5:3b --ollama-base-url http://localhost:11434 --image-backend ascii --ascii-size 200x80
```

List models:

```powershell
python recursive_artist_agent.py list-models --provider ollama
```

Check backend health:

```powershell
python recursive_artist_agent.py check-backends --artist qwen_artist --probe
```

## Runtime Policies

- `strict`: disallows deterministic local vision and all mock backends
- `hybrid`: hosted-first where configured, with explicit ASCII fallback support for image generation
- `offline`: enforces fully local execution (`vision=ollama`, `llm=ollama`, `image=ascii`)

## ASCII Fallback

If image generation fails and `image_fallback=ascii`:
- system asks the active LLM backend to generate ASCII art
- output is hard-enforced to exact `ascii_size` (`WxH`, default `160x60`)
- artifact is written as `.txt` and can be preserved in the artist gallery

## Run Workflow

Per run:
1. Generate one run vision from soul context (personality, obsession, memories, history).
2. Keep that run vision fixed for the entire run.
3. Iterate up to 5 attempts:
   - generate image from current iteration image prompt
   - critique/judge from artist model
   - refine only the iteration image prompt for next attempt
4. Persist best artifact + tier (`masterpiece` / `study` / `failure`).
5. Reflect and optionally edit soul fields (obsession, traits, memories).

## Ollama Setup

Install and run Ollama, then pull a model:

```powershell
ollama pull qwen2.5:3b
```

Check visibility from the app:

```powershell
python recursive_artist_agent.py list-models --provider ollama
```

## Testing

If `pytest` is available:

```powershell
python -m pytest -q
```

Compile-only sanity check:

```powershell
python -m compileall artist_agent tests
```

## Troubleshooting

Lock error (`Another awakening is already running...`):
- lock files are auto-reclaimed when stale
- manual fallback:

```powershell
Remove-Item -Force artists\<artist_id>\.awaken.lock
```

Gemini `404` model errors:
- use current model IDs, then verify with:

```powershell
python recursive_artist_agent.py list-models --provider gemini --contains gemini
```

Gemini `429 RESOURCE_EXHAUSTED`:
- key is valid but quota is exhausted; use another provider or local Ollama

## Notes for Maintainers

For architecture and extension guidance, see:

- `docs/AGENT_ONBOARDING.md`
- `TODO.md`
