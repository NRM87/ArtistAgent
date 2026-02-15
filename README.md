# Recursive Artist Agent

A multi-artist CLI system where each artist recursively:
- generates a vision
- creates an artifact (image or ASCII fallback)
- critiques and judges output
- updates memories, personality, and obsession over time

The project supports hosted providers (Gemini/OpenAI/Anthropic), local Ollama models, and robust fallback behavior.

## Key Features

- Multi-artist architecture (`artists/<artist_id>/...`)
- Profile-based model/runtime configuration (`profiles/<profile>.json`)
- CLI commands for setup, creation, configuration, running, and health checks
- Local Ollama support for vision + LLM reasoning
- LLM-driven ASCII fallback with enforced canvas size
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

- `strict`: requires hosted backends where configured; no hosted fallback unless explicitly configured via fallback mode
- `hybrid`: hosted first, with local/mock fallback paths where available
- `offline`: enforces fully local execution (`vision=ollama`, `llm=ollama`, `image=ascii/mock`)

## ASCII Fallback

If image generation fails and `image_fallback=ascii`:
- system asks the active LLM backend to generate ASCII art
- output is hard-enforced to exact `ascii_size` (`WxH`, default `160x60`)
- artifact is written as `.txt` and can be preserved in the artist gallery

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
python -m py_compile recursive_artist_agent.py artist_agent/*.py tests/*.py
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
