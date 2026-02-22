import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

from .configuration import (
    load_profile_config,
    profile_path,
    resolve_api_key,
    resolve_effective_profile_id,
)
from .state import load_config_file


def _http_get_json(url: str, headers: Dict[str, str], timeout: int = 30) -> Dict:
    req = urllib.request.Request(url=url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    out = json.loads(payload)
    if not isinstance(out, dict):
        raise ValueError("Response was not a JSON object.")
    return out


def _extract_model_rows(provider: str, payload: Dict, method_filter: str = "", contains: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    mf = method_filter.strip()
    needle = contains.strip().lower()

    if provider == "gemini":
        for model in payload.get("models", []) or []:
            name = str(model.get("name", ""))
            display = str(model.get("displayName", ""))
            methods = [str(m) for m in (model.get("supportedGenerationMethods", []) or [])]
            if mf and mf not in methods:
                continue
            haystack = f"{name} {display}".lower()
            if needle and needle not in haystack:
                continue
            rows.append({"id": name, "display": display, "methods": ", ".join(methods)})
        rows.sort(key=lambda x: x["id"])
        return rows

    if provider == "openai":
        for model in payload.get("data", []) or []:
            mid = str(model.get("id", ""))
            if needle and needle not in mid.lower():
                continue
            rows.append({"id": mid, "display": "", "methods": ""})
        rows.sort(key=lambda x: x["id"])
        return rows

    if provider == "anthropic":
        for model in payload.get("data", []) or []:
            mid = str(model.get("id", ""))
            display = str(model.get("display_name", ""))
            haystack = f"{mid} {display}".lower()
            if needle and needle not in haystack:
                continue
            rows.append({"id": mid, "display": display, "methods": ""})
        rows.sort(key=lambda x: x["id"])
        return rows

    if provider == "ollama":
        for model in payload.get("models", []) or []:
            name = str(model.get("name", "") or model.get("model", ""))
            if needle and needle not in name.lower():
                continue
            rows.append({"id": name, "display": "", "methods": "local"})
        rows.sort(key=lambda x: x["id"])
        return rows

    return rows


def _resolve_cli_executable(name: str) -> str:
    candidates = [name]
    if os.name == "nt":
        candidates = [f"{name}.cmd", f"{name}.exe", name]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    return ""


def _infer_provider_for_listing(args) -> str:
    if args.provider.strip():
        return args.provider.strip()
    profile_id = resolve_effective_profile_id(args)
    profile_cfg = load_profile_config(args, profile_id)
    return str(profile_cfg.get("llm_backend", "gemini"))


def list_models(args) -> None:
    provider = _infer_provider_for_listing(args)
    if provider in ("local", "mock", "ascii", "cli", "codex"):
        print(f"Provider '{provider}' is local-only and does not expose a hosted model catalog.")
        return

    api_key = ""
    if provider in ("gemini", "openai", "anthropic"):
        api_key = resolve_api_key(provider, "")
        if not api_key:
            env_hint = {
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }.get(provider, "API_KEY")
            print(f"No API key found for provider '{provider}'. Set {env_hint} or pass --provider with a configured profile.")
            return

    try:
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(api_key)}"
            payload = _http_get_json(url, {"Content-Type": "application/json"})
        elif provider == "openai":
            payload = _http_get_json("https://api.openai.com/v1/models", {"Authorization": f"Bearer {api_key}"})
        elif provider == "anthropic":
            payload = _http_get_json("https://api.anthropic.com/v1/models", {"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        elif provider == "ollama":
            base = str(getattr(args, "ollama_base_url", "http://localhost:11434")).rstrip("/")
            payload = _http_get_json(f"{base}/api/tags", {"Content-Type": "application/json"})
        else:
            print(f"Unsupported provider '{provider}'.")
            return
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            print(f"Model list request failed for {provider}: HTTP {exc.code} - {body}")
        else:
            print(f"Model list request failed for {provider}: HTTP {exc.code}")
        return
    except Exception as exc:
        print(f"Model list request failed for {provider}: {exc}")
        return

    rows = _extract_model_rows(provider, payload, method_filter=args.method, contains=args.contains)
    print(f"Provider: {provider}")
    if args.method:
        print(f"Method filter: {args.method}")
    if args.contains:
        print(f"Name filter: {args.contains}")
    print(f"Models found: {len(rows)}")
    for row in rows:
        line = f"- {row['id']}"
        if row.get("display"):
            line += f" ({row['display']})"
        if row.get("methods"):
            line += f" :: {row['methods']}"
        print(line)


def _provider_probe(provider: str, api_key: str, ollama_base_url: str) -> str:
    try:
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={urllib.parse.quote(api_key)}"
            _http_get_json(url, {"Content-Type": "application/json"}, timeout=20)
            return "ok"
        if provider == "openai":
            _http_get_json("https://api.openai.com/v1/models", {"Authorization": f"Bearer {api_key}"}, timeout=20)
            return "ok"
        if provider == "anthropic":
            _http_get_json("https://api.anthropic.com/v1/models", {"x-api-key": api_key, "anthropic-version": "2023-06-01"}, timeout=20)
            return "ok"
        if provider == "ollama":
            base = ollama_base_url.rstrip("/")
            payload = _http_get_json(f"{base}/api/tags", {"Content-Type": "application/json"}, timeout=3)
            count = len(payload.get("models", []) or [])
            return f"ok ({count} local models)"
        if provider == "cli":
            preferred = str(api_key).strip().lower()
            targets = [preferred] if preferred in ("gemini", "codex") else ["gemini", "codex"]
            found = []
            missing = []
            for t in targets:
                exe = _resolve_cli_executable(t)
                if not exe:
                    missing.append(t)
                    continue
                found.append(t)
            if found and not missing:
                return f"ok ({', '.join(found)})"
            if found and missing:
                return f"partial ({', '.join(found)} ok; missing {', '.join(missing)})"
            return "error: no supported CLI adapters found"
        if provider == "codex":
            exe = _resolve_cli_executable("codex")
            if exe:
                return "ok (codex)"
            return "error: codex CLI not found"
        return "n/a"
    except urllib.error.HTTPError as exc:
        return f"http {exc.code}"
    except Exception as exc:
        return f"error: {exc}"


def check_backends(args) -> None:
    profile_id = resolve_effective_profile_id(args)
    path = profile_path(args, profile_id)
    loaded = load_config_file(path) or {}
    profile_cfg = load_profile_config(args, profile_id)
    for k, v in profile_cfg.items():
        if hasattr(args, k) and k not in getattr(args, "_explicit_args", set()):
            setattr(args, k, v)

    entries = [
        ("vision", str(args.vision_backend).strip(), str(args.vision_model).strip()),
        ("llm", str(args.llm_backend).strip(), str(args.llm_model).strip()),
        ("image", str(args.image_backend).strip(), str(args.image_model).strip()),
    ]
    base_url = str(getattr(args, "ollama_base_url", "http://localhost:11434")).strip() or "http://localhost:11434"

    print(f"Profile: {profile_id}")
    if not path.exists():
        print(f"Warning: profile file not found at {path}; using default settings.")
    elif not loaded:
        print(f"Warning: profile file at {path} is empty/unreadable; using default settings.")
    print(f"Probe mode: {'on' if args.probe else 'off'}")
    seen = set()
    probe_cache: Dict[str, str] = {}
    ollama_names = None

    for role, provider, model in entries:
        line = f"- {role}: {provider}"
        if model:
            line += f":{model}"

        if provider in ("local", "mock", "ascii"):
            line += " -> ok (built-in)"
            print(line)
            continue

        if provider == "cli":
            adapter = str(getattr(args, "vision_cli" if role == "vision" else "llm_cli", "gemini")).strip().lower()
            if args.probe:
                line += f" -> {_provider_probe('cli', adapter, base_url)}"
            else:
                line += f" -> configured ({adapter})"
            print(line)
            continue

        if provider == "codex":
            if args.probe:
                line += f" -> {_provider_probe('codex', '', base_url)}"
            else:
                line += " -> configured (codex)"
            print(line)
            continue

        key = resolve_api_key(provider, "")
        if provider == "ollama":
            probe = probe_cache.get("ollama")
            if probe is None:
                probe = _provider_probe("ollama", "", base_url)
                probe_cache["ollama"] = probe
            line += f" -> {probe} @ {base_url}"
            if model and probe.startswith("ok"):
                try:
                    if ollama_names is None:
                        payload = _http_get_json(f"{base_url.rstrip('/')}/api/tags", {"Content-Type": "application/json"}, timeout=3)
                        ollama_names = {str(m.get('name', '') or m.get('model', '')) for m in (payload.get('models', []) or [])}
                    if model not in ollama_names:
                        line += " (model not pulled)"
                except Exception:
                    pass
            print(line)
            continue

        if not key:
            line += " -> missing API key"
            print(line)
            continue

        if args.probe:
            probe_key = provider
            if probe_key not in seen:
                result = _provider_probe(provider, key, base_url)
                seen.add(probe_key)
            else:
                result = "ok (shared provider probe)"
            line += f" -> {result}"
        else:
            line += " -> key present"
        print(line)
