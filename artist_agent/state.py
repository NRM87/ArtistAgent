import json
import datetime
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import DEFAULT_SOUL


def safe_default_soul() -> Dict:
    return json.loads(json.dumps(DEFAULT_SOUL))


def ensure_dirs(temp_dir: Path, gallery_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)


def clear_temp(temp_dir: Path) -> None:
    if temp_dir.exists():
        for item in temp_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except Exception as exc:
                print(f"Warning: could not clear temp item {item}: {exc}")


def atomic_write_json(path: Path, data: Dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    os.replace(tmp, path)


def load_soul(path: Path) -> Dict:
    if not path.exists():
        soul = safe_default_soul()
        atomic_write_json(path, soul)
        return soul

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            soul = json.load(f)
    except Exception as exc:
        print(f"Warning: soul file corrupted/unreadable ({exc}). Recreating defaults.")
        soul = safe_default_soul()
        atomic_write_json(path, soul)
        return soul

    for key, value in DEFAULT_SOUL.items():
        soul.setdefault(key, json.loads(json.dumps(value)))

    if not isinstance(soul.get("memories"), list):
        soul["memories"] = []
    if not isinstance(soul.get("text_memories"), list):
        soul["text_memories"] = []
    if not isinstance(soul.get("review_history"), list):
        soul["review_history"] = []
    if not isinstance(soul.get("cycle_history"), list):
        soul["cycle_history"] = []
    if not isinstance(soul.get("personality_traits"), list):
        soul["personality_traits"] = list(DEFAULT_SOUL["personality_traits"])
    return soul


def load_config_file(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            print(f"Warning: config file {path} must contain a JSON object. Ignoring config.")
            return {}
        return payload
    except Exception as exc:
        print(f"Warning: failed to parse config file {path} ({exc}). Ignoring config.")
        return {}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        print(f"Warning: failed to load .env file ({exc})")


def acquire_lock(lock_path: Path) -> Optional[int]:
    def _pid_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _read_lock_pid(path: Path) -> Optional[int]:
        try:
            return int(path.read_text(encoding="utf-8-sig").strip())
        except Exception:
            return None

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        return fd
    except FileExistsError:
        existing_pid = _read_lock_pid(lock_path)
        if existing_pid is None or not _pid_is_running(existing_pid):
            # Recover stale/corrupt lock files left by interrupted runs.
            # If another process races us here, re-acquire will fail and return None.
            try:
                lock_path.unlink(missing_ok=True)
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                return fd
            except Exception:
                return None
        return None


def release_lock(lock_fd: Optional[int], lock_path: Path) -> None:
    try:
        if lock_fd is not None:
            os.close(lock_fd)
    except Exception:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def merge_config(base: Dict, overlay: Dict) -> Dict:
    out = dict(base)
    out.update(overlay)
    return out


def apply_artist_manifest_to_soul(soul: Dict, manifest: Dict) -> Dict:
    if manifest.get("name"):
        soul["name"] = str(manifest["name"])

    is_fresh = (
        int(soul.get("creation_count", 0)) == 0
        and not soul.get("memories")
        and not soul.get("text_memories")
        and not soul.get("cycle_history")
    )
    enforce_personality = bool(manifest.get("enforce_personality", False))
    enforce_obsession = bool(manifest.get("enforce_obsession", False))

    if (is_fresh or enforce_personality) and "personality_traits" in manifest and isinstance(manifest.get("personality_traits"), list):
        soul["personality_traits"] = [str(t) for t in manifest["personality_traits"]]

    if (is_fresh or enforce_obsession) and "current_obsession" in manifest:
        soul["current_obsession"] = str(manifest["current_obsession"])

    return soul


def load_memory_sources(paths: List[Path]) -> Dict:
    extra = {"text_memories": [], "memories": []}
    for p in paths:
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict):
                t = obj.get("text_memories", [])
                a = obj.get("memories", [])
                if isinstance(t, list):
                    extra["text_memories"].extend(t)
                if isinstance(a, list):
                    extra["memories"].extend(a)
            elif isinstance(obj, list):
                extra["text_memories"].extend(obj)
        except Exception as exc:
            print(f"Warning: failed to load memory source {p} ({exc})")
    return extra


def move_to_gallery(temp_path: Path, creation_id: int, gallery_dir: Path) -> Path:
    suffix = temp_path.suffix if temp_path.suffix else ".png"
    target = gallery_dir / f"img_{creation_id:04d}{suffix}"
    if target.exists():
        import datetime

        stamp = datetime.datetime.now().strftime("%H%M%S")
        target = gallery_dir / f"img_{creation_id:04d}_{stamp}{suffix}"
    shutil.move(str(temp_path), str(target))
    return target


def cleanup_gallery_orphans(gallery_dir: Path, memories: List[Dict]) -> int:
    if not gallery_dir.exists():
        return 0

    gallery_root = gallery_dir.resolve()
    keep = set()
    for mem in memories:
        raw_path = str(mem.get("file_path", "")).strip()
        if not raw_path:
            continue
        p = Path(raw_path)
        candidates = []
        if p.is_absolute():
            candidates.append(p.resolve())
        else:
            candidates.append((Path.cwd() / p).resolve())
            candidates.append((gallery_root / p.name).resolve())
        for candidate in candidates:
            if candidate.exists() and (candidate == gallery_root or gallery_root in candidate.parents):
                keep.add(candidate)

    removed = 0
    for item in gallery_dir.iterdir():
        if not item.is_file():
            continue
        candidate = item.resolve()
        if candidate not in keep:
            try:
                item.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    return removed


def ensure_review_dirs(artist_dir: Path) -> Dict[str, Path]:
    root = artist_dir / "reviews"
    inbox = root / "inbox"
    outbox = root / "outbox"
    processed = root / "processed"
    root.mkdir(parents=True, exist_ok=True)
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return {"root": root, "inbox": inbox, "outbox": outbox, "processed": processed}


def _resolve_artifact_path(artist_dir: Path, file_path: str) -> Optional[Path]:
    raw = str(file_path).strip()
    if not raw:
        return None
    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        candidates.append((artist_dir / p).resolve())
        candidates.append((artist_dir / "gallery" / p.name).resolve())
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            continue
    return None


def list_review_candidates(
    artists_dir: Path,
    reviewer_artist_id: str,
    review_targets: List[str],
    authored_keys: set,
    limit: int = 20,
) -> List[Dict]:
    targets = {str(t).strip() for t in (review_targets or []) if str(t).strip()}
    out: List[Dict] = []
    if not artists_dir.exists():
        return out
    for artist_root in sorted([p for p in artists_dir.iterdir() if p.is_dir()]):
        target_artist = artist_root.name
        if target_artist == reviewer_artist_id:
            continue
        if targets and target_artist not in targets:
            continue
        soul = load_config_file(artist_root / "soul.json")
        memories = [m for m in (soul.get("memories", []) or []) if isinstance(m, dict)]
        for mem in reversed(memories):
            if str(mem.get("type", "")).strip().lower() != "artwork":
                continue
            art_id = int(mem.get("id", 0)) if str(mem.get("id", "")).strip() else 0
            review_key = f"{target_artist}:{art_id}"
            if review_key in authored_keys:
                continue
            path = _resolve_artifact_path(artist_root, str(mem.get("file_path", "")))
            if path is None:
                continue
            out.append(
                {
                    "target_artist": target_artist,
                    "target_artwork_id": art_id,
                    "target_artifact_path": str(path).replace("\\", "/"),
                    "target_vision": str(mem.get("vision", "")).strip(),
                    "target_score": int(mem.get("final_score", 0)),
                    "target_timestamp": str(mem.get("timestamp", "")).strip(),
                    "review_key": review_key,
                }
            )
            break
    out.sort(key=lambda x: x.get("target_timestamp", ""), reverse=True)
    return out[: max(0, int(limit))]


def persist_cross_artist_review(
    artists_dir: Path,
    author_artist_id: str,
    target_artist_id: str,
    review: Dict,
) -> Dict[str, Path]:
    author_dir = artists_dir / author_artist_id
    target_dir = artists_dir / target_artist_id
    author_paths = ensure_review_dirs(author_dir)
    target_paths = ensure_review_dirs(target_dir)

    review_id = str(review.get("review_id", "")).strip()
    if not review_id:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        review_id = f"rvw_{author_artist_id}_{target_artist_id}_{stamp}_{abs(hash(str(review))) % 100000:05d}"
        review["review_id"] = review_id

    outbox_path = author_paths["outbox"] / f"{review_id}.json"
    inbox_path = target_paths["inbox"] / f"{review_id}.json"
    atomic_write_json(outbox_path, review)
    atomic_write_json(inbox_path, review)
    return {"outbox": outbox_path, "inbox": inbox_path}


def load_inbox_reviews(artist_dir: Path, limit: int = 10) -> List[Tuple[Path, Dict]]:
    paths = ensure_review_dirs(artist_dir)
    files = sorted([p for p in paths["inbox"].iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    out: List[Tuple[Path, Dict]] = []
    for p in files[: max(0, int(limit))]:
        payload = load_config_file(p)
        if payload:
            out.append((p, payload))
    return out


def archive_inbox_review(artist_dir: Path, inbox_path: Path, payload: Dict) -> Optional[Path]:
    paths = ensure_review_dirs(artist_dir)
    name = inbox_path.name if inbox_path.name else f"review_{datetime.datetime.now().strftime('%H%M%S')}.json"
    target = paths["processed"] / name
    try:
        atomic_write_json(target, payload)
        inbox_path.unlink(missing_ok=True)
        return target
    except Exception:
        return None
