from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Iterator

import pytest

_LOCAL_TMP_ROOT = Path(__file__).resolve().parent / "_tmp_work"


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Workspace-local tmp_path fixture that avoids system temp ACL issues."""
    _LOCAL_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    case_dir = _LOCAL_TMP_ROOT / f"case_{uuid.uuid4().hex[:12]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
