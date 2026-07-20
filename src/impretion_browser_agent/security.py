from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, cast

from fastapi import HTTPException, Request

TOKEN_HEADER = "x-impretion-sidecar-token"


def authenticate_request(request: Request, expected: str) -> None:
    provided = request.headers.get(TOKEN_HEADER, "")
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Browser sidecar authentication required")


def path_within(root: Path, candidate: Path, *, must_exist: bool = True) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Path escapes the authorized workspace") from error
    return resolved


def validate_runtime_manifest(manifest_path: Path, executable: Path, build_target: str) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = manifest["targets"][build_target]
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Chromium runtime manifest is invalid") from error
    if manifest.get("schemaVersion") != 1 or manifest.get("playwrightVersion") != "1.59.0":
        raise RuntimeError("Chromium runtime manifest is incompatible")
    resource_root = manifest_path.parent.resolve(strict=True)
    expected_executable = (resource_root / str(target.get("executableRelativePath", ""))).resolve(strict=True)
    executable = executable.resolve(strict=True)
    try:
        expected_executable.relative_to(resource_root)
    except ValueError as error:
        raise RuntimeError("Chromium runtime manifest path escapes its resource root") from error
    if expected_executable != executable or not executable.is_file():
        raise RuntimeError("Bundled Chromium executable is missing")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    if not hmac.compare_digest(digest, str(target.get("sha256", ""))):
        raise RuntimeError("Bundled Chromium hash verification failed")
    return cast(dict[str, Any], manifest)
