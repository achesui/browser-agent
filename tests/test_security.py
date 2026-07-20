import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from impretion_browser_agent.security import path_within, validate_runtime_manifest


def test_path_validation_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "file.txt"
    inside.write_text("ok")
    assert path_within(root, inside) == inside
    outside = tmp_path / "outside.txt"
    outside.write_text("no")
    with pytest.raises(HTTPException):
        path_within(root, outside)
    link = root / "link"
    link.symlink_to(outside)
    with pytest.raises(HTTPException):
        path_within(root, link)


def test_runtime_manifest_requires_matching_hash(tmp_path: Path) -> None:
    executable = tmp_path / "chromium"
    executable.write_bytes(b"browser")
    manifest = tmp_path / "runtime-manifest.json"
    manifest.write_text(json.dumps({
        "schemaVersion": 1,
        "playwrightVersion": "1.59.0",
        "chromiumRevision": "1217",
        "chromiumVersion": "147.0.7727.15",
        "targets": {"target": {
            "executableRelativePath": "chromium",
            "sha256": hashlib.sha256(b"browser").hexdigest(),
        }},
    }))
    assert validate_runtime_manifest(manifest, executable, "target")["chromiumRevision"] == "1217"
    executable.write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        validate_runtime_manifest(manifest, executable, "target")
