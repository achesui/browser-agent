from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_sidecar.py"
SPEC = importlib.util.spec_from_file_location("build_sidecar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_sidecar)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_valid_package(
    tmp_path: Path,
    *,
    target: str = "aarch64-apple-darwin",
) -> tuple[Path, Path]:
    app = tmp_path / "src-tauri"
    resource_root = app / "resources" / "browser-agent"
    sidecar = app / "binaries" / (
        f"impretion-browser-agent-{target}{build_sidecar.executable_suffix(target)}"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"sidecar")
    source_lock = {
        "browserAgentCommit": "0123456789abcdef0123456789abcdef01234567",
        "protocolVersion": 1,
        "sidecarVersion": "0.1.0",
        "buildTarget": target,
        "sidecarBinarySha256": digest(sidecar),
    }
    lock_path = resource_root / "source-lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(source_lock), encoding="utf-8")
    return app, lock_path


def test_windows_executable_suffix() -> None:
    assert build_sidecar.executable_suffix("x86_64-pc-windows-msvc") == ".exe"
    assert build_sidecar.executable_suffix("aarch64-apple-darwin") == ""


def test_windows_target_detection() -> None:
    assert build_sidecar.current_target("Windows", "AMD64") == "x86_64-pc-windows-msvc"
    assert build_sidecar.current_target("Windows", "x86_64") == "x86_64-pc-windows-msvc"


def test_macos_arm_target_detection() -> None:
    assert build_sidecar.current_target("Darwin", "arm64") == "aarch64-apple-darwin"
    assert build_sidecar.current_target("Darwin", "aarch64") == "aarch64-apple-darwin"


def test_linux_is_rejected() -> None:
    with pytest.raises(build_sidecar.PackagingError, match="Unsupported"):
        build_sidecar.current_target("Linux", "x86_64")


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Darwin", "x86_64"), ("Windows", "ARM64"), ("Darwin", "mips64")],
)
def test_unsupported_architectures_are_rejected(system: str, machine: str) -> None:
    with pytest.raises(build_sidecar.PackagingError, match="Build natively"):
        build_sidecar.current_target(system, machine)


def test_project_version_resolution(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\nversion = "0.1.0"\n', encoding="utf-8")
    assert build_sidecar.project_version(pyproject) == "0.1.0"


def test_project_version_missing_is_rejected(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\n', encoding="utf-8")
    with pytest.raises(build_sidecar.PackagingError):
        build_sidecar.project_version(pyproject)


def test_missing_source_lock(tmp_path: Path) -> None:
    with pytest.raises(build_sidecar.PackagingError, match="Missing source lock"):
        build_sidecar.verify_artifacts(
            "aarch64-apple-darwin", "0.1.0", app_tauri=tmp_path
        )


def test_wrong_target_is_rejected(tmp_path: Path) -> None:
    app, _ = write_valid_package(tmp_path)
    with pytest.raises(build_sidecar.PackagingError, match="buildTarget"):
        build_sidecar.verify_artifacts(
            "x86_64-pc-windows-msvc", "0.1.0", app_tauri=app
        )


def test_incorrect_sidecar_hash_is_rejected(tmp_path: Path) -> None:
    app, _ = write_valid_package(tmp_path)
    sidecar = app / "binaries" / "impretion-browser-agent-aarch64-apple-darwin"
    sidecar.write_bytes(b"tampered")
    with pytest.raises(build_sidecar.PackagingError, match="sidecar hash"):
        build_sidecar.verify_artifacts(
            "aarch64-apple-darwin", "0.1.0", app_tauri=app
        )


def test_successful_packaging_validation(tmp_path: Path) -> None:
    app, _ = write_valid_package(tmp_path)
    build_sidecar.verify_artifacts(
        "aarch64-apple-darwin", "0.1.0", app_tauri=app
    )


def test_build_commands_use_active_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    output = tmp_path / "dist" / "impretion-browser-agent.exe"

    def run(command: list[str], **_: object) -> None:
        calls.append(command)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"sidecar")

    monkeypatch.setattr(build_sidecar.subprocess, "run", run)
    monkeypatch.setattr(build_sidecar, "ROOT", tmp_path)
    (tmp_path / "browser-agent.spec").write_text("# stub", encoding="utf-8")
    output.parent.mkdir()
    output.write_bytes(b"sidecar")
    build_sidecar.build_sidecar("x86_64-pc-windows-msvc")
    assert calls == [
        [
            build_sidecar.sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(tmp_path / "browser-agent.spec"),
        ]
    ]
