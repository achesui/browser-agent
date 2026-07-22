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
    executable_relative_path: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    app = tmp_path / "src-tauri"
    resource_root = app / "resources" / "browser-agent"
    chromium = resource_root / "chromium" / target / "chrome"
    chromium.parent.mkdir(parents=True)
    chromium.write_bytes(b"chromium")
    sidecar = app / "binaries" / (
        f"impretion-browser-agent-{target}{build_sidecar.executable_suffix(target)}"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"sidecar")
    runtime = {
        "schemaVersion": 1,
        "playwrightVersion": "1.59.0",
        "chromiumRevision": "1217",
        "chromiumVersion": "147.0.7727.15",
        "targets": {
            target: {
                "executableRelativePath": executable_relative_path
                or chromium.relative_to(resource_root).as_posix(),
                "sha256": digest(chromium),
            }
        },
    }
    runtime_path = resource_root / "runtime-manifest.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    source_lock = {
        "browserAgentCommit": "abc123",
        "protocolVersion": 1,
        "sidecarVersion": "0.1.0",
        "playwrightVersion": "1.59.0",
        "chromiumRevision": "1217",
        "buildTarget": target,
        "runtimeManifestSha256": digest(runtime_path),
        "sidecarBinarySha256": digest(sidecar),
    }
    lock_path = resource_root / "source-lock.json"
    lock_path.write_text(json.dumps(source_lock), encoding="utf-8")
    return app, runtime_path, lock_path, chromium


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


def test_playwright_version_resolution(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_text(
        '[project]\ndependencies = ["playwright==1.59.0", "pytest==9.0.2"]\n',
        encoding="utf-8",
    )
    lock.write_text(
        'version = 1\n[[package]]\nname = "playwright"\nversion = "1.59.0"\n',
        encoding="utf-8",
    )
    assert build_sidecar.locked_playwright_version(pyproject, lock) == "1.59.0"


def test_playwright_lock_mismatch_is_rejected(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    pyproject.write_text('[project]\ndependencies = ["playwright==1.59.0"]\n')
    lock.write_text('[[package]]\nname = "playwright"\nversion = "1.58.0"\n')
    with pytest.raises(build_sidecar.PackagingError, match="lock mismatch"):
        build_sidecar.locked_playwright_version(pyproject, lock)


def test_chromium_revision_resolution(tmp_path: Path) -> None:
    manifest = tmp_path / "browsers.json"
    manifest.write_text(
        json.dumps(
            {
                "browsers": [
                    {"name": "firefox", "revision": "1", "browserVersion": "1"},
                    {
                        "name": "chromium",
                        "revision": "1217",
                        "browserVersion": "147.0.7727.15",
                    },
                ]
            }
        )
    )
    assert build_sidecar.playwright_metadata(manifest) == (
        "1217",
        "147.0.7727.15",
    )


@pytest.mark.parametrize("contents", ["{", "[]", "{}", '{"browsers": []}'])
def test_missing_or_malformed_playwright_manifest(
    tmp_path: Path, contents: str
) -> None:
    manifest = tmp_path / "browsers.json"
    manifest.write_text(contents)
    with pytest.raises(build_sidecar.PackagingError):
        build_sidecar.playwright_metadata(manifest)


def test_missing_runtime_manifest(tmp_path: Path) -> None:
    with pytest.raises(build_sidecar.PackagingError, match="Missing runtime manifest"):
        build_sidecar.verify_artifacts(
            "aarch64-apple-darwin", "1.59.0", "1217", app_tauri=tmp_path
        )


def test_wrong_target_is_rejected(tmp_path: Path) -> None:
    app, _, _, _ = write_valid_package(tmp_path)
    with pytest.raises(build_sidecar.PackagingError, match="target"):
        build_sidecar.verify_artifacts(
            "x86_64-pc-windows-msvc", "1.59.0", "1217", app_tauri=app
        )


def test_incorrect_chromium_hash_is_rejected(tmp_path: Path) -> None:
    app, _, _, chromium = write_valid_package(tmp_path)
    chromium.write_bytes(b"tampered")
    with pytest.raises(build_sidecar.PackagingError, match="Chromium executable hash"):
        build_sidecar.verify_artifacts(
            "aarch64-apple-darwin", "1.59.0", "1217", app_tauri=app
        )


def test_incorrect_sidecar_hash_is_rejected(tmp_path: Path) -> None:
    app, _, _, _ = write_valid_package(tmp_path)
    sidecar = app / "binaries" / "impretion-browser-agent-aarch64-apple-darwin"
    sidecar.write_bytes(b"tampered")
    with pytest.raises(build_sidecar.PackagingError, match="sidecar hash"):
        build_sidecar.verify_artifacts(
            "aarch64-apple-darwin", "1.59.0", "1217", app_tauri=app
        )


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    app, _, _, _ = write_valid_package(
        tmp_path, executable_relative_path="../../../../outside/chrome"
    )
    with pytest.raises(build_sidecar.PackagingError, match="escapes"):
        build_sidecar.verify_artifacts(
            "aarch64-apple-darwin", "1.59.0", "1217", app_tauri=app
        )


def test_successful_packaging_validation(tmp_path: Path) -> None:
    app, _, _, _ = write_valid_package(tmp_path)
    build_sidecar.verify_artifacts(
        "aarch64-apple-darwin", "1.59.0", "1217", app_tauri=app
    )


def test_build_commands_use_active_python(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_: object) -> None:
        calls.append(command)

    monkeypatch.setattr(build_sidecar.subprocess, "run", run)
    monkeypatch.setattr(build_sidecar, "ROOT", tmp_path)
    output = tmp_path / "dist" / "impretion-browser-agent.exe"
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
            "browser-agent.spec",
        ]
    ]


def test_chromium_download_uses_active_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr(build_sidecar.subprocess, "run", run)
    monkeypatch.setattr(build_sidecar, "ROOT", tmp_path)
    monkeypatch.setenv("PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD", "1")
    executable = (
        tmp_path
        / "build"
        / "playwright-browsers"
        / "chromium-1217"
        / "chrome-mac-arm64"
        / "chrome"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"chromium")

    chromium_root, found_executable = build_sidecar.download_chromium("1217")

    assert chromium_root == executable.parent
    assert found_executable == executable
    assert calls[0][0] == [
        build_sidecar.sys.executable,
        "-m",
        "playwright",
        "install",
        "chromium",
    ]
    environment = calls[0][1]["env"]
    assert isinstance(environment, dict)
    assert "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD" not in environment
    assert environment["PLAYWRIGHT_BROWSERS_PATH"] == str(
        tmp_path / "build" / "playwright-browsers"
    )
