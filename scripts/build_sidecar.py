from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_TARGETS = frozenset(
    {"x86_64-pc-windows-msvc", "aarch64-apple-darwin"}
)


class PackagingError(RuntimeError):
    """Raised when the native Browser Agent package cannot be trusted."""


def resolve_app_tauri() -> Path:
    """
    Resolve impretion-app/src-tauri for both the local monorepo layout and CI.

    Optional override:
      IMPRETION_APP_ROOT=/absolute/path/to/impretion-app
    """
    override = os.environ.get("IMPRETION_APP_ROOT")
    if override:
        app_root = Path(override).expanduser().resolve()
        return app_root if app_root.name == "src-tauri" else app_root / "src-tauri"

    candidates = [
        ROOT.parents[1] / "impretion-app" / "src-tauri",
        ROOT.parent / "impretion-app" / "src-tauri",
    ]
    return next(
        (candidate for candidate in candidates if candidate.is_dir()),
        candidates[0],
    )


APP_TAURI = resolve_app_tauri()
RESOURCE_ROOT = APP_TAURI / "resources" / "browser-agent"


def current_target(
    system: str | None = None,
    machine: str | None = None,
) -> str:
    system = system or platform.system()
    machine = machine or platform.machine()
    normalized_machine = machine.lower()

    if system == "Windows" and normalized_machine in {"amd64", "x86_64"}:
        return "x86_64-pc-windows-msvc"

    if system == "Darwin" and normalized_machine in {"arm64", "aarch64"}:
        return "aarch64-apple-darwin"

    raise PackagingError(
        f"Unsupported Browser Agent build host: {system} {machine}. "
        f"Build natively on one of: {', '.join(sorted(SUPPORTED_TARGETS))}."
    )


def require_supported_target(target: str) -> None:
    if target not in SUPPORTED_TARGETS:
        raise PackagingError(
            f"Unsupported Browser Agent target {target!r}; supported targets are: "
            f"{', '.join(sorted(SUPPORTED_TARGETS))}."
        )


def executable_suffix(target: str) -> str:
    require_supported_target(target)
    return ".exe" if target == "x86_64-pc-windows-msvc" else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackagingError(f"Missing {description}: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackagingError(
            f"Malformed {description}: {path}: {error}"
        ) from error

    if not isinstance(value, dict):
        raise PackagingError(
            f"Malformed {description}: expected a JSON object at {path}"
        )

    return value


def load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackagingError(f"Missing {description}: {path}") from error
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise PackagingError(
            f"Cannot read {description}: {path}: {error}"
        ) from error

    if not isinstance(value, dict):
        raise PackagingError(
            f"Malformed {description}: expected a TOML table at {path}"
        )

    return value


def project_version(
    pyproject_path: Path | None = None,
) -> str:
    pyproject_path = pyproject_path or ROOT / "pyproject.toml"

    pyproject = load_toml(pyproject_path, "pyproject.toml")

    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise PackagingError(
            "pyproject.toml must contain a [project] table"
        )

    sidecar_version = project.get("version")
    if not isinstance(sidecar_version, str) or not sidecar_version:
        raise PackagingError(
            "pyproject.toml must define project.version"
        )

    return sidecar_version


def validate_locked_dependencies() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise PackagingError(
            "uv is required to validate uv.lock"
        )

    subprocess.run(
        [uv, "lock", "--check"],
        check=True,
        cwd=ROOT,
    )

    return project_version()


def build_sidecar(target: str) -> Path:
    require_supported_target(target)

    spec_path = ROOT / "browser-agent.spec"

    if not spec_path.is_file():
        raise PackagingError(
            f"Missing PyInstaller spec: {spec_path}"
        )

    output = (
        ROOT
        / "dist"
        / f"impretion-browser-agent{executable_suffix(target)}"
    )

    if output.exists():
        output.unlink()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(spec_path),
        ],
        check=True,
        cwd=ROOT,
    )

    if not output.is_file():
        raise PackagingError(
            f"PyInstaller output is missing: {output}"
        )

    return output


def copy_sidecar(
    target: str,
    sidecar_source: Path,
) -> Path:
    require_supported_target(target)

    if not APP_TAURI.is_dir():
        raise PackagingError(
            f"Impretion Tauri directory is missing: {APP_TAURI}. "
            "Set IMPRETION_APP_ROOT if the repositories use "
            "a different layout."
        )

    sidecar = (
        APP_TAURI
        / "binaries"
        / (
            f"impretion-browser-agent-{target}"
            f"{executable_suffix(target)}"
        )
    )

    sidecar.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        sidecar_source,
        sidecar,
    )

    if target != "x86_64-pc-windows-msvc":
        sidecar.chmod(0o755)

    return sidecar


def source_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PackagingError(
            "Cannot resolve Browser Agent source commit"
        ) from error

    if not commit:
        raise PackagingError(
            "Browser Agent source commit is empty"
        )

    return commit


def write_source_lock(
    target: str,
    sidecar: Path,
    sidecar_version: str,
) -> Path:
    require_supported_target(target)

    RESOURCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_lock_path = RESOURCE_ROOT / "source-lock.json"

    source_lock = {
        "browserAgentCommit": source_commit(),
        "protocolVersion": 1,
        "sidecarVersion": sidecar_version,
        "buildTarget": target,
        "sidecarBinarySha256": sha256(sidecar),
    }

    source_lock_path.write_text(
        json.dumps(source_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    return source_lock_path


def verify_artifacts(
    target: str,
    sidecar_version: str,
    *,
    app_tauri: Path | None = None,
) -> None:
    require_supported_target(target)

    app_root = app_tauri or APP_TAURI

    resource_root = (
        app_root
        / "resources"
        / "browser-agent"
    )

    source_lock_path = (
        resource_root
        / "source-lock.json"
    )

    source_lock = load_json(
        source_lock_path,
        "source lock",
    )

    expected_lock_values: dict[str, Any] = {
        "buildTarget": target,
        "sidecarVersion": sidecar_version,
        "protocolVersion": 1,
    }

    for name, expected in expected_lock_values.items():
        if source_lock.get(name) != expected:
            raise PackagingError(
                f"Source lock {name} does not match "
                "the package metadata"
            )

    sidecar = (
        app_root
        / "binaries"
        / (
            f"impretion-browser-agent-{target}"
            f"{executable_suffix(target)}"
        )
    )

    if not sidecar.is_file():
        raise PackagingError(
            "Bundled Browser Agent sidecar is missing: "
            f"{sidecar}"
        )

    if source_lock.get(
        "sidecarBinarySha256"
    ) != sha256(sidecar):
        raise PackagingError(
            "Bundled Browser Agent sidecar hash is incorrect"
        )

    commit = source_lock.get("browserAgentCommit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or not all(byte in "0123456789abcdefABCDEF" for byte in commit)
    ):
        raise PackagingError(
            "Source lock has no valid browserAgentCommit"
        )


def package() -> None:
    target = current_target()

    sidecar_version = validate_locked_dependencies()

    sidecar_output = build_sidecar(target)

    sidecar = copy_sidecar(
        target,
        sidecar_output,
    )

    write_source_lock(
        target,
        sidecar,
        sidecar_version,
    )

    verify_artifacts(
        target,
        sidecar_version,
    )

    print(
        f"Browser Agent package verified for {target}"
    )
    print(f"Sidecar: {sidecar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Package the native Browser Agent sidecar"
        )
    )

    parser.add_argument(
        "action",
        choices=("package",),
        help=(
            "Build, copy and verify "
            "the native package"
        ),
    )

    parser.parse_args()

    try:
        package()
    except (
        PackagingError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(
            1,
            f"Browser Agent packaging failed: {error}\n",
        )


if __name__ == "__main__":
    main()
