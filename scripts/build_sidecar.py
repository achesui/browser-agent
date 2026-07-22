from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


def project_metadata(
    pyproject_path: Path | None = None,
    lock_path: Path | None = None,
) -> tuple[str, str]:
    pyproject_path = pyproject_path or ROOT / "pyproject.toml"
    lock_path = lock_path or ROOT / "uv.lock"

    pyproject = load_toml(pyproject_path, "pyproject.toml")
    lock = load_toml(lock_path, "uv.lock")

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

    dependencies = project.get("dependencies")
    if not isinstance(dependencies, list):
        raise PackagingError(
            "pyproject.toml project.dependencies must be a list"
        )

    pins = [
        dependency.removeprefix("playwright==")
        for dependency in dependencies
        if isinstance(dependency, str)
        and dependency.startswith("playwright==")
    ]

    if len(pins) != 1 or not pins[0]:
        raise PackagingError(
            "pyproject.toml must contain exactly one exact "
            "playwright==VERSION dependency"
        )

    lock_versions = [
        package.get("version")
        for package in lock.get("package", [])
        if isinstance(package, dict)
        and package.get("name") == "playwright"
    ]

    if lock_versions != [pins[0]]:
        raise PackagingError(
            f"Playwright lock mismatch: pyproject.toml pins {pins[0]!r}, "
            f"uv.lock contains {lock_versions!r}"
        )

    return sidecar_version, pins[0]


def playwright_browser_manifest_path() -> Path:
    try:
        distribution = importlib.metadata.distribution("playwright")
    except importlib.metadata.PackageNotFoundError as error:
        raise PackagingError(
            "Playwright is not installed in the active Python environment"
        ) from error

    manifest_path = Path(
        str(
            distribution.locate_file(
                "playwright/driver/package/browsers.json"
            )
        )
    )

    if not manifest_path.is_file():
        raise PackagingError(
            "Installed Playwright browser manifest is missing: "
            f"{manifest_path}"
        )

    return manifest_path


def playwright_metadata(
    manifest_path: Path | None = None,
) -> tuple[str, str]:
    manifest = load_json(
        manifest_path or playwright_browser_manifest_path(),
        "installed Playwright browser manifest",
    )

    browsers = manifest.get("browsers")
    if not isinstance(browsers, list):
        raise PackagingError(
            "Malformed Playwright browser manifest: "
            "'browsers' must be a list"
        )

    chromium = next(
        (
            entry
            for entry in browsers
            if isinstance(entry, dict)
            and entry.get("name") == "chromium"
        ),
        None,
    )

    if chromium is None:
        raise PackagingError(
            "Installed Playwright browser manifest "
            "has no Chromium entry"
        )

    revision = chromium.get("revision")
    browser_version = chromium.get("browserVersion")

    if (
        not isinstance(revision, str)
        or not revision
        or not isinstance(browser_version, str)
        or not browser_version
    ):
        raise PackagingError(
            "Installed Playwright Chromium metadata is incomplete"
        )

    return revision, browser_version


def validate_locked_dependencies() -> tuple[str, str]:
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

    sidecar_version, locked_playwright = project_metadata()

    try:
        installed_playwright = importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError as error:
        raise PackagingError(
            "Playwright is not installed in the active Python environment"
        ) from error

    if installed_playwright != locked_playwright:
        raise PackagingError(
            f"Installed Playwright {installed_playwright} does not match "
            f"locked version {locked_playwright}"
        )

    return sidecar_version, locked_playwright


def find_chromium_executable(source_root: Path) -> Path:
    executable_names = {
        "chrome.exe",
        "Google Chrome for Testing",
        "chrome",
        "Chromium",
    }

    candidates = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.name in executable_names
        ),
        key=lambda path: path.as_posix(),
    )

    if not candidates:
        raise PackagingError(
            f"Chromium executable is missing below {source_root}"
        )

    return candidates[0]


def download_chromium(revision: str) -> tuple[Path, Path]:
    browser_cache = ROOT / "build" / "playwright-browsers"

    environment = dict(os.environ)
    environment.pop("PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD", None)
    environment["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "playwright",
            "install",
            "chromium",
        ],
        check=True,
        cwd=ROOT,
        env=environment,
    )

    source_root = browser_cache / f"chromium-{revision}"

    if not source_root.is_dir():
        raise PackagingError(
            "Playwright did not install expected Chromium revision "
            f"{revision}"
        )

    executable = find_chromium_executable(source_root)

    platform_root = executable
    while platform_root.parent != source_root:
        platform_root = platform_root.parent

    if platform_root == source_root:
        raise PackagingError(
            "Cannot determine Chromium platform directory below "
            f"{source_root}"
        )

    return platform_root, executable


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


def replace_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise PackagingError(
            f"Source directory is missing: {source}"
        )

    if destination.exists():
        shutil.rmtree(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        source,
        destination,
        symlinks=True,
    )


def copy_artifacts(
    target: str,
    sidecar_source: Path,
    chromium_source: Path,
    chromium_executable: Path,
) -> tuple[Path, Path]:
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

    chromium_parent = RESOURCE_ROOT / "chromium"

    if chromium_parent.exists():
        shutil.rmtree(chromium_parent)

    chromium_target = chromium_parent / target

    replace_tree(
        chromium_source,
        chromium_target,
    )

    relative_executable = chromium_executable.relative_to(
        chromium_source
    )

    installed_executable = (
        chromium_target / relative_executable
    )

    if not installed_executable.is_file():
        raise PackagingError(
            "Copied Chromium executable is missing: "
            f"{installed_executable}"
        )

    return sidecar, installed_executable


def safe_resource_relative(
    path: Path,
    resource_root: Path | None = None,
) -> Path:
    root = (resource_root or RESOURCE_ROOT).resolve()
    resolved = path.resolve()

    try:
        return resolved.relative_to(root)
    except ValueError as error:
        raise PackagingError(
            "Path escapes Browser Agent resource directory: "
            f"{path}"
        ) from error


def resolve_manifest_executable(
    relative_value: object,
    resource_root: Path | None = None,
) -> Path:
    root = (resource_root or RESOURCE_ROOT).resolve()

    if not isinstance(relative_value, str) or not relative_value:
        raise PackagingError(
            "Runtime manifest has an invalid "
            "Chromium executable path"
        )

    candidate = Path(relative_value)

    if candidate.is_absolute():
        raise PackagingError(
            "Runtime manifest Chromium path must be relative"
        )

    resolved = (root / candidate).resolve()

    safe_resource_relative(
        resolved,
        root,
    )

    return resolved


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


def write_manifests(
    target: str,
    sidecar: Path,
    chromium_executable: Path,
    sidecar_version: str,
    playwright_version: str,
    chromium_revision: str,
    chromium_version: str,
) -> tuple[Path, Path]:
    require_supported_target(target)

    executable_relative = safe_resource_relative(
        chromium_executable
    )

    RESOURCE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_path = RESOURCE_ROOT / "runtime-manifest.json"

    runtime = {
        "schemaVersion": 1,
        "playwrightVersion": playwright_version,
        "chromiumRevision": chromium_revision,
        "chromiumVersion": chromium_version,
        "targets": {
            target: {
                "executableRelativePath": (
                    executable_relative.as_posix()
                ),
                "sha256": sha256(chromium_executable),
            }
        },
    }

    runtime_path.write_text(
        json.dumps(runtime, indent=2) + "\n",
        encoding="utf-8",
    )

    source_lock_path = RESOURCE_ROOT / "source-lock.json"

    source_lock = {
        "browserAgentCommit": source_commit(),
        "protocolVersion": 1,
        "sidecarVersion": sidecar_version,
        "playwrightVersion": playwright_version,
        "chromiumRevision": chromium_revision,
        "chromiumVersion": chromium_version,
        "buildTarget": target,
        "runtimeManifestSha256": sha256(runtime_path),
        "sidecarBinarySha256": sha256(sidecar),
    }

    source_lock_path.write_text(
        json.dumps(source_lock, indent=2) + "\n",
        encoding="utf-8",
    )

    return runtime_path, source_lock_path


def verify_artifacts(
    target: str,
    sidecar_version: str,
    playwright_version: str,
    chromium_revision: str,
    chromium_version: str,
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

    runtime_path = (
        resource_root
        / "runtime-manifest.json"
    )

    source_lock_path = (
        resource_root
        / "source-lock.json"
    )

    runtime = load_json(
        runtime_path,
        "runtime manifest",
    )

    source_lock = load_json(
        source_lock_path,
        "source lock",
    )

    if runtime.get("schemaVersion") != 1:
        raise PackagingError(
            "Runtime manifest schemaVersion must be 1"
        )

    targets = runtime.get("targets")

    if not isinstance(targets, dict):
        raise PackagingError(
            "Runtime manifest targets must be an object"
        )

    if set(targets) != {target}:
        raise PackagingError(
            "Runtime manifest must contain only "
            "the current build target"
        )

    expected_runtime_values = {
        "playwrightVersion": playwright_version,
        "chromiumRevision": chromium_revision,
        "chromiumVersion": chromium_version,
    }

    for name, expected in expected_runtime_values.items():
        if runtime.get(name) != expected:
            raise PackagingError(
                f"Runtime manifest {name} does not match "
                "the package metadata"
            )

    expected_lock_values = {
        "buildTarget": target,
        "sidecarVersion": sidecar_version,
        "playwrightVersion": playwright_version,
        "chromiumRevision": chromium_revision,
        "chromiumVersion": chromium_version,
    }

    for name, expected in expected_lock_values.items():
        if source_lock.get(name) != expected:
            raise PackagingError(
                f"Source lock {name} does not match "
                "the package metadata"
            )

    target_entry = targets.get(target)

    if not isinstance(target_entry, dict):
        raise PackagingError(
            "Runtime manifest has no valid entry "
            f"for target {target}"
        )

    chromium = resolve_manifest_executable(
        target_entry.get("executableRelativePath"),
        resource_root,
    )

    if not chromium.is_file():
        raise PackagingError(
            "Bundled Chromium executable is missing: "
            f"{chromium}"
        )

    if target_entry.get("sha256") != sha256(chromium):
        raise PackagingError(
            "Bundled Chromium executable hash is incorrect"
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

    if source_lock.get(
        "runtimeManifestSha256"
    ) != sha256(runtime_path):
        raise PackagingError(
            "Runtime manifest hash is incorrect"
        )


def package() -> None:
    target = current_target()

    (
        sidecar_version,
        playwright_version,
    ) = validate_locked_dependencies()

    (
        chromium_revision,
        chromium_version,
    ) = playwright_metadata()

    (
        chromium_root,
        chromium_executable,
    ) = download_chromium(
        chromium_revision
    )

    sidecar_output = build_sidecar(target)

    (
        sidecar,
        installed_chromium,
    ) = copy_artifacts(
        target,
        sidecar_output,
        chromium_root,
        chromium_executable,
    )

    write_manifests(
        target,
        sidecar,
        installed_chromium,
        sidecar_version,
        playwright_version,
        chromium_revision,
        chromium_version,
    )

    verify_artifacts(
        target,
        sidecar_version,
        playwright_version,
        chromium_revision,
        chromium_version,
    )

    print(
        f"Browser Agent package verified for {target}"
    )
    print(f"Sidecar: {sidecar}")
    print(f"Chromium: {installed_chromium}")


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
            "Build, copy, manifest and verify "
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