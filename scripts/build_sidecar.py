from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_TAURI = ROOT.parents[1] / "impretion-app" / "src-tauri"
PLAYWRIGHT_VERSION = "1.59.0"
CHROMIUM_REVISION = "1217"
CHROMIUM_VERSION = "147.0.7727.15"


def target_triple() -> str:
    machine = {"arm64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}[platform.machine()]
    if platform.system() == "Darwin":
        return f"{machine}-apple-darwin"
    if platform.system() == "Windows":
        return f"{machine}-pc-windows-msvc"
    return f"{machine}-unknown-linux-gnu"


def prepare_chromium() -> None:
    destination = ROOT / "build" / "playwright-browsers"
    environment = {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": str(destination)}
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "playwright", "install", "chromium"], check=True, env=environment)
    source_root = destination / f"chromium-{CHROMIUM_REVISION}"
    executable = next(path for path in source_root.rglob("*") if path.is_file() and path.name in {"chrome", "Google Chrome for Testing", "chrome.exe"})
    platform_root = executable
    while platform_root.parent != source_root:
        platform_root = platform_root.parent
    resource_root = APP_TAURI / "resources/browser-agent"
    target_root = resource_root / "chromium" / target_triple()
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(platform_root, target_root)
    installed_executable = target_root / executable.relative_to(platform_root)
    relative = installed_executable.relative_to(resource_root).as_posix()
    digest = hashlib.sha256(installed_executable.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "playwrightVersion": PLAYWRIGHT_VERSION,
        "chromiumRevision": CHROMIUM_REVISION,
        "chromiumVersion": CHROMIUM_VERSION,
        "targets": {target_triple(): {"executableRelativePath": relative, "sha256": digest}},
    }
    resource_root.mkdir(parents=True, exist_ok=True)
    (resource_root / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_sidecar() -> None:
    subprocess.run([str(ROOT / ".venv/bin/pyinstaller"), "--noconfirm", "--clean", "browser-agent.spec"], check=True, cwd=ROOT)


def copy_binary() -> None:
    source = ROOT / "dist/impretion-browser-agent"
    suffix = ".exe" if platform.system() == "Windows" else ""
    destination = APP_TAURI / "binaries" / f"impretion-browser-agent-{target_triple()}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o755)


def source_lock() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest_path = APP_TAURI / "resources/browser-agent/runtime-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    suffix = ".exe" if platform.system() == "Windows" else ""
    binary_path = APP_TAURI / "binaries" / f"impretion-browser-agent-{target_triple()}{suffix}"
    lock = {
        "browserAgentCommit": commit,
        "protocolVersion": 1,
        "sidecarVersion": "0.1.0",
        "playwrightVersion": PLAYWRIGHT_VERSION,
        "chromiumRevision": CHROMIUM_REVISION,
        "buildTarget": target_triple(),
        "runtimeManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sidecarBinarySha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
    }
    (APP_TAURI / "resources/browser-agent/source-lock.json").write_text(json.dumps(lock, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-chromium", "build", "copy", "source-lock"))
    action = parser.parse_args().action
    {"prepare-chromium": prepare_chromium, "build": build_sidecar, "copy": copy_binary, "source-lock": source_lock}[action]()


if __name__ == "__main__":
    main()
