from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path
from typing import Any

import psutil
from playwright.async_api import async_playwright

from .config import SidecarConfig
from .security import validate_runtime_manifest


async def verify_runtime(config: SidecarConfig) -> dict[str, Any]:
    manifest = validate_runtime_manifest(config.manifest_path, config.chromium_executable, config.build_target)
    try:
        async with asyncio.timeout(30):
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=str(config.chromium_executable),
                    headless=True,
                    timeout=20_000,
                )
                try:
                    page = await browser.new_page()
                    await page.goto("data:text/html,<title>impretion-runtime-ready</title>", timeout=5_000)
                    if await page.title() != "impretion-runtime-ready":
                        raise RuntimeError("Bundled Chromium verification page failed")
                finally:
                    await browser.close()
    except TimeoutError as error:
        raise RuntimeError("Bundled Chromium runtime verification timed out") from error
    return manifest


def playwright_version() -> str:
    return importlib.metadata.version("playwright")


async def watch_parent(parent_pid: int, shutdown_event: asyncio.Event) -> None:
    while not shutdown_event.is_set():
        if parent_pid <= 1 or not psutil.pid_exists(parent_pid):
            shutdown_event.set()
            return
        await asyncio.sleep(2)
