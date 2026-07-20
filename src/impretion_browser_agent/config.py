from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _required_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return Path(value).expanduser().resolve(strict=True)


@dataclass(frozen=True)
class SidecarConfig:
    local_token: str
    ai_processor_base_url: str
    database_path: Path
    workspace_root: Path
    manifest_path: Path
    chromium_executable: Path
    parent_pid: int
    instance_id: str
    build_target: str
    max_concurrency: int = 1
    maximum_queue_length: int = 32

    @classmethod
    def from_environment(cls) -> "SidecarConfig":
        token = os.environ.get("IMPRETION_BROWSER_AGENT_TOKEN", "").strip()
        if len(token) < 43:
            raise RuntimeError("IMPRETION_BROWSER_AGENT_TOKEN must contain at least 256 bits")
        base_url = os.environ.get("IMPRETION_AI_PROCESSOR_BASE_URL", "").strip().rstrip("/")
        if not base_url.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise RuntimeError("IMPRETION_AI_PROCESSOR_BASE_URL is invalid")
        return cls(
            local_token=token,
            ai_processor_base_url=base_url,
            database_path=Path(os.environ["IMPRETION_BROWSER_AGENT_DATABASE_PATH"]).resolve(),
            workspace_root=_required_path("IMPRETION_BROWSER_AGENT_WORKSPACE_ROOT"),
            manifest_path=_required_path("IMPRETION_BROWSER_AGENT_RUNTIME_MANIFEST"),
            chromium_executable=_required_path("IMPRETION_BROWSER_AGENT_CHROMIUM_EXECUTABLE"),
            parent_pid=int(os.environ["IMPRETION_BROWSER_AGENT_PARENT_PID"]),
            instance_id=os.environ["IMPRETION_BROWSER_AGENT_INSTANCE_ID"],
            build_target=os.environ["IMPRETION_BROWSER_AGENT_BUILD_TARGET"],
            max_concurrency=max(1, min(int(os.environ.get("IMPRETION_BROWSER_AGENT_MAX_CONCURRENCY", "1")), 4)),
            maximum_queue_length=max(1, min(int(os.environ.get("IMPRETION_BROWSER_AGENT_MAX_QUEUE", "32")), 128)),
        )

