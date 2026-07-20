from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys

import uvicorn

from .app import create_app
from .config import SidecarConfig
from .protocol import BROWSER_AGENT_PROTOCOL_VERSION, READY_PREFIX, SIDECAR_VERSION
from .runtime import verify_runtime


def main() -> None:
    parser = argparse.ArgumentParser(prog="impretion-browser-agent")
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    config = SidecarConfig.from_environment()
    if args.verify_runtime:
        asyncio.run(verify_runtime(config))
        return
    asyncio.run(serve(config))


async def serve(config: SidecarConfig) -> None:
    bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bound_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    bound_socket.bind(("127.0.0.1", 0))
    bound_socket.listen(2048)
    bound_socket.setblocking(False)
    port = int(bound_socket.getsockname()[1])
    app = create_app(config)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False))
    task = asyncio.create_task(server.serve(sockets=[bound_socket]))
    while not server.started and not task.done():
        await asyncio.sleep(0.01)
    if task.done():
        await task
        return
    print(READY_PREFIX + json.dumps({
        "port": port,
        "instanceId": config.instance_id,
        "protocolVersion": BROWSER_AGENT_PROTOCOL_VERSION,
        "sidecarVersion": SIDECAR_VERSION,
    }, separators=(",", ":")), flush=True)
    await create_app_shutdown_wait(app.state.shutdown_event, server, task)


async def create_app_shutdown_wait(event: asyncio.Event, server: uvicorn.Server, task: asyncio.Task[None]) -> None:
    shutdown_wait = asyncio.create_task(event.wait())
    done, _ = await asyncio.wait({task, shutdown_wait}, return_when=asyncio.FIRST_COMPLETED)
    if shutdown_wait in done:
        server.should_exit = True
    await task
    shutdown_wait.cancel()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"event": "sidecar.fatal", "error": str(error)[:2000]}), file=sys.stderr, flush=True)
        raise
