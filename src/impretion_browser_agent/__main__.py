from __future__ import annotations

import argparse

from .worker import run


def main() -> None:
    parser = argparse.ArgumentParser(prog="impretion-browser-agent")
    # argv carries nothing by contract (see spawn_spec); anything present is
    # ignored so stray flags can never become behavior or leak into logs.
    parser.parse_known_args()
    raise SystemExit(run())


if __name__ == "__main__":
    main()
