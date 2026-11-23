#!/usr/bin/env python3
"""Command-line entry point.

  python tools/sitegen.py export           write web/src/data/lessons.json (run before the web app)
  python tools/sitegen.py check [FILTER]   run every code example and compare with its ```output block
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sitegen import config, export, validator  # noqa: E402


def cmd_export(_args: argparse.Namespace) -> int:
    data = export.export()
    total = sum(l["minutes"] for l in data["lessons"])
    print(f"Exported {len(data['lessons'])} lessons ({total} min of reading) to {config.WEB_DATA_FILE}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    return 1 if validator.check(args.filter) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="write the lesson data for the React app").set_defaults(func=cmd_export)
    p_check = sub.add_parser("check", help="run every code example (cpp needs g++, python needs python)")
    p_check.add_argument("filter", nargs="?", help="only lessons whose file name contains this text")
    p_check.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
