#!/usr/bin/env python3
"""Decant CLI — Build, run, and validate AI training configurations.

Usage:
    decant build <file.nx> [-o output.py]
    decant run <file.nx> [--timeout SECONDS]
    decant check <file.nx>
    decant version
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decant.parser import parse_file
from decant.generator import write_generated_code
from decant.executor import run_training
from decant import __version__


def _build_cmd(args: argparse.Namespace) -> int:
    """Handle ``decant build``."""
    try:
        config = parse_file(args.file)
        output = args.output or args.file.replace(".nx", ".py")
        out_path = write_generated_code(config, output)
        print(f"✔ Generated: {out_path}")
        print(f"  Model:    {config.model.base}")
        print(f"  Purpose:  {config.train.purpose}")
        print(f"  Precision: {config.train.precision}")
        print(f"  Compression: {config.train.compression}")
        print(f"  Context:  {config.model.context}", end="")
        if config.model.context_extend:
            print(f" → {config.model.context_extend} (YaRN)")
        else:
            print()
        return 0
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1


def _run_cmd(args: argparse.Namespace) -> int:
    """Handle ``decant run``."""
    try:
        config = parse_file(args.file)
        return run_training(config, generated_path=args.output, timeout=args.timeout)
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1


def _check_cmd(args: argparse.Namespace) -> int:
    """Handle ``decant check`` — validate configuration only."""
    try:
        config = parse_file(args.file)
        print(f"✔ Configuration is valid!")
        print(f"  Model:    {config.model.base if config.model.base else '(not set)'}")
        print(f"  Purpose:  {config.train.purpose}")
        if config.dataset.path:
            print(f"  Dataset:  {config.dataset.path}")
        print(f"  Precision: {config.train.precision}")
        print(f"  Compression: {config.train.compression}")
        print(f"  Context:  {config.model.context}", end="")
        if config.model.context_extend:
            print(f" → {config.model.context_extend} ({config.model.context_method.upper()})", end="")
        print()
        return 0
    except Exception as e:
        parts = str(e).split("\n")
        for part in parts:
            print(f"❌ {part}", file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decant",
        description="Decant DSL — A minimal language for AI model training & hosting",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = sub.add_parser("build", help="Generate Python training code (no execution)")
    build_p.add_argument("file", help="Path to .nx configuration file")
    build_p.add_argument("-o", "--output", default=None, help="Output .py file path")
    build_p.set_defaults(func=_build_cmd)

    # run
    run_p = sub.add_parser("run", help="Generate and execute training")
    run_p.add_argument("file", help="Path to .nx configuration file")
    run_p.add_argument("-o", "--output", default=None, help="Write generated .py before running")
    run_p.add_argument("--timeout", type=int, default=None, help="Max runtime in seconds")
    run_p.set_defaults(func=_run_cmd)

    # check
    check_p = sub.add_parser("check", help="Validate configuration without generating code")
    check_p.add_argument("file", help="Path to .nx configuration file")
    check_p.set_defaults(func=_check_cmd)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
