"""
Command line interface for PolarPandas.

Provides a lightweight entry point that surfaces basic
package information without importing heavy dependencies.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polarpandas",
        description="Utility commands for the PolarPandas package.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the PolarPandas version and exit.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a short summary of the project.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Entry point for the `polarpandas` console script.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit status code.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.version:
        print(__version__)
        return 0

    if args.summary:
        print(
            "PolarPandas – pandas-compatible API powered by Polars. "
            "Use '--version' to view the installed version."
        )
        return 0

    # No explicit option selected: show help to stderr to mirror argparse CLI UX.
    parser.print_help(sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

