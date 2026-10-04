"""Print a SHA-256 checksum for every raw XML file below a directory, sorted by path.

The Makefile compares this output with the previous run to detect added, removed or
changed raw files, independent of file modification times and of platform tools.

Usage: ``python -m energy_price.checksums data/ausgleichpreis``
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def checksums(raw_dir: str | Path) -> list[str]:
    """Return one ``<sha256>  <path>`` line per XML file below ``raw_dir``."""
    return [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.as_posix()}"
        for path in sorted(Path(raw_dir).rglob("*.xml"))
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_dir", type=Path, help="directory with the Swissgrid XML files")
    args = parser.parse_args(argv)

    for line in checksums(args.raw_dir):
        print(line)


if __name__ == "__main__":
    main()
