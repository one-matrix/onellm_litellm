#!/usr/bin/env python3
"""Auto-annotate every Prisma model/enum with ``@@schema("public")`` if missing.

Prisma's ``multiSchema`` preview requires every model/enum to declare which
PostgreSQL schema it belongs to via ``@@schema(...)``. Upstream BerriAI/litellm
doesn't use multiSchema and ships models without that annotation, so every
time we merge from upstream the newly added/modified models would fail
``prisma format`` until they're annotated.

This script idempotently adds ``@@schema("public")`` to any block that
doesn't already have a ``@@schema(...)`` line — letting the upstream merge
flow be:

    git pull upstream main
    cd backend && make schema-fix    # this script + prisma format
    git add backend/*schema.prisma && git commit ...

Our own models in the ``app`` schema (CreditWallet etc.) already have
explicit ``@@schema("app")`` annotations, so this script leaves them alone.

The script targets all three schema.prisma copies we keep in sync (see
CLAUDE.md "Keep schema files in sync"). Pass --check to fail (exit 1) instead
of editing — useful as a CI guard.

Brace-aware: respects string literals like ``Json @default("{}")`` so the
inner ``{}`` aren't counted as block delimiters.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ANNOTATION_LINE = '  @@schema("public")\n'

DEFAULT_TARGETS = [
    "schema.prisma",
    "litellm/proxy/schema.prisma",
    "litellm-proxy-extras/litellm_proxy_extras/schema.prisma",
]


def _annotate_text(text: str) -> tuple[str, int]:
    """Return (new_text, blocks_changed)."""
    out_lines: list[str] = []
    in_block = False
    block_buffer: list[str] = []
    block_depth = 0
    block_header_re = re.compile(r"^(model|enum)\s+\w+\s*\{")
    changed_blocks = 0

    for line in text.splitlines(keepends=True):
        if not in_block:
            if block_header_re.match(line):
                in_block = True
                block_depth = 1
                block_buffer = [line]
                continue
            out_lines.append(line)
            continue

        block_buffer.append(line)
        # Strip string literals so JSON-default {} don't shift the counter.
        stripped = re.sub(r'"([^"\\]|\\.)*"', '""', line)
        for ch in stripped:
            if ch == "{":
                block_depth += 1
            elif ch == "}":
                block_depth -= 1
                if block_depth == 0:
                    has_schema = any("@@schema(" in bl for bl in block_buffer)
                    if not has_schema:
                        out_lines.extend(block_buffer[:-1])
                        out_lines.append(ANNOTATION_LINE)
                        out_lines.append(block_buffer[-1])
                        changed_blocks += 1
                    else:
                        out_lines.extend(block_buffer)
                    in_block = False
                    block_buffer = []
                    break

    if in_block:
        raise RuntimeError("unbalanced braces — model/enum block did not close")
    return "".join(out_lines), changed_blocks


def process(path: Path, check_only: bool) -> int:
    text = path.read_text()
    new_text, n = _annotate_text(text)
    if n == 0:
        print(f"  {path}: already annotated")
        return 0
    if check_only:
        print(
            f"  {path}: {n} blocks missing @@schema annotation (run without "
            f"--check to fix)"
        )
        return 1
    path.write_text(new_text)
    print(f"  {path}: annotated {n} blocks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        help="schema.prisma file(s) to process; defaults to all three known copies",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any block needs annotation (CI guard mode)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="repo root when default paths are used (default: current dir)",
    )
    args = parser.parse_args()

    if args.paths:
        targets = [Path(p) for p in args.paths]
    else:
        root = Path(args.root)
        targets = [root / t for t in DEFAULT_TARGETS]

    print(
        ("Checking " if args.check else "Annotating ")
        + f"{len(targets)} schema.prisma file(s):"
    )
    rc = 0
    for p in targets:
        if not p.exists():
            print(f"  {p}: SKIP (not found)")
            continue
        rc |= process(p, args.check)
    return rc


if __name__ == "__main__":
    sys.exit(main())
