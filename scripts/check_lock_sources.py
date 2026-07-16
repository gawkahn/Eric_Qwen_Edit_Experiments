#!/usr/bin/env python3
"""Verify every uv.lock package comes from a hash-verified registry (quality-gate kit adoption 2026-07-16).

Tier-3 dependency hygiene (project CLAUDE.md §Dependency hygiene; global §11):
every external artifact must be installable only against a recorded integrity
hash. uv records a per-artifact sha256 for every `registry` source, and
`uv sync --locked` verifies them on install — so "no non-registry source" is
equivalent to "every external artifact is hash-verified". The sole allowed
exception is our own project (`editable`/`virtual = "."`), which is local source,
not a supply-chain artifact.

Exits non-zero if any package uses a git / path / url / directory source, which
would bypass digest verification. See
docs/decisions/ADR-031-license-policy.md (supply-chain gate; tier-3 rule per global §11).
"""
from __future__ import annotations

import sys
import tomllib

# Source kinds that carry (or are) hash-verified / local-trusted artifacts.
_REGISTRY = "registry"
_LOCAL_SELF = {"editable", "virtual"}  # our own project root, source "."


def find_offenders(lock: dict) -> list[tuple[str, str, object]]:
    """Return (name, source_kind, source_value) for every package that is NOT a
    hash-verified registry source or our own local project."""
    offenders = []
    for pkg in lock.get("package", []):
        source = pkg.get("source", {})
        kind = next(iter(source), None)
        if kind == _REGISTRY:
            continue
        if kind in _LOCAL_SELF and source.get(kind) == ".":
            continue
        offenders.append((pkg.get("name", "?"), kind or "none", source.get(kind) if kind else None))
    return offenders


def main() -> int:
    with open("uv.lock", "rb") as fh:
        lock = tomllib.load(fh)
    offenders = find_offenders(lock)
    if offenders:
        print(
            f"TIER-3 VIOLATION — {len(offenders)} package(s) from a non-registry "
            "(non-hash-verified) source:"
        )
        for name, kind, val in offenders:
            print(f"  {name}: {kind} = {val}")
        print(
            "\nEvery external dependency must be registry-sourced (hash-verified by "
            "`uv sync --locked`). A git/path/url source bypasses digest verification. "
            "See docs/decisions/ADR-031-license-policy.md (supply-chain gate; tier-3 rule per global §11)."
        )
        return 1
    n = len(lock.get("package", []))
    print(
        f"All {n} uv.lock packages are registry-sourced (hash-verified) or the "
        "local project — tier-3 holds."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
