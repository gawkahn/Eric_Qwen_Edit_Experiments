#!/usr/bin/env python3
"""Verify every uv.lock package comes from a hash-verified registry (quality-gate kit adoption 2026-07-16).

Tier-3 dependency hygiene (project CLAUDE.md §Dependency hygiene; global §11):
every external artifact must be installable only against a recorded integrity
hash. uv records a per-artifact sha256 for every `registry` source, and
`uv sync --locked` verifies them on install — so "no non-registry source" is
equivalent to "every external artifact is hash-verified". The allowed
exceptions are our own project (`editable`/`virtual = "."`), which is local
source rather than a supply-chain artifact, and our own runtime core pinned by
git+https to an immutable commit id (see `_SIBLING_CORE_*` below).

Exits non-zero if any package uses a git / path / url / directory source, which
would bypass digest verification. See
docs/decisions/ADR-031-license-policy.md (supply-chain gate; tier-3 rule per global §11).
"""
from __future__ import annotations

import re
import sys
import tomllib

# Source kinds that carry (or are) hash-verified / local-trusted artifacts.
_REGISTRY = "registry"
_LOCAL_SELF = {"editable", "virtual"}  # our own project root, source "."
# The one non-registry dependency this repo is allowed to carry: our own
# runtime core, pinned by git+https to an immutable commit (ADR-045).
_SIBLING_CORE_NAME = "comfyless-diffusion"
_SIBLING_CORE_URL = "https://github.com/gawkahn/comfyless_diffusion.git"
# uv writes `<url>?tag=<tag>#<40-hex commit>`. The commit fragment is the whole
# point of allowing this at all, so it is matched, not assumed.
_COMMIT_FRAGMENT = re.compile(r"#[0-9a-f]{40}$")


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
        url = source.get(kind)
        if (pkg.get("name") == _SIBLING_CORE_NAME
                and kind == "git"
                and isinstance(url, str)
                and url.startswith(_SIBLING_CORE_URL)
                # startswith ALONE is a hole: any continuation of the allowed
                # URL also matches. `…/comfyless_diffusion.git/../../evil/repo.git`
                # passes a naive prefix test, and git's libcurl transport squashes
                # the dot-segments, so it would actually fetch the attacker's repo
                # — with a real 40-hex commit of THEIR tree. Require the very next
                # character to end the URL proper. (Caught in review 2026-09-12;
                # the first version of this check shipped the hole.)
                and (len(url) == len(_SIBLING_CORE_URL)
                     or url[len(_SIBLING_CORE_URL)] in "?#")
                and _COMMIT_FRAGMENT.search(url)):
            # ADR-045: the runtime core is our own code, pinned by git+https to
            # an immutable commit. It carries no registry sha256, but a commit
            # id is itself content-addressed, so the artifact is pinned exactly
            # — which is what tier-3 is protecting.
            #
            # This REPLACED an `editable`+path exception (2026-09-12), and is
            # deliberately narrower: name AND exact repo URL AND a real 40-hex
            # commit must all match. A tag alone would NOT pass, because tags
            # are mutable and could be repointed at different code.
            #
            # Retire this exception entirely if comfyless-diffusion is ever
            # published to a registry (Vision open question: PyPI vs git+https).
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
