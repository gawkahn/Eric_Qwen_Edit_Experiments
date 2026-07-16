#!/usr/bin/env python3
"""Check dependency licenses against ADR-031 (quality-gate kit adoption).

Reads `pip-licenses --format=json` from stdin and exits non-zero if any
dependency (excluding our own package) carries a license that is not on the
ADR-031 allowlist. Default-deny: an unknown/unlisted license fails.

Compound-expression semantics (kit-inherited):
  - `A OR B`, `A; B`, `A, B`  → allowed if EITHER operand is allowed
  - `A AND B`                 → allowed only if BOTH operands are allowed

Per-package exceptions (ADR-031 §Decision-3/4): NVIDIA CUDA runtime wheels are
allowed by package-NAME scope (their license strings are proprietary but they
are torch's hard runtime companions), and two packages with missing license
metadata are excepted with their verified upstream license. Adding an
exception requires an ADR-031 Changelog append.

Policy: docs/decisions/ADR-031-license-policy.md. This is the enforcement arm
of that ADR; keep the two in sync.
"""
from __future__ import annotations

import json
import re
import sys

# Normalized allowlist tokens (ADR-031 §Decision-1).
_ALLOWED = {
    "MIT", "BSD", "APACHE", "ISC", "PSF", "MPL-2.0", "CNRI-PYTHON",
    "0BSD", "ZLIB", "CC0", "UNLICENSE",
}

# Our own package — the policy applies to dependencies, not to us.
_SELF = {"comfyui-eric-qwen-edit"}

# NVIDIA CUDA runtime wheels: allowed by name scope (ADR-031 §Decision-3).
_NVIDIA_NAME_RE = re.compile(r"^(nvidia-|cuda-)")

# Missing-metadata exceptions, name → verified upstream license (§Decision-4).
_PACKAGE_EXCEPTIONS = {
    "sentencepiece": "Apache-2.0 upstream (metadata says UNKNOWN)",
    "torchao": "BSD-3-Clause upstream (metadata says UNKNOWN)",
}


def _normalize_atom(atom: str) -> str | None:
    """Map one license string (SPDX id or classifier spelling) to an allowlist
    token, a ``__DENIED_GPL__`` sentinel, or None (unknown → default-deny).

    Matches are word-boundary/version-anchored, NOT bare substrings: the
    ``License`` metadata field is free prose, and ``"MIT" in a`` would pass
    "per**MIT**ted"/"li**MIT**ed" while ``"MPL" in a`` would pass "si**MPL**e".
    The permissive patterns are also pinned to the allowed *versions* —
    BSD-2/3-Clause but not BSD-4-Clause/BSD-Protection, Apache-2.0 but not
    Apache-1.x, MPL-2.0 but not MPL-1.1. Copyleft is matched first;
    over-denying is safe. An ``Apache/MIT WITH <exception>`` only adds
    permissions to a permissive base, so ignoring the exception is safe.
    """
    a = atom.strip().upper()
    if not a:
        return None
    # Strong/network copyleft first (GPL/AGPL/LGPL, any suffix: -only/-or-later/
    # -3.0/v3/ WITH exception). No allowlisted license contains "GPL".
    if re.search(r"\b[AL]?GPL", a):
        return "__DENIED_GPL__"
    # 0BSD before the MIT/BSD checks (it is its own SPDX id).
    if re.search(r"\b0BSD\b", a):
        return "0BSD"
    if re.search(r"\bMIT\b", a):
        return "MIT"
    # BSD-2/3-Clause or the classic "BSD License" classifier — NOT BSD-4-Clause
    # (advertising) or BSD-Protection (copyleft).
    if re.search(r"\bBSD-[23]-CLAUSE\b|\bBSD LICENSE\b|\b[23]-CLAUSE BSD\b", a) or a == "BSD":
        return "BSD"
    # Apache-2.0 / "Apache Software License" classifier (= 2.0) — NOT Apache-1.x.
    if re.search(r"\bAPACHE-2(\.0)?\b|\bAPACHE SOFTWARE LICENSE\b|APACHE LICENSE.{0,10}\b2|\bAPACHE 2(\.0)?\b", a):
        return "APACHE"
    if re.search(r"\bISC\b", a):
        return "ISC"
    # MPL-2.0 only — NOT MPL-1.1.
    if re.search(r"\bMPL-2(\.0)?\b|\bMPL 2(\.0)?\b|MOZILLA PUBLIC LICENSE 2", a):
        return "MPL-2.0"
    if re.search(r"\bCNRI\b", a):
        return "CNRI-PYTHON"
    if a in {"PSF-2.0", "PYTHON-2.0", "PSF"} or "PYTHON SOFTWARE FOUNDATION" in a:
        return "PSF"
    if re.search(r"\bZLIB\b", a):
        return "ZLIB"
    if re.search(r"\bCC0(-1\.0)?\b", a):
        return "CC0"
    if re.search(r"\bUNLICENSE\b", a):
        return "UNLICENSE"
    return None


def license_allowed(expr: str) -> bool:
    """True iff *expr* is acceptable under ADR-031. Empty/unknown → False."""
    if not expr or not expr.strip():
        return False
    # Some packages (e.g. tiktoken) embed the FULL license text in the License
    # field; the declared license is its first line ("MIT License\n\nCopyright
    # ..."). Only the first non-empty line is the SPDX-ish expression — parsing
    # the prose would split on the text's own ANDs and default-deny.
    expr = next(line for line in expr.splitlines() if line.strip())
    # Split on AND first (every AND-part must pass); within a part, OR/;/,
    # separators mean any single operand may satisfy it.
    for and_part in re.split(r"\bAND\b", expr, flags=re.IGNORECASE):
        or_atoms = re.split(r"\bOR\b|;|,", and_part, flags=re.IGNORECASE)
        tokens = [_normalize_atom(a) for a in or_atoms]
        if not any(t in _ALLOWED for t in tokens):
            return False
    return True


def main() -> int:
    data = json.load(sys.stdin)
    if not data:
        # A valid-but-empty scan means pip-licenses saw no packages — a
        # misconfigured/empty environment, not a clean bill. Refuse to pass.
        print("no packages in scan — refusing to pass (misconfigured environment?)")
        return 1
    violations = []
    for pkg in data:
        name = pkg.get("Name", "").lower().replace("_", "-")
        if name in _SELF:
            continue
        if _NVIDIA_NAME_RE.match(name):
            continue  # ADR-031 §Decision-3: CUDA runtime wheels, name-scoped
        if name in _PACKAGE_EXCEPTIONS:
            continue  # ADR-031 §Decision-4: verified-permissive metadata gaps
        lic = pkg.get("License", "") or ""
        if not license_allowed(lic):
            violations.append((pkg.get("Name", ""), pkg.get("Version", ""), lic))
    if violations:
        print(
            f"LICENSE POLICY VIOLATION — {len(violations)} package(s) not on the "
            "ADR-031 allowlist:"
        )
        for name, ver, lic in violations:
            print(f"  {name} {ver}: {lic!r}")
        print(
            "\nSee docs/decisions/ADR-031-license-policy.md — add to the allowlist "
            "(with rationale + Changelog append) or drop the dependency."
        )
        return 1
    print(f"All {len(data)} package licenses conform to ADR-031.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
