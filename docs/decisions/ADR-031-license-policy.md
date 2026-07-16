# ADR-031 — Dependency license policy (supply-chain gate)

Status:   accepted
Date:     2026-07-16

## Context

The quality-gate kit's supply-chain gate (adopted from
`~/.claude/templates/quality-gate-kit-python-uv/`, reference impl local_agents
ADR-006) includes a default-deny license check: `pip-licenses` output is piped
through `scripts/check_licenses.py`, which fails on any dependency whose
license is not on an explicit allowlist. Before wiring that gate, this repo
needs its own policy — the reference repo's allowlist (pure-OSI: MIT / BSD /
Apache / ISC / PSF / MPL-2.0 / CNRI) does not fit a CUDA project: `torch`
drags in ~17 NVIDIA runtime wheels carrying proprietary licenses
(`LicenseRef-NVIDIA-Proprietary`, `NVIDIA Proprietary Software`, and the
generic `Other/Proprietary License` classifier), and several packages ship
compound SPDX expressions with permissive atoms the reference set doesn't
name (0BSD, Zlib, CC0-1.0).

Measured baseline (2026-07-16, 103 locked packages): all non-conforming
entries are either (a) NVIDIA CUDA runtime wheels, or (b) missing-metadata
packages whose upstream license is verifiably permissive (`sentencepiece` =
Apache-2.0, `torchao` = BSD-3-Clause, `cuda-toolkit` = NVIDIA meta-package).

This is a solo personal project distributing a ComfyUI node pack under its
own LICENSE.txt; dependencies are consumed, not vendored or redistributed.

## Decision

1. **Allowlist (default-deny, license-string based):** MIT, BSD-2/3-Clause,
   Apache-2.0, ISC, PSF, MPL-2.0, CNRI-Python — the kit set — **plus** the
   permissive atoms 0BSD, Zlib, CC0-1.0, Unlicense. Compound-expression
   semantics unchanged from the kit script (`OR`/`;`/`,` = any operand;
   `AND` = all operands).
2. **Copyleft (GPL/AGPL/LGPL in any form) stays denied.** Nothing in the
   current tree needs it; if a future dep does, that's a new ADR, not an
   allowlist tweak.
3. **NVIDIA CUDA runtime wheels are allowed by package-name scope, not by
   license string.** Packages matching `nvidia-*`, `cuda-*` (and
   `cuda-bindings`/`cuda-toolkit`) are accepted regardless of their license
   metadata. Rationale: they are torch's hard runtime companions — there is
   no CUDA without them — and the licenses permit exactly this use
   (redistribution as installed runtime components). Allowing the generic
   `Other/Proprietary License` *string* globally would let any future
   proprietary package pass silently; name-scoping keeps default-deny for
   everything else.
4. **Named per-package exceptions for missing metadata** (each with its
   verified upstream license): `sentencepiece` (Apache-2.0), `torchao`
   (BSD-3-Clause). Exceptions live in `scripts/check_licenses.py`
   `_PACKAGE_EXCEPTIONS` with a rationale string; adding one requires
   appending to this ADR's Changelog.
5. **Own package exempt:** `comfyui-eric-qwen-edit` (policy governs
   dependencies, not us).

## Alternatives Rejected

- **Allowlist the `Other/Proprietary License` classifier string** — passes
  any future proprietary dep silently; breaks default-deny.
- **Drop the license gate entirely** (solo project, nothing redistributed
  beyond the node pack) — cheap to keep, and the gate's real value is
  catching a surprise GPL/AGPL transitively entering the tree.
- **Vendored license review per release** — heavier than the automated gate
  with no added assurance at this project's scale.

## Deferred / Out of Scope

- License scanning of model weights / LoRA files (data, not code deps).
- The node-pack pip path (`requirements.txt` consumers): the gate runs
  against the uv lock; the direct-dep agreement rule keeps the two manifests
  aligned, so coverage is equivalent at the top level.

## Changelog

- 2026-07-16 — accepted; initial policy written before wiring the
  supply-chain gate (quality-gate kit adoption).

AI-Disclosure: Claude (Fable 5) authored; Grant reviewed.
