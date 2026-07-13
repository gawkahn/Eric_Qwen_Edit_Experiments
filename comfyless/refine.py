"""comfyless iterative refinement loop (LLM-as-judge) — ADR-027.

Slice 1 (this file, so far): the security-critical verdict boundary + the judge
request-building pieces, in isolation. Later slices add catalog-name resolution
(F2/F3), the greedy hill-climb loop controller, and seed-image entry (F4/F5).

The keystone invariant (ADR-027, security review F1): the LLM judge's output is
gated by a CLOSED two-key allowlist — only `overrides.prompt` and
`overrides.loras[{name,action,weight}]` are ever honored. `COMFYLESS_SCHEMA` is
NOT the gate on planner output; it contains path fields (`model`,
`transformer_path`, `loras[].path`, ...) and gating on it would let the LLM
supply filesystem paths. LoRA `name`s captured here are opaque handles — they are
resolved to paths ONLY by the hardened ADR-015 in-memory resolver in a later
slice, never treated as paths in this module.

See docs/decisions/ADR-027-comfyless-refinement-loop.md and
docs/security/review-refinement-loop-2026-07-13.md.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, List, Optional

# ── Bounds (security review F5/F6) ───────────────────────────────────────────
#: Longest-side pixel cap for the image sent to the judge. Candidates can be
#: 17-50+ MP; a raw PNG at that size base64-encodes to hundreds of MB per call.
JUDGE_MAX_PX = 1536
#: Byte cap on a --seed-image file read (F5).
SEED_IMAGE_MAX_BYTES = 64 * 1024 ** 2
#: Pixel cap on a --seed-image (F5, decompression-bomb guard). ~67 MP, comfortably
#: above the project's ~50 MP legit output ceiling. Enforced on the lazy header
#: (Image.size) BEFORE full decode — the byte cap does not bound pixel count, and
#: PIL's own MAX_IMAGE_PIXELS is a mutable process-global we do not rely on.
SEED_IMAGE_MAX_PIXELS = 64 * 1024 ** 2
#: Judge scores are integers in this inclusive range.
SCORE_MIN, SCORE_MAX = 1, 10
#: Planner-supplied LoRA weights are clamped to this absolute magnitude (F6).
LORA_WEIGHT_ABS_MAX = 4.0
#: A planner-proposed prompt longer than this is ignored (kept the prior prompt).
OVERRIDE_PROMPT_MAX_CHARS = 20_000
#: Default per-call timeout (seconds) on the judge HTTP request (F5).
JUDGE_HTTP_TIMEOUT = 120
#: Max bytes read from a judge HTTP response (a misbehaving endpoint could stream
#: unbounded bytes into memory otherwise).
JUDGE_RESPONSE_MAX_BYTES = 8 * 1024 ** 2

_ALLOWED_LORA_ACTIONS = ("add", "remove", "set_weight")


class RefineError(Exception):
    """A refinement-loop error. A RefineError raised out of verdict parsing means
    the judge response is unusable for THIS iteration; the loop controller treats
    that as 'consume an iteration and continue' — the cap, not the parse, bounds
    the loop (ADR-027 F7)."""


# ── Verdict data model ───────────────────────────────────────────────────────
@dataclass
class LoraOp:
    """A single validated LoRA action from the planner. `name` is an OPAQUE
    catalog handle — resolved to a path only by the hardened resolver in a later
    slice, never treated as a path here."""
    name: str
    action: str  # one of _ALLOWED_LORA_ACTIONS
    weight: Optional[float] = None


@dataclass
class Verdict:
    """The sanitized judge+planner result. Only the closed-allowlist fields
    survive parsing; everything else is dropped into `notices`.

    `verdict` is the judge's ADVISORY self-report; the loop's authoritative pass
    decision is scores >= threshold (ADR-027), not this string, so a judge that
    lies "pass" cannot self-promote past the numeric gate."""
    prompt_adherence: int
    aesthetics: int
    verdict: str  # "pass" | "revise"
    critique: dict
    override_prompt: Optional[str]
    lora_ops: List[LoraOp]
    notices: List[str] = field(default_factory=list)


def _reject_nonfinite(token: str) -> Any:
    """json.loads parse_constant hook — raises on NaN/Infinity/-Infinity literals
    (F6). LLMs emit these; the stdlib default would silently produce float inf/nan."""
    raise RefineError(f"judge response contained non-finite JSON constant {token!r}")


def _extract_json_block(raw: str) -> str:
    """Return the outermost {...} slice of `raw`, tolerating ```json fences and
    surrounding prose that chat models commonly wrap around JSON.

    First-`{`/last-`}` slicing cannot silently select a *different* embedded
    object: a stray brace in surrounding prose, or two concatenated JSON objects,
    makes the slice invalid JSON → `parse_verdict` raises `RefineError` and the
    loop consumes an iteration (the ADR F7 fail mode). This trades a little
    robustness (a trailing `}` in post-JSON prose wastes an iteration) for that
    fail-closed guarantee — deliberate."""
    if not isinstance(raw, str):
        raise RefineError("judge response is not text")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RefineError(f"no JSON object found in judge response: {raw[:200]!r}")
    return raw[start:end + 1]


def _coerce_score(scores: dict, key: str, notices: List[str]) -> int:
    """Validate one score as a finite number, round to int, clamp to 1-10 (F6).
    Missing / non-numeric / non-finite → RefineError (a judge that can't score is
    a malformed verdict, not a recoverable one)."""
    val = scores.get(key)
    # bool is an int subclass — exclude it explicitly.
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise RefineError(f"judge score {key!r} is missing or non-numeric: {val!r}")
    # A huge bare-integer literal (e.g. 10**400) is a valid Python int that
    # parse_constant never sees; math.isfinite/round raise OverflowError on it.
    # Treat that as a malformed verdict, same as non-finite (F6/F7).
    try:
        if not math.isfinite(val):
            raise RefineError(f"judge score {key!r} is not finite: {val!r}")
        iv = int(round(val))
    except OverflowError as e:
        raise RefineError(f"judge score {key!r} is out of representable range: {val!r}") from e
    if iv != val:
        notices.append(f"score {key!r}={val!r} coerced to int {iv}")
    if iv < SCORE_MIN or iv > SCORE_MAX:
        clamped = min(SCORE_MAX, max(SCORE_MIN, iv))
        notices.append(f"score {key!r}={val!r} outside {SCORE_MIN}-{SCORE_MAX} — clamped to {clamped}")
        iv = clamped
    return iv


def _parse_lora_op(entry: Any, idx: int, notices: List[str]) -> Optional[LoraOp]:
    """Validate one planner LoRA op. Unknown per-entry keys (notably `path`) are
    dropped with a notice — an LLM cannot smuggle a path through a LoRA entry.
    A structurally-bad entry is dropped (warn-don't-block), not fatal."""
    if not isinstance(entry, dict):
        notices.append(f"loras[{idx}] is not an object — dropped")
        return None
    for k in entry:
        if k not in ("name", "action", "weight"):
            notices.append(f"loras[{idx}]: dropped disallowed key {k!r}")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        notices.append(f"loras[{idx}]: missing/empty 'name' — dropped")
        return None
    name = name.strip()
    # Defense-in-depth: the ADR-015 resolver (a later slice) is the real gate,
    # but reject the obvious path/control-char carriers here so the invariant
    # never rests solely on that wiring.
    if any(sep in name for sep in ("/", "\\")) or any(ord(c) < 32 for c in name):
        notices.append(f"loras[{idx}]: name {name!r} contains path/control chars — dropped")
        return None
    action = entry.get("action")
    if action not in _ALLOWED_LORA_ACTIONS:
        notices.append(f"loras[{idx}]: invalid action {action!r} — dropped")
        return None
    weight: Optional[float] = None
    raw_w = entry.get("weight")
    if raw_w is not None:
        if isinstance(raw_w, bool) or not isinstance(raw_w, (int, float)):
            notices.append(f"loras[{idx}]: non-numeric weight {raw_w!r} — weight dropped")
            raw_w = None
        else:
            # OverflowError guards the huge-bare-int case (see _coerce_score).
            try:
                finite = math.isfinite(raw_w)
            except OverflowError:
                finite = False
            if not finite:
                notices.append(f"loras[{idx}]: non-finite/out-of-range weight — weight dropped")
                raw_w = None
    if raw_w is not None:
        weight = float(raw_w)
        if abs(weight) > LORA_WEIGHT_ABS_MAX:
            clamped = math.copysign(LORA_WEIGHT_ABS_MAX, weight)
            notices.append(
                f"loras[{idx}]: weight {weight} exceeds |{LORA_WEIGHT_ABS_MAX}| — "
                f"clamped to {clamped}")
            weight = clamped
    if action == "set_weight" and weight is None:
        notices.append(f"loras[{idx}]: 'set_weight' without a valid weight — dropped")
        return None
    return LoraOp(name=name, action=action, weight=weight)


def parse_verdict(raw: str) -> Verdict:
    """Parse + sanitize a raw judge response into a Verdict (ADR-027 F1/F6/F7).

    Closed allowlist: only `scores`, `verdict`, `critique`, `overrides` are read
    at the top level, and within `overrides` only `prompt` and `loras`. Every
    other key at every level is dropped with a notice. Non-finite JSON constants
    are rejected. An unrecoverable response raises RefineError (the loop consumes
    an iteration and continues)."""
    notices: List[str] = []
    block = _extract_json_block(raw)
    try:
        data = json.loads(block, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as e:
        raise RefineError(f"judge response is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise RefineError("judge response JSON is not an object")

    scores = data.get("scores")
    if not isinstance(scores, dict):
        raise RefineError("judge verdict missing 'scores' object")
    prompt_adherence = _coerce_score(scores, "prompt_adherence", notices)
    aesthetics = _coerce_score(scores, "aesthetics", notices)
    for k in scores:
        if k not in ("prompt_adherence", "aesthetics"):
            notices.append(f"dropped unknown score key {k!r}")

    verdict = data.get("verdict")
    if verdict not in ("pass", "revise"):
        notices.append(f"verdict {verdict!r} not in pass/revise — treated as 'revise'")
        verdict = "revise"

    # Critique is allowlisted like scores (F7): only the two known keys, string
    # values only. Without this, an LLM's critique dict is a raw payload carrier
    # (arbitrary keys/nesting/non-finite numbers) that later slices persist to
    # disk and echo into the next planner call.
    critique_raw = data.get("critique")
    critique: dict = {}
    if isinstance(critique_raw, dict):
        for k, cv in critique_raw.items():
            if k not in ("prompt_adherence", "aesthetics"):
                notices.append(f"dropped unknown critique key {k!r}")
            elif not isinstance(cv, str):
                notices.append(f"critique {k!r} is not a string — ignored")
            else:
                critique[k] = cv
    elif critique_raw is not None:
        notices.append("'critique' is not an object — ignored")

    override_prompt: Optional[str] = None
    lora_ops: List[LoraOp] = []
    overrides = data.get("overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            notices.append("'overrides' is not an object — ignored")
        else:
            for k in overrides:
                if k not in ("prompt", "loras"):
                    notices.append(f"dropped disallowed override key {k!r}")
            op = overrides.get("prompt")
            if op is not None:
                if not isinstance(op, str):
                    notices.append("override 'prompt' is not a string — ignored")
                elif not op.strip():
                    notices.append("override 'prompt' is empty — ignored")
                elif len(op) > OVERRIDE_PROMPT_MAX_CHARS:
                    notices.append(
                        f"override 'prompt' exceeds {OVERRIDE_PROMPT_MAX_CHARS} chars — ignored")
                else:
                    override_prompt = op
            loras = overrides.get("loras")
            if loras is not None:
                if not isinstance(loras, list):
                    notices.append("override 'loras' is not a list — ignored")
                else:
                    for i, entry in enumerate(loras):
                        parsed = _parse_lora_op(entry, i, notices)
                        if parsed is not None:
                            lora_ops.append(parsed)

    for k in data:
        if k not in ("scores", "verdict", "critique", "overrides"):
            notices.append(f"dropped unknown verdict key {k!r}")

    return Verdict(
        prompt_adherence=prompt_adherence,
        aesthetics=aesthetics,
        verdict=verdict,
        critique=critique,
        override_prompt=override_prompt,
        lora_ops=lora_ops,
        notices=notices,
    )


# ── Judge request building (F5) ──────────────────────────────────────────────
def downscale_for_judge(img, max_px: int = JUDGE_MAX_PX):
    """Return `img` unchanged if its longest side <= max_px, else a resized copy
    (LANCZOS) with the longest side == max_px. Caps the payload sent to the judge."""
    from PIL import Image
    w, h = img.size
    longest = max(w, h)
    if longest <= max_px:
        return img
    scale = max_px / longest
    return img.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.LANCZOS)


def load_seed_image_capped(path: str, max_bytes: int = SEED_IMAGE_MAX_BYTES,
                           max_pixels: int = SEED_IMAGE_MAX_PIXELS):
    """Open a seed image after BOTH a byte gate and a pixel gate (F5). Returns an
    RGB PIL image. The pixel gate is enforced on the lazy header (Image.size)
    BEFORE the full `.convert("RGB")` decode, so a decompression bomb (small file,
    huge declared dimensions) is rejected without ever allocating its pixels — the
    byte cap alone does not bound pixel count, and we do not rely on PIL's mutable
    process-global MAX_IMAGE_PIXELS."""
    from PIL import Image
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise RefineError(f"cannot stat seed image {path!r}: {e}") from e
    if size > max_bytes:
        raise RefineError(
            f"seed image {path!r} is {size} bytes, exceeds cap {max_bytes}")
    try:
        img = Image.open(path)  # lazy — reads header, not pixels
        w, h = img.size
    except Exception as e:  # noqa: BLE001 — PIL raises a zoo of errors on bad files
        raise RefineError(f"cannot open seed image {path!r}: {e}") from e
    if w * h > max_pixels:
        raise RefineError(
            f"seed image {path!r} is {w}x{h} ({w * h} px), exceeds pixel cap {max_pixels}")
    try:
        return img.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise RefineError(f"cannot decode seed image {path!r}: {e}") from e


def image_to_data_uri(img) -> str:
    """PNG-encode a (already downscaled) PIL image to a base64 data: URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_judge_payload(model: str, system_prompt: str, user_text: str,
                        image_data_uri: str, temperature: float = 0.0) -> dict:
    """Build the OpenAI-compatible chat/completions payload with a vision content
    array (text + image_url). Judge runs at temperature 0 for reproducible scoring."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]},
        ],
        "temperature": temperature,
        "stream": False,
    }


def _post_judge(endpoint: str, payload: dict, key: str = "",
                timeout: int = JUDGE_HTTP_TIMEOUT) -> str:
    """POST one judge request; return the single choice's message content.

    Thin transport wrapper (mirrors enhance._post_chat) — not unit-tested; needs a
    live endpoint. The pure request-building and response-parsing around it are the
    tested surface."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Cap the read so a misbehaving endpoint can't stream unbounded bytes.
            data = json.loads(resp.read(JUDGE_RESPONSE_MAX_BYTES).decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RefineError(f"judge endpoint HTTP {e.code} from {endpoint}: {detail}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RefineError(f"judge endpoint cannot reach {endpoint}: {e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RefineError(
            f"judge response missing choices[0].message.content: {str(data)[:200]}") from e


# ── Catalog integration (ADR-027 F2/F3) ──────────────────────────────────────
#
# Two planes, kept strictly separate (security review F2):
#   • name→path resolution goes through the ADR-015 in-memory resolver ONLY
#     (basename-strip, forbidden-char gate, kind enforcement, existence,
#     union-containment). refine.py never reads a load path from the DB and never
#     trusts a path from the LLM.
#   • the ADR-022 SQLite DB is consulted for METADATA/FTS only. Its rows carry
#     abs_path/root/relative_path; those NEVER reach the planner. `_safe_lora_view`
#     and the exact-name query below are an ALLOWLIST — path columns can't leak (F3).

#: Entry columns the planner may see. Allowlist — abs_path/root/relative_path and
#: audit columns (sha256, classification, reason, ...) are excluded by omission.
_SAFE_ENTRY_FIELDS = ("name", "kind", "model_family")
#: Description columns the planner may see.
_SAFE_DESC_FIELDS = ("description", "usage_tips", "trigger_words", "strength_rec",
                     "sampler_rec")


@dataclass
class ResolvedLoraOp:
    """A planner LoraOp whose catalog name has been resolved to a server-side path
    via the ADR-015 resolver. `abs_path` is the load target for the generate wiring
    (a later slice) — it is NEVER sent to the LLM. `resolved_name` is the NFC
    catalog name."""
    op: LoraOp
    resolved_name: str
    abs_path: str


def resolve_lora_ops(catalog, roots, lora_ops: List[LoraOp],
                     notices: Optional[List[str]] = None):
    """Resolve each planner LoRA op's NAME to a path via the ADR-015 in-memory
    resolver (F2) — the ONLY name→path path in the loop. Unresolvable, wrong-kind,
    or path-shaped names are dropped with an operator notice (warn-don't-block);
    a name is never fabricated into a path.

    Returns (resolved: List[ResolvedLoraOp], notices)."""
    if notices is None:
        notices = []
    from comfyless.catalog import resolve_reference
    roots_t = (roots,) if isinstance(roots, str) else tuple(roots)
    resolved: List[ResolvedLoraOp] = []
    for op in lora_ops:
        res = resolve_reference(catalog, op.name, roots_t, expected_kind="lora")
        if not res.ok or res.abs_path is None or res.name is None:
            notices.append(
                f"lora {op.name!r}: not resolvable as a catalog LoRA "
                f"(cause: {res.cause}) — dropped")
            continue
        resolved.append(ResolvedLoraOp(op=op, resolved_name=res.name,
                                       abs_path=res.abs_path))
    return resolved, notices


def open_catalog_db(db_path: str):
    """Open the ADR-022 metadata DB read-only, or return None. Metadata is
    best-effort: the loop resolves and generates fine without it, the planner just
    gets no LoRA descriptions.

    An ABSENT DB is normal — return None quietly. A PRESENT-but-unusable DB
    (schema mismatch, or a corrupt/foreign non-SQLite file, which makes
    connect_readonly's PRAGMA probe raise sqlite3.DatabaseError, not CatalogDBError)
    degrades to name-only but WARNS — running the planner blind on a DB the operator
    thinks is live should not be silent (warn-don't-block)."""
    from comfyless import catalog_db
    if not os.path.isfile(db_path):
        return None
    try:
        return catalog_db.connect_readonly(db_path)
    except (catalog_db.CatalogDBError, sqlite3.Error) as e:
        print(f"[refine] WARNING: catalog metadata DB {db_path!r} is unusable "
              f"({e}); planner will see LoRA names only", file=sys.stderr)
        return None


def _safe_desc_view(desc: Optional[dict], view: dict) -> None:
    """Merge the allowlisted description fields of `desc` (a dict) into `view`."""
    if not desc:
        return
    for f in _SAFE_DESC_FIELDS:
        val = desc.get(f)
        # Skip empties incl. the "[]"/"{}" that trigger_words sanitizes to when a
        # LoRA has no triggers — don't feed empty-container noise to the planner.
        if val and val not in ("[]", "{}"):
            view[f] = val


def _safe_lora_view(row: dict) -> dict:
    """Project a catalog_db.search() row (entries.* + best_description) to the
    planner allowlist (F3). Path/audit columns are dropped by omission."""
    view = {f: row[f] for f in _SAFE_ENTRY_FIELDS if row.get(f) is not None}
    _safe_desc_view(row.get("best_description"), view)
    return view


def lora_metadata(conn, name: str) -> Optional[dict]:
    """Safe metadata for an exact LoRA name (F3). The SELECT lists ONLY allowlisted
    columns — abs_path/root/relative_path are never queried. Returns a safe dict, or
    None if the name isn't in the DB (best-effort)."""
    row = conn.execute(
        "SELECT id, name, kind, model_family FROM entries "
        "WHERE kind = 'lora' AND name = ? AND excluded = 0 AND stale = 0",
        (name,)).fetchone()
    if row is None:
        return None
    view = {"name": row["name"], "kind": row["kind"]}
    if row["model_family"]:
        view["model_family"] = row["model_family"]
    desc = conn.execute(
        "SELECT description, usage_tips, trigger_words, strength_rec, sampler_rec "
        "FROM descriptions WHERE entry_id = ? "
        "ORDER BY CASE source WHEN 'sidecar' THEN 0 WHEN 'civitai_api' THEN 1 "
        "WHEN 'web' THEN 2 ELSE 3 END LIMIT 1",
        (row["id"],)).fetchone()
    _safe_desc_view(dict(desc) if desc else None, view)
    return view


def search_loras(conn, term: str, *, limit: int = 5) -> List[dict]:
    """Search LoRAs by effect/name via the ADR-022 FTS (reused). Every row is
    projected through `_safe_lora_view` before return — search() SELECTs e.* incl.
    abs_path, so the allowlist is what keeps paths out of the planner (F3)."""
    from comfyless import catalog_db
    rows = catalog_db.search(conn, term, kind="lora", limit=limit)
    return [_safe_lora_view(r) for r in rows]


def assemble_planner_loras(conn, names) -> List[dict]:
    """Safe metadata for a set of in-play LoRA names, for the planner context. A
    name with no DB row still yields a name-only entry so the planner knows it is
    active. `conn` may be None (no metadata DB) — then every entry is name-only."""
    out: List[dict] = []
    for n in names:
        md = lora_metadata(conn, n) if conn is not None else None
        out.append(md if md is not None else {"name": n})
    return out
