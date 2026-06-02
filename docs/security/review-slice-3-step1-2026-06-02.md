# Security Review — ADR-015 Slice 3 Step 1: `resolve_reference`

**AI-Disclosure:** Claude (Opus 4.8, 1M context) authored this security review; Grant reviews and owns the Red Zone sign-off.
**Date:** 2026-06-02
**Scope:** the request-time resolver landed in `comfyless/catalog.py` (`resolve_reference`, `ResolveResult`, `ResolveCause`) + its unit tests in `test_mcp_server.py`. NO handler wiring (Step 2) reviewed — out of scope per the slice plan.

## Summary

Step 1 adds a pure, read-only request-time resolver that converts an LLM-agent-supplied reference value into either a server-side `abs_path` (success) or a structured operator-audit-only failure cause. The function never renders an agent-facing string; the uniform-error contract (HIGH-1) is the *handler's* job in Step 2, and this module correctly confines itself to returning a discriminated `ResolveResult`. The threat model is the same-uid stdio trust boundary: a (possibly prompt-injected) agent supplies `raw_ref`; the operator supplies `catalog` and `model_base` at spawn. The resolver must (a) never return an `abs_path` outside `--model-base`, (b) never return distinguishing structure that the *agent* could observe (it only returns it operator-side), (c) never raise on agent input, and (d) keep request-time normalization byte-symmetric with build-time catalog keys.

I traced every branch of `resolve_reference` against the six audit questions, checked the `os.path.exists` → `_within` ordering for TOCTOU, verified the malformed-gate runs before `normalize_name` and before `catalog.get`, verified each failure branch nulls `abs_path`/`name`/`kind`, and checked `_within`'s actual implementation (it re-`realpath`s, so symlink-swap is re-resolved at request time). I read the build-side gate (`_add_entry`, `_FORBIDDEN_NAME_CHARS`) to confirm the request candidate goes through the *same* forbidden-char regex and the *same* `normalize_name` as catalog keys. **Verdict: CLEAN for Step 1 as a standalone pure function.** No HIGH or MEDIUM findings. The two refinements vs the ADR (KindMismatch added, HFCacheMiss subsumed by PathMoved) are security-sound and do not need an ADR amendment. All real risk in this surface is deferred to Step 2 (the handler must actually fold every cause into one byte-identical frame, and must not log/return `abs_path`); I record those as carry-forward obligations, not Step-1 findings.

## Coverage

Reviewed:
- `comfyless/catalog.py:625-814` — the entire request-time resolution block (ResolveCause, ResolveResult, resolve_reference). Read in full.
- `comfyless/catalog.py:194-332` — `_FORBIDDEN_NAME_CHARS`, `normalize_name`, `_add_entry` (build-side gate, for symmetry verification).
- `comfyless/server.py:158-188` — `_within` and `_check_paths` (the reused containment helper).
- `nodes/eric_diffusion_utils.py:45-150` — `infer_model_family`, `_is_hf_repo_id`, `resolve_hf_path` (the HF-path contract the subsumed-HFCacheMiss reasoning rests on).
- `test_mcp_server.py:3041-3168` — the Step-1 resolver unit tests.
- `docs/decisions/ADR-015-*.md` §1/§2/§4 and `docs/vision/slice-3-mcp-generate-catalog.md` invariants 1-15 (design source of truth, for the two-refinement soundness check).

Not reviewed (and why):
- `comfyless/mcp_server.py` handler wiring — Step 2, explicitly not in this commit. The load-bearing uniform-frame rendering, audit-line emission, and `resolved_params`/`notices` construction all live there; their correctness is NOT established by this review.
- The other 8 test suites — untouched per the slice plan.
- Build-side scan/manifest internals beyond the symmetry-relevant gate — frozen slice 2, already reviewed.

## Findings

### Question 1 — Path-traversal / escape: CLEAN

No `raw_ref` shape can make the resolver return an `abs_path` outside `--model-base`. Reasoning:

- **Traversal / absolute / separators (`../`, `/etc/passwd`, UNC-ish `\\host\share`):** the resolver does **not** join `raw_ref` to anything. A value containing `/` (or `os.sep`) is reduced to `os.path.basename(raw_ref)` (`catalog.py:748-750`); the directory component is discarded and never touches the filesystem. The basename is then looked up as a *catalog key* — it can only succeed if it byte-matches an operator-minted key. `../../../etc/passwd` basenames to `passwd`, which is a catalog miss → `UnknownName`. There is no path the agent supplies that becomes a load target; the load target is always `entry["abs_path"]`, which build-time `_within` already constrained to `--model-base`.
- **The returned `abs_path` is the catalog's, never the agent's** (`catalog.py:783`, `807-813`). Confirmed by test at `test_mcp_server.py:3083-3084`.
- **Symlink swap / request-time escape:** even though the catalog `abs_path` is build-time-checked, the resolver re-runs `_within(abs_path, model_base)` at request time (`catalog.py:801`), and `_within` itself calls `os.path.realpath` on both operands (`server.py:160-161`). So a post-spawn symlink that redirects an in-base catalog path to an out-of-base target is caught by the request-time re-resolution → `WithinFailure`. Test `test_mcp_server.py:3144-3163` exercises an out-of-base entry directly.
- **TOCTOU between `os.path.exists` (line 789) and `_within` (line 801):** there is a window — `exists` realpaths/stats once, `_within` realpaths again — but it is **not exploitable to escape containment**, because `_within` is the authoritative gate and it re-resolves immediately before the success return. The worst a race achieves is a `PathMoved`-vs-`WithinFailure` cause flip or a stale pass, but `_within` still fires on the final realpath. The order is correct: existence-check first (cheap, distinguishes the common moved/evicted case for the operator audit), containment-check last and load-deciding.
- **TOCTOU between this resolve and the eventual load in Step 2** is real but is a **Step-2 obligation**, not a Step-1 finding: the resolver returns an `abs_path` that Step 2's `_load_pipeline` will open later, and the filesystem can change in between. The slice-1 `_check_paths`/`_within` machinery is retained as request-time defense-in-depth (Vision invariant 9); the load path must re-validate. **Carry-forward note for Step 2:** the resolved `abs_path` must pass through the existing `_check_paths`/`_within` net at the load boundary, not be trusted because the resolver blessed it. The ADR §2 step 3 and Vision invariant 4 already require this; flag it explicitly so it is not lost when the handler is wired.

### Question 2 — Enumeration oracle (the load-bearing property): CLEAN at this layer

The resolver preserves the ability to render exactly one uniform agent frame, because **it returns no agent-facing string at all** — only a structured `ResolveResult`. The distinguishing information lives solely in `cause` (a `ResolveCause` Literal) and in the success-only fields. By construction:

- All five failure causes produce `ResolveResult(ok=False, cause=<X>, abs_path=None, name=None, kind=None)`. The *only* field that differs between failure modes is `cause`, which the docstring (`catalog.py:636-647`) and Vision invariant 3 designate operator-audit-only. Whether the oracle actually closes depends entirely on Step 2 mapping every `ok=False` to the byte-identical frame and writing `cause` to stderr only — **that is the load-bearing test (N5), and it is not in this commit.** Step 1 does not *leak* an oracle; it also cannot *guarantee* closure alone. Correctly scoped.
- **`path_was_discarded` is set on failure branches too** (e.g. `catalog.py:762-765, 771-774`). This is fine: it never crosses to the agent (it drives the operator audit and, on *success*, the path-discard INFO notice keyed on the resolved catalog name). Step 2 must not let `path_was_discarded` influence the agent-facing *failure* frame — but on failure there is no notice to emit, so this is a non-issue as long as Step 2 only emits the discard notice on `ok=True`. **Carry-forward note for Step 2.**

The cause-set folding is oracle-free *as designed*: KindMismatch and PathMoved-subsumes-HFCacheMiss are additional reasons the *operator* audit can name, all collapsing to the same agent frame. See Question 6 for the soundness argument.

### Question 3 — `abs_path` leakage on failure branches: CLEAN

Every failure return was checked individually:

- Non-str input (line 744-745): `ResolveResult(ok=False, cause="MalformedReference")` — defaults make `abs_path/name/kind=None`. ✓
- Malformed gate (line 762-765): `abs_path/name/kind` default None. ✓
- UnknownName (line 771-774): None. ✓
- KindMismatch (line 778-781): None. ✓
- PathMoved (line 790-793): None. ✓ (test 3142 asserts no fallback path)
- WithinFailure (line 802-805): None. ✓ (test 3162 asserts no abs_path)

`name`, `kind`, and `abs_path` are populated *only* on the single success return (line 807-813), and there `name` is the NFC catalog key (agent-presentable, not a path). No failure branch returns a server path. Confirmed by tests at 3103, 3110, 3142, 3162.

### Question 4 — Normalization symmetry: CLEAN

The request candidate goes through the **same** `_FORBIDDEN_NAME_CHARS` regex (`catalog.py:760`) and the **same** `normalize_name` (`catalog.py:768`) as catalog keys get at build time (`_add_entry`, lines 298 and 306). Critically:

- **Order is correct on the request side:** forbidden-char gate (line 757-760) runs **before** `normalize_name` (line 768), mirroring the build side where the gate runs before `normalize_name` (line 298 before 306). The shared design rationale — every codepoint in `_FORBIDDEN_NAME_CHARS` is NFC-stable, so gating pre-normalization cannot let a decomposed forbidden char slip through — is documented at lines 288-297 and holds for the request side identically. A forbidden char therefore cannot reach `catalog.get`.
- **NUL is double-gated:** `"\x00" in candidate` (line 759) AND `\x00-\x1f` is inside `_FORBIDDEN_NAME_CHARS`. Redundant but harmless; the explicit check also documents intent and pre-empts the `realpath`-raises-bare-ValueError class of bug (though this resolver never realpaths the candidate, only the catalog abs_path).
- **NFC symmetry:** because both sides call the identical `normalize_name`, a request candidate and a catalog key cannot disagree on NFC form. A confusable that NFC-normalizes to a key matches it (intended — that *is* the lookup); one that does not, misses. No order bug lets a forbidden char reach lookup.

One observation, not a finding: the build side additionally enforces **case-insensitive collision rejection** at insert time, while the request side does an exact (case-sensitive) `dict.get`. This is by design and is *not* an asymmetry vulnerability: the build side guarantees no two keys casefold-collide, so case-sensitive lookup is unambiguous; a request whose case differs from the stored key simply misses (`UnknownName`) — which is correct, since the operator chose case-sensitive matching (ADR-015 §1). No oracle, no escape.

### Question 5 — DoS / exceptions on agent input: CLEAN

No agent-supplied `raw_ref` can raise instead of returning `MalformedReference`:

- **Non-str** (int, None, dict, bytes): guarded first at line 744 with `isinstance(raw_ref, str)`. Test 3128-3130 passes `12345`. ✓ (Note: `bytes` would also hit this branch — `isinstance(b"x", str)` is False — so no `TypeError` from `"/" in raw_ref`.)
- **NUL byte:** caught at line 759 → MalformedReference, no exception. The resolver never calls `os.path.realpath`/`basename` *on the candidate* in a way that would raise on NUL — `os.path.basename` (line 749) is called on `raw_ref` only when a separator is present, and `os.path.basename` does NOT raise on embedded NUL (it is pure string slicing); the NUL is caught by the line-759 gate afterward. Test 3113 confirms.
- **Empty after basename-strip** (`"/foo/bar/"` → `""`): line 758 `candidate == ""` → MalformedReference. Test 3116. ✓
- **Huge string:** `os.path.basename`, the regex search, and `normalize_name` are all linear; there is no catastrophic-backtracking pattern in `_FORBIDDEN_NAME_CHARS` (it is a simple character class, no alternation/quantifier nesting). A multi-MB `raw_ref` costs O(n) — acceptable; no unbounded amplification, no recursion. There is **no length cap** on `raw_ref`, but the realistic ceiling is the MCP JSON-RPC frame size the transport already bounds, and the per-call cost is linear. Not a finding for a same-uid stdio peer; if HTTP transport ever lands (ADR-015 §4 INFO-3), revisit a length cap there.
- **`os.path.exists` / `_within`** on the catalog `abs_path` (operator-controlled, already realpath'd and in-base): `os.path.exists` swallows OSError internally and returns False; `_within`'s `os.path.realpath` does not raise on a normal stored abs_path. The catalog `abs_path` is not agent-influenced, so agent input cannot steer these into a raising state.

No branch raises on agent input. The docstring's promise (line 736-737) "Raises nothing for agent-supplied input" holds.

### Question 6 — Soundness of the two refinements vs the ADR: SOUND, no amendment needed

**(a) KindMismatch added.** The ADR's five-cause list (UnknownName/PathMoved/HFCacheMiss/WithinFailure + MalformedReference) did not enumerate wrong-kind. Adding it as a distinct *operator-audit* cause that folds into the same agent frame **strengthens** HIGH-1: without it, a lora name supplied in a `model` field would resolve to an abs_path and fall through to `_load_pipeline`, surfacing a *different* (load-time InternalError) frame — a mild existence/kind oracle. Folding it at the resolver closes that path before the load boundary. This is a refinement *within* the spirit of HIGH-1 (one agent frame for every resolution failure), not a contract change. The Vision already flags it (invariant 2a) and Grant owns the sign-off. **Soundness condition for Step 2:** the handler must pass the correct `expected_kind` for each field (`model`/cascade `stage_*` → `"model"`, `loras[].name` → `"lora"`, `transformer` → `"transformer"`), or the kind oracle reopens via a wrong field accepting a wrong kind. That is a Step-2 wiring obligation; flag it.

**(b) HFCacheMiss subsumed by PathMoved.** Sound. Catalog entries store the already-resolved *local* cache path (build-time `resolve_hf_path(allow_download=False)`, slice-2 invariant 6), and the entry does **not** retain the originating repo ID (`CatalogEntry` has no repo-id field — confirmed at `catalog.py:156-174`). So at request time there is nothing to re-resolve via HF; a post-spawn cache eviction simply makes the stored local path vanish, which is exactly `os.path.exists(abs_path) == False` → `PathMoved`. Request-time HF re-resolution would be a no-op on an absolute local path (`resolve_hf_path` returns non-repo-id paths unchanged — `eric_diffusion_utils.py:123-124`). The agent-facing uniformity holds identically. Build-time HF-cache-miss remains a slice-2 startup failure (operator channel), unaffected. **No ADR amendment required** — the ADR's `HFCacheMiss` was a *request-time audit label*, and the resolver's design makes that state physically identical to PathMoved; the Vision (invariant 2b) records the subsumption. Note Vision invariant 4 still says "HF-sourced entries re-resolve via `resolve_hf_path(...)`" at request time — that sentence is now vestigial given the subsumption; not a security defect, but worth a one-line Vision clarification when Step 2 lands so the implementer does not add a dead re-resolution call.

## Carry-forward obligations for Step 2 (not Step-1 findings)

These are where the actual HIGH-1 risk lives; Step 1 is clean but cannot discharge them:

1. **Uniform frame:** every `ok=False` (all five causes) must map to one byte-identical agent error class + message (`"reference not available"`). The keystone test N5 proves this; it is not in this commit.
2. **Cause to stderr only:** `cause` must never appear in any agent-facing frame; written only to the audit line (Vision invariant 3, N6/N12).
3. **`abs_path` never in response/notice/error/audit** (Vision invariant 5/10, N11/N13).
4. **Correct `expected_kind` per field** (else the KindMismatch closure is moot).
5. **Discard notice only on success**, keyed on resolved catalog `name`, never raw `R` (INFO-2, N9).
6. **Re-validate the resolved `abs_path` at the load boundary** (TOCTOU between resolve and load) via the retained `_check_paths`/`_within` net.

## Verdict

**CLEAN — no HIGH/MEDIUM/LOW findings in Step 1.** The resolver is a correctly-scoped, fail-closed, exception-free pure function; path-traversal is structurally impossible (basename-strip + catalog-key lookup + request-time `_within` re-realpath); every failure branch nulls the path/name/kind; normalization is byte-symmetric with the build side; and the two ADR refinements are security-sound without amendment. The load-bearing oracle-closure property is correctly deferred to the Step-2 handler, with the obligations enumerated above.

---

*Note: line numbers above are from the auditor's read at review time (2026-06-02, pre-commit); the three code-reviewer-recommended unit tests were added to `test_mcp_server.py` after this review, shifting later line numbers in that file. The resolver in `comfyless/catalog.py` was unchanged by that addition.*
