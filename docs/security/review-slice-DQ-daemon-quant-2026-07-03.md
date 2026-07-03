# Security Review — slice DQ: daemon quant carriage (design stage)

AI-Disclosure: Claude (Opus 4.8) authored; Grant reviewed.
Date: 2026-07-03
Scope: §12 delta review of the proposed change in `docs/vision/slice-DQ-daemon-quant.md` against `comfyless/server.py`, `comfyless/generate.py`, `comfyless/params_validation.py`, and the quant helpers in `nodes/eric_diffusion_utils.py`. Code NOT yet written — findings are binding implementation requirements.
Reviewer: security-auditor (Opus)
Threat model: single-user local desktop; the daemon socket is same-uid only (0700 dir, 0600 socket). "Attacker" = any same-uid local process able to connect to the socket, plus the future `--json`/LLM-agent client whose request contents are model-driven. Availability of the single-threaded accept loop is the primary asset; secondary assets are path-confinement (no read/write outside `--model-base`/`--output-dir`) and the "never serve the wrong artifact" cache invariant.

## Threat model and what I checked

The delta widens the daemon wire request by three fields (`quant`, `quant_skip`, `quant_only`), removes the client-side branch that kept `--quant` runs out of the daemon entirely, and tightens the pipeline cache key so a quantized pipeline can never be served under mismatched quant settings and can never take the incremental LoRA-diff path (which cannot undo a direct merge into requantized weights, per ADR-019 slice DMR). I traced each new field from `_delegate_to_server` (generate.py:1391) across the socket, through the canonical validator (`params_validation.validate_machine_request`), through the server's `_validate_request`/`_check_paths` guards, into `_handle_generate`'s cache key and load calls, and finally into `build_quant_config` / `resolve_quant_components`. The core question for each field was: does it reach a filesystem API, a compute op, or the error/eviction control flow in a way the existing guards don't already close?

The confinement story holds well. `quant_skip`/`quant_only` are already type- and hygiene-validated (str, no NUL, no `/`/`\`, ≤32 entries) and — verified by reading `resolve_quant_components` — they are consumed only as dict keys and membership tests against `model_index`-derived role names; the only `os.path.join` in that path (`component_weight_bytes`) is fed model-directory keys, never client strings, so a separatorless `..` slips the validator but is inert (a dict miss → "unknown component" notice). The one place the design actively works against the existing hardened posture is item 2: importing the torch-heavy `QUANT_MODES` **inside** `_validate_request`, which is on an unguarded call path. That reintroduces the precise accept-loop-kill class the prior slice's H-2 closed. The cache-key changes are sound in direction (a strict-superset discriminator can only over-evict, never under-evict, so N1/N2 are structurally upheld) but carry three implementation conditions that, if missed, silently break N2 or N4. Details below.

## Findings

### F1 [BLOCKER] — Heavy import inside `_validate_request` reintroduces the H-2 accept-loop-kill class
**Invariant protected:** Accept-loop survivability (design N5; prior-review H-2). No unhandled exception may escape `_handle_connection` — it exits the `with conn` block, bubbles out of the `while keep_running` loop, and kills the daemon.
**Risk:** `_validate_request` is called unguarded at `server.py:279`; today it only calls the pure, torch-free `validate_machine_request`, which cannot raise. Adding `from nodes.eric_diffusion_utils import QUANT_MODES` inside the function (as the design's item 2 specifies, "imported inside the function (heavy module)") introduces an `import` of a torch/CUDA-touching module on that unguarded path. Any exception during that import — `ImportError`, a CUDA-init failure, a partially-installed torchao — propagates out and one-shots the daemon. The semantic membership check itself is safe; the *import* is the hazard.
**Requirement:** Do NOT import `eric_diffusion_utils` inside `_validate_request`. `QUANT_MODES` is the trivial tuple `("none", "fp8")`; define the allowed set as a light constant with no torch in its import graph — either a module-level tuple in `server.py` or (preferred, keeps "one validator" spirit) a boundary constant in `params_validation.py` next to the existing hygiene constants, and do a pure `req.get("quant") not in QUANT_MODES` string check. If for some reason a heavy import is unavoidable, wrap it in `try/except Exception` that returns a `ValidationError` **string** rather than propagating. Add a test that a `quant` value outside the set returns a `ValidationError` response and that the validator raises no exception for any `quant` input type the canonical validator already accepts.

### F2 [SHOULD] — LoRA-set cache-key contribution must be gated on `quant != "none"`
**Invariant protected:** N4 — unquantized daemon behavior (incremental LoRA-diff semantics) unchanged.
**Risk:** The design adds the LoRA `(path, weight)` set to the cache key "when quant != none." If the implementation adds it unconditionally, every unquantized LoRA swap becomes a full cold evict+reload (30–90s) instead of the incremental `delete_adapters`/`add` path, silently regressing the unquantized hot path — an availability regression and a violation of the N4 invariant.
**Requirement:** Guard the LoRA-set inclusion behind `if quant != "none":`. For `quant == "none"` the cache key's quant triple is the constant `("none", (), ())` and the LoRA set is absent — byte-for-byte equivalent eviction behavior to today. Add a test asserting an unquantized LoRA-set change does NOT change the cache key (still takes the diff path), and a quantized one does.

### F3 [SHOULD] — Quantized LoRA-set key must include weight and be order-/type-normalized
**Invariant protected:** N2 — no incremental LoRA add/remove on a quantized pipeline; under DMR a weight-only change cannot be undone incrementally.
**Risk:** If the cache-key LoRA component keys on path only, two requests with the same LoRA paths but different weights produce the same key → warm-cache hit → the add-loop skips (path already in `loaded_paths`) → the pipeline keeps the *previously merged* weight while reporting success. That is exactly the "merge silently left baked in" failure N2 exists to prevent. Reordering the same set must NOT force a reload (spurious churn), and a raw `list` inside the key is mutable and order-sensitive.
**Requirement:** Under quant, the LoRA component of the cache key = `tuple(sorted((l["path"], float(l.get("weight", 1.0))) for l in requested_loras))`. Weight MUST be present; sort for order-insensitivity (consistent with `test_lora_order_insensitive.py`). Normalize `quant_skip`/`quant_only` in the key to `tuple(sorted(...))` as well — immutable, comparable, stable. Add tests: same paths + different weight under quant → key differs (reload); same set reordered → key identical (cache hit).

### F4 [NOTE] — Client abspath (not realpath) in the cache key: redundant reload on symlink-aliased paths, no security impact
**Invariant protected:** N1 — never serve a bf16 pipeline for an fp8 request or the wrong LoRA artifact.
**Risk/assessment:** The cache key uses client-sent abspaths; `_check_paths` realpath-validates separately. Two symlink-distinct abspaths to the same file yield different keys → an extra cold reload under quant. This is self-inflicted, single-user, availability-only. Critically it is *over*-eviction: a strict-superset key can never collide two genuinely-different configs onto one entry, so N1 holds structurally.
**Requirement:** None. Do NOT realpath into the cache key — that would add a filesystem touch to the hot compare path and buys nothing against this threat model. Optionally document the symlink-alias reload as expected.

### F5 [NOTE] — `quant_skip`/`quant_only` confirmed non-filesystem-reaching (N3 upheld by construction)
**Invariant protected:** N3 — quant slot-name strings never reach a filesystem API in server.py.
**Assessment:** Verified in `resolve_quant_components` (utils ~1318): client `only`/`skip` entries are used only as `roles.get(name)` lookups and `name in selected` / `name not in roles` tests. The sole `os.path.join` in the quant path (`component_weight_bytes`, utils ~1304) is called with `name` drawn from `model_index.items()` keys (model-directory-derived, trusted), never from client `skip`/`only`. A separatorless `..` passes the validator (which rejects `/` and `\`) but is inert here — a dict miss producing an "unknown component ... ignored" notice.
**Requirement:** Preserve this property — the implementation must not begin feeding client `skip`/`only` names into any `os.path.*`, `open`, `os.scandir`, etc. Add a regression test asserting `quant_only=[".."]` yields an "unknown component" notice and triggers no filesystem access.

### F6 [NOTE] — ValidationError-vs-LoadError taxonomy: no info leak, defense-in-depth ordering is correct
**Invariant protected:** Error-taxonomy cleanliness; cheap rejection before expensive load.
**Assessment:** Both a boundary `ValidationError` and the fallback `LoadError` (from `build_quant_config`'s own `ValueError`, utils:1444) would echo the allowed set `QUANT_MODES` — not sensitive. `build_quant_config`'s raise is already caught inside `_load_pipeline`'s `try/except` (server.py:379/424 → `LoadError`), so the boundary check is *not* the sole crash-guard; it is defense-in-depth plus correct ordering (reject before `_check_paths`/load). No audit-posture change. Prior-review H-5 (validation rejections not logged to server stderr) now also covers this new "bad quant mode" rejection — acceptable under the single-user model, but fold it into the H-5 flag for the `--json`-bridge scope-change gate.
**Requirement:** Keep the boundary check ordered before `_check_paths`/`_handle_generate` (as designed) and keep the redundant `build_quant_config` raise inside the load `try/except` (it currently is).

### F7 [NOTE] — DoS churn: no new capability class
**Invariant protected:** Accept-loop availability; bounded memory.
**Assessment:** Under quant, any LoRA change forces a full cold reload. A same-uid client alternating LoRA sets under quant can force repeated cold loads — but this is the same magnitude and mechanism as the already-possible model-switch churn (`model` is in the cache key today). The single-threaded accept loop serializes it; the eviction path does `del` + `torch.cuda.empty_cache()` + `server_state.clear()` before reload, so no memory growth.
**Requirement:** None.

### F8 [NOTE] — Prior-review deferrals H-1 / H-3 and this "trigger" commit
**Assessment:** Prior review deferred H-1 (`_socket_dir` should `lstat` and reject a symlinked `/tmp/comfyless-$UID`) "to the next server-touching commit" — this slice IS a server-touching commit, but the design doc doesn't mention it. H-3 (server-side `lora_weight` type check) appears already resolved by ADR-012's `validate_machine_request`/`validate_lora_entry` (canonical float) — verify rather than re-fix. Flagging so the H-1 re-deferral is a conscious choice, not drift.
**Requirement:** None binding on this slice's security posture. Either pick up the two-line H-1 `lstat` fix here or record an explicit re-deferral in TECH_DEBT so the "next server-touching commit" trigger doesn't silently roll forward again.

## Verdict

**The design is sound and may proceed to implementation**, conditional on clearing F1 (blocker) and satisfying F2/F3 as written. The confinement and cache-invariant reasoning is correct: the new fields add no path-traversal, injection, or filesystem reach (F5), the cache-key change is a strict-superset discriminator that structurally upholds N1/N2 (F3/F4), and no new DoS capability class is introduced (F7). The single real hazard is an implementation trap in the design's own wording — pulling a torch-heavy import onto the unguarded validation path (F1) — which reintroduces the exact accept-loop-kill the prior hardening slice closed. Edit scope in the Vision is clean and matches the surfaces touched; no scope creep observed.

## Binding implementation checklist (the code must satisfy all)

1. **[F1]** `_validate_request` gains a pure `quant not in QUANT_MODES` membership check with **no** import of `eric_diffusion_utils` or any torch-touching module; `QUANT_MODES` sourced from a light constant (server.py module scope or `params_validation.py`). Returns a `ValidationError` string; cannot raise. Ordered before `_check_paths`.
2. **[F2]** LoRA `(path, weight)` set joins the cache key **only** when `quant != "none"`; the `quant == "none"` path is byte-for-byte unchanged (incremental LoRA diff preserved).
3. **[F3]** Under quant, the cache-key LoRA component = `tuple(sorted((path, float(weight)) …))` — weight included, order-normalized; `quant_skip`/`quant_only` in the key normalized to `tuple(sorted(...))`.
4. **[F3]** Cache key adds the quant triple `(quant, tuple(sorted(quant_skip)), tuple(sorted(quant_only)))` for all requests (constant for `quant="none"`).
5. **[F4]** No `realpath` in the cache key; client abspaths only.
6. **[F5]** No client `quant_skip`/`quant_only` string reaches any filesystem API; regression test with `quant_only=[".."]` asserts "unknown component" + no FS access.
7. **[F6]** `build_quant_config`'s `ValueError` stays inside the `_load_pipeline` `try/except` (redundant `LoadError` guard retained).
8. **Tests** (extend `test_server_robustness.py`): bogus `quant` mode → `ValidationError` (no raise); quant-vs-none key discrimination; skip/only key discrimination; quant LoRA-set change → reload; unquantized LoRA-set change → still diff path; quant LoRA reorder → cache hit; quant weight-only change → reload; delegation-skip branch removed and wire request carries the triple.
9. **[F8]** Either land the H-1 `lstat` fix or record an explicit TECH_DEBT re-deferral.
