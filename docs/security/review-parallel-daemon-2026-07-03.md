# Security Review — Parallel comfyless Daemon (one per GPU, device-keyed sockets)

**Date:** 2026-07-03
**Reviewer:** Claude (Opus 4.8), security-auditor pass
**Design under review:** ADR-020 (`docs/decisions/ADR-020-parallel-daemon-per-gpu.md`), slice Vision (`docs/vision/slice-parallel-daemon-per-gpu.md`)
**Baseline trust model:** ADR-001 (`docs/decisions/ADR-001-daemon-socket-security.md`)
**Code grounded against:** `comfyless/server.py` (`_socket_dir`/`socket_path` L69-88, `_validate_request` L95-151, `_within`/`_check_paths` L158-189, `_handle_generate` device handling L341/L531, output-counter L494-501, `run_server` L567-608); `comfyless/generate.py` (`--device` default L1249, `_send_server_command`/`_send_unload`/`_delegate_to_server` L1453-1546, `--unload` dispatch L2160)
**AI-Disclosure:** Review authored by Claude (Opus 4.8); reviewed by Grant Kahn.

## Summary

The change spawns one daemon per GPU and gives each a device-keyed socket name (`comfyless-cuda0.sock`, `comfyless-cuda1.sock`, `comfyless-cpu.sock`) inside the existing ADR-001 per-UID `0700` directory, routing each client to the socket for its `--device`. The threat model is unchanged from ADR-001 and correctly scoped: a single-user workstation, local `AF_UNIX` sockets in a `0700` directory, where the only realistic adversary is same-uid mistakes/footguns — cross-uid enumeration, connection, and replacement are already precluded by the directory mode, and nothing in this design touches that control. Design A is the right call: it keeps `_handle_generate` byte-identical and buys GPU isolation from OS process separation rather than from new in-process locking, which is genuinely lower-risk on a §12 surface than design B.

**The socket-naming trust model is sound.** ADR-001 §1–§6 all carry over: sockets stay in the same `0700` dir at `0600`, the server still owns output-path resolution, still validates model/LoRA paths against `--model-base`, still sanitizes adapter names, still schema-validates at the boundary, and there is still no autostart. Adding more *names* in an already-access-controlled directory adds no cross-uid exposure. Device-scoped `--unload` is strictly no-worse than the ADR-001-deferred no-`SO_PEERCRED` item — a same-uid actor could always stop the one daemon; now they stop N daemons one socket at a time, which is the same capability, not a new one. The device-string path-injection primitive is real but is closed at its source by the whitelist, *provided the implementation gets the ordering and the regex right* (Finding 3). The two findings that actually matter are not in the socket layer at all: the output-dir counter collision (a data-loss bug the parallel design newly exposes) and the fact that the daemon still honors the request payload's `device` field over its own pinned `--device`, which quietly defeats the per-GPU isolation the ADR is built to provide.

## Findings

### [HIGH] Concurrent daemons to the same `--output-dir` will silently overwrite each other's auto-numbered output

**Location:** `comfyless/server.py:494-501`

**Scenario:** The auto-numbering path scans `comfyless{counter:04d}.png` and picks the first name for which `os.path.exists()` is false, then hands that path to `generate()`, which writes the file 30–90s later at the *end* of inference. This is a check-to-use (TOCTOU) window the length of an entire generation. With one daemon it was harmless — serialization guaranteed the previous file existed before the next scan. The whole point of ADR-020 is to run two daemons concurrently, and the canonical invocation in ADR-020 §1 points both at the *same* `~/gen-output`. Fire one gen on cuda:0 and one on cuda:1 at the same moment: both scan an empty-ish directory, both select `comfyless0001.png`, both spend a minute generating, both write `comfyless0001.png`. One image is silently lost. This is guaranteed, not racy-if-unlucky, under the design's headline use case. It is a correctness/data-loss issue, not a privilege boundary break, hence HIGH not CRITICAL.

**Fix:** Make the counter allocation atomic against concurrent writers. Smallest targeted change: replace the `exists()`-then-write pattern with an atomic-create reservation — open the candidate with `os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)` inside the loop, advancing `counter` on `FileExistsError`, and keep the fd/zero-byte placeholder so the name is reserved for the duration of generation. Alternatively (and simpler if you accept the constraint) document that concurrent daemons MUST use distinct `--output-dir` per device and have `run_server` refuse to start if its device-keyed output subdir is shared — but the atomic-create fix is preferable because it also hardens the single-daemon case and needs no operator discipline. Note this is outside the `_handle_generate`-must-not-change edit-scope boundary in the Vision (L88-90); it should be its own follow-up slice — flag it, don't fold it in silently (see Finding 5).

### [MEDIUM] Daemon honors the request payload's `device` over its own pinned `--device`, breaking per-GPU isolation

**Location:** `comfyless/server.py:341` (`req_device = req.get("device") or device`), consumed at L531

**Scenario:** ADR-020 makes each daemon "pinned via `--device cuda:N`" and invariant 1 asserts "single-writer per daemon … each daemon owns exactly one GPU." But `_handle_generate` (which the ADR deliberately keeps byte-identical) lets the *request payload's* `device` field override the daemon's pinned device. The client normally sends `device == args.device`, matching the socket it routed to, so the happy path is consistent. Nothing enforces it, though: a stale client, the MCP/mcpo wiring, or any confused caller can connect to the `cuda:0` socket and send `{"device": "cuda:1", ...}`. The cuda:0 daemon then runs inference on cuda:1 — touching the GPU the *other* daemon owns, and because device is part of the cache key, evicting/reloading in a way that re-introduces exactly the cross-GPU thrash ADR-020 exists to eliminate. Threat model is same-uid, so this is an accidental-footgun / isolation-correctness break rather than an exploit, hence MEDIUM. It matters because it contradicts a stated invariant and the "byte-for-byte identical `_handle_generate`" decision is what leaves it open.

**Fix:** Pin the device server-side: in `_handle_generate`, ignore the request's `device` in favor of the daemon's launch `device` (drop the `req.get("device") or` fallback so it becomes `req_device = device`), or reject with a `PathError`-style structured error when `req.get("device")` is present and does not match the daemon's device. The first is smaller and matches the "the daemon owns its GPU" mental model. This edits `_handle_generate`, which the Vision fences off — so it belongs in its own slice with an ADR-020 Changelog note, not this one (see Finding 5).

### [MEDIUM] Device normalization must run *after* an anchored whitelist, and must canonicalize the integer — otherwise aliasing and trailing-byte gaps

**Location:** design of `socket_path(device)` (ADR-020 §3; Vision §3, L94-96) — not yet code

**Scenario:** Three related canonicalization gaps, all same-uid footguns (LOW individually, grouped as MEDIUM because they undermine the two invariants the design leans on — "whitelist before filename" and "`cuda`==`cuda:0`, nothing else collides"):

1. **Ordering.** Vision §3 describes "normalize device → canonical form … whitelist-reject anything else, then build slug" — i.e. normalize *before* whitelist. If "normalize" is implemented as any transformation broader than an exact-string match (a `startswith`, a `split(":")`, a strip/lower), a crafted input could be massaged into a passing string before the regex ever sees the raw value. The safe order is: apply the anchored whitelist to the **raw** input first, *then* normalize the survivors. Normalizing a set already restricted to `{cpu, cuda, cuda:\d+}` is safe; normalizing arbitrary input is not.

2. **Anchor.** If the whitelist uses `re.match(r"^(cpu|cuda(:\d+)?)$", s)`, note that Python's `$` also matches just before a trailing newline, so `"cuda:0\n"` is accepted and yields socket name `comfyless-cuda0\n.sock` — a distinct file from `comfyless-cuda0.sock` (client/daemon mismatch) and a newline embedded in a path. Use `re.fullmatch(r"(cpu|cuda(:\d+)?)", s)` (or `\Z` instead of `$`). This is exactly the class of "device string can never contribute path components" the ADR claims to close; a fullmatch closes it, `re.match`+`$` leaves a crack.

3. **Integer canonicalization.** `\d+` accepts `cuda:00`, `cuda:0`, `cuda:007`. Normalization only rewrites bare `cuda`→`cuda:0`; it does not fold leading zeros. So `cuda:00` → `comfyless-cuda00.sock`, a *different* socket from `comfyless-cuda0.sock`, while PyTorch resolves both to physical GPU 0. Result: a daemon started as `--device cuda:00` and a client saying `--device cuda:0` miss each other, and worse, two daemons (`cuda:0` and `cuda:00`) can both bind to physical GPU 0 — VRAM contention and the very thrash the design forbids, with no error. This is the "two different device strings map unexpectedly" gap the prompt asked about.

Very large `cuda:<digits>` (e.g. `cuda:99999999`) is *not* a resource concern in itself — no traversal, no allocation; the only effect is that an absurdly long digit run could push the socket path past the `AF_UNIX` `sun_path` ~108-byte limit and make `bind()` fail loudly, which is fail-closed and acceptable.

**Fix:** Implement `socket_path` as: (a) `re.fullmatch(r"(cpu|cuda(:\d+)?)", device)` on the raw string, raise on miss; (b) then canonicalize by parsing the index — `slug = "cpu" if device == "cpu" else f"cuda{int(device.split(':')[1]) if ':' in device else 0}"` — which folds `cuda`, `cuda:0`, `cuda:00`, `cuda:007→7` correctly and cannot carry any non-`[a-z0-9]` byte into the filename. Add the negative tests the Vision already lists (invariant 4) plus one for `cuda:00 == cuda:0` and one for a trailing-newline rejection.

### [INFO] `unlink`-then-`bind` on the socket path is unchanged and remains safe under the same-uid model

**Location:** `comfyless/server.py:582-586`

**Observation:** `run_server` does `if sock_path.exists(): sock_path.unlink()` then `bind()`. `exists()` follows symlinks and `unlink()` removes the link itself (not its target), so a symlink planted at the socket name is deleted rather than followed, and a dangling symlink makes `bind()` fail loudly. The only actor who can plant anything at that path is the same uid (the dir is `0700`), so this is a same-uid self-inflicted footgun, not a cross-uid TOCTOU. The parallel design adds more socket *names* but every one lives in the same `0700` dir and is subject to the same same-uid-only reachability — no new symlink/replacement surface. No change required; calibrated to the single-user-workstation threat model this is not a finding, recorded only so the reviewer sees it was checked.

### [INFO] Scope-creep watch: the two must-fix behavioral changes fall outside the declared edit boundary

**Location:** Vision §2 (L88-90) fences `_handle_generate` generation/cache/LoRA logic as "Must NOT change"

**Observation:** Findings 1 and 2 both require edits inside `_handle_generate`/the output-numbering block, which the Vision explicitly declares off-limits for this slice (that fence is the whole basis for the "byte-for-byte identical" safety argument). That is not a reason to skip them — it means they should each land as their own slice with an ADR-020 Changelog append, rather than being quietly folded into the socket-naming change. Doing them inline would break the "A changes only the socket name + routing" property that makes this slice easy to review. Flagging the scope tension itself per §4/§6 discipline.

## Bottom line

The socket-naming and routing design preserves ADR-001 §1–§6 in full and introduces no new cross-uid exposure; device-scoped `--unload` is no-worse than baseline. Ship the socket-naming slice once Finding 3 (fullmatch + whitelist-before-normalize + integer canonicalization) is baked into the `socket_path` implementation and its negative tests — that is the one item that must be right *in this slice* because it defines the new path-construction primitive. Findings 1 (output-dir counter collision — the real data-loss exposure the parallel design creates) and 2 (payload device overriding the daemon's pinned device) are must-fix but belong in their own follow-up slices against `_handle_generate`, each with an ADR-020 Changelog note, so the isolation the ADR promises is actually enforced and not merely intended.
