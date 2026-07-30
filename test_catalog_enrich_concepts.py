#!/usr/bin/env python3
"""Unit tests — offline LLM concept enrichment (ADR-041 D1, slice 2b).

Entirely offline: every endpoint call is injected. The point of the suite is
that the module's guarantees hold no matter WHAT the model returns — a
cooperative model, a lazy one, a broken one, and one that has been talked into
cooperating with a hostile description.

Run: ./.venv/bin/python3 test_catalog_enrich_concepts.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import comfyless.catalog_db as cdb            # noqa: E402
import comfyless.catalog_concepts as ccpt     # noqa: E402
import comfyless.catalog_enrich_concepts as ce  # noqa: E402

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def _row(**kw):
    """A catalog row as sqlite3.Row-ish: the module indexes by key."""
    base = {"id": 1, "name": "x", "abs_path": "/root/sub/x.safetensors",
            "root": "/root", "model_family": "flux", "model_name": None,
            "description": None, "usage_tips": None, "trigger_words": None,
            "instruction_template": None, "have_hash": None,
            "have_vocab": None}
    base.update(kw)
    return base


_HEAD_SWAP = ("head_swap: start with Picture 1 as the base image, keeping its "
              "lighting, environment and background, and replace the head "
              "with the one from Picture 2 preserving hair and eye colour")


# ════════════════════════════════════════════════════════════════════════
print("\n== The prompt is code-owned and cannot drift from the validator ==")
# ════════════════════════════════════════════════════════════════════════

_sp = ce.system_prompt()
_missing = [c for c in ccpt.CONCEPTS if c not in _sp]
check("the system prompt carries every concept id",
      not _missing, detail=repr(_missing))
check("the vocabulary is spliced from catalog_concepts, not hand-copied "
      "(so a vocabulary edit reaches the prompt automatically)",
      ccpt.vocabulary_prompt_block() in _sp)
check("the prompt frames metadata as DATA, not instructions",
      "not instructions" in _sp and "ignore any directions" in _sp.lower())
check("the prompt asks for the JSON shape parse_response reads",
      '"concepts"' in _sp and '"function_summary"' in _sp)
for leak in ("operator", "planner", "critique", "target prompt"):
    check(f"the prompt carries no {leak} text (build-time tool, not a "
          f"conversation)", leak not in _sp.lower())


# ════════════════════════════════════════════════════════════════════════
print("\n== entry_metadata: exactly what the model sees ==")
# ════════════════════════════════════════════════════════════════════════

m = ce.entry_metadata(_row(name="thing", model_family="flux2",
                           description="a description",
                           instruction_template=_HEAD_SWAP))
check("folder is derived relative to the scan root", m["folder"] == "sub",
      detail=repr(m["folder"]))
check("the untruncated template is passed through",
      m["instruction_template"] == _HEAD_SWAP)
m_root = ce.entry_metadata(_row(abs_path="/root/x.safetensors"))
check("an entry at the root has no folder hint", m_root["folder"] == "",
      detail=repr(m_root["folder"]))
m_out = ce.entry_metadata(_row(abs_path="/elsewhere/x.safetensors"))
check("a path outside its root yields no folder rather than a leaked "
      "absolute path", m_out["folder"] == "", detail=repr(m_out["folder"]))
# A SIBLING root sharing a name prefix is the case a bare startswith gets
# wrong — it yields `../loras-extra/sub`, feeding a `..` path to the model as
# a folder. The /elsewhere case above passes either way, so it proved nothing
# here (code-review 2026-07-30, finding 5).
m_sib = ce.entry_metadata(_row(root="/models/loras",
                               abs_path="/models/loras-extra/sub/a.safetensors"))
check("a SIBLING root sharing a prefix yields no folder (not a '..' path)",
      m_sib["folder"] == "", detail=repr(m_sib["folder"]))
m_deep = ce.entry_metadata(_row(root="/models/loras/",
                                abs_path="/models/loras/a/b/x.safetensors"))
check("a trailing separator on the root does not break folder derivation",
      m_deep["folder"] == os.path.join("a", "b"), detail=repr(m_deep["folder"]))

# Duplicate trained words are the norm after slice 1's 64 B cap: two
# near-identical templates truncate to the SAME string, and sending it twice
# tells the model nothing.
m_dup = ce.entry_metadata(_row(trigger_words=json.dumps(
    ["head_swap: start with Picture 1 as the base image, keeping its l",
     "head_swap: start with Picture 1 as the base image, keeping its l",
     "other"])))
check("identical trigger words are deduped before they reach the prompt",
      m_dup["trigger_words"] == [
          "head_swap: start with Picture 1 as the base image, keeping its l",
          "other"], detail=repr(m_dup["trigger_words"]))
check("a corrupt trigger_words blob degrades to no triggers, never raises",
      ce.entry_metadata(_row(trigger_words="{not json"))["trigger_words"] == [])
check("non-list trigger_words JSON degrades to no triggers",
      ce.entry_metadata(_row(trigger_words="42"))["trigger_words"] == [])
big = ce.entry_metadata(_row(description="d" * 9000, usage_tips="t" * 9000))
check("description is capped for the prompt (one huge description must not "
      "crowd out the vocabulary block)",
      len(big["description"]) == ce._PROMPT_DESC_CAP)
check("usage_tips is capped for the prompt",
      len(big["usage_tips"]) == ce._PROMPT_TIPS_CAP)

um = ce.render_user_message(ce.entry_metadata(
    _row(name="thing", description="a description")))
check("the user message fences the untrusted block",
      "<lora-metadata>" in um and "</lora-metadata>" in um)
check("empty fields are omitted rather than sent as blanks",
      "usage tips:" not in um, detail=um)
check("the folder hint is labelled WEAK where it appears",
      "WEAK hint" in ce.render_user_message(
          ce.entry_metadata(_row(description="d"))))


# ════════════════════════════════════════════════════════════════════════
print("\n== source_hash: what invalidates an enrichment ==")
# ════════════════════════════════════════════════════════════════════════

meta_a = ce.entry_metadata(_row(description="alpha"))
meta_b = ce.entry_metadata(_row(description="beta"))
check("the hash is stable across calls",
      ce.source_hash(meta_a) == ce.source_hash(meta_a))
check("changed metadata changes the hash",
      ce.source_hash(meta_a) != ce.source_hash(meta_b))

_real_vocab = ccpt.VOCAB_VERSION
try:
    ccpt.VOCAB_VERSION = _real_vocab + 1
    _bumped = ce.source_hash(meta_a)
finally:
    ccpt.VOCAB_VERSION = _real_vocab
check("a VOCABULARY bump changes the hash (else the corpus freezes at "
      "whatever vocabulary enriched it first, and nothing reports the split)",
      _bumped != ce.source_hash(meta_a))

_real_sp = ce._SYSTEM_PROMPT
try:
    ce._SYSTEM_PROMPT = _real_sp + "\nOne more instruction.\n"
    _pbump = ce.source_hash(meta_a)
finally:
    ce._SYSTEM_PROMPT = _real_sp
check("changing the SYSTEM PROMPT TEXT changes the hash (content-hashed, so "
      "a prompt revision re-enriches without anyone remembering to bump a "
      "version integer)",
      _pbump != ce.source_hash(meta_a))

# The bug that killed PROMPT_VERSION: an ALIAS-only vocabulary edit changes
# the prompt the model sees, but VOCAB_VERSION's documented bump rule covers
# concept IDS only — so under integer versioning this edit invalidated
# nothing and every affected entry kept a stale enrichment forever.
_real_hair = ccpt._VOCAB["hair"]
try:
    ccpt._VOCAB["hair"] = _real_hair + ("wig",)
    _alias_hash = ce.source_hash(meta_a)
finally:
    ccpt._VOCAB["hair"] = _real_hair
check("an ALIAS-ONLY vocabulary edit changes the hash — the edit the "
      "id-based bump rule does NOT cover, and the common kind in practice",
      _alias_hash != ce.source_hash(meta_a))


# ════════════════════════════════════════════════════════════════════════
print("\n== parse_response: tolerant envelope, strict nothing else ==")
# ════════════════════════════════════════════════════════════════════════

_GOOD = '{"concepts": ["hair", "face"], "function_summary": "Swaps heads."}'
c, s = ce.parse_response(_GOOD)
check("bare JSON parses", c == ["hair", "face"] and s == "Swaps heads.")
c, s = ce.parse_response("```json\n" + _GOOD + "\n```")
check("a fenced code block parses", c == ["hair", "face"])
c, s = ce.parse_response("Sure! Here is the answer:\n" + _GOOD + "\nHope that helps!")
check("prose on both sides parses (models do this constantly)",
      c == ["hair", "face"] and s == "Swaps heads.")
for junk, label in ((None, "None"), (42, "an int"), ("", "empty string"),
                    ("no braces here", "prose with no object"),
                    ("{ not json }", "malformed object"),
                    ("[1,2,3]", "a JSON array")):
    c, s = ce.parse_response(junk)
    check(f"{label} yields (None, None) and never raises",
          c is None and s is None)
c, s = ce.parse_response('{"function_summary": "only a summary"}')
check("a missing concepts key is None, not a crash",
      c is None and s == "only a summary")
c, s = ce.parse_response('{"concepts": [], "function_summary": "x"}')
check("an EXPLICIT empty list is a list, not None — the batch relies on this "
      "to tell an honest 'affects nothing' from a half-shaped reply",
      c == [] and s == "x")


# ════════════════════════════════════════════════════════════════════════
print("\n== call_model: sampling is code-owned ==")
# ════════════════════════════════════════════════════════════════════════

_seen = {}


def _fake_post(endpoint, payload, key):
    _seen["endpoint"] = endpoint
    _seen["payload"] = payload
    _seen["key"] = key
    return [_GOOD]


cfg = {"type": "openai-endpoint", "url": "http://localhost:8019/v1",
       "model": "gemma-test", "top_k": 20, "repetition_penalty": 1.05,
       "temperature": 0.9, "batch_variations": True}
out = ce.call_model(dict(cfg), "user text", _post=_fake_post)
check("returns the first choice's content", out == _GOOD)
check("posts to /chat/completions on the configured url",
      _seen["endpoint"] == "http://localhost:8019/v1/chat/completions")
p = _seen["payload"]
check("temperature is 0 — this is extraction, not creative writing "
      "(and a stable result is what makes source_hash skipping meaningful)",
      p["temperature"] == 0.0, detail=repr(p.get("temperature")))
check("the backend's CREATIVE sampling knobs are NOT inherited",
      "top_k" not in p and "repetition_penalty" not in p
      and "batch_variations" not in p and "n" not in p,
      detail=repr(sorted(p)))
check("max_tokens is bounded", isinstance(p.get("max_tokens"), int)
      and 0 < p["max_tokens"] <= 2048)
check("the system message is the code-owned prompt",
      p["messages"][0]["role"] == "system"
      and p["messages"][0]["content"] == ce.system_prompt())
check("the untrusted metadata goes in the USER turn, never the system turn",
      p["messages"][1]["role"] == "user"
      and p["messages"][1]["content"] == "user text")


def _boom_post(endpoint, payload, key):
    from comfyless import enhance
    raise enhance.EnhanceError("connection refused")


try:
    ce.call_model(dict(cfg), "x", _post=_boom_post)
    _raised = ""
except ce.ConceptEnrichError as e:
    _raised = str(e)
except Exception as e:  # noqa: BLE001
    _raised = f"WRONG TYPE {type(e).__name__}"
check("an endpoint error surfaces as ConceptEnrichError",
      _raised == "connection refused", detail=_raised)


def _empty_post(endpoint, payload, key):
    return []


try:
    ce.call_model(dict(cfg), "x", _post=_empty_post)
    _ok = False
except ce.ConceptEnrichError:
    _ok = True
check("a response with no choices is an error, not an empty enrichment", _ok)


# ════════════════════════════════════════════════════════════════════════
print("\n== enrich_concepts: the batch, with the model injected ==")
# ════════════════════════════════════════════════════════════════════════

def _mkdb(td, n=3):
    """A catalog with n LoRA entries + one excluded + one stale."""
    conn = cdb.connect(os.path.join(td, "c.sqlite"))
    cdb.upsert_family(conn, name="flux", hf_local_path="/hf/flux")
    ids = {}
    for i in range(n):
        eid = cdb.upsert_entry(conn, name=f"lora{i}", kind="lora",
                               abs_path=f"/root/style/lora{i}.safetensors",
                               root="/root")
        cdb.upsert_description(conn, entry_id=eid, source="civitai_api",
                               description=f"description number {i}")
        cdb.set_entry_family(conn, eid, "flux")
        ids[f"lora{i}"] = eid
    ex = cdb.upsert_entry(conn, name="excluded_one", kind="lora",
                          abs_path="/root/x.safetensors", root="/root")
    conn.execute("UPDATE entries SET excluded = 1 WHERE id = ?", (ex,))
    st = cdb.upsert_entry(conn, name="stale_one", kind="lora",
                          abs_path="/root/s.safetensors", root="/root")
    conn.execute("UPDATE entries SET stale = 1 WHERE id = ?", (st,))
    # A transformer must never be a candidate — Grant's scope decision.
    cdb.upsert_entry(conn, name="a_transformer", kind="transformer",
                     abs_path="/root/t.safetensors", root="/root")
    conn.commit()
    conn.close()
    return os.path.join(td, "c.sqlite"), ids


_calls = []


def _good_call(cfg, user_message):
    _calls.append(user_message)
    return _GOOD


_real_resolve = ce.resolve_backend
ce.resolve_backend = lambda backend, registry_path=None: {
    "type": "openai-endpoint", "url": "http://x/v1", "model": "m"}

try:
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td)
        _calls.clear()
        st = ce.enrich_concepts(dbp, _call=_good_call)
        check("only non-excluded, non-stale LoRAs are candidates "
              "(transformers are out of scope by Grant's decision)",
              st["candidates"] == 3, detail=repr(st))
        check("all candidates enriched", st["enriched"] == 3
              and st["failures"] == 0 and st["unparseable"] == 0,
              detail=repr(st))

        conn = cdb.connect(dbp)
        rows = conn.execute("SELECT * FROM enrichment").fetchall()
        check("one enrichment row per entry", len(rows) == 3)
        check("concepts stored as validated vocabulary ids",
              json.loads(rows[0]["concepts"]) == ["face", "hair"],
              detail=rows[0]["concepts"])
        check("function_summary stored", rows[0]["function_summary"]
              == "Swaps heads.")
        # Exact, not a substring: "m" appears in almost any wrong value,
        # including one that lost the model half entirely (code-review
        # 2026-07-30, finding 6).
        check("the model id is recorded for provenance, backend-qualified",
              rows[0]["model"] == "gemma-moe-nvfp4:m",
              detail=repr(rows[0]["model"]))
        check("source_hash recorded", bool(rows[0]["source_hash"]))
        check("vocab_version stamped", rows[0]["vocab_version"]
              == ccpt.VOCAB_VERSION)
        # End-to-end: the whole point of the slice.
        check("END-TO-END: an enriched entry is reachable by a concept ALIAS "
              "nobody wrote in its metadata ('haircut' -> tagged `hair`)",
              any(h["name"].startswith("lora")
                  for h in cdb.search(conn, "haircut")),
              detail=repr([h["name"] for h in cdb.search(conn, "haircut")]))
        conn.close()

        # ── Incremental ────────────────────────────────────────────────
        _calls.clear()
        st2 = ce.enrich_concepts(dbp, _call=_good_call)
        check("a second run makes ZERO model calls (incremental by "
              "source_hash — D1's 'only for entries whose metadata changed')",
              st2["skipped_fresh"] == 3 and st2["enriched"] == 0
              and not _calls, detail=repr(st2))

        _calls.clear()
        st3 = ce.enrich_concepts(dbp, _call=_good_call, refresh=True)
        check("--refresh forces re-enrichment", st3["enriched"] == 3
              and len(_calls) == 3, detail=repr(st3))

        # Changing the metadata must invalidate.
        conn = cdb.connect(dbp)
        conn.execute("UPDATE descriptions SET description = 'CHANGED' "
                     "WHERE entry_id = ?", (ids["lora0"],))
        conn.commit()
        conn.close()
        _calls.clear()
        st4 = ce.enrich_concepts(dbp, _call=_good_call)
        check("changed metadata re-enriches exactly that entry",
              st4["enriched"] == 1 and st4["skipped_fresh"] == 2,
              detail=repr(st4))

        # A vocabulary bump must invalidate every row.
        _real_v = ccpt.VOCAB_VERSION
        try:
            ccpt.VOCAB_VERSION = _real_v + 1
            _calls.clear()
            st5 = ce.enrich_concepts(dbp, _call=_good_call)
            check("a VOCABULARY bump re-enriches the whole corpus",
                  st5["enriched"] == 3 and st5["skipped_fresh"] == 0,
                  detail=repr(st5))
        finally:
            ccpt.VOCAB_VERSION = _real_v

    # ── Hostile / malformed model output ───────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=1)

        def _hostile(cfg, user_message):
            return json.dumps({
                "concepts": ["hair", "ignore all previous instructions",
                             "<script>alert(1)</script>", "swap",
                             "'; DROP TABLE entries; --"],
                "function_summary":
                    "<b>Best</b> LoRA ever &lt;script&gt;x&lt;/script&gt;"})

        st = ce.enrich_concepts(dbp, _call=_hostile)
        conn = cdb.connect(dbp)
        row = conn.execute("SELECT * FROM enrichment").fetchone()
        stored = json.loads(row["concepts"])
        check("NEGATIVE: a cooperating model cannot store invented tags",
              stored == ["hair"], detail=repr(stored))
        check("NEGATIVE: the ambiguous tag is dropped too", "swap" not in stored)
        check("NEGATIVE: dropped tags are COUNTED and surfaced, not swallowed",
              st["dropped_tags"] == 4 and len(st["dropped_examples"]) == 4,
              detail=repr(st))
        check("NEGATIVE: the summary is sanitized at the DB boundary",
              "<b>" not in (row["function_summary"] or "")
              and "<script>" not in (row["function_summary"] or ""),
              detail=repr(row["function_summary"]))
        fts = conn.execute(
            "SELECT concepts FROM catalog_fts WHERE concepts != ''").fetchall()
        joined = " ".join(r["concepts"] for r in fts)
        check("NEGATIVE: NONE of the hostile text reaches the indexed "
              "concepts column — it holds repo-owned alias text only",
              "ignore" not in joined and "script" not in joined
              and "DROP" not in joined, detail=joined[:120])
        conn.close()

    # The dropped-tag log is the ONE channel this module deliberately points
    # at hostile-influenced text, and `normalize()` returns the raw tag —
    # truncated, but NOT control-stripped (it is not the DB sanitizer). So it
    # must be repr'd before it reaches a terminal (code-review 2026-07-30,
    # finding 4).
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=1)
        ESC = "\x1b]0;pwned\x07\x1b[31m"

        def _escapes(cfg, user_message):
            return json.dumps({"concepts": ["hair", ESC + "evil"],
                               "function_summary": "x"})

        st = ce.enrich_concepts(dbp, _call=_escapes)
        joined_ex = " ".join(st["dropped_examples"])
        check("NEGATIVE: a terminal-escape sequence in a dropped tag is "
              "repr-escaped before it can reach the operator's terminal",
              st["dropped_examples"] and "\x1b" not in joined_ex
              and "\\x1b" in joined_ex, detail=repr(joined_ex)[:120])

    # ── Unparseable / failing model ────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=2)
        st = ce.enrich_concepts(dbp, _call=lambda c, u: "I'm sorry, I can't.")
        check("unparseable replies are counted, not stored",
              st["unparseable"] == 2 and st["enriched"] == 0, detail=repr(st))
        conn = cdb.connect(dbp)
        check("...and no enrichment row is written for them",
              conn.execute("SELECT COUNT(*) FROM enrichment").fetchone()[0] == 0)
        conn.close()

    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=3)

        def _empty_concepts(cfg, user_message):
            return '{"concepts": [], "function_summary": "Does something."}'

        st = ce.enrich_concepts(dbp, _call=_empty_concepts)
        check("an honest empty-concepts answer is stored and counted "
              "(a thin description should not be force-tagged)",
              st["no_concepts"] == 3 and st["enriched"] == 3, detail=repr(st))

    # ...but a reply MISSING the concepts key entirely is a half-shaped
    # response, not an honest empty one. Storing it would record a
    # source_hash and the entry would never be retried, leaving it
    # indistinguishable afterwards from a real "affects nothing" answer.
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=2)
        st = ce.enrich_concepts(
            dbp, _call=lambda c, u: '{"function_summary": "Adds things."}')
        check("a reply MISSING the concepts key counts as unparseable "
              "(retryable) rather than being stored as an empty answer",
              st["unparseable"] == 2 and st["enriched"] == 0, detail=repr(st))
        conn = cdb.connect(dbp)
        check("...and no row is written, so a later run retries it",
              conn.execute(
                  "SELECT COUNT(*) FROM enrichment").fetchone()[0] == 0)
        conn.close()

    # ── Endpoint failure semantics ─────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=10)
        _n = {"i": 0}

        def _always_fail(cfg, user_message):
            _n["i"] += 1
            raise ce.ConceptEnrichError("connection refused")

        try:
            ce.enrich_concepts(dbp, _call=_always_fail)
            _aborted = False
        except ce.ConceptEnrichError:
            _aborted = True
        check("a dead endpoint aborts after N consecutive failures rather "
              "than hammering all 10 entries", _aborted
              and _n["i"] == ce._CONSECUTIVE_FAILURE_ABORT,
              detail=f"{_n['i']} calls made")

        # One bad entry among good ones must NOT abort the run.
        _n2 = {"i": 0}

        def _one_bad(cfg, user_message):
            _n2["i"] += 1
            if _n2["i"] == 2:
                raise ce.ConceptEnrichError("transient")
            return _GOOD

        st = ce.enrich_concepts(dbp, _call=_one_bad)
        check("an isolated failure is counted and the run continues "
              "(warn-don't-block)",
              st["failures"] == 1 and st["enriched"] == 9, detail=repr(st))

    # The consecutive-failure COUNTER RESET, which the isolated-failure test
    # above cannot see: with only one injected failure, `consecutive` never
    # approaches the threshold whether or not the reset exists. A FLAKY
    # endpoint is the case that distinguishes them — fail every other call,
    # and without the reset the count accumulates to 5 and aborts a run that
    # is making steady progress, with a message claiming the failures were
    # "consecutive" (code-review 2026-07-30, finding 2).
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=12)
        _flap = {"i": 0}

        def _alternating(cfg, user_message):
            _flap["i"] += 1
            if _flap["i"] % 2 == 0:
                raise ce.ConceptEnrichError("flaky")
            return _GOOD

        try:
            st = ce.enrich_concepts(dbp, _call=_alternating)
            _aborted2 = False
        except ce.ConceptEnrichError:
            st = {}
            _aborted2 = True
        check("a FLAKY endpoint (fail every other call) does NOT abort — the "
              "counter resets on success, so 6 scattered failures are not 5 "
              "consecutive ones",
              not _aborted2 and st.get("failures", 0) > 1
              and st.get("enriched", 0) > 1, detail=repr(st))

    # The FTS rebuild that runs BEFORE the abort raises. The always-failing
    # test above commits nothing, so it cannot tell whether that rebuild
    # happened; only a run with successes THEN a death can (code-review
    # 2026-07-30, finding 3). Without it, an overnight run that enriched 200
    # entries and then lost the endpoint leaves all 200 un-searchable until
    # some later run completes.
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=9)
        _seq = {"i": 0}

        def _good_then_dead(cfg, user_message):
            _seq["i"] += 1
            if _seq["i"] <= 3:
                return _GOOD
            raise ce.ConceptEnrichError("endpoint died")

        try:
            ce.enrich_concepts(dbp, _call=_good_then_dead)
            _died = False
        except ce.ConceptEnrichError:
            _died = True
        check("an endpoint that dies mid-run still aborts", _died)
        conn = cdb.connect(dbp)
        check("the 3 entries enriched BEFORE the abort are committed",
              conn.execute("SELECT COUNT(*) FROM enrichment").fetchone()[0] == 3)
        check("...and they are actually SEARCHABLE by concept alias — the "
              "abort path rebuilds FTS before raising, so committed work is "
              "not stranded until some later completing run",
              len(cdb.search(conn, "haircut")) == 3,
              detail=repr([h["name"] for h in cdb.search(conn, "haircut")]))
        conn.close()

    # `--limit` is about MODEL CALLS, not rows scanned: on a resume, entries
    # already fresh must be skipped WITHOUT consuming the limit, or a
    # limited resume would make no progress at all.
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=3)
        st1 = ce.enrich_concepts(dbp, _call=_good_call, limit=2)
        st2 = ce.enrich_concepts(dbp, _call=_good_call, limit=2)
        check("a limited RESUME skips the fresh entries without spending the "
              "limit on them, and finishes the remainder",
              st1["enriched"] == 2 and st2["skipped_fresh"] == 2
              and st2["enriched"] == 1, detail=f"{st1} then {st2}")

    # include_excluded is a declared flag; the default side was pinned above.
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=2)
        st = ce.enrich_concepts(dbp, _call=_good_call, include_excluded=True)
        check("--include-excluded widens the candidate set to excluded "
              "entries (stale ones stay out)",
              st["candidates"] == 3 and st["enriched"] == 3, detail=repr(st))

    # ── dry-run and limit ──────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        dbp, ids = _mkdb(td, n=3)
        st = ce.enrich_concepts(dbp, _call=_good_call, dry_run=True)
        conn = cdb.connect(dbp)
        check("--dry-run calls the model but writes NOTHING",
              st["enriched"] == 3
              and conn.execute(
                  "SELECT COUNT(*) FROM enrichment").fetchone()[0] == 0,
              detail=repr(st))
        conn.close()
        st = ce.enrich_concepts(dbp, _call=_good_call, limit=2)
        check("--limit bounds the run (resumable)", st["enriched"] == 2,
              detail=repr(st))
finally:
    ce.resolve_backend = _real_resolve


# ════════════════════════════════════════════════════════════════════════
print("\n== Load-plane independence (ADR-022 invariant 7) ==")
# ════════════════════════════════════════════════════════════════════════

_REPO = os.path.dirname(os.path.abspath(__file__))
for fname in ("comfyless/generate.py", "comfyless/server.py",
              "comfyless/catalog.py"):
    tree = ast.parse(open(os.path.join(_REPO, fname), encoding="utf-8").read())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    bad = [i for i in imports if "catalog_enrich_concepts" in i]
    check(f"{fname} never imports the enrichment tool", not bad,
          detail=repr(bad))


# ════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────")
print(f"  {passed} passed, {failed} failed")
print("──────────────────────────────────────────────────")
sys.exit(0 if failed == 0 else 1)
