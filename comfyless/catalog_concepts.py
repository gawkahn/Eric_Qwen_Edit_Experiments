# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""The CLOSED concept vocabulary for catalog enrichment (ADR-041 D1/D5).

WHY A CLOSED VOCABULARY. Slice 2 feeds third-party text (civitai marketing
prose, uploader-authored trained words) into a local LLM and reads structured
output back. That output then lands in a column the refine planner searches.
If the model could write arbitrary strings into `concepts`, a hostile or merely
confused description would be able to place text of its choosing into the field
that decides which LoRA gets offered. So `concepts` is not free text: the model
may only SELECT from the list below, and anything it returns that is not in the
list is DROPPED, not stored (ADR-041 D5, negative test 1).

The security property is structural, not diligent: `normalize()` is the only
way in, `catalog_db.upsert_enrichment` is the only caller, and the strings that
reach the FTS index come from `expand_for_index()` — which reads THIS FILE and
never the model's output. Hostile text cannot reach the concepts column even if
the model cooperates with it, because the model's contribution is reduced to a
set membership decision before storage.

HOW `haircut` REACHES A HEAD-SWAP LORA. Each concept carries query-side
aliases. `expand_for_index()` renders a tagged entry's concepts into
repo-owned alias TEXT which is what gets indexed, so a query for "haircut"
matches an entry tagged `hair` even though no third party ever wrote the word
"haircut" anywhere in its metadata. That is ADR-041's whole thesis in one
function, and it is why the aliases are query vocabulary (what an operator or a
judge critique would say) rather than synonyms of the tag name.

DERIVATION (2026-07-30, against the post-slice-1 live catalog: 308 searchable
LoRAs, 244 with description text). The corpus was measured before this list was
written, and every concept below has textual support in it — the smallest,
`inpaint`, is mentioned by 8 entries; the largest, `accelerator`, by 104. The
axes come from three sources that agree with each other:

  * the operator's own LoRA folder taxonomy (`style/`, `concept/anatomy/…`,
    `action/nsfw`, `tool/faceswap`, `accelerators/`, `realism/`) — functional,
    operator-authored, and applied to all 308 entries. Grant's read, taken as
    written: those hierarchies "aren't worthless, but they are inconsistent",
    so they are a DERIVATION source and (slice 2b) one low-weight hint among
    several in the enrichment prompt — never an authoritative tag feed;
  * functional tokens surviving a stopword + process-boilerplate + model-family
    strip of name/description/trigger text (`style` 56, `realistic` 34,
    `skin` 33, `realism` 29, `character` 28, `body`/`anatomy`/`face` 27 …);
  * the verbs the 8 instruction templates expose (head swap, base image,
    preserve lighting/environment/background, expression transfer).

FROZEN means frozen (ADR-041 D5): growing this list is a deliberate edit with a
`VOCAB_VERSION` bump, not a model decision. Bumping the version marks every
stored enrichment row as vocabulary-stale so a rebuild can re-tag it.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

#: Bumped whenever a concept id is added, removed, or renamed. Stored per
#: enrichment row (`enrichment.vocab_version`) so slice 2b can tell
#: "enriched under an older vocabulary" from "never enriched" — the two need
#: different handling and conflating them would silently freeze the corpus at
#: whatever vocabulary existed the first time it was enriched.
VOCAB_VERSION = 1

#: At most this many concepts per entry. A LoRA that "does" more than eight
#: things is not being described, it is being keyword-stuffed — and every extra
#: tag dilutes ranking for the entries that genuinely carry it.
MAX_CONCEPTS = 8

# ── The vocabulary ──────────────────────────────────────────────────────
# id -> query-side aliases. Aliases are what an OPERATOR or a JUDGE CRITIQUE
# would say, not synonyms of the id: they are the query half of the semantic
# bridge. An alias may legitimately be ambiguous across concepts ("swap" fits
# both face-swap and head-swap); `normalize()` refuses to guess and drops it,
# while `expand_for_index()` indexes it under both. See _ALIAS_INDEX.
_VOCAB: Dict[str, Tuple[str, ...]] = {
    # ── render style ────────────────────────────────────────────────────
    "photorealism": ("photorealistic", "photoreal", "realism", "realistic",
                     "lifelike", "photograph", "photographic", "real"),
    "illustration": ("illustrated", "drawing", "drawn", "painting",
                     "painterly", "artwork", "sketch"),
    "anime": ("manga", "cartoon", "toon"),
    "cinematic": ("cinema", "filmic", "movie", "cinematography"),
    "stylization": ("stylized", "stylised", "style", "styles", "aesthetic",
                    "artstyle"),
    # ── render quality / image properties ───────────────────────────────
    "detail": ("detailed", "detailing", "crisp", "crispness", "sharp",
               "sharpness", "microdetail"),
    "texture": ("textured", "surface", "grain", "roughness"),
    "skin": ("complexion", "pores", "pore", "freckles", "blemish"),
    "lighting": ("light", "lit", "relight", "relighting", "shadow",
                 "shadows", "highlights", "exposure", "backlit"),
    "color": ("colour", "saturation", "hue", "tone", "tonality", "grading",
              "palette"),
    "composition": ("framing", "crop", "cropping", "angle", "perspective",
                    "viewpoint", "shot", "pov"),
    # Aliases here must name the DEFECT, never a quality someone might want.
    # "bokeh", bare "blur" and bare "flat" were dropped after review: bokeh and
    # motion blur are things prompts REQUEST, and "flat" is a legitimate
    # illustration style — each would have surfaced slop-removal LoRAs to
    # someone asking for the opposite. "blurry"/"flatness" keep the defect
    # reading (code-review 2026-07-30, vocabulary advisories).
    "anti-slop": ("slop", "plastic", "waxy", "airbrushed", "blurry",
                  "oversaturated", "flatness"),
    # ── human subject ───────────────────────────────────────────────────
    "face": ("facial", "features", "eyes", "eye", "nose", "mouth", "lips",
             "teeth", "cheeks", "jawline"),
    "hair": ("hairstyle", "hairstyles", "haircut", "braid", "ponytail",
             "bald", "beard", "moustache"),
    "expression": ("expressions", "smile", "smiling", "frown", "emotion",
                   "gaze"),
    "hands-feet": ("hands", "hand", "fingers", "finger", "feet", "foot",
                   "toes", "nails"),
    "body": ("anatomy", "anatomical", "physique", "figure", "build",
             "proportions", "torso", "waist", "hips", "legs", "curvy",
             "petite", "slim", "muscular"),
    "breasts": ("breast", "bust", "chest", "nipples", "nipple", "cleavage",
                "tits"),
    "genitalia": ("pussy", "vagina", "vulva", "labia", "penis", "cock",
                  "genital", "genitals", "pubic"),
    # "same" was dropped as an alias: it is a filler word that appears in
    # ordinary prose and would have made every identity-tagged LoRA a magnet
    # for unrelated critiques (code-review 2026-07-30).
    "identity": ("likeness", "resemblance", "consistency", "consistent",
                 "recognizable"),
    "character": ("persona", "characters"),
    "face-swap": ("faceswap", "swap"),
    "head-swap": ("headswap", "head", "swap"),
    "nudity": ("nude", "naked", "topless", "undressed", "undress"),
    "sex-act": ("sex", "sexual", "intercourse", "blowjob", "oral",
                "cunnilingus", "deepthroat", "penetration", "doggy",
                "cowgirl", "anal", "masturbation", "orgasm"),
    "nsfw": ("explicit", "porn", "pornographic", "hentai", "adult", "lewd"),
    # ── scene contents ──────────────────────────────────────────────────
    "clothing": ("clothes", "outfit", "outfits", "dress", "lingerie",
                 "shirt", "garment", "costume", "uniform", "swimsuit"),
    "object": ("objects", "prop", "props", "item", "product"),
    "background": ("environment", "scene", "scenery", "backdrop", "setting",
                   "location"),
    # ── edit operations (edit-model LoRAs) ──────────────────────────────
    "edit-instruction": ("edit", "editing", "instruct", "instruction",
                         "transform", "replace", "remove", "insert"),
    "inpaint": ("inpainting", "mask", "masked", "outpaint", "outpainting"),
    "upscale": ("upscaling", "resolution", "enlarge", "hires"),
    "restore": ("restoration", "repair", "denoise", "deblur", "artifacts",
                "artifact", "fix", "cleanup"),
    # ── generation mechanics ────────────────────────────────────────────
    "accelerator": ("turbo", "lightning", "distilled", "distill", "hyper",
                    "lcm", "steps", "speed", "fast"),
    "slider": ("sliders", "adjust", "adjustable", "bidirectional", "dial"),
    "text-render": ("text", "typography", "lettering", "logo", "signage",
                    "caption", "writing"),
    "pose": ("poses", "posing", "posture", "stance", "kneeling", "standing",
             "sitting", "crouching"),
    "action": ("motion", "movement", "dynamic", "running", "walking",
               "gesture"),
}

#: The closed set. Anything not in here is not storable, full stop.
CONCEPTS: FrozenSet[str] = frozenset(_VOCAB)

#: Canonical order for stored/serialized concept lists — declaration order, so
#: a stored JSON array is deterministic. Two enrichment runs that pick the same
#: set produce byte-identical rows, which is what lets 2b treat an unchanged
#: source hash as "nothing to do" without diffing semantics.
_ORDER: Dict[str, int] = {cid: i for i, cid in enumerate(_VOCAB)}

# alias -> every concept it could mean. Built rather than asserted-unique on
# purpose: shared aliases are real ("swap" belongs to face-swap AND head-swap),
# and the two consumers want opposite things from them. `normalize()` must not
# guess, so it drops an ambiguous alias; the index wants maximum reach, so it
# emits the token under both concepts. Encoding that as one map keeps the
# ambiguity visible instead of resolving it silently in whichever direction the
# first writer happened to prefer.
_ALIAS_INDEX: Dict[str, Set[str]] = {}
for _cid, _aliases in _VOCAB.items():
    for _a in _aliases:
        _ALIAS_INDEX.setdefault(_a, set()).add(_cid)
# A concept id always wins over any alias claim on the same string — an id is
# never ambiguous.
for _cid in _VOCAB:
    _ALIAS_INDEX[_cid] = {_cid}


def _canon(raw: object) -> str:
    """Model output -> comparable key: casefold, unify hyphen/underscore/space,
    strip surrounding junk. Deliberately narrow — this normalizes SPELLING, and
    must never widen into interpreting arbitrary strings."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip().casefold()
    for ch in ("_", " ", "/"):
        s = s.replace(ch, "-")
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-.,;:\"'")


def normalize(raw: object, *, max_concepts: int = MAX_CONCEPTS
              ) -> Tuple[List[str], List[str]]:
    """Model-proposed concepts -> (accepted ids, dropped raw tags).

    The parse boundary of ADR-041 D5. Accepts a sequence (the documented
    output shape) or a comma/newline-separated string (the shape a model
    produces when it ignores the shape it was asked for). Everything else
    yields no concepts at all — a scalar, a dict, or None is a malformed
    response, not a tag.

    Accepted: an exact concept id, or an UNAMBIGUOUS alias of one. Dropped:
    unknown text, ambiguous aliases, duplicates, non-strings, and anything
    past `max_concepts`. Dropped tags are RETURNED rather than swallowed so
    the build tool can log them — a description trying to inject text is
    exactly the event an operator should be able to see, and silence there
    would make a hostile corpus indistinguishable from a cooperative one.

    Output is in canonical vocabulary order, deduplicated.
    """
    if isinstance(raw, str):
        items: Sequence[object] = [
            p for p in raw.replace("\n", ",").split(",") if p.strip()]
    elif isinstance(raw, (list, tuple, set, frozenset)):
        items = list(raw)
    else:
        return [], []

    accepted: Set[str] = set()
    dropped: List[str] = []
    # Bound the work regardless of what came back: a model that returns 10k
    # tags gets its first few hundred looked at, not an unbounded loop.
    for item in list(items)[:256]:
        key = _canon(item)
        owners = _ALIAS_INDEX.get(key)
        if not owners or len(owners) != 1:
            # Unknown, or ambiguous ("swap"). Either way it is not a decision
            # this function is allowed to make.
            dropped.append(str(item)[:64])
            continue
        cid = next(iter(owners))
        if cid in accepted:
            continue
        if len(accepted) >= max_concepts:
            dropped.append(str(item)[:64])
            continue
        accepted.add(cid)
    return sorted(accepted, key=lambda c: _ORDER[c]), dropped


def expand_for_index(concept_ids: Sequence[str]) -> str:
    """Validated concept ids -> the repo-owned TEXT that gets FTS-indexed.

    This is the retrieval half of the semantic bridge: an entry tagged `hair`
    is indexed as "hair hairstyle hairstyles haircut braid ponytail bald
    beard moustache", so `haircut` finds it. Hyphenated ids are emitted as
    separate words too (`head-swap` -> "head swap headswap …") because
    unicode61 tokenizes on the hyphen anyway and the bare words are what a
    query actually contains.

    EVERY byte returned here comes from this module. Unknown ids are ignored
    rather than echoed, so even a caller that skipped `normalize()` cannot get
    third-party text into the index through this function. That is what makes
    the concepts column structurally clean rather than merely validated.
    """
    words: List[str] = []
    seen: Set[str] = set()
    if not isinstance(concept_ids, (list, tuple)):
        return ""
    for cid in concept_ids:
        # Non-strings are skipped BEFORE the membership test. A stored blob is
        # not necessarily a list of strings — a hand-edited or corrupted
        # `enrichment.concepts` can decode to `[["x"]]`, and `["x"] in _VOCAB`
        # raises TypeError: unhashable. Guarding at the type level rather than
        # relying on the caller keeps the "no caller can break this" promise
        # honest for availability as well as for injection (code-review
        # 2026-07-30, finding 1).
        if not isinstance(cid, str) or cid not in _VOCAB:
            continue
        for token in (cid.replace("-", " "), *_VOCAB[cid]):
            for w in token.split():
                if w not in seen:
                    seen.add(w)
                    words.append(w)
    return " ".join(words)


def vocabulary_prompt_block() -> str:
    """The concept list as it is shown to the enrichment model (slice 2b).

    Lives here so the prompt and the validator cannot drift apart: a model
    asked for tags that `normalize()` would reject produces nothing but
    dropped tags, and the failure looks like a bad model rather than a bad
    prompt. One line per concept, id first, aliases as the gloss.
    """
    return "\n".join(
        f"- {cid}: {', '.join(_VOCAB[cid])}" for cid in _VOCAB)
