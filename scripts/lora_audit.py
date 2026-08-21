"""LoRA audit / classify / manifest tool — S1 of ADR-014.

See docs/decisions/ADR-014-lora-audit-tool.md and
docs/vision/slice-lora-audit.md for the contract.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import sys
import tomllib
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import safetensors.torch  # third-party: convert-output writer (ADR §8, §10)

# ── folder_paths stub (ADR §2, F-4) ────────────────────────────────────
# MUST be installed BEFORE any node-module load — eric_qwen_edit_lora.py
# does `import folder_paths` at module level. The stub satisfies the
# import without pulling in ComfyUI's runtime; any real `folder_paths`
# on sys.path is never consulted.
_fp = types.ModuleType("folder_paths")
_fp.get_folder_paths = lambda _category: []
_fp.get_full_path = lambda _category, _name: None
sys.modules["folder_paths"] = _fp

# ── Classifier reuse ───────────────────────────────────────────────────
# These modules moved to comfyless/core/ (ADR-045 slice 1b).  That package
# is import-safe -- its __init__ is a docstring, unlike nodes/__init__.py
# which pulls in every node class and ComfyUI's `comfy.*`.  So the old
# fake-package + spec_from_file_location scaffolding is no longer needed:
# a plain import resolves the modules AND their relative imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"

# Run as a script, sys.path[0] is scripts/, so `comfyless` is not importable
# from here.  (The folder_paths stub above is already installed, and
# comfyless/__init__ only fills gaps, so it will not be displaced.)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Modules still under nodes/ (e.g. eric_qwen_edit_lora) still need the fake
# package: nodes/__init__.py imports every node class and ComfyUI's `comfy.*`.
_nodes_pkg = types.ModuleType("nodes")
_nodes_pkg.__path__ = [str(_NODES_DIR)]
sys.modules.setdefault("nodes", _nodes_pkg)


def _load_node_module(modname: str):
    """Import a classifier module, wherever slice 1b left it."""
    if (_REPO_ROOT / "comfyless" / "core" / f"{modname}.py").exists():
        return importlib.import_module(f"comfyless.core.{modname}")
    dotted = f"nodes.{modname}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(
        dotted, _NODES_DIR / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order: convert first, then convert_apply (relative
# import from convert), then check, then qwen-edit-lora.
_load_node_module("eric_lora_format_convert")
_convert_mod = _load_node_module("eric_lora_format_convert_apply")
_check_mod = _load_node_module("eric_diffusion_lora_check")
_qwen_mod = _load_node_module("eric_qwen_edit_lora")
_convert_base_mod = sys.modules["comfyless.core.eric_lora_format_convert"]

check_lora = _check_mod.check_lora
build_param_dict_from_dir = _check_mod.build_param_dict_from_dir
LoRACheckResult = _check_mod.LoRACheckResult
_read_safetensors_header = _check_mod._read_safetensors_header
_strip_adapter_suffix = _check_mod._strip_adapter_suffix
find_matching_plan = _convert_mod.find_matching_plan
convert_state_dict = _convert_mod.convert_state_dict  # S3: convert-path writer
detect_lora_format = _convert_base_mod.detect_lora_format
_load_state_dict = _qwen_mod._load_state_dict
load_lora_with_key_fix = _qwen_mod.load_lora_with_key_fix
unload_adapters = _qwen_mod.unload_adapters

# ── Tool contract constants ────────────────────────────────────────────
_TOOL_VERSION = "0.3.0"  # 0.2.0: kind:"transformer" entries (ADR-021)
#                        # 0.3.0: ok_native_convert + native_convert field
#                        #        (ADR-014 amendment 2026-07-28)
_AUDIT_VERSION = 1

EXIT_OK = 0
EXIT_STARTUP_FAIL = 1
EXIT_FILE_ERRORS = 2

_SCAN_EXTENSIONS = (".safetensors", ".pt", ".bin", ".pth")
_BASE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_SIZE_CAP_BYTES = 5 * 1024 * 1024 * 1024
_PATH_FLAG_NAMES = frozenset({
    "--audit-root", "--config", "--base", "--override-base", "-o", "--output",
    "--output-dir", "--output-allowlist-prefix", "--transformer-root",
})

KIND_LORA = "lora"
KIND_TRANSFORMER = "transformer"
CLASS_USABLE = "usable"
CLASS_CONVERTABLE = "convertable"
CLASS_UNCONVERTABLE = "unconvertable"
CLASS_DELETABLE = "deletable"
CLASS_ERROR = "error"

# Reason codes (per ADR §3 closed sets)
R_OK = "ok"
R_NORM_TARGETING = "norm_targeting"
R_DIM_MISMATCH_PARTIAL = "dim_mismatch_partial"
# ADR-014 amendment 2026-07-28: matched only after the base family's own
# diffusers LoraLoaderMixin converted the key layout (Kohya -> diffusers).
# Still CLASS_USABLE — the runtime loader performs this same conversion, so
# these files load today. Granularity lives in FileEntry.native_convert.
R_OK_NATIVE_CONVERT = "ok_native_convert"
# Minimum share of the SOURCE adapter layers a conversion must retain before
# its post-conversion match is trusted. Guards the partial-conversion case:
# converters that drop unrecognised keys with a warning would otherwise let a
# barely-understood file score 100% on the sliver that survived.
_COVERAGE_FLOOR = 0.5
R_LORA_PASSTHROUGH = "lora_passthrough"
R_LORA_QKV_SPLIT = "lora_qkv_split"
R_LOKR_TO_LORA_SVD = "lokr_to_lora_svd"
R_WRONG_ARCH = "wrong_arch"
R_POOR_MATCH_NO_PLAN = "poor_match_no_plan"
R_LOHA_UNSUPPORTED = "loha_unsupported"
R_FORMAT_UNKNOWN = "format_unknown"
R_ARCH_MISMATCH_DIFFUSERS_ONLY = "arch_mismatch_diffusers_only"
R_ZERO_BYTE = "zero_byte"
R_TRUNCATED_HEADER = "truncated_header"
R_UNPARSEABLE_HEADER = "unparseable_header"
R_UNRECOGNIZED_EXT_ZERO_CONTENT = "unrecognized_extension_zero_content"
R_SIZE_CAP_EXCEEDED = "size_cap_exceeded"

# Per-base verdict reason when a base couldn't be classified against
R_BASE_UNAVAILABLE = "base_unavailable"

# Transformer reason codes (ADR-021 §2 mapping table)
R_T_AIO_BUNDLE = "aio_bundle"
R_T_NO_MATCHING_BASE = "no_matching_base"
# prognosis-based usable reasons are minted as f"prognosis_{verdict.lower()}"
# (prognosis_hi-prec / prognosis_scaled / prognosis_plainfp8 / prognosis_cq-fp8);
# unsupported-quant unconvertable reasons as f"quant_unsupported_{verdict.lower()}".

# Transformer matching / duplicate-detection constants (ADR-021 §3/§4)
_T_MATCH_THRESHOLD = 0.90       # shape-multiset overlap for a base match
_T_DUP_THRESHOLD = 0.999        # overlap floor before duplicate sampling
_T_DUP_SAMPLE_K = 4             # unique-shape tensor pairs compared
_T_DUP_SAMPLE_CAP = 1024 * 1024  # bytes read per tensor per side (1 MiB)
# Usable prognosis verdicts are the complement of the unsupported set +
# UNREADABLE (HI-PREC / SCALED / PLAINFP8 / CQ-FP8 / AIO) — enforced by
# the branch order in _classify_transformer.
_T_UNSUPPORTED_VERDICTS = ("BNB", "SVDQ", "NVFP4")  # + CQ-<non-fp8> by prefix
# Warning codes (transformer audit)
W_DUP_CHECK_INCONCLUSIVE = "dup_check_inconclusive"  # ADR-021 §4 (<2 pairs)

# Convert-path outcome reasons (ADR §8; recorded in FileEntry.convert_reason)
R_CONVERT_COLLISION = "collision"
R_CONVERT_FAILED = "convert_failed"

# Delete-path outcome reasons (ADR §9; recorded in FileEntry.delete_reason).
# delete_reason is None on a performed deletion (entry.deleted == True) and on
# files the delete path never touched (non-deletable / preview mode).
R_DELETE_CLASSIFICATION_CHANGED = "classification_changed"  # F-5 reclassify skip
R_DELETE_CONTAINMENT_FAILED = "containment_failed"          # gate 2 fail
R_DELETE_UNLINK_FAILED = "unlink_failed"                    # stat/unlink OSError

# Warning codes
W_EXCLUDED_SYMLINK_ESCAPE = "excluded_symlink_escape"
W_DANGLING_SYMLINK = "dangling_symlink"
W_UNREADABLE = "unreadable"
W_STALE_TMP_FILE = "stale_tmp_file"  # ADR §10 F-2 Option A
W_DRY_LOAD_BASE_FAILED = "dry_load_base_failed"  # ADR §7
W_DRY_LOAD_VRAM_CASCADE = "dry_load_vram_cascade_possible"  # ADR §7 F-3
W_DRY_LOAD_UNLOAD_FAILED = "dry_load_unload_failed"  # ADR §7
W_DELETE_SKIPPED_RECLASSIFY = "delete_skipped_classification_changed"  # ADR §9 F-5
W_DELETE_CONTAINMENT_FAILED = "delete_skipped_containment_failed"  # ADR §9 gate 2
W_DELETE_FAILED = "delete_failed"  # ADR §9 unlink/stat OSError

_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "lora_audit.toml"


# ── Stderr helpers (ADR §11; machine-caller contract Vision §13) ───────
def _emit(level: str, msg: str) -> None:
    print(f"[{level}] {msg}", file=sys.stderr, flush=True)


def _emit_info(msg: str) -> None:
    _emit("INFO", msg)


def _emit_warn(msg: str) -> None:
    _emit("WARN", msg)


def _emit_error(msg: str) -> None:
    _emit("ERROR", msg)


# ── Data classes ───────────────────────────────────────────────────────
@dataclass
class BaseSpec:
    name: str
    path: Path
    param_count: int = 0
    param_dict: Optional[dict] = None  # populated lazily by _prepare_bases
    param_names: tuple = ()
    dry_load_attempted: bool = False  # flipped per-base by _dry_load_per_base (S2)


@dataclass
class FileEntry:
    relative_path: str
    classification: str
    reason: str
    kind: str = KIND_LORA
    sha256: Optional[str] = None
    size_bytes: int = 0
    verdicts_by_base: dict[str, dict[str, Any]] = field(default_factory=dict)
    convert_plan: Optional[dict[str, Any]] = None
    convert_output: Optional[str] = None
    convert_reason: Optional[str] = None  # S3: "collision" | "convert_failed" | None
    deleted: bool = False  # S4: True iff this file was unlinked this run
    delete_reason: Optional[str] = None  # S4: None | classification_changed |
    #                                      containment_failed | unlink_failed
    error: Optional[str] = None
    # ADR-021 transformer-entry fields. root_index is the IDENTITY + sort
    # discriminator (index into the manifest's transformer_roots array;
    # security F-2 — basename `root` is display-only). LoRA entries keep -1.
    root_index: int = -1
    root: Optional[str] = None
    prognosis: Optional[dict[str, Any]] = None
    matched_bases: Optional[list[str]] = None
    duplicate_of: Optional[str] = None
    # ADR-014 amendment 2026-07-28. Set only when the on-disk key layout
    # failed the direct shape-match but the base family's own diffusers
    # LoraLoaderMixin converted it into a matching one:
    #   {"mixin": str, "base": str, "verdict": {<LoRACheckResult dict>},
    #    "source_layers": int, "converted_layers": int,
    #    "matched_bases": [str]}
    # `verdict` carries the post-conversion result so OK / NORM_TARGETING /
    # DIM_MISMATCH_PARTIAL granularity survives the single reason code;
    # source/converted layer counts make the _COVERAGE_FLOOR ratio
    # auditable; `matched_bases` lists EVERY base that matched, not just
    # the winner in `base` (family-conflict signal).
    native_convert: Optional[dict[str, Any]] = None

    def to_json(self) -> dict[str, Any]:
        out = {
            "classification": self.classification,
            "convert_output": self.convert_output,
            "convert_plan": self.convert_plan,
            "convert_reason": self.convert_reason,
            "deleted": self.deleted,
            "delete_reason": self.delete_reason,
            "error": self.error,
            "kind": self.kind,
            "reason": self.reason,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verdicts_by_base": self.verdicts_by_base,
        }
        # Transformer-only keys are emitted only on transformer entries so
        # kind:"lora" entries stay byte-identical to tool 0.1.0 output
        # (ADR-021 §5 additivity; consumers branch on kind per F-3).
        if self.kind == KIND_TRANSFORMER:
            out["root_index"] = self.root_index
            out["root"] = self.root
            out["prognosis"] = self.prognosis
            out["matched_bases"] = self.matched_bases or []
            out["duplicate_of"] = self.duplicate_of
        # Same additivity discipline as the transformer keys above: emitted
        # only on the entries it applies to, so every LoRA entry that did not
        # need conversion stays byte-identical to tool 0.2.0 output.
        if self.native_convert is not None:
            out["native_convert"] = self.native_convert
        return out


@dataclass
class Warning_:
    file: Optional[str]
    code: str
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "file": self.file}


# ── Argument parsing ───────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lora_audit",
        description=(
            f"LoRA audit / classify / manifest tool (v{_TOOL_VERSION}). "
            "ADR-014 — shape-match classification, optional dry-load, "
            "optional --convert, and optional --delete (triple-gated)."
        ),
    )
    p.add_argument(
        "--audit-root", required=True, type=Path,
        help="Root directory of the LoRA tree to scan.",
    )
    p.add_argument(
        "--transformer-root", action="append", type=Path, default=[],
        dest="transformer_roots",
        help="Directory tree of single-file transformers to audit as "
             "kind:'transformer' (ADR-021). Repeatable. Must be disjoint "
             "from --audit-root and from each other (startup abort "
             "otherwise). Read-classify-report only: --convert/--delete "
             "never touch these trees.",
    )
    cfg_grp = p.add_mutually_exclusive_group()
    cfg_grp.add_argument(
        "--config", type=Path, default=None,
        help=f"TOML config path (default: {_DEFAULT_CONFIG_PATH}).",
    )
    cfg_grp.add_argument(
        "--no-config", action="store_true",
        help="Skip config-file ingestion entirely.",
    )
    p.add_argument(
        "--base", action="append", default=[], metavar="name=PATH",
        help="Add a base model. Repeatable. Name must match [a-zA-Z0-9_-]+.",
    )
    p.add_argument(
        "--override-base", action="append", default=[], metavar="name=PATH",
        help="Override a base from config with this path. Repeatable.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Manifest output path (default: <audit_root>/lora_audit.json).",
    )
    p.add_argument(
        "--print-manifest", action="store_true",
        help="Print manifest to stdout instead of writing to file.",
    )
    out_grp = p.add_mutually_exclusive_group()
    out_grp.add_argument(
        "--allow-output-outside-root", action="store_true",
        help="Interactive-only: allow --output-dir outside audit-root "
             "(blacklist applied). Mutually exclusive with "
             "--require-output-allowlist.",
    )
    out_grp.add_argument(
        "--require-output-allowlist", action="store_true",
        help="Machine-caller: require --output-allowlist-prefix for any "
             "output outside audit-root.",
    )
    p.add_argument(
        "--output-allowlist-prefix", action="append", default=[],
        type=Path, metavar="PATH",
        help="Permitted output prefix (used with --require-output-allowlist). "
             "Repeatable.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Convert output directory (for --convert). Converted siblings "
             "are written under <output-dir>/<relative_source_dir>/ instead "
             "of next to the source. Validated for containment (ADR §6).",
    )
    p.add_argument(
        "--dry-load", action="store_true",
        help="Sequentially load each base pipeline (diffusers, "
             "local_files_only) and attempt to load every non-WRONG_ARCH "
             "LoRA against it. Records loaded/applied_modules per base in "
             "the manifest. Default off; [defaults] dry_load = true in "
             "config flips the default.",
    )
    p.add_argument(
        "--convert", action="store_true",
        help="For each 'convertable' file, write a converted sibling "
             "<source_stem>.<target_family>.safetensors (atomic; pre-write "
             "collision skip; source never modified). ADR §8. Default off.",
    )
    p.add_argument(
        "--delete", action="store_true",
        help="Delete files classified 'deletable' (zero-byte / truncated / "
             "unparseable header / unrecognized-garbage). Triple-gated: "
             "deletable classification + audit-root containment + --yes. "
             "Without --yes, prints a preview and exits 0 with NO I/O. "
             "Re-checks the deletable signature at unlink time (ADR §9). "
             "Default off.",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Authorize destructive operations (--delete) non-interactively. "
             "The ONLY confirmation signal; its absence is preview-mode, never "
             "a prompt (machine-caller contract, Vision #13). No effect "
             "without --delete.",
    )
    return p


# ── Config + base ingestion ────────────────────────────────────────────
def _load_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        if not _DEFAULT_CONFIG_PATH.exists():
            return {}
        path = _DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise SystemExit(
            f"[ERROR] config file not found: {path}"
        )
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"[ERROR] config TOML parse failed ({path}): {e}")


def _parse_base_flag(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(
            f"[ERROR] --base / --override-base requires name=PATH form, got {spec!r}"
        )
    name, raw_path = spec.split("=", 1)
    if not _BASE_NAME_RE.fullmatch(name):
        raise SystemExit(
            f"[ERROR] base name {name!r} invalid; must match [a-zA-Z0-9_-]+"
        )
    return name, Path(raw_path)


def _resolve_bases(
    config: dict[str, Any],
    base_flags: list[str],
    override_flags: list[str],
) -> list[BaseSpec]:
    bases: dict[str, Path] = {}
    for name, raw in (config.get("bases") or {}).items():
        if not _BASE_NAME_RE.fullmatch(name):
            raise SystemExit(
                f"[ERROR] config base name {name!r} invalid; must match [a-zA-Z0-9_-]+"
            )
        bases[name] = Path(raw)
    for spec in base_flags:
        name, path = _parse_base_flag(spec)
        if name in bases:
            raise SystemExit(
                f"[ERROR] base {name!r} already declared by config; "
                f"use --override-base to replace"
            )
        bases[name] = path
    for spec in override_flags:
        name, path = _parse_base_flag(spec)
        bases[name] = path
    if not bases:
        raise SystemExit(
            "[ERROR] no bases specified (config has no [bases] and no --base passed)"
        )
    resolved: list[BaseSpec] = []
    for name, path in sorted(bases.items()):
        try:
            real = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as e:
            raise SystemExit(f"[ERROR] base {name!r} path unresolvable: {e}")
        if not real.is_dir():
            raise SystemExit(f"[ERROR] base {name!r} path is not a directory: {real}")
        shards = list(real.glob("*.safetensors"))
        if not shards:
            raise SystemExit(
                f"[ERROR] base {name!r} dir has no .safetensors shards: {real}"
            )
        resolved.append(BaseSpec(name=name, path=real))
    return resolved


# ── Output-dir policy (ADR §6 F-17 default-deny + mutual exclusion) ────
def _validate_output_dir(
    output_dir: Optional[Path],
    audit_root: Path,
    allow_outside: bool,
    require_allowlist: bool,
    allowlist_prefixes: list[Path],
) -> Optional[Path]:
    if output_dir is None:
        return None
    real = output_dir.resolve()
    try:
        real.relative_to(audit_root)
        return real
    except ValueError:
        pass
    if not allow_outside and not require_allowlist:
        raise SystemExit(
            f"[ERROR] --output-dir is outside --audit-root and neither "
            f"--allow-output-outside-root (interactive) nor "
            f"--require-output-allowlist (machine) was passed"
        )
    if allow_outside:
        # `Path("/")` deliberately excluded — every absolute path
        # `is_relative_to("/")` is True, which would make this branch a
        # deny-all and render --allow-output-outside-root unusable. The
        # blacklist is documented as gappy (ADR §6) and is not a security
        # control machine callers may rely on; --require-output-allowlist
        # is the authoritative gate for them.
        _BLACKLIST = (
            Path("/etc"), Path("/usr"), Path("/var"),
            Path("/sys"), Path("/proc"), Path("/dev"),
            Path.home() / ".ssh", Path.home() / ".gnupg",
        )
        if real in _BLACKLIST or any(
            real == bl or real.is_relative_to(bl) for bl in _BLACKLIST
        ):
            raise SystemExit(
                f"[ERROR] --output-dir resolves to system-blacklisted path: {real}"
            )
        if ".." in str(output_dir).split(os.sep):
            raise SystemExit(f"[ERROR] --output-dir contains .. components: {output_dir}")
        return real
    if not allowlist_prefixes:
        raise SystemExit(
            "[ERROR] --require-output-allowlist requires at least one "
            "--output-allowlist-prefix"
        )
    if ".." in str(output_dir).split(os.sep):
        raise SystemExit(
            f"[ERROR] --output-dir contains .. components: {output_dir}"
        )
    resolved_prefixes = []
    for pfx in allowlist_prefixes:
        if not pfx.is_absolute():
            raise SystemExit(
                f"[ERROR] --output-allowlist-prefix must be absolute: {pfx}"
            )
        try:
            resolved = pfx.resolve(strict=True)
        except (FileNotFoundError, OSError) as e:
            raise SystemExit(
                f"[ERROR] --output-allowlist-prefix must exist and be resolvable: {pfx} ({e})"
            )
        if not resolved.is_dir():
            raise SystemExit(
                f"[ERROR] --output-allowlist-prefix must be a directory: {resolved}"
            )
        resolved_prefixes.append(resolved)
    if not any(
        real == p or real.is_relative_to(p) for p in resolved_prefixes
    ):
        raise SystemExit(
            f"[ERROR] --output-dir not within any --output-allowlist-prefix: {real}"
        )
    return real


# ── Manifest builder + atomic writer (ADR §5, §10) ─────────────────────
# Short-option path flags accepting argparse's concatenated form (-oVALUE).
# Kept in sync with the path flags declared in _build_parser; any new
# short-form flag that takes a path argument MUST be added here.
_SHORT_PATH_FLAGS = ("-o",)


def _redact_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        eq_form = "=" in token and token.split("=", 1)[0] in _PATH_FLAG_NAMES
        short_concat = next(
            (s for s in _SHORT_PATH_FLAGS
             if token.startswith(s) and len(token) > len(s) and token != s),
            None,
        )
        if token in _PATH_FLAG_NAMES:
            out.append(token)
            if i + 1 < len(argv):
                out.append("<redacted>")
                i += 2
                continue
        elif eq_form:
            flag = token.split("=", 1)[0]
            out.append(f"{flag}=<redacted>")
        elif short_concat is not None:
            out.append(f"{short_concat}<redacted>")
        else:
            out.append(token)
        i += 1
    return out


def _config_sha256(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(
    audit_root: Path,
    bases: list[BaseSpec],
    files: list[FileEntry],
    warnings: list[Warning_],
    output_dir: Optional[Path],
    argv: list[str],
    config_path: Optional[Path],
    transformer_roots: Optional[list[Path]] = None,
) -> dict[str, Any]:
    totals = {
        "files_scanned": len(files),
        CLASS_USABLE: 0,
        CLASS_CONVERTABLE: 0,
        CLASS_UNCONVERTABLE: 0,
        CLASS_DELETABLE: 0,
        CLASS_ERROR: 0,
    }
    for entry in files:
        if entry.classification in totals:
            totals[entry.classification] += 1
    # ADR-021 §5: sort key (root_index_or_-1, relative_path). LoRA entries
    # carry -1, so their ordering is byte-identical to tool 0.1.0 output;
    # the index (never the collision-prone basename) discriminates
    # transformer roots (security F-2).
    files_sorted = sorted(
        files,
        key=lambda e: (e.root_index if e.kind == KIND_TRANSFORMER else -1,
                       e.relative_path),
    )
    warnings_sorted = sorted(
        warnings, key=lambda w: ((w.file or ""), w.code)
    )
    manifest: dict[str, Any] = {
        "audit_root": str(audit_root),
        "audit_version": _AUDIT_VERSION,
        "audited_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bases": {
            b.name: {
                "dry_load_attempted": b.dry_load_attempted,
                "param_count": b.param_count,
                "path": str(b.path),
            }
            for b in bases
        },
        "files": [e.to_json() for e in files_sorted],
        "tool_invocation": {
            "argv_redacted": _redact_argv(argv),
            "config_path": str(config_path) if config_path else None,
            "config_sha256": _config_sha256(config_path),
        },
        "tool_version": _TOOL_VERSION,
        "totals": totals,
        "warnings": [w.to_json() for w in warnings_sorted],
    }
    if output_dir is not None:
        manifest["output_dir"] = str(output_dir)
    if transformer_roots:
        # Resolved paths, CLI order — root_index on entries indexes into
        # this array (ADR-021 §5).
        manifest["transformer_roots"] = [str(t) for t in transformer_roots]
    return manifest


def _write_manifest_atomic(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, out_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


# ── Path policy (ADR §6 — realpath authoritative, fd-open narrowing) ──
def _rel(path: Path, audit_root: Path) -> str:
    try:
        return str(path.relative_to(audit_root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


def _passes_scan_containment(
    path: Path, audit_root: Path, warnings: list[Warning_]
) -> bool:
    try:
        real = Path(os.path.realpath(path))
    except OSError as e:
        warnings.append(Warning_(_rel(path, audit_root), W_UNREADABLE, str(e)))
        return False
    try:
        real.relative_to(audit_root)
    except ValueError:
        warnings.append(Warning_(_rel(path, audit_root), W_EXCLUDED_SYMLINK_ESCAPE,
                                 f"realpath {real} not under audit_root"))
        return False
    if not real.exists():
        warnings.append(Warning_(_rel(path, audit_root), W_DANGLING_SYMLINK, ""))
        return False
    return True


def _open_no_follow(
    path: Path, audit_root: Path, warnings: list[Warning_]
) -> bool:
    """Defense-in-depth narrowing per ADR §6. Realpath check from
    _passes_scan_containment is the authoritative control; this only adds
    O_NOFOLLOW + Linux /proc/self/fd narrowing for paths that aren't
    deliberately symlinks. Intentional symlinks (`path.is_symlink()`)
    bypass O_NOFOLLOW (it would reject them with ELOOP) and rely on the
    realpath check, which already passed."""
    if path.is_symlink():
        return True
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as e:
        if e.errno == errno.ELOOP:
            warnings.append(Warning_(_rel(path, audit_root),
                                     W_EXCLUDED_SYMLINK_ESCAPE,
                                     "O_NOFOLLOW rejected terminal symlink"))
        else:
            warnings.append(Warning_(_rel(path, audit_root), W_UNREADABLE,
                                     str(e)))
        return False
    try:
        if sys.platform == "linux":
            real_fd = Path(os.path.realpath(f"/proc/self/fd/{fd}"))
            try:
                real_fd.relative_to(audit_root)
            except ValueError:
                warnings.append(Warning_(_rel(path, audit_root),
                                         W_EXCLUDED_SYMLINK_ESCAPE,
                                         f"fd realpath {real_fd} not under audit_root"))
                return False
        return True
    finally:
        os.close(fd)


# ── Deletable-signature probe (ADR §3) ─────────────────────────────────
def _probe_safetensors_garbage(path: Path, size: int) -> Optional[str]:
    if size == 0:
        return R_ZERO_BYTE
    if size < 8:
        return R_TRUNCATED_HEADER
    try:
        with open(path, "rb") as f:
            n_bytes = f.read(8)
            if len(n_bytes) < 8:
                return R_TRUNCATED_HEADER
            n = struct.unpack("<Q", n_bytes)[0]
            # 100 MB header cap (mirrors audit_single_files._header /
            # safetensors' own bound). Without it, a crafted file whose
            # declared header length is tens of GB (with matching on-disk
            # padding) makes the f.read(n) below load it all into RAM —
            # unbounded-read DoS. The LoRA path was implicitly bounded by
            # the 5 GB size cap; the ADR-021 transformer path EXEMPTS that
            # cap, so this read needs its own bound (security-audit
            # 2026-07-06 F-T1). A >100 MB header is garbage regardless.
            if n > 100_000_000:
                return R_UNPARSEABLE_HEADER
            if size < 8 + n:
                return R_TRUNCATED_HEADER
            header_bytes = f.read(n)
            if len(header_bytes) < n:
                return R_TRUNCATED_HEADER
            try:
                json.loads(header_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return R_UNPARSEABLE_HEADER
    except OSError:
        return R_UNPARSEABLE_HEADER
    return None


def _probe_pickle_magic(path: Path, size: int) -> Optional[str]:
    # Magic-byte probe only — does NOT load. Torch checkpoints are either
    # zip archives (PK\x03\x04) or pickle streams (starts with PROTO opcode
    # 0x80). A file matching neither is confidently garbage. Real load
    # failures happen later in the classifier via _load_state_dict (whose
    # torch.load uses weights_only=True per the reused nodes module);
    # those surface as classification=error via per-file fault isolation.
    if size < 4:
        return R_UNRECOGNIZED_EXT_ZERO_CONTENT
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return R_UNRECOGNIZED_EXT_ZERO_CONTENT
    if head[:2] == b"PK" or head[:1] == b"\x80":
        return None
    return R_UNRECOGNIZED_EXT_ZERO_CONTENT


def _classify_deletable(path: Path, size: int) -> Optional[str]:
    if size == 0:
        return R_ZERO_BYTE
    ext = path.suffix.lower()
    if ext == ".safetensors":
        return _probe_safetensors_garbage(path, size)
    if ext in (".pt", ".bin", ".pth"):
        return _probe_pickle_magic(path, size)
    return None


# ── Classifier reuse (ADR §3 precedence) ──────────────────────────────
def _verdict_to_dict(r) -> dict[str, Any]:
    return {
        "dim_ok_pct": round(r.dim_ok_pct, 2),
        "key_match_pct": round(r.key_match_pct, 2),
        "matched": r.matched,
        "total_layers": r.total_layers,
        "verdict": r.verdict,
    }


def _is_usable_verdict(r) -> tuple[bool, str]:
    """Map a LoRACheckResult into (usable?, reason). OQ-1: ADR's
    dim_mismatch_partial definition is self-contradictory (DIM_MISMATCH
    requires dim_ok<90, but ADR also requires dim_ok>=90); interpreted
    pragmatically here, flagged for ADR Changelog clarification."""
    v = r.verdict
    if v == "OK":
        return True, R_OK
    if v == "NORM_TARGETING":
        return True, R_NORM_TARGETING
    if v == "DIM_MISMATCH" and r.dim_ok_pct >= 50.0:
        return True, R_DIM_MISMATCH_PARTIAL
    return False, ""


def _native_hit_rank(record: dict[str, Any]) -> tuple:
    """Rank a native-convert hit so the BEST-matching base wins, not the
    alphabetically-first one.

    Coverage outranks the verdict LABEL, deliberately: the motivating case
    had `flux` at 100% but NORM_TARGETING (a real Flux LoRA that also
    touches norm layers) versus `chroma` at 84.62% with a clean OK.
    Ranking the label first re-selects `chroma` — the file is a Flux LoRA,
    and coverage is the signal that says so.

    `matched` (absolute layers that resolved) is primary, not
    `key_match_pct`. Percentage is computed over the POST-conversion dict,
    whose size each mixin decides, so comparing percentages ACROSS mixins
    divides by different denominators. A converter that drops what it does
    not recognise can retain half the source layers (passing
    `_COVERAGE_FLOOR`), match all of that sliver, and score 100% — beating
    the correct family's fuller conversion at 90%. Absolute matched-layer
    count is denominator-free and does not invert that way.

    The tradeoff, named rather than hidden: `matched` carries the opposite
    bias, since a converter that splits fused projections (qkv into
    to_q/to_k/to_v) inflates the count. That bias is the weaker one —
    split layers still have to resolve against the base index to be
    counted, so inflation only helps a base the file genuinely fits.
    Within a single mixin the two metrics are monotonically equivalent
    (shared denominator), so this changes nothing for the flux-vs-chroma
    case that motivated the fix.
    """
    v = record.get("verdict") or {}
    return (
        v.get("matched", 0),
        v.get("key_match_pct", 0.0),
        v.get("dim_ok_pct", 0.0),
        1 if v.get("verdict") == "OK" else 0,
    )


def _adapter_layer_count(keys) -> int:
    """Count distinct adapter target layers in a LoRA key namespace.

    Same grouping `check_lora` uses (a layer is one `_strip_adapter_suffix`
    base with a recognised adapter suffix), so the pre/post-conversion
    counts in `_try_native_convert_match` are directly comparable.
    """
    layers = set()
    for k in keys:
        base_key, sfx = _strip_adapter_suffix(k)
        if sfx is not None:
            layers.add(base_key)
    return len(layers)


def _resolve_lora_mixin(base: BaseSpec):
    """Resolve the diffusers `*LoraLoaderMixin` the RUNTIME loader would use
    for this base, or None (ADR-014 amendment 2026-07-28).

    Data-driven, so a new family gains conversion coverage with no edit here:
    `<base>/../model_index.json` -> `_class_name` -> `getattr(diffusers, cls)`
    -> first `*LoraLoaderMixin` in the MRO. This is the same detection the
    generic loader uses (CLAUDE.md "Auto-detection"), which is what makes the
    audit's answer the loader's answer rather than a parallel guess.

    Best-effort enrichment only. Every failure mode — no `model_index.json`,
    unreadable/!JSON, absent `_class_name`, class not in this diffusers
    version, no mixin in the MRO — returns None, and the caller then reports
    exactly what the direct shape-match already concluded. A base whose family
    diffusers cannot convert must never become an audit ERROR.
    """
    index_path = base.path.parent / "model_index.json"
    try:
        with open(index_path, "rb") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return None
    # A model_index.json that parses to a list/str/number is malformed for
    # our purposes; `.get` on it would raise AttributeError and escape.
    if not isinstance(loaded, dict):
        return None
    class_name = loaded.get("_class_name")
    if not isinstance(class_name, str) or not class_name:
        return None
    try:
        import diffusers
    except Exception:
        return None
    try:
        # diffusers' _LazyModule raises (not AttributeError) when a named
        # export's backend is unavailable, so the getattr default is not
        # sufficient on its own.
        pipeline_cls = getattr(diffusers, class_name, None)
    except Exception:
        return None
    if pipeline_cls is None or not isinstance(pipeline_cls, type):
        return None
    for ancestor in pipeline_cls.__mro__:
        if ancestor.__name__.endswith("LoraLoaderMixin") and hasattr(
            ancestor, "lora_state_dict"
        ):
            return ancestor
    return None


def _try_native_convert_match(
    path: Path, state_dict: dict, base: BaseSpec
) -> Optional[tuple[str, dict[str, Any]]]:
    """Re-run the shape-match against `state_dict` as the base family's own
    diffusers converter would rewrite it (ADR-014 amendment 2026-07-28).

    Returns (reason, native_convert_record) on a usable post-conversion
    verdict, else None. The conversion is attempted and then the SAME
    `check_lora` matcher adjudicates the result — a foreign layout (Wan
    against a Flux base) passes through the converter untouched and still
    fails the match.

    Two guards keep a PARTIAL conversion from being promoted. Some converters
    (`_convert_kohya_flux2_lora_to_diffusers`) drop unrecognised source keys
    with only a warning and return what they understood. Since `check_lora`
    computes `key_match_pct` over the POST-conversion dict, those dropped
    keys are invisible to it: a file the converter barely understood could
    otherwise score 100% on the sliver that survived.
      1. `_COVERAGE_FLOOR` — the converted dict must retain at least half the
         source's adapter layers.
      2. The post-conversion verdict must be OK or NORM_TARGETING. The
         DIM_MISMATCH-at-50% acceptance `_is_usable_verdict` allows on the
         direct path is deliberately NOT honoured here, because a partial
         conversion plus a partial dim match compounds two weak signals.
    Neither guard is a proof — a converter that splits fused projections
    (qkv -> to_q/to_k/to_v) inflates the converted layer count, so the ratio
    is a heuristic floor, not a conservation law. See the ADR-014 amendment
    and the TECH_DEBT entry for the named residual.
    """
    try:
        mixin = _resolve_lora_mixin(base)
    except Exception:
        # Belt and braces: a base misconfiguration must never become a
        # per-FILE audit error across the whole corpus (Vision invariant 9).
        return None
    if mixin is None:
        return None
    try:
        # Defensive copy: diffusers' `lora_state_dict` mutates a dict
        # argument in place — every `load_lora_weights` implementation
        # copies before calling it for exactly this reason. The Qwen
        # converter pops the caller's keys and can drain the dict to empty,
        # which would corrupt both the remaining bases in this loop and the
        # `find_matching_plan` probe that consumes `state_dict` afterwards.
        converted = mixin.lora_state_dict(dict(state_dict))
    except Exception:
        # Converters raise on layouts they do not recognise. That is a
        # "no" for this base, not a tool failure.
        return None
    if isinstance(converted, tuple):  # SDXL-style (state_dict, network_alphas)
        converted = converted[0]
    if not converted:
        return None

    source_layers = _adapter_layer_count(state_dict)
    converted_layers = _adapter_layer_count(converted)
    if source_layers and converted_layers < _COVERAGE_FLOOR * source_layers:
        return None

    try:
        r = check_lora(path, param_dict=base.param_dict, log_prefix="",
                       state_dict=converted)
    except Exception:
        return None
    if r.verdict not in ("OK", "NORM_TARGETING"):
        return None
    return R_OK_NATIVE_CONVERT, {
        "mixin": mixin.__name__,
        "base": base.name,
        "verdict": _verdict_to_dict(r),
        # Recorded so the residual above is auditable from the manifest
        # rather than hidden behind a boolean.
        "source_layers": source_layers,
        "converted_layers": converted_layers,
    }


def _classify_lora(
    path: Path, bases: list[BaseSpec]
) -> tuple[str, str, dict[str, dict[str, Any]], Optional[dict[str, Any]],
           Optional[dict[str, Any]]]:
    """Run shape-match against every base; if no usable, attempt the native
    convert probe, then convertable.
    Returns (classification, reason, verdicts_by_base, convert_plan_or_None,
    native_convert_or_None).
    Raises if every available base errored on this file — caller (_classify_one)
    converts that into classification=error per Vision invariant 9."""
    verdicts: dict[str, dict[str, Any]] = {}
    best_usable_reason: Optional[str] = None
    classified_bases = 0
    errored_bases = 0
    last_error: Optional[str] = None

    for base in bases:
        if base.param_dict is None:
            verdicts[base.name] = {"verdict": "BASE_UNAVAILABLE",
                                   "reason": R_BASE_UNAVAILABLE}
            continue
        try:
            r = check_lora(path, param_dict=base.param_dict, log_prefix="")
        except Exception as e:
            verdicts[base.name] = {"verdict": "ERROR", "error": str(e)[:200]}
            errored_bases += 1
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        verdicts[base.name] = _verdict_to_dict(r)
        classified_bases += 1
        if best_usable_reason is None:
            ok, reason = _is_usable_verdict(r)
            if ok:
                best_usable_reason = reason

    if classified_bases == 0 and errored_bases > 0:
        # Every available base errored — this is a per-file classifier
        # failure, not an "unconvertable" file. Vision invariant 9 says
        # mark classification=error and let the loop continue.
        raise RuntimeError(
            f"all bases errored during classification; last: {last_error}"
        )

    if best_usable_reason is not None:
        # Fast path: the on-disk layout already matched. The state dict is
        # never loaded and no converter runs — the ~241 already-usable LoRAs
        # pay nothing for the amendment below.
        return CLASS_USABLE, best_usable_reason, verdicts, None, None

    # No usable verdict — probe convertable. Requires loading state dict.
    try:
        header = _read_safetensors_header(path) if path.suffix.lower() == ".safetensors" else None
    except Exception:
        header = None

    if header is not None:
        fmt = detect_lora_format(header.keys())
        if fmt == "loha":
            return CLASS_UNCONVERTABLE, R_LOHA_UNSUPPORTED, verdicts, None, None

    # Try find_matching_plan against each base.
    try:
        state_dict = _load_state_dict(str(path))
    except Exception as e:
        # Can't load — surface as unconvertable with arch hint if any.
        reason = _pick_unconvertable_reason(verdicts, header_present=header is not None)
        return CLASS_UNCONVERTABLE, reason, verdicts, None, None

    # ADR-014 amendment 2026-07-28: before declaring this unconvertable,
    # ask whether the RUNTIME loader would have made it work. Kohya-format
    # files (the dominant civitai Flux layout) fail the direct shape-match
    # but `load_lora_weights` converts them natively, so comfyless loads
    # them today. Runs on the state dict already loaded above — no extra
    # read.
    #
    # EVERY base is probed, not just up to the first hit. Architecturally
    # related families match the same file (Chroma is a Flux.1 derivative,
    # so a Flux LoRA converts and matches both), and `_resolve_bases` hands
    # us bases ALPHABETICALLY — so returning on first hit reported whichever
    # family sorted earliest, not the one that actually fit. Measured: a
    # Kohya Flux LoRA matched `flux` at 100% and `chroma` at 84.62%, and
    # first-hit reported `chroma`. The catalog's `model_family` tag comes
    # from this, so that was a live mislabel on 58 of 138 files.
    native_hits = []
    for base in bases:
        if base.param_dict is None:
            continue
        hit = _try_native_convert_match(path, state_dict, base)
        if hit is not None:
            native_hits.append(hit[1])
    if native_hits:
        # Best match wins. `max` over an alphabetically-ordered list keeps
        # ties deterministic (Vision invariant 8).
        best = dict(max(native_hits, key=_native_hit_rank))
        # Preserve the multi-family signal the direct path keeps in
        # `verdicts_by_base`; reporting only the winner would discard it.
        #
        # CARRIED FOR, NOT YET READ BY, the catalog. `catalog_builder.py`
        # derives a lora's `ok_bases` solely from `verdicts_by_base`
        # (OK/NORM_TARGETING) and never looks at `native_convert` — and a
        # native-convert entry's direct verdicts are all WRONG_ARCH by
        # construction. So until that consumer is wired, these families
        # reach the manifest but NOT catalog search / MCP `list_loras` /
        # refine offers. See TECH_DEBT 2026-07-28 "native_convert families
        # never reach the catalog" — it blocks ADR-041.
        best["matched_bases"] = sorted(h["base"] for h in native_hits)
        return CLASS_USABLE, R_OK_NATIVE_CONVERT, verdicts, None, best

    for base in bases:
        if base.param_dict is None:
            continue
        try:
            plan = find_matching_plan(state_dict, base.param_names)
        except Exception:
            plan = None
        if plan is not None:
            return (
                CLASS_CONVERTABLE,
                R_LOKR_TO_LORA_SVD if plan.source_family.startswith(("bfl_", "lokr"))
                else R_LORA_PASSTHROUGH,
                verdicts,
                {
                    "source_family": plan.source_family,
                    "target_family": plan.target_family,
                    "target_base": base.name,
                },
                None,
            )

    return (
        CLASS_UNCONVERTABLE,
        _pick_unconvertable_reason(verdicts, header_present=header is not None),
        verdicts,
        None,
        None,
    )


def _pick_unconvertable_reason(
    verdicts: dict[str, dict[str, Any]], header_present: bool
) -> str:
    if not header_present:
        return R_FORMAT_UNKNOWN
    for v in verdicts.values():
        if v.get("verdict") == "WRONG_ARCH":
            return R_WRONG_ARCH
    for v in verdicts.values():
        if v.get("verdict") == "POOR_MATCH":
            return R_POOR_MATCH_NO_PLAN
    return R_FORMAT_UNKNOWN


# ── sha256 of file ──────────────────────────────────────────────────────
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Prepare base param_dicts (single call per base) ────────────────────
def _prepare_bases(bases: list[BaseSpec], warnings: list[Warning_]) -> None:
    for base in bases:
        try:
            pd = build_param_dict_from_dir(str(base.path))
        except Exception as e:
            warnings.append(Warning_(None, R_BASE_UNAVAILABLE,
                                     f"base {base.name!r}: {e}"))
            continue
        names = tuple(k for k in pd if not k.startswith("_"))
        base.param_dict = pd
        base.param_names = names
        base.param_count = len(names)


# ── Dry-load loop (ADR §7) ─────────────────────────────────────────────
# The `applied=` token format matches the loader's three direct-merge log
# lines (lora_direct, lokr_direct, loha_direct in eric_qwen_edit_lora.py
# at lines 1034, 438, 605 — search 'direct merge .weight='). PEFT and
# pipeline fast-paths don't emit a count; `applied_modules` is None in
# those cases. If the loader's print format changes, this regex is the
# single failure point — re-validate against the loader on each upgrade.
_APPLIED_RE = re.compile(r"applied=(\d+)")
_ADAPTER_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_]")


def _parse_applied_modules(captured: str) -> Optional[int]:
    matches = _APPLIED_RE.findall(captured)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _safe_adapter_name(rel_path: str) -> str:
    stem = Path(rel_path).stem
    safe = _ADAPTER_NAME_SAFE_RE.sub("_", stem)
    return safe or "lora"


def _load_dry_load_pipeline(base_dir: Path):
    """Lazy-import diffusers + torch and load the full base pipeline.

    Imports are deferred so non-dry-load runs do not pay the diffusers
    import cost and so tests can monkey-patch this function. Per ADR §7
    + project CLAUDE.md `Important Constraints`, `local_files_only=True`
    is non-negotiable — the audit tool must never phone home.

    DO NOT add a `**kwargs` passthrough or accept a caller-supplied
    `local_files_only` argument — a future regression that lets a caller
    shadow this kwarg silently re-opens the network surface.
    """
    import torch  # noqa: F401  — used via torch.bfloat16
    from diffusers import AutoPipelineForText2Image
    return AutoPipelineForText2Image.from_pretrained(
        str(base_dir),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )


def _empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _dry_load_per_base(
    audit_root: Path,
    base: BaseSpec,
    files: list[FileEntry],
    warnings: list[Warning_],
) -> bool:
    """Sequentially dry-load every applicable LoRA against *base*.

    Returns True iff the base pipeline loaded. On success, mutates
    each candidate's `verdicts_by_base[base.name]` to include a
    `dry_load` sub-object `{loaded, applied_modules, reason}` and sets
    `base.dry_load_attempted = True`. On base-load failure, emits a
    WARN and returns False without mutating any file's verdict.

    Per-LoRA fault isolation (Vision invariant 9): any exception from
    `load_lora_with_key_fix` for an individual file is caught and
    recorded as `loaded=false, reason=<summary>` for that file only;
    the per-base loop continues.
    """
    base_pipeline_root = base.path.parent
    try:
        pipe = _load_dry_load_pipeline(base_pipeline_root)
    except Exception as e:
        detail = f"{type(e).__name__}: {str(e)[:200]}"
        warnings.append(Warning_(None, W_DRY_LOAD_BASE_FAILED,
                                 f"base {base.name!r}: {detail}"))
        _emit_warn(
            f"dry-load skipped for base {base.name!r}: {detail}. "
            f"Falling back to shape-match."
        )
        return False

    candidates: list[FileEntry] = []
    for entry in files:
        if entry.kind != KIND_LORA:
            continue
        if entry.classification in (CLASS_DELETABLE, CLASS_ERROR):
            continue
        v = entry.verdicts_by_base.get(base.name) or {}
        verdict = v.get("verdict")
        if verdict in ("WRONG_ARCH", "BASE_UNAVAILABLE", "ERROR", None):
            continue
        candidates.append(entry)

    _emit_info(
        f"base {base.name!r}: pipeline loaded; dry-loading "
        f"{len(candidates)} candidates"
    )

    used_names: set[str] = set()
    loaded_adapter_names: list[str] = []
    for entry in candidates:
        base_name_safe = _safe_adapter_name(entry.relative_path)
        adapter_name = base_name_safe
        suffix = 1
        while adapter_name in used_names:
            suffix += 1
            adapter_name = f"{base_name_safe}_{suffix}"
        used_names.add(adapter_name)

        abs_path_p = audit_root / entry.relative_path
        # Re-run scan-time containment per security-auditor M-1: the
        # symlink target may have changed between scan and dry-load.
        # `_passes_scan_containment` is realpath + relative_to(audit_root);
        # the inner TOCTOU between this check and the loader's open is
        # the same same-uid residual ADR §6 already accepts.
        recheck_warnings: list[Warning_] = []
        if not _passes_scan_containment(abs_path_p, audit_root, recheck_warnings):
            entry.verdicts_by_base[base.name]["dry_load"] = {
                "applied_modules": None,
                "loaded": False,
                "reason": "containment_changed",
            }
            continue

        captured = io.StringIO()
        loaded = False
        applied_raw: Optional[int] = None
        reason: Optional[str] = None
        try:
            with contextlib.redirect_stdout(captured):
                loaded = bool(load_lora_with_key_fix(
                    pipe, str(abs_path_p), adapter_name=adapter_name,
                ))
            applied_raw = _parse_applied_modules(captured.getvalue())
        except Exception as e:
            # Broad except (not BaseException) preserves KeyboardInterrupt
            # and SystemExit pass-through while isolating per-file faults
            # per Vision invariant 9.
            loaded = False
            reason = f"{type(e).__name__}: {str(e)[:200]}"

        # `applied_modules` is the merged-module count *on success*. The
        # direct-merge loader paths print `applied=0, skipped=N` when no
        # module matched AND return False; folding code-reviewer H-1, we
        # null out the count whenever `loaded` is False so the downstream
        # catalog reads `applied_modules` as "what got merged" only.
        applied = applied_raw if loaded else None

        entry.verdicts_by_base[base.name]["dry_load"] = {
            "applied_modules": applied,
            "loaded": loaded,
            "reason": reason,
        }
        if loaded:
            loaded_adapter_names.append(adapter_name)

    # Unload is logically redundant (the immediately-following `del pipe`
    # discards the transformer anyway) but kept for loader-faithfulness
    # per Vision invariant 7 — the unload path is what production code
    # runs after every LoRA stack swap.
    if loaded_adapter_names:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                unload_adapters(pipe, loaded_adapter_names)
        except Exception as e:
            warnings.append(Warning_(
                None, W_DRY_LOAD_UNLOAD_FAILED,
                f"base {base.name!r}: {type(e).__name__}: {str(e)[:200]}",
            ))

    del pipe
    _empty_cuda_cache()
    base.dry_load_attempted = True
    return True


def _run_dry_load(
    audit_root: Path,
    bases: list[BaseSpec],
    files: list[FileEntry],
    warnings: list[Warning_],
) -> None:
    """Iterate bases and run dry-load per-base. Emits VRAM-cascade warning
    (ADR §7 F-3) on any base failure that follows an earlier base failure.
    """
    runnable = [b for b in bases if b.param_dict is not None]
    total = len(runnable)
    any_base_failed = False
    for idx, base in enumerate(runnable, start=1):
        _emit_info(f"base {base.name!r} ({idx}/{total}): loading pipeline...")
        ok = _dry_load_per_base(audit_root, base, files, warnings)
        if not ok:
            if any_base_failed:
                warnings.append(Warning_(
                    None, W_DRY_LOAD_VRAM_CASCADE,
                    f"prior base load failed; base {base.name!r} failure "
                    f"may be a downstream effect, not the base's own problem",
                ))
                _emit_warn(
                    "vram_cascade_possible: prior base load failed; "
                    "subsequent base failures may be downstream effects, "
                    "not the base's own problem"
                )
            any_base_failed = True


# ── Convert path (ADR §8 + §10; S3) ────────────────────────────────────
def _convert_one(
    entry: FileEntry,
    audit_root: Path,
    output_dir: Optional[Path],
    base_by_name: dict[str, BaseSpec],
) -> tuple[Optional[str], Optional[str]]:
    """Convert one `convertable` FileEntry to a sibling safetensors file.

    Returns `(convert_output_relative, convert_reason)`:
      - `(rel_path, None)`            — written successfully.
      - `(None, "collision")`        — target already exists; skipped, source
                                        untouched (ADR §8 pre-write check).
      - `(None, "convert_failed")`   — load / plan / convert / write error;
                                        skipped, source untouched.

    Containment chain (security): the output path is
    `<base_dir>/<relative_source_dir>/<stem>.<target_family>.safetensors`,
    where `base_dir` is either `audit_root` (already realpath-contained at
    scan time) or the `--output-dir` (already validated by
    `_validate_output_dir`, ADR §6). `entry.relative_path` is an rglob result
    that passed `_passes_scan_containment` and therefore contains no `..`
    component, so the join cannot escape `base_dir`. A defensive post-join
    re-check is applied anyway.

    Source is NEVER modified (Vision invariant 3): the target path always
    carries the inserted `.<target_family>` infix, so it can never equal the
    source path, and the write goes via a `.tmp` in the target directory.
    """
    plan_dict = entry.convert_plan or {}
    target_family = plan_dict.get("target_family")
    target_base_name = plan_dict.get("target_base")
    if not target_family or not target_base_name:
        # A convertable entry always carries both; treat absence as a failure
        # rather than silently skipping (keeps the manifest diagnosable).
        return None, R_CONVERT_FAILED

    base = base_by_name.get(target_base_name)
    if base is None or base.param_dict is None:
        # The base that yielded the plan at scan time is gone/unavailable.
        return None, R_CONVERT_FAILED

    rel = PurePosixPath(entry.relative_path)
    target_name = f"{rel.stem}.{target_family}.safetensors"
    parent = rel.parent
    out_rel = (
        target_name if str(parent) == "."
        else f"{parent.as_posix()}/{target_name}"
    )

    base_dir = output_dir if output_dir is not None else audit_root
    target_path = base_dir / out_rel
    source_path = audit_root / entry.relative_path

    # Defensive containment re-check (no `..` is possible in out_rel, but the
    # security posture is "verify the join, don't trust the derivation").
    try:
        target_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        _emit_warn(f"convert_skipped_escape: {out_rel}")
        return None, R_CONVERT_FAILED

    # Pre-write collision check (ADR §8). os.replace() below would otherwise
    # clobber an existing target; this pre-check is the no-overwrite guard.
    # Source can never be the target (different name), but a samefile guard
    # is kept for defence in depth.
    if target_path.exists():
        _emit_warn(f"convert_skipped_collision: {out_rel}")
        return None, R_CONVERT_COLLISION

    try:
        source_sd = _load_state_dict(str(source_path))
        # Re-derive the live ConversionPlan (first-match-wins is deterministic
        # within one process; the FileEntry only carries the serialized dict).
        plan = find_matching_plan(source_sd, base.param_names)
        if plan is None:
            return None, R_CONVERT_FAILED
        converted = convert_state_dict(source_sd, plan)
        if not converted:
            # Empty result means no module matched — nothing useful to write.
            return None, R_CONVERT_FAILED
    except Exception as e:  # noqa: BLE001 — per-file fault isolation (inv. 9)
        _emit_warn(
            f"convert_failed: {out_rel}: {type(e).__name__}: {str(e)[:200]}"
        )
        return None, R_CONVERT_FAILED

    # Atomic write. The `.tmp` lives in the TARGET directory, so os.replace()
    # is always intra-filesystem — ADR §10's cross-filesystem shutil.move()
    # fallback is moot here (we never replace across directories). A SIGKILL
    # between save_file and os.replace leaves an orphan `*.safetensors.tmp`,
    # surfaced as `stale_tmp_file` on the next scan (ADR §10 F-2 Option A);
    # the tool never auto-deletes it.
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_name(target_path.name + ".tmp")
    try:
        safetensors.torch.save_file(converted, str(tmp))
        os.replace(tmp, target_path)
    except Exception as e:  # noqa: BLE001 — per-file fault isolation (inv. 9)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        _emit_warn(
            f"convert_failed: {out_rel}: {type(e).__name__}: {str(e)[:200]}"
        )
        return None, R_CONVERT_FAILED

    return out_rel, None


def _run_convert(
    audit_root: Path,
    output_dir: Optional[Path],
    bases: list[BaseSpec],
    files: list[FileEntry],
    warnings: list[Warning_],
) -> None:
    """Iterate `convertable` entries and populate convert_output/convert_reason.

    Per-file fault isolation (Vision invariant 9): a single conversion that
    raises must not abort the loop. `_convert_one` already catches its own
    expected failure modes; this loop adds a backstop `except` so any
    unforeseen error still maps to `convert_failed` and the run continues.
    """
    base_by_name = {b.name: b for b in bases}
    n_done = n_skip = n_fail = 0
    for entry in files:
        if entry.classification != CLASS_CONVERTABLE:
            continue
        try:
            out_rel, reason = _convert_one(
                entry, audit_root, output_dir, base_by_name
            )
        except Exception as e:  # noqa: BLE001 — backstop fault isolation
            out_rel, reason = None, R_CONVERT_FAILED
            _emit_warn(
                f"convert_failed: {entry.relative_path}: "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
        entry.convert_output = out_rel
        entry.convert_reason = reason
        if reason is None:
            n_done += 1
        elif reason == R_CONVERT_COLLISION:
            n_skip += 1
        else:
            n_fail += 1
    _emit_info(
        f"convert: {n_done} written, {n_skip} skipped (collision), "
        f"{n_fail} failed"
    )


# ── Delete path (ADR §9; Vision invariants 3/4/13) ─────────────────────
def _safe_unlink(
    path: Path, audit_root: Path, warnings: list[Warning_]
) -> tuple[bool, Optional[str]]:
    """Dir-fd-relative unlink with F-5 pre-unlink re-classification (ADR §9).

    Returns `(deleted, skip_reason)`:
      - `(True, None)`                        — file unlinked.
      - `(False, R_DELETE_CLASSIFICATION_CHANGED)` — file no longer matches a
        `deletable` signature at unlink time (F-5 narrowing); skipped.
      - `(False, R_DELETE_CONTAINMENT_FAILED)` — gate 2: parent dir failed the
        §6 realpath descendancy control, or the parent fd's /proc realpath is
        not under `audit_root`, or O_NOFOLLOW rejected a parent symlink.
      - `(False, R_DELETE_UNLINK_FAILED)`     — stat/unlink raised OSError.

    The unlink targets `path.name` relative to a `parent_fd` opened
    O_NOFOLLOW|O_DIRECTORY, so a terminal-component symlink swap between the
    fd-check and the unlink syscall cannot redirect to a different inode, and
    an intermediate-directory swap on the parent cannot redirect the open.
    Residual reclassify→unlink content TOCTOU is named-and-accepted (ADR §9
    Option B); an attacker with same-uid write under `audit_root` could already
    unlink directly, so it grants no new capability.
    """
    parent = path.parent
    # Gate 2 (authoritative): §6 realpath containment control on the parent.
    if not _passes_scan_containment(parent, audit_root, warnings):
        warnings.append(Warning_(_rel(path, audit_root),
                                 W_DELETE_CONTAINMENT_FAILED,
                                 "parent failed scan containment"))
        return False, R_DELETE_CONTAINMENT_FAILED
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_NOFOLLOW
                            | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as e:
        warnings.append(Warning_(_rel(path, audit_root),
                                 W_DELETE_CONTAINMENT_FAILED, str(e)))
        return False, R_DELETE_CONTAINMENT_FAILED
    try:
        if sys.platform == "linux":
            real_parent = Path(os.path.realpath(f"/proc/self/fd/{parent_fd}"))
            try:
                real_parent.relative_to(audit_root)
            except ValueError:
                # F-8 leak class: real_parent here is an absolute path OUTSIDE
                # audit_root (e.g. a swapped-symlink target). Do NOT embed it in
                # the manifest detail — the `file` field (via _rel) already
                # identifies the entry; a fixed token avoids disclosing the
                # escape target if the manifest is shared for diagnostics.
                warnings.append(Warning_(
                    _rel(path, audit_root), W_DELETE_CONTAINMENT_FAILED,
                    "parent fd realpath escaped audit_root"))
                return False, R_DELETE_CONTAINMENT_FAILED
        # Gate 1 re-check (F-5): re-run the cheap deletable signature on the
        # file as it exists right now. The manifest classification was computed
        # at scan time; the file may have been content-swapped since.
        try:
            cur_size = path.stat().st_size
        except OSError as e:
            warnings.append(Warning_(_rel(path, audit_root),
                                     W_DELETE_FAILED, str(e)))
            return False, R_DELETE_UNLINK_FAILED
        if _classify_deletable(path, cur_size) is None:
            warnings.append(Warning_(_rel(path, audit_root),
                                     W_DELETE_SKIPPED_RECLASSIFY, ""))
            return False, R_DELETE_CLASSIFICATION_CHANGED
        try:
            os.unlink(path.name, dir_fd=parent_fd)
        except OSError as e:
            warnings.append(Warning_(_rel(path, audit_root),
                                     W_DELETE_FAILED, str(e)))
            return False, R_DELETE_UNLINK_FAILED
        return True, None
    finally:
        os.close(parent_fd)


def _run_delete(
    audit_root: Path,
    files: list[FileEntry],
    warnings: list[Warning_],
    confirmed: bool,
) -> None:
    """Preview (no `--yes`) or execute (`--yes`) deletion of `deletable` files.

    Triple-gate (ADR §9), enforced in code order:
      1. classification == 'deletable'  — the loop filter; the flag NEVER
         promotes a non-deletable file (Vision invariant 3, no-promotion).
      2. fd-based realpath descendancy of audit_root — inside `_safe_unlink`.
      3. confirmed (`--yes`)            — the preview-vs-execute branch below;
         absence is preview-mode, never an interactive prompt (Vision #13).

    Per-file fault isolation (Vision invariant 9): a single unlink failure must
    not abort the loop. `_safe_unlink` catches its own OSErrors; this loop adds
    a backstop `except` so any unforeseen error still maps to a skip reason and
    the run continues.
    """
    # kind == KIND_LORA filter (ADR-021 Vision invariants 7/11): transformer
    # entries are report-only; even a `deletable`-classified garbage file
    # under a transformer root must NEVER be unlinked. Belt-and-suspenders
    # with the §1 root-disjointness startup invariant (a transformer entry's
    # relative_path is rooted at a DIFFERENT tree, so audit_root/rel would
    # be wrong anyway — but the explicit kind gate is the contract).
    deletable = [e for e in files
                 if e.classification == CLASS_DELETABLE
                 and e.kind == KIND_LORA]
    by_reason: dict[str, int] = {}
    for e in deletable:
        by_reason[e.reason] = by_reason.get(e.reason, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(by_reason.items()))

    if not confirmed:
        # Preview mode: zero I/O (Vision invariant 4). No FileEntry mutation.
        _emit_info(f"would delete {len(deletable)} files ({summary})")
        for e in deletable:
            _emit_info(f"would_delete: {e.relative_path}")
        return

    n_deleted = n_skipped = n_failed = 0
    for e in deletable:
        path = audit_root / e.relative_path
        try:
            ok, skip_reason = _safe_unlink(path, audit_root, warnings)
        except Exception as ex:  # noqa: BLE001 — backstop fault isolation
            ok, skip_reason = False, R_DELETE_UNLINK_FAILED
            # Mirror _safe_unlink's own OSError paths: a backstop-caught fault
            # must also surface in the manifest warnings[], not stderr alone,
            # so the catalog sees a consistent record for every failure mode.
            warnings.append(Warning_(_rel(path, audit_root), W_DELETE_FAILED,
                                     f"{type(ex).__name__}: {str(ex)[:200]}"))
            _emit_warn(
                f"delete_failed: {e.relative_path}: "
                f"{type(ex).__name__}: {str(ex)[:200]}"
            )
        e.deleted = ok
        e.delete_reason = skip_reason
        if ok:
            n_deleted += 1
        elif skip_reason == R_DELETE_CLASSIFICATION_CHANGED:
            n_skipped += 1
        else:
            n_failed += 1
    _emit_info(
        f"delete: {n_deleted} removed, {n_skipped} skipped "
        f"(classification changed), {n_failed} failed"
    )


# ── Scan loop (Vision invariant 9: per-file fault isolation) ───────────
# ── Transformer audit (ADR-021) ────────────────────────────────────────
# Read-classify-report ONLY: no write, no delete, no dry-load on these
# trees. Prognosis reuses audit_single_files.py (header-only, reviewed
# under ADR-019 C-d); base matching is a name-agnostic shape-multiset
# comparison; duplicate detection is bounded sampled reads (§4).

_ASF_MODULE = None  # lazy singleton


def _load_audit_single_files():
    """Load repo-root audit_single_files.py via importlib (ADR-021 §2).

    Import-time execution verified side-effect-free (security F-7): stdlib
    imports + constants/functions, main() behind __main__ guard.
    """
    global _ASF_MODULE
    if _ASF_MODULE is None:
        path = _REPO_ROOT / "audit_single_files.py"
        spec = importlib.util.spec_from_file_location("audit_single_files", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ASF_MODULE = mod
    return _ASF_MODULE


def _dtype_class(dtype: str) -> str:
    """Collapse float precisions so bf16/fp16/fp32 variants shape-match;
    quant dtypes (F8_*, I8, …) keep their identity (ADR-021 §3)."""
    return "float" if dtype in ("BF16", "F16", "F32", "F64") else dtype


def _tensor_nbytes(info: dict) -> int:
    lo, hi = info.get("data_offsets", (0, 0))
    return max(0, hi - lo)


def _base_transformer_index(base: BaseSpec,
                            warnings: list[Warning_]) -> dict[str, Any]:
    """Header-only index of a base's transformer shards (ADR-021 §3/§4).

    Returns {"multiset": {(shape, dtype_class): count},
             "unique":  {(shape, dtype): (key, shard_path, hdr_len, lo, hi)}
             — (shape, EXACT dtype) pairs occurring exactly once,
             "total": int}.

    An unreadable shard marks the WHOLE base unavailable for transformer
    matching (empty index; files fall to no_matching_base) with a loud
    warning. Skipping just the shard would SHRINK |B| and INFLATE the
    overlap ratio |T∩B|/|B| — a partial base could manufacture a false
    `usable` match or a false `duplicate_of`, the exact outcome ADR §4's
    fail-toward-inclusion posture forbids (code-review 2026-07-06
    finding 1: the prior per-shard skip was NOT conservative).
    """
    asf = _load_audit_single_files()
    multiset: dict[tuple, int] = {}
    seen: dict[tuple, list] = {}
    total = 0
    for shard in sorted(Path(base.path).glob("*.safetensors")):
        try:
            hdr, hlen = asf._header(str(shard))
        except Exception as e:  # noqa: BLE001
            warnings.append(Warning_(
                None, W_UNREADABLE,
                f"base {base.name!r} shard {shard.name!r} unreadable "
                f"({type(e).__name__}); base excluded from transformer "
                f"matching (partial |B| would inflate overlap)"))
            return {"multiset": {}, "unique": {}, "total": 0}
        for k, v in hdr.items():
            if k == "__metadata__" or not isinstance(v, dict):
                continue
            shape = tuple(v.get("shape", ()))
            dtype = v.get("dtype", "?")
            lo, hi = v.get("data_offsets", (0, 0))
            multiset[(shape, _dtype_class(dtype))] = (
                multiset.get((shape, _dtype_class(dtype)), 0) + 1)
            seen.setdefault((shape, dtype), []).append(
                (k, str(shard), hlen, lo, hi))
            total += 1
    unique = {sk: entries[0] for sk, entries in seen.items()
              if len(entries) == 1}
    return {"multiset": multiset, "unique": unique, "total": total}


def _read_tensor_prefix(path: str, hdr_len: int, lo: int, hi: int,
                        cap: int = _T_DUP_SAMPLE_CAP) -> Optional[bytes]:
    """Guarded bounded read of a tensor's leading bytes (ADR-021 §4 F-4).

    n = max(0, min(hi - lo, cap, file_size - (8 + hdr_len + lo))) — a
    crafted data_offsets (hi < lo, offsets past EOF, hostile hdr_len) can
    never produce a negative/unbounded read. Returns None on ANY short,
    empty, or errored read; callers treat None as NOT byte-equal (fail
    toward inclusion).
    """
    try:
        fsize = os.path.getsize(path)
        n = max(0, min(hi - lo, cap, fsize - (8 + hdr_len + lo)))
        if n <= 0:
            return None
        with open(path, "rb") as f:
            f.seek(8 + hdr_len + lo)
            data = f.read(n)
        return data if len(data) == n else None
    except OSError:
        return None


def _t_shape_multiset(hdr: dict, dit_only: bool) -> dict[tuple, int]:
    """Shape multiset of a transformer file's header; for AIO bundles the
    comparison restricts to the DiT component groups (ADR-021 §3)."""
    asf = _load_audit_single_files()
    dit_prefixes = asf._AIO_GROUPS["dit"]
    out: dict[tuple, int] = {}
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        if dit_only and not k.startswith(dit_prefixes):
            continue
        shape = tuple(v.get("shape", ()))
        key = (shape, _dtype_class(v.get("dtype", "?")))
        out[key] = out.get(key, 0) + 1
    return out


def _multiset_overlap(t_ms: dict, b_ms: dict) -> float:
    """|T ∩ B| / |B| as multiset intersection (ADR-021 §3)."""
    total_b = sum(b_ms.values())
    if total_b == 0:
        return 0.0
    inter = sum(min(c, t_ms.get(sk, 0)) for sk, c in b_ms.items())
    return inter / total_b


def _t_duplicate_check(path: str, hdr: dict, hlen: int,
                       base_idx: dict[str, Any],
                       warnings: list[Warning_], rel: str) -> bool:
    """Sampled-content duplicate determination vs one base (ADR-021 §4).

    Unique-(shape, EXACT dtype) pairing sidesteps key-name mapping; K=4
    largest by tensor size with KEY-NAME tie-break (deterministic — round-2
    NEW-3); guarded 1-MiB reads on both sides; any None/short read or
    mismatch → not duplicate (fail toward inclusion).
    """
    t_unique: dict[tuple, tuple] = {}
    t_seen: dict[tuple, list] = {}
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        sk = (tuple(v.get("shape", ())), v.get("dtype", "?"))
        t_seen.setdefault(sk, []).append(
            (k, v.get("data_offsets", (0, 0))))
    for sk, entries in t_seen.items():
        if len(entries) == 1:
            t_unique[sk] = entries[0]

    pair_keys = [sk for sk in t_unique if sk in base_idx["unique"]]
    if len(pair_keys) < 2:
        warnings.append(Warning_(rel, W_DUP_CHECK_INCONCLUSIVE,
                                 "fewer than 2 unique-shape tensor pairs; "
                                 "duplicate_of left null (ADR-021 §4)"))
        return False
    # K largest by tensor byte-size; tie-break on the TRANSFORMER-side key
    # name so the sampled set is deterministic (round-2 NEW-3).
    pair_keys.sort(
        key=lambda sk: (-(t_unique[sk][1][1] - t_unique[sk][1][0]),
                        t_unique[sk][0]))
    for sk in pair_keys[:_T_DUP_SAMPLE_K]:
        t_key, (t_lo, t_hi) = t_unique[sk]
        b_key, b_shard, b_hlen, b_lo, b_hi = base_idx["unique"][sk]
        t_bytes = _read_tensor_prefix(path, hlen, t_lo, t_hi)
        b_bytes = _read_tensor_prefix(b_shard, b_hlen, b_lo, b_hi)
        if t_bytes is None or b_bytes is None or t_bytes != b_bytes:
            return False
    return True


def _classify_transformer(path: Path, rel: str, root_index: int,
                          root_name: str, bases: list[BaseSpec],
                          base_indices: dict[str, dict],
                          warnings: list[Warning_]) -> FileEntry:
    """Per-file transformer classification (ADR-021 §2 mapping table).

    NOTE: the LoRA 5 GB _SIZE_CAP_BYTES deliberately does NOT apply —
    transformers are routinely 20-40 GB, the cap existed to bound pickle
    disk-fill, and transformer roots scan .safetensors only (no pickle
    surface). Every read on this path is bounded WITHOUT the size cap:
    _probe_safetensors_garbage + asf._header both cap headers at 100 MB
    (the former added per security-audit F-T1 — the size cap had been
    implicitly bounding it on the LoRA path), and content reads are the
    §4 guarded 1-MiB samples.
    """
    asf = _load_audit_single_files()
    try:
        size = path.stat().st_size
    except OSError as e:
        return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                         root_index=root_index, root=root_name,
                         classification=CLASS_ERROR, reason="stat_failed",
                         error=str(e)[:200])

    deletable_reason = _classify_deletable(path, size)
    if deletable_reason is not None:
        # REPORT-ONLY (Vision invariants 7/11): _run_delete filters on
        # kind == KIND_LORA, so this entry can never be unlinked.
        return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                         root_index=root_index, root=root_name,
                         classification=CLASS_DELETABLE,
                         reason=deletable_reason, size_bytes=size)

    prognosis = asf.audit_file(str(path))
    prog_out = {"verdict": prognosis["verdict"],
                "detail": prognosis["detail"],
                "family": prognosis["family"],
                "n_fp8": prognosis["n_fp8"]}
    verdict = prognosis["verdict"]

    if verdict == "UNREADABLE":
        return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                         root_index=root_index, root=root_name,
                         classification=CLASS_ERROR,
                         reason="prognosis_unreadable", size_bytes=size,
                         prognosis=prog_out, error=prognosis["detail"][:200])

    if verdict in _T_UNSUPPORTED_VERDICTS or (
            verdict.startswith("CQ-") and verdict != "CQ-FP8"):
        return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                         root_index=root_index, root=root_name,
                         classification=CLASS_UNCONVERTABLE,
                         reason=f"quant_unsupported_{verdict.lower()}",
                         size_bytes=size, prognosis=prog_out,
                         matched_bases=[])

    # Shape-fingerprint matching (§3) — dit-keys only for AIO bundles.
    try:
        hdr, hlen = asf._header(str(path))
    except Exception as e:  # noqa: BLE001
        return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                         root_index=root_index, root=root_name,
                         classification=CLASS_ERROR, reason="header_reread",
                         size_bytes=size, prognosis=prog_out,
                         error=f"{type(e).__name__}: {str(e)[:160]}")
    t_ms = _t_shape_multiset(hdr, dit_only=(verdict == "AIO"))

    matched: list[str] = []
    duplicate_of: Optional[str] = None
    for b in bases:
        idx = base_indices.get(b.name)
        if idx is None or idx["total"] == 0:
            continue
        overlap = _multiset_overlap(t_ms, idx["multiset"])
        if overlap >= _T_MATCH_THRESHOLD:
            matched.append(b.name)
            if duplicate_of is None and overlap >= _T_DUP_THRESHOLD:
                if _t_duplicate_check(str(path), hdr, hlen, idx,
                                      warnings, rel):
                    duplicate_of = b.name

    if matched:
        reason = (R_T_AIO_BUNDLE if verdict == "AIO"
                  else f"prognosis_{verdict.lower()}")
        return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                         root_index=root_index, root=root_name,
                         classification=CLASS_USABLE, reason=reason,
                         size_bytes=size, prognosis=prog_out,
                         matched_bases=sorted(matched),
                         duplicate_of=duplicate_of)

    reason = (R_FORMAT_UNKNOWN if prognosis["family"] == "?"
              else R_T_NO_MATCHING_BASE)
    return FileEntry(relative_path=rel, kind=KIND_TRANSFORMER,
                     root_index=root_index, root=root_name,
                     classification=CLASS_UNCONVERTABLE, reason=reason,
                     size_bytes=size, prognosis=prog_out, matched_bases=[])


def _scan_transformer_roots(
    transformer_roots: list[Path], bases: list[BaseSpec],
    warnings: list[Warning_],
) -> tuple[list[FileEntry], bool]:
    """Scan each transformer root (containment rooted at ITSELF, ADR-021 §1)
    and classify every .safetensors. Returns (entries, any_error)."""
    base_indices: dict[str, dict] = {}
    for b in bases:
        try:
            base_indices[b.name] = _base_transformer_index(b, warnings)
        except Exception as e:  # noqa: BLE001 — a broken base must not
            warnings.append(Warning_(None, W_UNREADABLE,     # kill the scan
                                     f"base {b.name} fingerprint failed: "
                                     f"{type(e).__name__}"))

    entries: list[FileEntry] = []
    any_error = False
    for root_index, troot in enumerate(transformer_roots):
        root_name = troot.name
        for path in sorted(troot.rglob("*.safetensors")):
            if not path.is_file() and not path.is_symlink():
                continue
            rel = _rel(path, troot)
            if not _passes_scan_containment(path, troot, warnings):
                continue
            if not _open_no_follow(path, troot, warnings):
                continue
            try:
                entry = _classify_transformer(
                    path, rel, root_index, root_name, bases,
                    base_indices, warnings)
            except Exception as e:  # noqa: BLE001 — per-file isolation
                entry = FileEntry(
                    relative_path=rel, kind=KIND_TRANSFORMER,
                    root_index=root_index, root=root_name,
                    classification=CLASS_ERROR,
                    reason="exception_during_classify",
                    error=f"{type(e).__name__}: {str(e)[:200]}")
            entries.append(entry)
            if entry.classification == CLASS_ERROR:
                any_error = True
    return entries, any_error


def _check_root_disjointness(audit_root: Path,
                             transformer_roots: list[Path]) -> Optional[str]:
    """ADR-021 §1 startup invariant (security F-1, HIGH): every transformer
    root must be disjoint from audit_root (not equal / ancestor / descendant)
    and transformer roots must be pairwise disjoint (F-5).

    HARD BLOCK, deliberately NOT warn-don't-block (round-2 NEW-2): a
    transformer root nested under audit_root lets a garbage transformer
    file classify `deletable` as a LoRA and be unlinked by `--delete --yes`
    — irreversible delete under a transformer root, Vision invariant 7.
    Comparison is on PATH-COMPONENT boundaries via os.path.commonpath
    (round-2 NEW-1: /a/checkpoints vs /a/checkpoints_old are siblings, not
    nested). Returns an error message, or None when all roots are disjoint.

    Assumption (security-audit F-T2): roots live on case-SENSITIVE
    filesystems (this deployment: ext4/mergerfs). On a case-insensitive
    mount, /models vs /MODELS would evade the string comparison; an
    inode-based (samefile-style) check would be needed for portability.
    """
    def _nested(a: Path, b: Path) -> bool:
        try:
            return os.path.commonpath([str(a), str(b)]) in (str(a), str(b))
        except ValueError:
            # Unreachable in production (both sides resolve(strict=True)d
            # first), but this guard protects a HIGH invariant — fail
            # CLOSED, treat as overlapping (code-review 2026-07-06
            # finding 2: returning False here would be fail-open).
            return True

    for t in transformer_roots:
        if _nested(audit_root, t):
            return (f"--transformer-root {t} overlaps --audit-root "
                    f"{audit_root} (equal/ancestor/descendant) — the delete "
                    f"path is rooted at --audit-root, so nesting would let "
                    f"'--delete --yes' unlink under a transformer root")
    for i, a in enumerate(transformer_roots):
        for b in transformer_roots[i + 1:]:
            if _nested(a, b):
                return (f"--transformer-root values overlap: {a} vs {b} "
                        f"(pairwise disjointness required)")
    return None


def _scan(
    audit_root: Path, bases: list[BaseSpec], warnings: list[Warning_]
) -> tuple[list[FileEntry], int]:
    files: list[FileEntry] = []
    any_error = False

    for path in sorted(audit_root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = _rel(path, audit_root)

        # Only surface .tmp files we know we created — `.safetensors.tmp`
        # (atomic convert output, ADR §10 F-2 Option A) and the manifest's
        # own atomic-write tmp. Arbitrary user `.tmp` files (editor swap,
        # downloader mid-flight, etc.) are silently ignored to avoid
        # false stale_tmp_file warnings and to keep concurrent invocations
        # of this tool safe.
        if rel.endswith(".safetensors.tmp") or path.name == "lora_audit.json.tmp":
            warnings.append(Warning_(rel, W_STALE_TMP_FILE,
                                     "left over from interrupted convert (no auto-cleanup; ADR §10)"))
            continue

        if path.suffix.lower() not in _SCAN_EXTENSIONS:
            continue

        if not _passes_scan_containment(path, audit_root, warnings):
            continue
        if not _open_no_follow(path, audit_root, warnings):
            continue

        try:
            entry = _classify_one(path, rel, bases)
            files.append(entry)
            if entry.classification == CLASS_ERROR:
                any_error = True
        except Exception as e:
            any_error = True
            files.append(FileEntry(
                relative_path=rel,
                classification=CLASS_ERROR,
                reason="exception_during_classify",
                size_bytes=path.stat().st_size if path.exists() else 0,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            ))

    return files, EXIT_FILE_ERRORS if any_error else EXIT_OK


def _classify_one(path: Path, rel: str, bases: list[BaseSpec]) -> FileEntry:
    """Per-file fault-isolated classification. Returns a FileEntry."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return FileEntry(relative_path=rel, classification=CLASS_ERROR,
                         reason="stat_failed", error=str(e)[:200])

    if size > _SIZE_CAP_BYTES:
        return FileEntry(relative_path=rel, classification=CLASS_ERROR,
                         reason=R_SIZE_CAP_EXCEEDED,
                         size_bytes=size,
                         error=f"file size {size} exceeds 5 GB cap")

    deletable_reason = _classify_deletable(path, size)
    if deletable_reason is not None:
        sha = _sha256_file(path) if size > 0 else None
        return FileEntry(
            relative_path=rel,
            classification=CLASS_DELETABLE,
            reason=deletable_reason,
            sha256=sha,
            size_bytes=size,
        )

    (classification, reason, verdicts, convert_plan,
     native_convert) = _classify_lora(path, bases)
    return FileEntry(
        relative_path=rel,
        classification=classification,
        reason=reason,
        sha256=_sha256_file(path),
        size_bytes=size,
        verdicts_by_base=verdicts,
        convert_plan=convert_plan,
        native_convert=native_convert,
    )


# ── Main ───────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    # argparse exits with status 2 on parser errors (unrecognized flag,
    # mutex violation, etc.); remap to EXIT_STARTUP_FAIL=1 so the exit-
    # code contract documented in Vision §13 / ADR §12 holds uniformly.
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        if e.code == 0:
            raise
        return EXIT_STARTUP_FAIL

    try:
        audit_root = args.audit_root.resolve(strict=True)
    except (FileNotFoundError, OSError) as e:
        _emit_error(f"audit-root unresolvable: {e}")
        return EXIT_STARTUP_FAIL
    if not audit_root.is_dir():
        _emit_error(f"audit-root is not a directory: {audit_root}")
        return EXIT_STARTUP_FAIL

    # ADR-021 §1: transformer roots — resolve fail-closed, then enforce
    # the disjointness startup invariant (security F-1 HIGH; hard block,
    # see _check_root_disjointness for why warn-don't-block yields here).
    transformer_roots: list[Path] = []
    for traw in args.transformer_roots:
        try:
            troot = traw.resolve(strict=True)
        except (FileNotFoundError, OSError) as e:
            _emit_error(f"transformer-root unresolvable: {e}")
            return EXIT_STARTUP_FAIL
        if not troot.is_dir():
            _emit_error(f"transformer-root is not a directory: {troot}")
            return EXIT_STARTUP_FAIL
        transformer_roots.append(troot)
    disjoint_err = _check_root_disjointness(audit_root, transformer_roots)
    if disjoint_err is not None:
        _emit_error(disjoint_err)
        return EXIT_STARTUP_FAIL

    config_path: Optional[Path] = None
    config: dict[str, Any] = {}
    if not args.no_config:
        config_path = args.config
        if config_path is None and _DEFAULT_CONFIG_PATH.exists():
            config_path = _DEFAULT_CONFIG_PATH
        if config_path is not None:
            config = _load_config(config_path)

    # [defaults] dry_load = true flips the default when CLI didn't pass
    # --dry-load (per ADR §7). CLI presence always wins; config only
    # promotes, never demotes.
    if not args.dry_load and bool(
        (config.get("defaults") or {}).get("dry_load", False)
    ):
        args.dry_load = True

    bases = _resolve_bases(config, args.base, args.override_base)

    output_dir = _validate_output_dir(
        args.output_dir,
        audit_root,
        args.allow_output_outside_root,
        args.require_output_allowlist,
        args.output_allowlist_prefix,
    )

    if args.output is not None:
        out_path = args.output.resolve()
        try:
            out_path.relative_to(audit_root)
        except ValueError:
            _emit_error(
                f"-o / --output must resolve inside --audit-root in S1 "
                f"(got {out_path})"
            )
            return EXIT_STARTUP_FAIL
    else:
        out_path = audit_root / "lora_audit.json"

    _emit_info(
        f"scanning {audit_root} with {len(bases)} base(s); "
        f"output={out_path if not args.print_manifest else '<stdout>'}"
    )

    warnings: list[Warning_] = []
    # Reused classifier functions print() diagnostics to stdout; suppress
    # to honor the machine-caller stdout contract (Vision §13: stdout
    # is empty unless --print-manifest is set, in which case it is ONLY
    # the manifest JSON). Diagnostic surfacing is a future-slice concern.
    with contextlib.redirect_stdout(io.StringIO()):
        _prepare_bases(bases, warnings)
        files, error_exit = _scan(audit_root, bases, warnings)
        if transformer_roots:
            t_entries, t_error = _scan_transformer_roots(
                transformer_roots, bases, warnings)
            files.extend(t_entries)
            if t_error and error_exit == EXIT_OK:
                error_exit = EXIT_FILE_ERRORS

    if args.dry_load:
        # `_load_dry_load_pipeline` invokes diffusers' from_pretrained
        # which prints warnings/info to stdout. Wrap the whole loop in
        # redirect_stdout so the machine-caller stdout contract (Vision
        # §13 — empty unless --print-manifest, JSON-only otherwise) is
        # preserved. Per-call loader output is captured separately by
        # `_dry_load_per_base` for `_applied_modules` parsing.
        with contextlib.redirect_stdout(io.StringIO()):
            _run_dry_load(audit_root, bases, files, warnings)

    if args.convert:
        # convert_state_dict / the reused loaders print diagnostics to stdout;
        # wrap the loop to preserve the machine-caller stdout contract (Vision
        # §13 — empty unless --print-manifest, JSON-only otherwise).
        with contextlib.redirect_stdout(io.StringIO()):
            _run_convert(audit_root, output_dir, bases, files, warnings)

    if args.delete:
        # Mutates the filesystem under audit_root (or previews under it).
        # Classification already ran in _scan; _run_delete re-checks the
        # deletable signature per-file at unlink time (ADR §9 F-5) and
        # enforces fd-based audit_root containment. Emits only [INFO]/[WARN]
        # to stderr — no stdout — so no redirect is needed here.
        _run_delete(audit_root, files, warnings, confirmed=args.yes)

    manifest = _build_manifest(
        audit_root=audit_root,
        bases=bases,
        files=files,
        warnings=warnings,
        output_dir=output_dir,
        argv=argv,
        config_path=config_path,
        transformer_roots=transformer_roots,
    )

    if args.print_manifest:
        sys.stdout.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    else:
        _write_manifest_atomic(manifest, out_path)
        _emit_info(f"wrote manifest: {out_path}")

    return error_exit if error_exit != EXIT_OK else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
