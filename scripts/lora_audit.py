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
from pathlib import Path
from typing import Any, Optional

# ── folder_paths stub (ADR §2, F-4) ────────────────────────────────────
# MUST be installed BEFORE any node-module load — eric_qwen_edit_lora.py
# does `import folder_paths` at module level. The stub satisfies the
# import without pulling in ComfyUI's runtime; any real `folder_paths`
# on sys.path is never consulted.
_fp = types.ModuleType("folder_paths")
_fp.get_folder_paths = lambda _category: []
_fp.get_full_path = lambda _category, _name: None
sys.modules["folder_paths"] = _fp

# ── Classifier reuse via fake-package + spec_from_file_location ────────
# `from nodes.* import …` triggers nodes/__init__.py which transitively
# requires ComfyUI's `comfy.*` package. We register a minimal `nodes`
# package object in sys.modules (no __init__ execution) with its
# __path__ set so relative imports inside loaded modules
# (`from .eric_lora_format_convert import …`) resolve correctly. Each
# module is then loaded with its dotted name `nodes.<modname>`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"

_nodes_pkg = types.ModuleType("nodes")
_nodes_pkg.__path__ = [str(_NODES_DIR)]
sys.modules["nodes"] = _nodes_pkg


def _load_node_module(modname: str):
    """Load nodes/<modname>.py as the dotted module `nodes.<modname>`."""
    dotted = f"nodes.{modname}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    path = _NODES_DIR / f"{modname}.py"
    spec = importlib.util.spec_from_file_location(dotted, path)
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
_convert_base_mod = sys.modules["nodes.eric_lora_format_convert"]

check_lora = _check_mod.check_lora
build_param_dict_from_dir = _check_mod.build_param_dict_from_dir
LoRACheckResult = _check_mod.LoRACheckResult
_read_safetensors_header = _check_mod._read_safetensors_header
find_matching_plan = _convert_mod.find_matching_plan
detect_lora_format = _convert_base_mod.detect_lora_format
_load_state_dict = _qwen_mod._load_state_dict
load_lora_with_key_fix = _qwen_mod.load_lora_with_key_fix
unload_adapters = _qwen_mod.unload_adapters

# ── Tool contract constants ────────────────────────────────────────────
_TOOL_VERSION = "0.1.0"
_AUDIT_VERSION = 1

EXIT_OK = 0
EXIT_STARTUP_FAIL = 1
EXIT_FILE_ERRORS = 2

_SCAN_EXTENSIONS = (".safetensors", ".pt", ".bin", ".pth")
_BASE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_SIZE_CAP_BYTES = 5 * 1024 * 1024 * 1024
_PATH_FLAG_NAMES = frozenset({
    "--audit-root", "--config", "--base", "--override-base", "-o", "--output",
    "--output-dir", "--output-allowlist-prefix",
})

KIND_LORA = "lora"
CLASS_USABLE = "usable"
CLASS_CONVERTABLE = "convertable"
CLASS_UNCONVERTABLE = "unconvertable"
CLASS_DELETABLE = "deletable"
CLASS_ERROR = "error"

# Reason codes (per ADR §3 closed sets)
R_OK = "ok"
R_NORM_TARGETING = "norm_targeting"
R_DIM_MISMATCH_PARTIAL = "dim_mismatch_partial"
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

# Warning codes
W_EXCLUDED_SYMLINK_ESCAPE = "excluded_symlink_escape"
W_DANGLING_SYMLINK = "dangling_symlink"
W_UNREADABLE = "unreadable"
W_STALE_TMP_FILE = "stale_tmp_file"  # ADR §10 F-2 Option A
W_DRY_LOAD_BASE_FAILED = "dry_load_base_failed"  # ADR §7
W_DRY_LOAD_VRAM_CASCADE = "dry_load_vram_cascade_possible"  # ADR §7 F-3
W_DRY_LOAD_UNLOAD_FAILED = "dry_load_unload_failed"  # ADR §7

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
    error: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "convert_output": self.convert_output,
            "convert_plan": self.convert_plan,
            "error": self.error,
            "kind": self.kind,
            "reason": self.reason,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verdicts_by_base": self.verdicts_by_base,
        }


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
            "S1+S2 of ADR-014 — shape-match classification and optional "
            "dry-load; --convert, --delete reject at runtime."
        ),
    )
    p.add_argument(
        "--audit-root", required=True, type=Path,
        help="Root directory of the LoRA tree to scan.",
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
        help="Convert output directory (for --convert; S3, not S1).",
    )
    # S3/S4 flags — parse but reject at runtime
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
        help="(S3 of ADR-014; not implemented in S2.)",
    )
    p.add_argument(
        "--delete", action="store_true",
        help="(S4 of ADR-014; not implemented in S1.)",
    )
    p.add_argument(
        "--yes", action="store_true",
        help="(S4 of ADR-014; not implemented in S1.)",
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
    files_sorted = sorted(files, key=lambda e: e.relative_path)
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


def _classify_lora(
    path: Path, bases: list[BaseSpec]
) -> tuple[str, str, dict[str, dict[str, Any]], Optional[dict[str, Any]]]:
    """Run shape-match against every base; if no usable, attempt convertable.
    Returns (classification, reason, verdicts_by_base, convert_plan_or_None).
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
        return CLASS_USABLE, best_usable_reason, verdicts, None

    # No usable verdict — probe convertable. Requires loading state dict.
    try:
        header = _read_safetensors_header(path) if path.suffix.lower() == ".safetensors" else None
    except Exception:
        header = None

    if header is not None:
        fmt = detect_lora_format(header.keys())
        if fmt == "loha":
            return CLASS_UNCONVERTABLE, R_LOHA_UNSUPPORTED, verdicts, None

    # Try find_matching_plan against each base.
    try:
        state_dict = _load_state_dict(str(path))
    except Exception as e:
        # Can't load — surface as unconvertable with arch hint if any.
        reason = _pick_unconvertable_reason(verdicts, header_present=header is not None)
        return CLASS_UNCONVERTABLE, reason, verdicts, None

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
            )

    return (
        CLASS_UNCONVERTABLE,
        _pick_unconvertable_reason(verdicts, header_present=header is not None),
        verdicts,
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


# ── Scan loop (Vision invariant 9: per-file fault isolation) ───────────
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

    classification, reason, verdicts, convert_plan = _classify_lora(path, bases)
    return FileEntry(
        relative_path=rel,
        classification=classification,
        reason=reason,
        sha256=_sha256_file(path),
        size_bytes=size,
        verdicts_by_base=verdicts,
        convert_plan=convert_plan,
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

    if args.convert:
        _emit_error("--convert not implemented in S2 — see ADR-014 §15")
        return EXIT_STARTUP_FAIL
    if args.delete or args.yes:
        _emit_error("--delete / --yes not implemented in S2 — see ADR-014 §15")
        return EXIT_STARTUP_FAIL

    try:
        audit_root = args.audit_root.resolve(strict=True)
    except (FileNotFoundError, OSError) as e:
        _emit_error(f"audit-root unresolvable: {e}")
        return EXIT_STARTUP_FAIL
    if not audit_root.is_dir():
        _emit_error(f"audit-root is not a directory: {audit_root}")
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

    if args.dry_load:
        # `_load_dry_load_pipeline` invokes diffusers' from_pretrained
        # which prints warnings/info to stdout. Wrap the whole loop in
        # redirect_stdout so the machine-caller stdout contract (Vision
        # §13 — empty unless --print-manifest, JSON-only otherwise) is
        # preserved. Per-call loader output is captured separately by
        # `_dry_load_per_base` for `_applied_modules` parsing.
        with contextlib.redirect_stdout(io.StringIO()):
            _run_dry_load(audit_root, bases, files, warnings)

    manifest = _build_manifest(
        audit_root=audit_root,
        bases=bases,
        files=files,
        warnings=warnings,
        output_dir=output_dir,
        argv=argv,
        config_path=config_path,
    )

    if args.print_manifest:
        sys.stdout.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    else:
        _write_manifest_atomic(manifest, out_path)
        _emit_info(f"wrote manifest: {out_path}")

    return error_exit if error_exit != EXIT_OK else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
