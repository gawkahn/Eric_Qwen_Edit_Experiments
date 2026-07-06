# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
"""comfyless catalog CLI (ADR-022) — the METADATA plane, never the load plane.

    python -m comfyless.catalog_cli build --model-base … --lora-path … \
        --transformer-path … --audit-manifest lora_audit.json
    python -m comfyless.catalog_cli show <name>

S1 ships `build` + `show`; `search`/`enrich`/`annotate`/`exclude` land in
S2-S4 per the ADR-022 slice plan.
"""
from __future__ import annotations

import json
import sys

import click

from . import catalog_db
from .catalog_builder import build as _build


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """LoRA/transformer catalog — SQLite metadata plane (ADR-022)."""


@cli.command(name="build")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True,
              help="Catalog DB path. Must be OFF the mergerfs union "
                   "(FUSE breaks SQLite locking).")
@click.option("--model-base", required=True,
              type=click.Path(file_okay=False, dir_okay=True))
@click.option("--lora-path", "lora_paths", multiple=True,
              type=click.Path(file_okay=False, dir_okay=True),
              help="ADR-018 kind-typed LoRA scan root. Repeatable.")
@click.option("--transformer-path", "transformer_paths", multiple=True,
              type=click.Path(file_okay=False, dir_okay=True),
              help="ADR-018 kind-typed transformer scan root. Repeatable.")
@click.option("--audit-manifest", "audit_manifests", multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help="lora_audit.json manifest(s) to join (ADR-014/021). "
                   "Repeatable.")
@click.option("--catalog", "catalog_path", default=None,
              type=click.Path(dir_okay=False),
              help="Optional operator manifest (ADR-015) forwarded to the "
                   "scan.")
@click.option("--force-fs", is_flag=True,
              help="Proceed even if --db sits on a FUSE filesystem "
                   "(SQLite may hang; you asked for it).")
def build_cmd(db_path: str, model_base: str, lora_paths, transformer_paths,
              audit_manifests, catalog_path, force_fs: bool) -> None:
    """Scan roots + join audit manifests into the catalog DB (no network)."""
    from comfyless.catalog import CatalogBuildError
    try:
        stats = _build(
            db_path, model_base,
            lora_paths=lora_paths,
            transformer_paths=transformer_paths,
            audit_manifests=audit_manifests,
            catalog_path=catalog_path,
            force_fs=force_fs,
        )
    except (catalog_db.CatalogDBError, CatalogBuildError) as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    click.echo(
        f"[catalog] build ok: {stats['entries']} entries "
        f"({stats['audited']} with audit evidence), "
        f"{stats['stale']} newly stale, {stats['fts_rows']} FTS rows "
        f"-> {db_path}")


@cli.command(name="enrich")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True)
@click.option("--limit", default=None, type=click.IntRange(min=1),
              help="Max lookups this run (resumable).")
@click.option("--rate", "rate_s", default=1.0, show_default=True,
              type=click.FloatRange(min=0.0),
              help="Seconds between requests (civitai rate courtesy).")
@click.option("--refresh", is_flag=True,
              help="Re-query entries that already have a civitai_api row.")
@click.option("--include-excluded", is_flag=True,
              help="Also enrich excluded entries (default: candidates only).")
@click.option("--hash-missing", "do_hash", is_flag=True,
              help="First compute sha256 for LoRA entries lacking one "
                   "(local IO only).")
@click.option("--force-fs", is_flag=True)
def enrich_cmd(db_path: str, limit, rate_s: float, refresh: bool,
               include_excluded: bool, do_hash: bool,
               force_fs: bool) -> None:
    """Tier-2 enrichment: civitai hash lookups (THE only network step).

    Exit codes: 0 clean · 2 partial (some lookups failed / network abort;
    run again to resume).
    """
    from .catalog_enrich import enrich, hash_missing, EnrichError
    if do_hash:
        hs = hash_missing(db_path, force_fs=force_fs)
        click.echo(f"[catalog] hashed {hs['hashed']} "
                   f"({hs['errors']} errors)")
    try:
        stats = enrich(db_path, limit=limit, rate_s=rate_s,
                       refresh=refresh, include_excluded=include_excluded,
                       force_fs=force_fs)
    except catalog_db.CatalogDBError as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    except EnrichError as e:
        click.echo(f"[catalog] enrich aborted: {e}", err=True)
        sys.exit(2)
    click.echo(f"[catalog] enrich: {stats}")
    sys.exit(0 if stats["failures"] == 0 else 2)


@cli.command(name="search")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True)
@click.option("--kind", default=None,
              type=click.Choice(["lora", "transformer", "model"]))
@click.option("--family", default=None,
              help="Filter to one model family (e.g. qwen-image, flux2).")
@click.option("--limit", default=20, show_default=True,
              type=click.IntRange(min=1))
@click.option("--include-excluded", is_flag=True,
              help="Also show excluded/stale entries (hidden by default).")
@click.argument("term")
def search_cmd(db_path: str, kind, family, limit: int,
               include_excluded: bool, term: str) -> None:
    """Search by description terms (FTS) or name / partial name.

    Examples: search "cinematic"  ·  search "mystic" --kind lora
    """
    try:
        conn = catalog_db.connect(db_path)
    except catalog_db.CatalogDBError as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    try:
        rows = catalog_db.search(conn, term, kind=kind, family=family,
                                 limit=limit,
                                 include_excluded=include_excluded)
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        if not rows:
            click.echo(f"[catalog] no hits for {term!r}", err=True)
    finally:
        conn.close()


@cli.command(name="worklist")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True)
@click.option("--kind", default=None,
              type=click.Choice(["lora", "transformer"]))
@click.option("--family", default=None)
@click.option("--limit", default=50, show_default=True,
              type=click.IntRange(min=1))
def worklist_cmd(db_path: str, kind, family, limit: int) -> None:
    """Candidates still BARE after tiers 1-2 (no description text from any
    source) — the research worklist for web/ai_authored enrichment
    (ADR-022 §6 tiers 3-4)."""
    try:
        conn = catalog_db.connect(db_path)
    except catalog_db.CatalogDBError as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    try:
        q = """
        SELECT e.name, e.kind, e.model_family, e.classification, e.sha256
        FROM entries e
        WHERE e.excluded = 0 AND e.stale = 0 AND e.kind != 'model'
          AND NOT EXISTS (SELECT 1 FROM descriptions d
                          WHERE d.entry_id = e.id
                            AND d.description IS NOT NULL)
        """
        args: list = []
        if kind:
            q += " AND e.kind = ?"
            args.append(kind)
        if family:
            q += " AND e.model_family = ?"
            args.append(family)
        q += " ORDER BY e.model_family, e.name LIMIT ?"
        args.append(limit)
        rows = [dict(r) for r in conn.execute(q, args).fetchall()]
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        click.echo(f"[catalog] {len(rows)} bare candidates", err=True)
    finally:
        conn.close()


@cli.command(name="annotate")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True)
@click.option("--kind", default=None,
              type=click.Choice(["lora", "transformer", "model"]))
@click.option("--source", required=True,
              type=click.Choice(["web", "ai_authored"]),
              help="Provenance tier. The machine tiers (sidecar, "
                   "civitai_api) cannot be written by hand.")
@click.option("--description", default=None)
@click.option("--usage-tips", default=None)
@click.option("--trigger-word", "trigger_words", multiple=True)
@click.option("--strength", "strength_rec", default=None,
              help="e.g. '0.8 for style, 1.0 for character'")
@click.option("--sampler", "sampler_rec", default=None,
              help="e.g. 'euler, 28 steps, cfg 4'")
@click.option("--url", "provenance_url", default=None,
              help="Source URL (required for --source web).")
@click.argument("name")
def annotate_cmd(db_path: str, kind, source: str, description, usage_tips,
                 trigger_words, strength_rec, sampler_rec, provenance_url,
                 name: str) -> None:
    """Write tier-3 (web research) / tier-4 (AI-authored) enrichment for
    one entry. All text passes the sanitizer; provenance is mandatory
    (ADR-022 §6).

    NOTE: re-annotating REPLACES the whole (entry, source) row — omitted
    fields become NULL. Pass every field you want kept on re-annotate.
    """
    if source == "web" and not provenance_url:
        click.echo("[catalog] ERROR: --url is required for --source web "
                   "(provenance, Vision invariant 4)", err=True)
        sys.exit(1)
    try:
        conn = catalog_db.connect(db_path)
    except catalog_db.CatalogDBError as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    try:
        q = "SELECT id, kind FROM entries WHERE name = ?"
        args = [name]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        rows = conn.execute(q, args).fetchall()
        if not rows:
            click.echo(f"[catalog] no entry named {name!r}", err=True)
            sys.exit(2)
        if len(rows) > 1:
            click.echo(f"[catalog] ambiguous name {name!r} "
                       f"({', '.join(r['kind'] for r in rows)}) — pass "
                       f"--kind", err=True)
            sys.exit(2)
        catalog_db.upsert_description(
            conn, entry_id=rows[0]["id"], source=source,
            description=description, usage_tips=usage_tips,
            trigger_words=list(trigger_words) or None,
            strength_rec=strength_rec, sampler_rec=sampler_rec,
            provenance_url=provenance_url)
        catalog_db.rebuild_fts(conn)
        conn.commit()
        click.echo(f"[catalog] annotated {name!r} ({source})")
    finally:
        conn.close()


@cli.command(name="exclude")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True)
@click.option("--kind", default=None,
              type=click.Choice(["lora", "transformer"]))
@click.option("--clear", is_flag=True,
              help="Clear an operator exclusion (the next build's policy "
                   "pass re-evaluates the entry normally).")
@click.argument("name")
def exclude_cmd(db_path: str, kind, clear: bool, name: str) -> None:
    """Operator exclusion — never touched by rebuilds until --clear."""
    try:
        conn = catalog_db.connect(db_path)
    except catalog_db.CatalogDBError as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    try:
        q = "SELECT id FROM entries WHERE name = ?"
        args = [name]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        rows = conn.execute(q, args).fetchall()
        if len(rows) != 1:
            click.echo(f"[catalog] {'no' if not rows else 'ambiguous'} "
                       f"entry {name!r}"
                       + (" — pass --kind" if len(rows) > 1 else ""),
                       err=True)
            sys.exit(2)
        if clear:
            # Scoped to operator exclusions (security F-2): clearing an
            # AUDIT exclusion would transiently surface a deletable/
            # duplicate asset in search until the next build re-applies
            # policy. Audit exclusions clear themselves when evidence
            # changes on rebuild.
            cur = conn.execute(
                "UPDATE entries SET excluded = 0, excluded_reason = NULL "
                "WHERE id = ? AND excluded_reason = 'operator'",
                (rows[0]["id"],))
            if cur.rowcount == 0:
                click.echo(f"[catalog] {name!r} has no OPERATOR exclusion "
                           f"to clear (audit exclusions clear on rebuild "
                           f"when evidence changes)", err=True)
                sys.exit(2)
        else:
            conn.execute("UPDATE entries SET excluded = 1, "
                         "excluded_reason = 'operator' WHERE id = ?",
                         (rows[0]["id"],))
        conn.commit()
        click.echo(f"[catalog] {name!r} "
                   f"{'un-excluded' if clear else 'excluded (operator)'}")
    finally:
        conn.close()


@cli.command(name="show")
@click.option("--db", "db_path", default=catalog_db.DEFAULT_DB_PATH,
              show_default=True)
@click.option("--kind", default=None,
              type=click.Choice(["lora", "transformer", "model"]))
@click.argument("name")
def show_cmd(db_path: str, kind, name: str) -> None:
    """Print one entry (+ its description rows) as JSON."""
    try:
        conn = catalog_db.connect(db_path)
    except catalog_db.CatalogDBError as e:
        click.echo(f"[catalog] ERROR: {e}", err=True)
        sys.exit(1)
    try:
        q = "SELECT * FROM entries WHERE name = ?"
        args = [name]
        if kind:
            q += " AND kind = ?"
            args.append(kind)
        rows = [dict(r) for r in conn.execute(q, args).fetchall()]
        if not rows:
            click.echo(f"[catalog] no entry named {name!r}"
                       + (f" of kind {kind!r}" if kind else ""), err=True)
            sys.exit(2)
        for row in rows:
            row["descriptions"] = [
                dict(d) for d in conn.execute(
                    "SELECT * FROM descriptions WHERE entry_id = ? "
                    "ORDER BY source", (row["id"],)).fetchall()
            ]
        click.echo(json.dumps(rows, indent=2, ensure_ascii=False))
    finally:
        conn.close()


if __name__ == "__main__":
    cli()
