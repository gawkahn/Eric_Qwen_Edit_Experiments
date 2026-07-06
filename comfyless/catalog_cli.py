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
