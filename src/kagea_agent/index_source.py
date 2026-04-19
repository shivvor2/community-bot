"""
index_source.py

Index a local documentation directory into a PageIndex artifact.

Usage:
    python -m kagea_agent.index_source [--vault-dir PATH] [--name NAME]
    python -m kagea_agent.index_source  # uses config defaults

The artifact is saved to <index_dir>/<name>.json (configurable via config.yaml).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from kagea_agent.qna.indexing import index_vault
from kagea_agent.config import load_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()  # Load .env for OPENROUTER_API_KEY etc.
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Index a markdown documentation vault into a PageIndex artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index from default vault_dir in config.yaml
  python -m kagea_agent.index_source

  # Index a specific directory with a custom name
  python -m kagea_agent.index_source --vault-dir ./docs --name hyperliquid-v1

  # Index with summaries disabled (faster)
  python -m kagea_agent.index_source --no-summaries
        """,
    )
    parser.add_argument(
        "--vault-dir",
        type=str,
        default=None,
        help="Root directory of markdown documentation (overrides config).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Artifact name (default: 'latest').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Skip LLM-generated folder/vault summaries (faster, cheaper).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-index even if already indexed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model for indexing/summaries (overrides config).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent indexing tasks (default: 5).",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)

    # ── Resolve parameters ───────────────────────────────────────────────
    vault_dir = args.vault_dir or config.indexing.vault_dir
    if not vault_dir:
        logging.error(
            "No vault directory specified. Use --vault-dir or set indexing.vault_dir in config.yaml"
        )
        sys.exit(1)

    vault_dir = Path(vault_dir).resolve()
    if not vault_dir.is_dir():
        logging.error(f"Vault directory not found: {vault_dir}")
        sys.exit(1)

    artifact_name = args.name or config.bot_settings.qna.source_name
    index_dir = Path(config.indexing.index_dir)
    model = args.model or config.indexing.pageindex_model
    max_concurrency = args.concurrency or config.ingestion.concurrency or 4
    generate_summaries = not args.no_summaries

    # ── Create workspace (PageIndex uses this for persistence) ────────────
    workspace_dir = index_dir / f".workspace_{artifact_name}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # ── Index ─────────────────────────────────────────────────────────────
    logging.info(f"Indexing vault: {vault_dir}")
    logging.info(f"  Model: {model}")
    logging.info(f"  Summaries: {generate_summaries}")
    logging.info(f"  Output: {index_dir / artifact_name}.json")

    if not os.getenv("OPENROUTER_API_KEY"):
        logging.warning("OPENROUTER_API_KEY not set — LLM calls will fail.")

    artifact = asyncio.run(
        index_vault(
            vault_dir=vault_dir,
            workspace=workspace_dir,
            model=model,
            output_path=index_dir / f"{artifact_name}.json",
            max_concurrency=max_concurrency,
            force_reindex=args.force,
            generate_summaries=generate_summaries,
        )
    )

    doc_count = len(artifact.get("documents", {}))
    folder_count = len(artifact.get("vault", {}).get("folders", {}))
    logging.info(f"Done. Indexed {doc_count} documents across {folder_count} folders.")
    logging.info(f"Artifact saved to: {index_dir / f'{artifact_name}.json'}")


if __name__ == "__main__":
    main()
