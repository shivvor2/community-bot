"""
indexing.py

Production indexer for a markdown documentation vault using PageIndex.
Produces a single hierarchical JSON artifact with:
  - vault-level summary
  - folder-level summaries
  - per-document PageIndex trees + full markdown text

Usage:
    artifact = asyncio.run(index_vault(
        vault_dir=Path("./docs"),
        workspace=Path("./workspace"),
        model="gpt-4o-2024-11-20",
    ))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from pageindex import PageIndexClient
from pageindex.utils import llm_acompletion, remove_fields, structure_to_list

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path metadata
# ---------------------------------------------------------------------------


def _path_metadata(md_path: Path, root: Path) -> dict[str, Any]:
    rel = md_path.relative_to(root)
    folder_chain = list(rel.parts[:-1])
    return {
        "source_path": rel.as_posix(),
        "file_name": rel.name,
        "stem": rel.stem,
        "parent_folder": rel.parent.as_posix() if rel.parent != Path(".") else "",
        "folder_chain": folder_chain,
        "top_level_folder": folder_chain[0] if folder_chain else "",
    }


# TODO: (Planned) implement staleness checking (possibly based on hash check)
def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


async def _generate_summary(text: str, model: str, kind: str = "folder") -> str:
    """Ask the LLM for a one-sentence summary. Uses litellm via pageindex's utils."""

    if kind == "vault":
        prompt = (
            "You are given a list of folder-level summaries from a documentation vault. "
            "Write a single concise paragraph describing what this entire documentation corpus covers.\n\n"
            f"{text}\n\n"
            "Return only the summary paragraph."
        )
    elif kind == "folder":
        prompt = (
            "You are given a list of document titles and descriptions from a single folder "
            "in a documentation site. Write one concise sentence describing what this folder covers.\n\n"
            f"{text}\n\n"
            "Return only the summary sentence."
        )
    else:
        raise ValueError(f"Unknown summary kind: {kind}")

    return await llm_acompletion(model=model, prompt=prompt)


# ---------------------------------------------------------------------------
# Core indexer
# ---------------------------------------------------------------------------


async def index_vault(
    vault_dir: str | Path,
    workspace: str | Path,
    model: str = "openrouter/mini",
    output_path: str | Path | None = None,
    max_concurrency: int = 5,
    force_reindex: bool = False,
    generate_summaries: bool = True,
) -> dict[str, Any]:
    """
    Index an entire markdown vault and return a hierarchical artifact.

    Parameters
    ----------
    vault_dir : Path
        Root directory of the markdown documentation.
    workspace : Path
        PageIndexClient workspace directory for persistence.
    model : str
        LLM model for indexing and summary generation.
    output_path : Path, optional
        If provided, writes the final artifact JSON here.
    max_concurrency : int
        Max concurrent PageIndex indexing calls.
    force_reindex : bool
        If True, re-indexes files even if already in workspace.
    generate_summaries : bool
        If True, generates folder and vault summaries via LLM.

    Returns
    -------
    dict
        The complete vault artifact.
    """
    vault_dir = Path(vault_dir).resolve()
    workspace = Path(workspace).resolve()

    if not vault_dir.is_dir():
        raise FileNotFoundError(f"Vault directory not found: {vault_dir}")

    client = PageIndexClient(workspace=str(workspace), model=model)

    md_files = sorted(vault_dir.rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found under {vault_dir}")

    logger.info(f"Found {len(md_files)} markdown files in {vault_dir}")

    # ── Build a map of already-indexed files (by absolute path) ──────────
    existing_by_path: dict[str, str] = {}
    for doc_id, doc in client.documents.items():
        p = doc.get("path", "")
        if p:
            existing_by_path[p] = doc_id

    # ── Index each file ──────────────────────────────────────────────────
    sem = asyncio.Semaphore(max_concurrency)
    documents: list[dict[str, Any]] = []

    async def _index_file(md_path: Path) -> dict[str, Any]:
        abs_path = str(md_path.resolve())
        meta = _path_metadata(md_path, vault_dir)
        sha = _file_sha256(md_path)

        # Check if already indexed
        if not force_reindex and abs_path in existing_by_path:
            doc_id = existing_by_path[abs_path]
            logger.info(f"[cached] {meta['source_path']} -> {doc_id}")
        else:
            async with sem:
                # PageIndexClient.index is sync internally (handles its own event loop),
                # so run in executor to avoid blocking
                loop = asyncio.get_running_loop()
                doc_id = await loop.run_in_executor(
                    None, client.index, str(md_path), "md"
                )
            logger.info(f"[indexed] {meta['source_path']} -> {doc_id}")

        # Retrieve the indexed data via client API
        doc_meta_json = client.get_document(doc_id)
        doc_meta = json.loads(doc_meta_json)

        structure_json = client.get_document_structure(doc_id)
        structure = json.loads(structure_json)

        # Read full markdown for whole-document retrieval
        raw_markdown = md_path.read_text(encoding="utf-8")

        return {
            **meta,
            "doc_id": doc_id,
            "sha256": sha,
            "doc_name": doc_meta.get("doc_name", ""),
            "doc_description": doc_meta.get("doc_description", ""),
            "line_count": doc_meta.get("line_count", 0),
            "markdown": raw_markdown,
            "pageindex_structure": structure,
        }

    tasks = [_index_file(p) for p in md_files]
    documents = await asyncio.gather(*tasks)

    # ── Build folder hierarchy ───────────────────────────────────────────
    folders_map: dict[str, dict[str, Any]] = {}
    docs_by_folder: dict[str, list[dict]] = defaultdict(list)

    for doc in documents:
        folder = doc["parent_folder"]
        docs_by_folder[folder].append(doc)

    # Collect all unique folder paths (including intermediate ones)
    all_folder_paths: set[str] = set()
    for doc in documents:
        parts = doc["folder_chain"]
        for i in range(len(parts)):
            all_folder_paths.add("/".join(parts[: i + 1]))
    all_folder_paths.add("")  # root

    for fp in sorted(all_folder_paths):
        parts = fp.split("/") if fp else []
        children = [
            f
            for f in all_folder_paths
            if f != fp
            and f.startswith(fp + "/" if fp else "")
            and f.count("/") == (fp.count("/") + 1 if fp else 0)
        ]
        folders_map[fp] = {
            "path": fp,
            "name": parts[-1] if parts else "(root)",
            "children": sorted(children),
            "files": sorted([d["source_path"] for d in docs_by_folder.get(fp, [])]),
            "summary": "",
        }

    # ── Generate folder summaries ────────────────────────────────────────
    if generate_summaries:
        folder_summary_tasks = []
        folder_keys_for_tasks = []

        for fp, folder_info in sorted(folders_map.items()):
            folder_docs = docs_by_folder.get(fp, [])
            if not folder_docs and not folder_info["children"]:
                continue

            # Build context from file titles + descriptions
            lines = []
            for d in folder_docs:
                desc = d.get("doc_description", "")
                line = f"- {d['file_name']}"
                if desc:
                    line += f": {desc}"
                lines.append(line)

            # Include child folder names for context
            for child_fp in folder_info["children"]:
                child_name = child_fp.split("/")[-1] if "/" in child_fp else child_fp
                lines.append(f"- [folder] {child_name}/")

            if lines:
                context = "\n".join(lines)
                folder_summary_tasks.append(
                    _generate_summary(context, model, kind="folder")
                )
                folder_keys_for_tasks.append(fp)

        if folder_summary_tasks:
            folder_summaries = await asyncio.gather(*folder_summary_tasks)
            for fp, summary in zip(folder_keys_for_tasks, folder_summaries):
                folders_map[fp]["summary"] = summary

        # ── Generate vault summary ───────────────────────────────────────
        vault_summary_context = "\n".join(
            f"- {fp or '(root)'}: {info['summary']}"
            for fp, info in sorted(folders_map.items())
            if info["summary"]
        )
        if vault_summary_context:
            vault_summary = await _generate_summary(
                vault_summary_context, model, kind="vault"
            )
        else:
            vault_summary = ""
    else:
        vault_summary = ""

    # ── Assemble final artifact ──────────────────────────────────────────
    artifact = {
        "vault": {
            "root": str(vault_dir),
            "summary": vault_summary,
            "document_count": len(documents),
            "folders": folders_map,
        },
        "documents": {doc["source_path"]: doc for doc in documents},
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)
        logger.info(f"Artifact written to {output_path}")

    return artifact


# If no (optional) name arg, index to "index/latest", otherwise, index to "index/name"
