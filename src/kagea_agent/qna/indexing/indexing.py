"""
indexing.py

Production indexer for a markdown documentation vault.
Produces a single hierarchical JSON artifact with:
  - vault-level summary
  - folder-level summaries
  - per-document heading tree + full markdown text

The tree structure mimics PageIndex's output format so downstream
tools (use_artifacts.py, qna_module.py) work unchanged.

Usage:
    artifact = asyncio.run(index_vault(
        vault_dir=Path("./docs"),
        model="openrouter/minimax/minimax-m2.7",
        output_path=Path("./indexed/latest.json"),
    ))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Markdown heading tree generation
# ---------------------------------------------------------------------------


def _parse_heading(line: str) -> tuple[int, str] | None:
    """Parse a markdown heading line. Returns (level, title) or None."""
    m = re.match(r"^(#{1,6})\s+(.+)$", line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def _build_heading_tree(markdown: str) -> list[dict[str, Any]]:
    """
    Build a hierarchical tree from markdown headings.

    Each node:
        {
            "node_id": "uuid-like string",
            "title": "Heading text",
            "level": 1-6,
            "line_num": int (1-indexed),
            "text": "Content under this heading until next same/higher level",
            "nodes": [child nodes]
        }
    """
    lines = markdown.split("\n")
    root_nodes: list[dict[str, Any]] = []
    # Stack of (level, node) — tracks current nesting path
    stack: list[tuple[int, dict[str, Any]]] = []

    # Collect content between headings
    current_content: list[str] = []
    current_line_start: int = 1

    def _flush_content(node: dict | None):
        """Attach accumulated content to a node."""
        if node is not None:
            node["text"] = "\n".join(current_content).strip()

    node_counter = 0

    for i, line in enumerate(lines):
        heading = _parse_heading(line)

        if heading:
            level, title = heading
            line_num = i + 1  # 1-indexed

            # Flush content into previous node
            if stack:
                _flush_content(stack[-1][1])

            # Create new node
            node_counter += 1
            node: dict[str, Any] = {
                "node_id": f"node_{node_counter}",
                "title": title,
                "level": level,
                "line_num": line_num,
                "text": "",
                "nodes": [],
            }

            # Find parent (first node on stack with lower level)
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                stack[-1][1]["nodes"].append(node)
            else:
                root_nodes.append(node)

            stack.append((level, node))
            current_content = []
            current_line_start = line_num + 1
        else:
            current_content.append(line)

    # Flush remaining content into last node
    if stack:
        _flush_content(stack[-1][1])

    return root_nodes


def _strip_tree_text(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove 'text' fields from tree for structure-only view."""
    result = []
    for node in nodes:
        stripped = {k: v for k, v in node.items() if k != "text"}
        if "nodes" in stripped:
            stripped["nodes"] = _strip_tree_text(stripped["nodes"])
        result.append(stripped)
    return result


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


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Summary generation (via litellm)
# ---------------------------------------------------------------------------


async def _generate_summary(
    text: str, model: str, kind: str = "folder"
) -> str:
    """Ask the LLM for a one-sentence summary. Uses litellm."""

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

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("No OPENROUTER_API_KEY — returning placeholder summary")
        return f"[Summary unavailable — no API key]"

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content
        return content.strip() if content else "[No content returned]"
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return f"[Summary generation failed: {e}]"


# ---------------------------------------------------------------------------
# Core indexer
# ---------------------------------------------------------------------------


async def index_vault(
    vault_dir: str | Path,
    workspace: str | Path | None = None,
    model: str = "openrouter/qwen/qwen-2.5-7b-instruct",
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
    workspace : Path, optional
        Legacy parameter (ignored — kept for API compatibility).
    model : str
        LLM model for summary generation (litellm format).
    output_path : Path, optional
        If provided, writes the final artifact JSON here.
    max_concurrency : int
        Max concurrent LLM calls for summaries.
    force_reindex : bool
        Kept for API compatibility (no caching in this implementation).
    generate_summaries : bool
        If True, generates folder and vault summaries via LLM.

    Returns
    -------
    dict
        The complete vault artifact.
    """
    vault_dir = Path(vault_dir).resolve()

    if not vault_dir.is_dir():
        raise FileNotFoundError(f"Vault directory not found: {vault_dir}")

    md_files = sorted(vault_dir.rglob("*.md"))
    # Filter out Zone.Identifier files (Windows metadata)
    md_files = [f for f in md_files if not f.name.endswith(":Zone.Identifier")]

    if not md_files:
        raise FileNotFoundError(f"No .md files found under {vault_dir}")

    logger.info(f"Found {len(md_files)} markdown files in {vault_dir}")

    # ── Index each file ──────────────────────────────────────────────────
    documents: list[dict[str, Any]] = []

    for md_path in md_files:
        meta = _path_metadata(md_path, vault_dir)
        sha = _file_sha256(md_path)
        raw_markdown = md_path.read_text(encoding="utf-8")

        # Build heading tree
        tree = _build_heading_tree(raw_markdown)

        # Extract doc name and description from first heading + first paragraph
        doc_name = meta["stem"]
        doc_description = ""
        first_heading = tree[0] if tree else None
        if first_heading:
            doc_name = first_heading["title"]
            # Use first ~200 chars of content under first heading as description
            text = first_heading.get("text", "").strip()
            if text:
                doc_description = text[:200].split("\n")[0]

        documents.append({
            **meta,
            "doc_id": f"doc_{sha[:12]}",
            "sha256": sha,
            "doc_name": doc_name,
            "doc_description": doc_description,
            "line_count": len(raw_markdown.splitlines()),
            "markdown": raw_markdown,
            "pageindex_structure": tree,
            "pageindex_structure_no_text": _strip_tree_text(tree),
        })

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
        sem = asyncio.Semaphore(max_concurrency)

        async def _safe_summary(context: str, kind: str) -> str:
            async with sem:
                return await _generate_summary(context, model, kind)

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
                    _safe_summary(context, kind="folder")
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
            vault_summary = await _safe_summary(
                vault_summary_context, kind="vault"
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
