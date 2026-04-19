"""
Functions for using the indexed artifact

Output artifact shape (Example):

{
  "vault": {
    "root": "/abs/path/to/hyperliquid-docs",
    "summary": "This documentation covers the Hyperliquid L1 blockchain...",
    "document_count": 42,
    "folders": {
      "": {
        "path": "",
        "name": "(root)",
        "children": ["about-hyperliquid", "builder-tools", "for-developers"],
        "files": ["audits.md", "brand-kit.md", "index.md", ...],
        "summary": "Top-level docs covering audits, onboarding, trading, ..."
      },
      "for-developers": {
        "path": "for-developers",
        "name": "for-developers",
        "children": ["for-developers/api"],
        "files": ["for-developers/api.md", "for-developers/hyperevm.md", "for-developers/nodes.md"],
        "summary": "Developer documentation for the Hyperliquid API, HyperEVM, and node operation."
      },
      "for-developers/api": {
        "path": "for-developers/api",
        "name": "api",
        "children": [],
        "files": ["for-developers/api/error-responses.md", ...],
        "summary": "API reference covering endpoints, error formats, asset IDs, ..."
      }
    }
  },
  "documents": {
    "for-developers/api/error-responses.md": {
      "source_path": "for-developers/api/error-responses.md",
      "file_name": "error-responses.md",
      "stem": "error-responses",
      "parent_folder": "for-developers/api",
      "folder_chain": ["for-developers", "api"],
      "top_level_folder": "for-developers",
      "doc_id": "a1b2c3d4-...",
      "sha256": "...",
      "doc_name": "error-responses",
      "doc_description": "Explains API error response formats and common failure cases.",
      "line_count": 87,
      "markdown": "# Error Responses\n\n...",
      "pageindex_structure": [ /* tree nodes without text */ ]
    }
  }
}

"""

import json

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


# From pageindex.utils, define it here to avoid importing
def remove_fields(data: list | dict, fields: list[str] = ["text"]) -> list | dict:
    """
    Recursively removes specified keys from a nested dictionary/list structure.
    """
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]


# ---------------------------------------------------------------------------
# Artifact funcs
# ---------------------------------------------------------------------------


def browse_vault(artifact: dict, folder_path: str = "") -> str:
    """
    Browse the vault folder hierarchy.
    Returns folder summary, child folders, and file list.
    """
    folder = artifact["vault"]["folders"].get(folder_path)
    if not folder:
        return json.dumps({"error": f"Folder '{folder_path}' not found"})
    return json.dumps(folder, indent=2)


def list_documents(artifact: dict, folder_path: str | None = None) -> str:
    """
    List documents, optionally filtered to a folder.
    Returns source_path, doc_description, and parent_folder for each.
    """
    docs = artifact["documents"]
    if folder_path is not None:
        docs = {k: v for k, v in docs.items() if v["parent_folder"] == folder_path}
    listing = [
        {
            "source_path": d["source_path"],
            "doc_description": d.get("doc_description", ""),
            "parent_folder": d["parent_folder"],
        }
        for d in docs.values()
    ]
    return json.dumps(listing, indent=2)


def get_document_structure(artifact: dict, source_path: str) -> str:
    """
    Returns the PageIndex tree structure (without text) for a document.
    The agent reasons over this to identify relevant sections/line numbers.
    """
    doc = artifact["documents"].get(source_path)
    if not doc:
        return json.dumps({"error": f"Document '{source_path}' not found"})

    structure = doc.get("pageindex_structure")
    if structure is None:
        return json.dumps({"error": f"No structure found for '{source_path}'"})

    # Strip text fields for the structure view
    def _strip_text(nodes: list | dict) -> list | dict:
        if isinstance(nodes, dict):
            return {k: _strip_text(v) for k, v in nodes.items() if k != "text"}
        elif isinstance(nodes, list):
            return [_strip_text(item) for item in nodes]
        return nodes

    structure = _strip_text(structure)

    return json.dumps(
        {
            "source_path": source_path,
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "line_count": doc.get("line_count", 0),
            "structure": structure,
        },
        indent=2,
        ensure_ascii=False,
    )


def get_section_content(artifact: dict, source_path: str, lines: str) -> str:
    """
    Retrieve content of specific sections by line number ranges.

    lines format: '1-50', '10,25', '12'
    Line numbers correspond to line_num values in the tree structure.
    Each returned section is the node whose header starts at that line.

    Falls back to slicing raw markdown if structure has no text fields.
    """
    doc = artifact["documents"].get(source_path)
    if not doc:
        return json.dumps({"error": f"Document '{source_path}' not found"})

    # ── Parse line spec ──────────────────────────────────────────────
    requested: set[int] = set()
    try:
        for part in lines.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo, hi = int(lo.strip()), int(hi.strip())
                if lo > hi:
                    return json.dumps({"error": f"Invalid range '{part}': start > end"})
                requested.update(range(lo, hi + 1))
            else:
                requested.add(int(part))
    except (ValueError, AttributeError) as e:
        return json.dumps(
            {
                "error": f"Invalid lines format: {lines!r}. Use '5-7', '3,8', or '12'. ({e})"
            }
        )

    min_line, max_line = min(requested), max(requested)

    # ── Traverse tree structure ──────────────────────────────────────
    structure = doc.get("pageindex_structure", [])
    results: list[dict] = []
    seen: set[int] = set()

    def _traverse(nodes: list[dict]):
        for node in nodes:
            ln = node.get("line_num")
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                text = node.get("text", "")
                results.append(
                    {
                        "line_num": ln,
                        "title": node.get("title", ""),
                        "node_id": node.get("node_id", ""),
                        "content": text,
                    }
                )
            if node.get("nodes"):
                _traverse(node["nodes"])

    _traverse(structure)

    # ── Fallback: slice raw markdown if nodes had no text ────────────
    if results and all(not r["content"] for r in results):
        md_lines = doc.get("markdown", "").splitlines()
        # For each matched node, extract from its line_num to the next
        # node's line_num (or EOF)
        all_line_nums = sorted(seen)
        for r in results:
            ln = r["line_num"]
            # Find the next node's line_num after this one
            idx = all_line_nums.index(ln)
            start = ln - 1  # 0-indexed
            if idx + 1 < len(all_line_nums):
                end = all_line_nums[idx + 1] - 1
            else:
                end = len(md_lines)
            r["content"] = "\n".join(md_lines[start:end]).strip()

    results.sort(key=lambda x: x["line_num"])
    return json.dumps(results, indent=2, ensure_ascii=False)


def get_full_document(artifact: dict, source_path: str) -> str:
    """
    Returns the complete markdown content of a document.
    Use for short documents where full context is needed.
    """
    doc = artifact["documents"].get(source_path)
    if not doc:
        return json.dumps({"error": f"Document '{source_path}' not found"})
    return json.dumps(
        {
            "source_path": doc["source_path"],
            "doc_description": doc.get("doc_description", ""),
            "parent_folder": doc["parent_folder"],
            "markdown": doc["markdown"],
        },
        ensure_ascii=False,
    )


def get_vault_context(artifact: dict) -> str:
    """
    Returns vault summary and all folder summaries.
    Useful for the agent's system prompt or initial context.
    """
    vault = artifact["vault"]
    context = {
        "vault_summary": vault["summary"],
        "folders": {
            fp: {"summary": info["summary"], "files": info["files"]}
            for fp, info in vault["folders"].items()
        },
    }
    return json.dumps(context, indent=2, ensure_ascii=False)
