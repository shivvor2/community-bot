import json
from pathlib import Path

import dspy
from typing import Optional

# Internal imports
from kagea_agent.config import load_config
from kagea_agent.qna.indexing import use_artifacts as indexing

_cfg = load_config()

# ---------------------------------------------------------------------------
# Lazy artifact loader
# ---------------------------------------------------------------------------

_artifact = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        index_dir = _cfg.indexing.index_dir
        source_name = _cfg.bot_settings.qna.source_name
        artifact_path = Path(index_dir) / f"{source_name}.json"
        with open(artifact_path, "r") as f:
            _artifact = json.load(f)
    return _artifact


# ---------------------------------------------------------------------------
# DSPy Tools (plain functions, closure over artifact)
# ---------------------------------------------------------------------------


def browse_vault(folder_path: str = ""):
    """
    Browse the vault folder hierarchy.
    Returns folder summary, child folders, and file list.
    """
    return indexing.browse_vault(_load_artifact(), folder_path)


def list_documents(folder_path: str | None = None):
    """
    List documents, optionally filtered to a folder.
    Returns source_path, doc_description, and parent_folder for each.
    """
    return indexing.list_documents(_load_artifact(), folder_path)


def get_document_structure(source_path: str):
    """
    Returns the PageIndex tree structure (without text) for a document.
    The agent reasons over this to identify relevant sections/line numbers.
    """
    return indexing.get_document_structure(_load_artifact(), source_path)


def get_section_content(source_path: str, lines: str) -> str:
    """
    Retrieve content of specific sections by line number ranges.

    lines format: '1-50', '10,25', '12'
    Line numbers correspond to line_num values in the tree structure.
    Each returned section is the node whose header starts at that line.

    Falls back to slicing raw markdown if structure has no text fields.
    """
    return indexing.get_section_content(_load_artifact(), source_path, lines)


def get_full_document(source_path: str) -> str:
    """
    Returns the complete markdown content of a document.
    Use for short documents where full context is needed.
    """
    return indexing.get_full_document(_load_artifact(), source_path)


def get_vault_context() -> str:
    """
    Returns vault summary and all folder summaries.
    Useful for the agent's system prompt or initial context.
    """
    return indexing.get_vault_context(_load_artifact())


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------

instructions = """
You are a QnA agent for a group chat, answering questions using a documentation knowledge base (the vault).

Process:
1. Use get_vault_context to understand what the vault covers.
2. Use browse_vault and list_documents to find relevant files.
3. Use get_document_structure to identify relevant sections.
4. Use get_section_content or get_full_document to retrieve content.
5. Synthesize an answer from retrieved content.

Rules:
- Base answers ONLY on vault content. If the vault doesn't contain the answer, set answer_found to false.
- Keep answers concise and suitable for group chat. Use Markdown formatting.
- The question may include images; consider them for context but answer from documentation.
- Consider chat_history to resolve ambiguous references.
- Cite source documents when possible.
"""


# TODO: Change chat history typing (if we want images from history)
class DocQA(dspy.Signature):
    """
    DSPy signature of the kagea qna agent
    """

    question: str = dspy.InputField()
    question_images: Optional[list[dspy.Image]] = dspy.InputField(
        desc="Images relevant to the question"
    )
    chat_history: str = dspy.InputField(desc="Recent Chat history")
    vault_context: str = dspy.InputField(
        desc="Summary of the Knowledge base and directory information"
    )
    answer: str = dspy.OutputField()
    answer_found: bool = dspy.OutputField(
        desc="Whether or not the agent is able to write an answer, even partial, **Using information from the Vault**"
    )


# Looks stupid, unfortunately the only way to supply
DocQA = DocQA.with_instructions(instructions)

# Use a smaller subset of tools if amount of lines is small
qna_agent_tools = [
    browse_vault,
    list_documents,
    get_document_structure,
    get_section_content,
    get_full_document,
]

max_iters = _cfg.bot_settings.qna.max_iter

qna_agent = dspy.ReAct(signature=DocQA, tools=qna_agent_tools, max_iters=max_iters)
