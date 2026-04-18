import json
from pathlib import Path

import dspy
import yaml

# Internal imports
from kagea_agent.qna.indexing import use_artifacts as indexing

# Assume ran from TLD, correct if wrong
with open("config.yaml", "r") as file:
    # Load content into a Python object (dictionary/list)
    config = yaml.safe_load(file)

# Load the artifact
index_dir = config.get("indexing", {}).get("index_dir", "indexed")
source_name = config.get("bot_settings", {}).get("qna", {}).get("source_name", "latest")
artifact_path = Path(index_dir, source_name)

# report if error
with open(artifact_path, "r") as file:
    artifact = json.load(file)


# ---------------------------------------------------------------------------
# DSPy Tools (plain functions for now, closure version of use_artifacts.py)
# ---------------------------------------------------------------------------

# Including original docstring as it is used by DSPy


def browse_vault(folder_path: str = ""):
    """
    Browse the vault folder hierarchy.
    Returns folder summary, child folders, and file list.
    """
    return indexing.browse_vault(artifact, folder_path)


def list_documents(folder_path: str | None = None):
    """
    List documents, optionally filtered to a folder.
    Returns source_path, doc_description, and parent_folder for each.
    """
    return indexing.list_documents(artifact, folder_path)


def get_document_structure(source_path: str):
    """
    Returns the PageIndex tree structure (without text) for a document.
    The agent reasons over this to identify relevant sections/line numbers.
    """
    return indexing.get_document_structure(artifact, source_path)


def get_section_content(source_path: str, lines: str) -> str:
    """
    Retrieve content of specific sections by line number ranges.

    lines format: '1-50', '10,25', '12'
    Line numbers correspond to line_num values in the tree structure.
    Each returned section is the node whose header starts at that line.

    Falls back to slicing raw markdown if structure has no text fields.
    """
    return indexing.get_section_content(artifact, source_path, lines)


def get_full_document(source_path: str) -> str:
    """
    Returns the complete markdown content of a document.
    Use for short documents where full context is needed.
    """
    return indexing.get_full_document(artifact, source_path)


def get_vault_context() -> str:
    """
    Returns vault summary and all folder summaries.
    Useful for the agent's system prompt or initial context.
    """
    return indexing.get_vault_context(artifact)


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------


# TODO: Write a Proper prompt, read from an external file (not in the code)
instructions = """
    You are Kagea agent, an AI agent designed for Document QnA in
    a groupchat setting.
"""


# TODO: Change chat history typing according to Telegram API modelling
class DocQA(dspy.Signature):
    """
    DSPy signature of the kagea qna agent
    """

    question: str = dspy.InputField()
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

qna_agent_tools = [
    browse_vault,
    list_documents,
    get_document_structure,
    get_section_content,
    get_full_document,
]

max_iters = int(config.get("bot_settings", {}).get("qna", {}).get("max_iter", 25))

qna_agent = dspy.ReAct(signature=DocQA, tools=qna_agent_tools, max_iters=max_iters)
