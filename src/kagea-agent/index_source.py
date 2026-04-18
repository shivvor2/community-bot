import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from qna.indexing import index_vault
from qna.ingestion import download_gitbook

# ---------------------------------------------------------------------------
# Load Config
# ---------------------------------------------------------------------------

# Assume ran from TLD, correct if wrong
with open("config.yaml", "r") as file:
    # Load content into a Python object (dictionary/list)
    data = yaml.safe_load(file)


# TODO: Get args from parser
# TODO: Add path to index directly from `sources` folder (skips ingestion step)
def main():
    url = ...  # Must be supplied
    name = ...  # Supplied by CLI, or "Latest" if not provided
    output_folder = data.get("ingestion", {}).get("output_dir", "")
    output_dir = Path(output_folder, name)

    download_gitbook(base_url=url, output_dir=output_dir)

    temp_workspace_dir_obj = TemporaryDirectory()
    pageindex_model = data.get("indexing", {}).get(
        "pageindex_model", "openrouter/minimax/minimax-m2.7"
    )

    artifact = asyncio.run(
        index_vault(
            vault_dir=Path("./docs"),
            workspace=Path(temp_workspace_dir_obj.name),
            model="pageindex_model",
        )
    )


# ---------------------------------------------------------------------------
# Load Summary
# ---------------------------------------------------------------------------
