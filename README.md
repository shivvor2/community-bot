# Kagea Community Agent

A group-chat-facing AI agent suite currently implementing **QnA** (documentation-grounded question answering) and **Moderation** (automated spam/violation detection). Built with [DSPy](https://dspy.ai) and [python-telegram-bot](https://python-telegram-bot.org).

## Features

- **QnA Agent** — Answers questions by searching an indexed documentation vault. Supports markdown documents from GitBook or local sources.
- **Moderation Agent** — Scans incoming messages for scams, unsolicited promotions, and impersonation. Operates as a first-pass filter before messages are processed.
- **Document Ingestion** — Download documentation from GitBook sites or use local markdown directories.
- **Vault Indexing** — Builds a hierarchical index with LLM-generated summaries for efficient retrieval.

## Installation

Requires Python 3.12+.

```bash
git clone https://github.com/shivvor2/community-bot
cd community-ai-agent
pip install -e ".[dev]"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
OPENROUTER_API_KEY=your-openrouter-api-key
```

### config.yaml

Edit `config.yaml` to configure the agent:

```yaml
bot_settings:
  api_base: "https://openrouter.ai/api/v1"      # LLM API endpoint
  api_key_varname: "OPENROUTER_API_KEY"           # env var name for the API key
  model: "google/gemma-4-31b-it"                  # model identifier (litellm format)
  max_hist_msg: 50                                # max chat history messages to retain
  qna:
    source_name: "latest"                         # artifact filename (without .json)
    max_iter: 25                                  # max ReAct iterations for QnA agent
  spam:
    mode: "delete"

org_info:
  name: "Hyperliquid"
  context: "Hyperliquid is a Layer 1 blockchain for DeFi trading"

indexing:
  pageindex_model: "openrouter/google/gemma-4-31b-it"
  index_dir: "indexed"
  vault_dir: ""                                   # path to local markdown docs directory

ingestion:
  concurrency: 4
  delay: 0.3
  output_dir: "sources"
```

## Usage

### 1. Ingest Documentation

**From GitBook (remote):**

```bash
kagea-ingest https://hyperliquid.gitbook.io/hyperliquid-docs -o ./sources/hyperliquid-docs
```

Options:
- `-o, --output` — Output directory (default: `./<slug>`)
- `-c, --concurrency` — Parallel downloads (default: 4)
- `-d, --delay` — Request delay in seconds (default: 0.3)

**From local markdown files:**

Place your `.md` files in a directory and set `indexing.vault_dir` in `config.yaml` to that path.

### 2. Index the Vault

```bash
# Uses vault_dir from config.yaml
kagea-index

# Or specify a directory and artifact name
kagea-index --vault-dir ./sources/hyperliquid-docs --name latest
```

Options:
- `--vault-dir` — Path to markdown directory (overrides config)
- `--name` — Artifact name (default: `latest`)
- `--no-summaries` — Skip LLM-generated summaries (faster)
- `--model` — LLM model for summaries (overrides config)

The indexed artifact is saved to `<index_dir>/<name>.json`.

### 3. Run the Bot

```bash
kagea-bot
```

The bot connects to Telegram and listens in group chats:
- All messages are first scanned by the **moderation agent** (handler group -1).
- Users can ask questions with `/ask <question>` (handler group 0).
- All messages are silently recorded for chat history context (handler group 1).

## Project Structure

```
src/kagea_agent/
├── config.py              # Pydantic-based config models
├── handlers.py            # Telegram message handlers
├── main.py                # Bot entry point
├── utils.py               # Formatting utilities
├── index_source.py        # CLI for vault indexing
├── moderation/
│   └── moderation_module.py   # Spam detection agent
└── qna/
    ├── qna_module.py          # QnA agent + tool wrappers
    ├── tools.py
    ├── ingestion/
    │   └── gitbook.py         # GitBook downloader
    └── indexing/
        ├── indexing.py        # Vault indexer
        └── use_artifacts.py   # Indexed artifact query functions
```
