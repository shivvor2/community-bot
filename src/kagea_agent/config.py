# src/kagea_agent/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel


class QnAConfig(BaseModel):
    source_name: str = "latest"
    max_iter: int = 25


class SpamConfig(BaseModel):
    mode: str = "warn"


class BotSettings(BaseModel):
    api_base: str = "https://api.openai.com/v1"  # OpenAI's endpoint
    # APIkey Varname for inference provider endpoint
    api_key_varname: str = "OPENAI_API_KEY"
    model: str = "gpt-5.4-mini"
    max_hist_msg: int = 50
    qna: QnAConfig = QnAConfig()
    spam: SpamConfig = SpamConfig()


class OrganizationInfo(BaseModel):
    name: str = ""
    context: str = ""


class IndexingConfig(BaseModel):
    pageindex_model: str = "openai/gpt-5.4-mini"
    index_dir: str = "indexed"
    vault_dir: str = ""


class IngestionConfig(BaseModel):
    sitemap_index: str = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
    user_agent: str = "Mozilla/5.0 (compatible; GitBookDownloader/1.0)"
    concurrency: int = 4
    delay: float = 0.3
    output_dir: str = "sources"


class AppConfig(BaseModel):
    bot_settings: BotSettings = BotSettings()
    org_info: OrganizationInfo = OrganizationInfo()
    indexing: IndexingConfig = IndexingConfig()
    ingestion: IngestionConfig = IngestionConfig()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)
