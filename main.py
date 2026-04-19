import os
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ApplicationHandlerStop,
)
from telegram import Update
import dspy

from kagea_agent.config import load_config
from kagea_agent.utils import get_org_context
from kagea_agent.handlers import spam_scanner, handle_ask, record_message

cfg = load_config()

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
MAX_HISTORY = 50

# TODO: Make into multiple models
bot_model = dspy.LM(
    cfg.bot_settings.model,
    api_key=os.getenv(cfg.bot_settings.api_key_varname),
    api_base=cfg.bot_settings.api_base,
)

dspy.configure(lm=bot_model)


# ── Admin cache on startup ─────────────────────────────────────────
async def post_init(application):
    """Cache information for configured chats."""
    # target chat IDs could be read from config
    # For now, admin list are fetched per-chat on first message

    # Also put everything
    application.bot_data["max_hist_msg"] = cfg.bot_settings.max_hist_msg
    application.bot_data["org_context"] = get_org_context()


# ── Build & Run ────────────────────────────────────────────────────
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    # Group -1: spam scanner on all group text messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & filters.ChatType.GROUPS & ~filters.COMMAND, spam_scanner
        ),
        group=-1,
    )

    # Group 0: /ask command
    app.add_handler(CommandHandler("ask", handle_ask), group=0)

    # Group 1: silent message recorder for history context
    app.add_handler(
        MessageHandler(filters.ChatType.GROUPS & ~filters.COMMAND, record_message),
        group=1,
    )

    app.run_polling()


if __name__ == "__main__":
    main()
