from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ApplicationHandlerStop,
)
from telegram import Update, Message
import dspy

from kagea_agent.config import load_config
from kagea_agent.utils import format_history_for_llm, format_admins_for_llm
from kagea_agent.qna import qna_agent, get_vault_context
from kagea_agent.moderation import moderation_agent

cfg = load_config()

# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------


async def extract_images_from_message(bot, message: "Message") -> list[bytes]:
    """Download all photos from a message, handling albums."""
    images: list[bytes] = []

    if not message or not message.photo:
        return images

    if message.media_group_id:
        # Album: fetch all messages in the group
        media_group = await bot.get_media_group(message.chat_id, message.message_id)
        for msg in media_group:
            if msg.photo:
                photo_file = await bot.get_file(msg.photo[-1].file_id)
                images.append(bytes(await photo_file.download_as_bytearray()))
    else:
        # Single photo
        photo_file = await bot.get_file(message.photo[-1].file_id)
        images.append(bytes(await photo_file.download_as_bytearray()))

    return images


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


# ── Spam handler (group -1, runs first on every message) ──────────
async def get_admin_list(bot, chat_id: int) -> str:
    """Fetch admins and format for LLM in one call."""
    admins = await bot.get_chat_administrators(chat_id)
    return format_admins_for_llm(admins)


async def spam_scanner(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Load required info from config
    msg = update.effective_message
    if not msg or not msg.text:
        return

    # Fetch fresh admin list
    admin_list = await get_admin_list(context.bot, msg.chat_id)

    # Extract images if any
    images = await extract_images_from_message(context.bot, msg)
    dspy_images = [dspy.Image(img) for img in images] if images else []

    # Get chat history
    history = context.chat_data.get("history", [])
    history_str = format_history_for_llm(history)

    # Run moderation
    result = moderation_agent(
        message=msg.text or "",
        message_images=dspy_images,
        chat_history=history_str,
        admin_list=admin_list,
    )

    if result.verdict != "legal":
        # Delete the spam message, quote the message, and "120"
        await msg.reply_text(
            f"Message deleted for: {result.verdict} \n Reasoning: {result.verdict_reasoning}"
        )
        try:
            await context.bot.delete_message(msg.chat.id, msg.message_id)
        except Exception:
            pass
        raise ApplicationHandlerStop


# ── /ask command (group 0) ─────────────────────────────────────────
async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    max_history = context.bot_data["max_hist_msg"]
    question = " ".join(context.args) if context.args else ""

    images: list[bytes] = []

    # If replying to a message, grab photos and/or text from it
    reply_msg = update.message.reply_to_message
    if reply_msg:
        if not question:
            question = reply_msg.text or reply_msg.caption or ""
        images.extend(await extract_images_from_message(context.bot, reply_msg))

    # Also grab photos from the /ask message itself
    images.extend(await extract_images_from_message(context.bot, update.message))

    if not question:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    history = context.chat_data.get("history", [])
    history_str = format_history_for_llm(history)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    dspy_images = [dspy.Image(img) for img in images] if images else []

    result = qna_agent(
        question=question,
        question_images=dspy_images,
        chat_history=history_str,
        vault_context=get_vault_context(),
    )

    if result.answer_found:
        await update.message.reply_text(result.answer, parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "I couldn't find a relevant answer in the documentation."
        )


# ── Catch-all message recorder (group 1, silent) ──────────────────
async def record_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Group 0 catch-all: records all messages for history context."""
    max_history = context.bot_data["max_hist_msg"]

    if "history" not in context.chat_data:
        context.chat_data["history"] = []

    msg = update.effective_message
    context.chat_data["history"].append(
        {
            "message_id": msg.message_id,
            "user_id": msg.from_user.id if msg.from_user else None,
            "username": msg.from_user.username if msg.from_user else None,
            "display_name": msg.from_user.first_name if msg.from_user else "Unknown",
            "text": msg.text or msg.caption or "",
            "has_photo": bool(msg.photo),
            "photo_file_id": msg.photo[-1].file_id
            if msg.photo
            else None,  # Might be problematic, we are storing only 1 fileID?
            "date": msg.date.isoformat(),
            "reply_to": msg.reply_to_message.message_id
            if msg.reply_to_message
            else None,
        }
    )

    # Trim
    context.chat_data["history"] = context.chat_data["history"][-max_history:]
