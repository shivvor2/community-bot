import time
from telegram import ChatMember
from telegram.constants import ChatMemberStatus

from kagea_agent.config import load_config

cfg = load_config()


# TODO: Download photos from history in previous use case
def format_history_for_llm(history: list[dict]) -> str:
    """
    Formats chat history for LLM context, grouping consecutive messages by the same user.
    """
    if not history:
        return ""

    formatted_lines = []
    last_user_id = None

    for msg in history:
        current_user_id = msg.get("user_id")

        # Create a new user header if the user changes
        if current_user_id != last_user_id:
            display_name = msg.get("display_name", "Unknown")
            username = msg.get("username")

            user_header = display_name
            if username:
                user_header += f" (@{username})"
            user_header += f" [User ID: {current_user_id}]"

            formatted_lines.append(f"\n{user_header}:")
            last_user_id = current_user_id

        # Compile message metadata
        meta_parts = [msg.get("date", "")]

        msg_id = msg.get("message_id")
        if msg_id is not None:
            meta_parts.append(f"MsgID: {msg_id}")

        reply_to = msg.get("reply_to")
        if reply_to is not None:
            meta_parts.append(f"ReplyTo: {reply_to}")

        meta_str = f"[{', '.join(meta_parts)}]"

        # Handle message text, indenting multi-line text to align under metadata
        text = msg.get("text", "")
        indented_text = text.replace("\n", "\n    ")

        formatted_lines.append(f"  {meta_str} {indented_text}")

    return "\n".join(formatted_lines).strip()


def format_admins_for_llm(admins: list[ChatMember]) -> str:
    """Pretty-print admin list for LLM consumption."""
    if not admins:
        return "No administrators found."

    lines = []
    for member in admins:
        user = member.user
        parts = []

        # Name
        name = user.full_name or user.first_name or "Unknown"
        parts.append(f"Name: {name}")

        # Username
        if user.username:
            parts.append(f"Username: @{user.username}")

        # User ID (useful for matching)
        parts.append(f"User ID: {user.id}")

        # Role
        if member.status == ChatMemberStatus.OWNER:
            parts.append("Role: Owner/Creator")
        else:
            parts.append("Role: Administrator")

        # Custom title if set
        if member.custom_title:
            parts.append(f"Custom Title: {member.custom_title}")

        lines.append("  - " + ", ".join(parts))

    return "Chat Administrators:\n" + "\n".join(lines)


def get_org_context() -> str:
    return f"Organization Context \n name: {cfg.org_info.name} \n description: {cfg.org_info.context}"
