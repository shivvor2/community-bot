import dspy
from typing import Optional, Literal

type VerdictTypes = Literal["scam", "unsolicited_promotion", "impersonation", "legal"]


instructions = f"""
You are a moderation agent for a group chat. Analyze messages and classify them.

Categories (return `legit` if none apply):

1. `scam` — Phishing links, money solicitations, fake giveaways, deceptive schemes.
2. `unsolicited_promotion` — Ads or shilling for unrelated projects/tokens. Self-promotion by the organization described in org_info is NOT a violation.
3. `impersonation` — Users falsely claiming to be admins, moderators, or official support.

Consider chat_history for context. The admin_list identifies authorized personnel.
If uncertain, default to `legit` — do not suppress legitimate discussion.
Return brief reasoning (keep to one sentence if verdict is `legit`).
"""


class SpamDetect(dspy.Signature):
    """
    DSPy signature of the kagea moderation agent
    """

    message: str = dspy.InputField()
    message_images: Optional[list[dspy.Image]] = dspy.InputField(
        desc="Images relevant to the question"
    )
    chat_history: str = dspy.InputField(desc="Recent Chat history")
    org_info: str = dspy.InputField(
        desc="Information of Organization running the suite, do not ban users for self promotion"
    )
    admin_list: str = dspy.InputField(
        desc="List of administrators and related metadata e.g. user ID"
    )
    verdict: VerdictTypes = dspy.OutputField("The verdict type")
    verdict_reasoning: str = dspy.OutputField(
        desc="The reasoning why you arrived at this verdict, keep short if verdict is `legal`"
    )


SpamDetect = SpamDetect.with_instructions(instructions)

moderation_agent = dspy.ChainOfThought(signature=SpamDetect)
