import dspy
from typing import Optional, Literal

type VerdictTypes = Literal["scam", "unsolicited_promotion", "impersonation", "legal"]


# TODO: Write a Proper prompt, read from an external file (not in the code)
instructions = """
You are Kagea Moderation agent, an AI agent designed for Moderation in
a groupchat setting.

Below are Categories of violations you need to detect,
if there are no violations, return a verdict of `legit`

### Category 1: Obvious Scams (`scam`)
**Examples:**
* Phishing attempts (e.g., malicious links designed to steal credentials).
* Direct solicitations for money.
* Deceptive promotional schemes (e.g., fake giveaways).

### Category 2: Unsolicited Promotions (`unsolicited_promotion`)
**Examples:**
* Must identify advertisements for unrelated projects.
* Must detect cryptocurrency or token shilling.
Discussing the organization's/ telegram group's projects should be allowed, and should never be flagged as such.

### Category 3: Impersonation (`impersonation`)
**Criteria:**
* Accounts/ Users falsely assuming authoritative roles (e.g., fake administrators).
* Accounts/ Users masquerading as official customer support.

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
