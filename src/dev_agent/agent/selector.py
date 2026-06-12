"""Auto-selector: uses a classifier LLM to pick the best profile for a prompt."""

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from dev_agent.config import LLMProfile

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)

_CLASSIFIER_PROMPT = """\
You are an LLM router. Given the user's task and the available LLM profiles below,
reply with ONLY the name of the most suitable profile — nothing else, no explanation.

Available profiles:
{profiles}

User task: {prompt}"""


async def select_profile(
    prompt: str,
    profiles: dict[str, LLMProfile],
    classifier_llm: "BaseChatModel",
) -> str:
    """Call the classifier LLM and return the selected profile name.

    Falls back to the first profile in `profiles` if the classifier fails
    or returns an unknown name.
    """
    fallback = next(iter(profiles))
    profile_lines = "\n".join(
        f"  {name}: {p.description.strip()}" for name, p in profiles.items()
    )
    classifier_input = _CLASSIFIER_PROMPT.format(profiles=profile_lines, prompt=prompt)

    try:
        response = await classifier_llm.ainvoke([HumanMessage(content=classifier_input)])
        chosen = response.content.strip().strip('"').strip("'").strip()
        if chosen not in profiles:
            log.warning("Classifier returned unknown profile '%s', falling back to '%s'", chosen, fallback)
            return fallback
        log.info("Auto-selected profile: %s", chosen)
        return chosen
    except Exception as exc:
        log.error("Profile selector failed (%s), falling back to '%s'", exc, fallback)
        return fallback
