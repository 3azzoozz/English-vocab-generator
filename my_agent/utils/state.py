"""State definition for the Lesson Prep Agent graph."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph import MessagesState


def merge_flagged(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """State reducer: accumulate flagged terms across retries, order-preserving,
    de-duplicated (case-insensitive)."""
    out = list(existing or [])
    lower = {t.lower() for t in out}
    for t in new or []:
        if t.lower() not in lower:
            out.append(t)
            lower.add(t.lower())
    return out


class LessonState(MessagesState):
    """Message history plus the accumulating list of above-level flagged terms.

    `flagged_terms` is written by the check_difficulty tool and read (from
    injected state) by assemble_lesson_pack, so the teacher disclosure is
    guaranteed rather than left to the LLM.
    """

    flagged_terms: Annotated[list[str], merge_flagged]
