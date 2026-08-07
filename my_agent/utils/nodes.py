"""Node functions for the Lesson Prep Agent graph.

The planner (agent_node) runs on Claude Haiku 4.5 — the tool-calling model that
decides which tool to invoke next. The tools it calls do their heavy lifting on
a local Ollama model (see tools.py).
"""

from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import END

from my_agent.utils.state import LessonState
from my_agent.utils.tools import TOOLS

# Cheapest current Claude model — used only for planning/tool-calling.
PLANNER_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a Lesson Prep Agent. Given a topic and a student CEFR level, you "
    "autonomously build an English vocabulary lesson pack.\n\n"
    "Valid CEFR levels are A1, A2, B1, B2, C1, C2. If the requested level is not "
    "one of these, or the topic is clearly inappropriate for the requested level, "
    "do NOT produce a pack - explain the problem to the user and stop.\n\n"
    "Otherwise follow this workflow:\n"
    "1. Call generate_vocabulary for the topic and level.\n"
    "2. Call generate_practice_questions using that vocabulary.\n"
    "3. Call check_difficulty on the vocabulary, and again on the questions. It "
    "returns JSON with a 'pass' boolean and 'flagged_terms'.\n"
    "4. If a check_difficulty result has \"pass\": false, regenerate that artifact "
    "ONCE and re-check it. Do not loop endlessly - if it still fails, proceed "
    "with the best version (flagged terms are disclosed automatically).\n"
    "5. When both artifacts are as close to the level as possible, call "
    "assemble_lesson_pack with the vocabulary and questions, then present the "
    "result. (The teacher note about flagged terms is added by the tool "
    "automatically; you do not manage it.)\n\n"
    "If any tool returns a string starting with 'ERROR:', stop and report it to "
    "the user rather than continuing."
)

# The planner LLM: tools bound so it can plan the sequence.
_planner = ChatAnthropic(
    model=PLANNER_MODEL, max_tokens=1024, temperature=0.0
).bind_tools(TOOLS)


def agent_node(state: LessonState) -> dict:
    """The LLM decides the next tool call (or produces the final answer)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    return {"messages": [_planner.invoke(messages)]}


def route(state: LessonState) -> Literal["tools", "__end__"]:
    """If the last message asked for a tool, run it; otherwise we're done."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END
