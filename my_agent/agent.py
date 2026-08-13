"""Construct the Lesson Prep Agent graph.

`graph` is the compiled graph LangGraph loads (see langgraph.json).
`run(topic, level)` is a convenience wrapper for scripts and the eval harness.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from my_agent.utils.nodes import agent_node, route
from my_agent.utils.observability import flush_langfuse, get_langfuse_callbacks
from my_agent.utils.state import LessonState
from my_agent.utils.tools import TOOLS

# Load a local .env if present, so ANTHROPIC_API_KEY is available for scripts.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional
    pass


def build_graph():
    """Compile and return the Lesson Prep Agent graph."""
    builder = StateGraph(LessonState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route)
    builder.add_edge("tools", "agent")
    return builder.compile()


# Module-level compiled graph — this is what langgraph.json points to.
graph = build_graph()


def run(topic: str, level: str, *, recursion_limit: int = 25) -> dict:
    """Run the agent for one topic/level and return the final state.

    If Langfuse is configured (see my_agent/utils/observability.py), the whole
    run — planner and tool LLM calls — is traced automatically.
    """
    user = f"Build a vocabulary lesson pack. Topic: {topic}. Student level: {level}."
    config = {
        "recursion_limit": recursion_limit,
        "callbacks": get_langfuse_callbacks(),
        "run_name": "lesson_prep_agent",
        "metadata": {
            "topic": topic,
            "level": level,
            "langfuse_tags": ["lesson-prep-agent", f"level:{level}"],
        },
    }
    return graph.invoke(
        {"messages": [("user", user)], "flagged_terms": []},
        config=config,
    )


if __name__ == "__main__":
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment (or .env) first.")
    final = run("food vocabulary", "A2")
    print(final["messages"][-1].content)
    print("\nAccumulated flagged terms:", final.get("flagged_terms"))
    flush_langfuse()  # ensure the trace is sent before the script exits
