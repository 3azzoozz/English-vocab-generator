"""Langfuse observability wiring for the Lesson Prep Agent.

Traces every graph run — planner (Claude) calls and the tools' local Ollama
calls — to Langfuse. Everything here degrades gracefully: if the Langfuse keys
are absent or the SDK isn't installed, the agent runs untraced with no errors.

Configuration (in .env):
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or LANGFUSE_HOST
"""

from __future__ import annotations

import os
from functools import lru_cache


def _base_url() -> str | None:
    return os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")


def langfuse_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


@lru_cache(maxsize=1)
def _callbacks() -> tuple:
    """Build (and cache) the Langfuse callback handler, or () if unavailable."""
    if not langfuse_enabled():
        return ()
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Configure the singleton client the handler will use. base_url covers
        # the user's LANGFUSE_BASE_URL as well as the SDK's default LANGFUSE_HOST.
        Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=_base_url(),
        )
        return (CallbackHandler(),)
    except Exception as e:  # missing SDK, bad config, etc. — never block the agent
        print(f"[observability] Langfuse disabled: {e}")
        return ()


def get_langfuse_callbacks() -> list:
    """Callbacks to pass in a LangChain/LangGraph config (empty if not configured)."""
    return list(_callbacks())


def flush_langfuse() -> None:
    """Flush pending traces — call before a short-lived script exits."""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass
