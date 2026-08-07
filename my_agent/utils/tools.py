"""Tools for the Lesson Prep Agent graph — one job each.

Generation and difficulty-judging run on a local Ollama model (free, private).
The planner node that *sequences* these tools uses Claude (see nodes.py), because
tool-calling needs to be reliable.

  generate_vocabulary(topic, level, count)              -> word list + definitions
  generate_practice_questions(topic, level, vocab_list) -> practice questions
  check_difficulty(text, target_level)                  -> {"pass", "flagged_terms", "reason"}
  assemble_lesson_pack(vocab, questions)                -> final Markdown lesson pack

check_difficulty returns structured output and accumulates flagged terms into
graph state. assemble_lesson_pack reads that accumulated list from *injected
state*, so a non-empty "Note for teacher" section is guaranteed to appear.
"""

from __future__ import annotations

import json
import re
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from my_agent.utils.state import merge_flagged

# Local model used for the heavy generation + judging calls (no API cost).
OLLAMA_MODEL = "qwen2.5:3b"
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# Valid CEFR levels. Anything outside this set is a hard "flag/refuse" case.
VALID_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _normalise_level(level: str) -> str:
    return (level or "").strip().upper()


def _text(message) -> str:
    """Extract plain text from an LLM response message (version-safe)."""
    return str(message.text).strip()


def _parse_json_object(raw: str) -> dict:
    """Best-effort parse of a JSON object from a model response."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def assess_difficulty(text: str, target_level: str) -> dict:
    """Core difficulty judgement. Returns a dict:
    {"pass": bool, "flagged_terms": [str, ...], "reason": str}.
    Factored out of the tool so it can be tested in isolation."""
    lvl = _normalise_level(target_level)
    if lvl not in VALID_LEVELS:
        return {
            "pass": False,
            "flagged_terms": [],
            "reason": f"'{target_level}' is not a valid CEFR level.",
        }
    prompt = (
        f"You are a strict CEFR levelling assessor. Assess whether the text below "
        f"is appropriate for CEFR level {lvl} English learners, judging vocabulary "
        f"difficulty, grammar complexity, and sentence length.\n\n"
        f"TEXT:\n{text}\n\n"
        f"Respond with ONLY a JSON object (no markdown, no prose) with keys:\n"
        f'  "pass": true if the text fits {lvl}, false if it is clearly above or below.\n'
        f'  "flagged_terms": array of the specific words or short phrases in the '
        f"text that are ABOVE {lvl} (empty array if pass is true).\n"
        f'  "reason": one short sentence.\n'
    )
    data = _parse_json_object(_text(llm.invoke(prompt)))
    terms = data.get("flagged_terms") or []
    return {
        "pass": bool(data.get("pass", False)),
        "flagged_terms": [str(t).strip() for t in terms if str(t).strip()],
        "reason": str(data.get("reason", "")).strip() or "No reason provided.",
    }


@tool
def generate_vocabulary(topic: str, level: str, count: int) -> str:
    """Generate a vocabulary word list with short definitions for a topic at a
    given CEFR level.

    Args:
        topic: The lesson topic, e.g. "food vocabulary".
        level: Target CEFR level (A1, A2, B1, B2, C1, or C2).
        count: How many words to produce (e.g. 8).

    Returns a numbered list of "word - short definition" lines, or an error
    string beginning with "ERROR:" if the level is not a valid CEFR level.
    """
    lvl = _normalise_level(level)
    if lvl not in VALID_LEVELS:
        return (
            f"ERROR: '{level}' is not a valid CEFR level. "
            f"Valid levels are {', '.join(sorted(VALID_LEVELS))}. "
            "Cannot generate vocabulary."
        )
    prompt = (
        f"You are an ESL curriculum writer. Produce exactly {count} English "
        f"vocabulary words for the topic '{topic}', appropriate for CEFR level "
        f"{lvl} learners. Each word must genuinely sit at {lvl} difficulty - "
        f"not easier, not harder. Format each as a numbered line:\n"
        f"N. word - a short, {lvl}-appropriate definition (one clause).\n"
        f"Output only the list, no preamble."
    )
    return _text(llm.invoke(prompt))


@tool
def generate_practice_questions(topic: str, level: str, vocab_list: str) -> str:
    """Generate practice questions that use the supplied vocabulary.

    Args:
        topic: The lesson topic.
        level: Target CEFR level (A1..C2).
        vocab_list: The vocabulary list produced by generate_vocabulary.

    Returns a set of practice questions, or an error string beginning with
    "ERROR:" if the level is invalid or the vocab list is missing/an error.
    """
    lvl = _normalise_level(level)
    if lvl not in VALID_LEVELS:
        return (
            f"ERROR: '{level}' is not a valid CEFR level. Cannot generate "
            "practice questions."
        )
    if not vocab_list or vocab_list.strip().startswith("ERROR:"):
        return "ERROR: No valid vocabulary list was provided; generate vocabulary first."
    prompt = (
        f"You are an ESL curriculum writer. Using ONLY these vocabulary words:\n\n"
        f"{vocab_list}\n\n"
        f"Write 5 practice questions for the topic '{topic}' at CEFR level {lvl}. "
        f"Mix formats (gap-fill, matching, short answer). Keep the language at "
        f"{lvl} level. Number the questions. Output only the questions."
    )
    return _text(llm.invoke(prompt))


@tool
def check_difficulty(
    text: str,
    target_level: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Check whether generated text actually matches a target CEFR level.

    Args:
        text: The vocabulary or questions text to assess.
        target_level: The CEFR level it is supposed to match (A1..C2).

    Returns a JSON object with keys "pass" (bool), "flagged_terms" (words above
    the level), and "reason". Any flagged terms are also accumulated in the
    lesson state so they can be disclosed to the teacher. Regenerate the content
    when "pass" is false.
    """
    result = assess_difficulty(text, target_level)
    return Command(
        update={
            "flagged_terms": result["flagged_terms"],
            "messages": [
                ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def assemble_lesson_pack(
    vocab: str,
    questions: str,
    flagged_terms: Annotated[list[str], InjectedState("flagged_terms")],
) -> str:
    """Format the vocabulary and questions into the final lesson-pack deliverable.

    Args:
        vocab: The (difficulty-checked) vocabulary list.
        questions: The (difficulty-checked) practice questions.

    Returns a formatted Markdown lesson pack. If any terms were flagged as
    above-level during difficulty checking, a "Note for teacher" section listing
    them is ALWAYS appended — the caller cannot suppress it. Only call this once
    both the vocabulary and the questions are as close to the target level as
    the retries could get them.
    """
    pack = (
        "# Lesson Pack\n\n"
        "## Vocabulary\n\n"
        f"{vocab.strip()}\n\n"
        "## Practice Questions\n\n"
        f"{questions.strip()}\n"
    )
    flagged = merge_flagged([], flagged_terms)  # de-dup defensively
    if flagged:
        pack += (
            "\n## Note for teacher\n\n"
            "These terms were flagged during generation as above the target "
            "level. Review or pre-teach them before use:\n\n"
            + "\n".join(f"- {t}" for t in flagged)
            + "\n"
        )
    return pack


TOOLS = [
    generate_vocabulary,
    generate_practice_questions,
    check_difficulty,
    assemble_lesson_pack,
]
