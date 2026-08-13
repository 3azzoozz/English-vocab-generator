"""Eval harness for the Lesson Prep Agent.

Runs a small scenario set, saves the full trajectory for each run, and scores:
  * task success      — did it produce a usable lesson pack (or correctly refuse)?
  * tool-call order    — right tools, in a sensible sequence?
  * difficulty caught  — did check_difficulty actually flag anything off-level?

Each scenario declares its EXPECTED tool-call sequence up front (written before
running), so we can compare intent vs. behaviour.

Economy: the planner runs on Claude Haiku 4.5 (billed); generation and judging
run on a local Ollama model (free). Only Claude tokens are priced below.

Requires ANTHROPIC_API_KEY in the environment (or .env) and a running Ollama.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback

from my_agent.agent import run
from my_agent.utils.nodes import PLANNER_MODEL
from my_agent.utils.observability import flush_langfuse, langfuse_enabled
from my_agent.utils.tools import OLLAMA_MODEL

OUT_DIR = Path(__file__).parent / "eval_runs"

# Haiku 4.5 pricing, USD per token (input $1 / output $5 per million).
# Applied only to Claude calls; local Ollama calls are free.
PRICE_IN = 1.0 / 1_000_000
PRICE_OUT = 5.0 / 1_000_000


# --------------------------------------------------------------------------- #
# Scenarios — expected tool sequences written BEFORE running.                  #
# --------------------------------------------------------------------------- #
# expect_pack: True  -> a usable lesson pack should be produced.
#              False -> the agent should flag/refuse and NOT produce a pack.

SCENARIOS = [
    {
        "name": "1_food_A2",
        "topic": "food vocabulary",
        "level": "A2",
        "expected_tools": [
            "generate_vocabulary",
            "generate_practice_questions",
            "check_difficulty",
            "check_difficulty",
            "assemble_lesson_pack",
        ],
        "expect_pack": True,
    },
    {
        "name": "2_travel_B1",
        "topic": "travel and transport vocabulary",
        "level": "B1",
        "expected_tools": [
            "generate_vocabulary",
            "generate_practice_questions",
            "check_difficulty",
            "check_difficulty",
            "assemble_lesson_pack",
        ],
        "expect_pack": True,
    },
    {
        "name": "3_business_idioms_C1",
        "topic": "business idioms",
        "level": "C1",
        "expected_tools": [
            "generate_vocabulary",
            "generate_practice_questions",
            "check_difficulty",
            "check_difficulty",
            "assemble_lesson_pack",
        ],
        "expect_pack": True,
    },
    {
        # Invalid CEFR level -> should be flagged/refused, no pack.
        "name": "4_invalid_level_Z9",
        "topic": "animal vocabulary",
        "level": "Z9",
        "expected_tools": [],  # ideally refuses before any content tool
        "expect_pack": False,
    },
    {
        # Topic far too advanced for the requested level -> should be flagged.
        "name": "5_mismatch_quantum_A1",
        "topic": "quantum field theory terminology",
        "level": "A1",
        "expected_tools": [],  # agent should flag the mismatch rather than build
        "expect_pack": False,
    },
    {
        # Deliberate difficulty-miss trap: a teachable topic (so the agent won't
        # refuse it outright) whose natural vocabulary — plaintiff, verdict,
        # subpoena — sits well above A2. The point is to see whether
        # check_difficulty flags the first draft and the agent REGENERATES,
        # or whether it plows straight through to assembly with off-level words.
        "name": "6_legal_A2_trap",
        "topic": "legal and courtroom vocabulary",
        "level": "A2",
        "expected_tools": [
            "generate_vocabulary",
            "check_difficulty",
            # ideally: FAIL here -> a second generate_vocabulary (regeneration)
        ],
        "expect_pack": True,  # a usable A2 pack, ideally after a retry
    },
]


# --------------------------------------------------------------------------- #
# Trajectory extraction + scoring.                                            #
# --------------------------------------------------------------------------- #


def _msg_type(m) -> str:
    return getattr(m, "type", m.__class__.__name__)


def extract_tool_sequence(messages) -> list[str]:
    """Ordered list of tool names the agent actually called."""
    seq = []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            seq.append(call["name"])
    return seq


def tool_outputs(messages) -> list[tuple[str, str]]:
    """(tool_name, output_text) pairs, in order, from ToolMessages."""
    out = []
    for m in messages:
        if _msg_type(m) == "tool":
            out.append((getattr(m, "name", "?"), str(m.content)))
    return out


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if every expected tool appears in `actual` in the same order
    (extra tools in between are allowed — e.g. a regeneration pass)."""
    it = iter(actual)
    return all(tool in it for tool in expected)


def render_trajectory(scenario: dict, messages) -> str:
    lines = [
        f"=== Scenario: {scenario['name']} ===",
        f"Topic: {scenario['topic']} | Level: {scenario['level']}",
        f"Expected tools: {scenario['expected_tools'] or '(refuse / flag — no build)'}",
        "",
    ]
    for i, m in enumerate(messages):
        kind = _msg_type(m)
        lines.append(f"--- [{i}] {kind} ---")
        calls = getattr(m, "tool_calls", None)
        if calls:
            for c in calls:
                lines.append(f"  TOOL CALL: {c['name']}({json.dumps(c['args'])[:200]})")
        content = m.content
        if isinstance(content, list):
            content = json.dumps(content)[:1500]
        if content:
            lines.append(f"  {str(content)[:1500]}")
        lines.append("")
    return "\n".join(lines)


def score(scenario: dict, messages) -> dict:
    actual = extract_tool_sequence(messages)
    outs = tool_outputs(messages)

    produced_pack = any(name == "assemble_lesson_pack" for name, _ in outs)
    any_error = any(txt.strip().startswith("ERROR:") for _, txt in outs)

    # check_difficulty now returns a JSON object {"pass", "flagged_terms", "reason"}.
    difficulty_runs = difficulty_fails = 0
    for name, txt in outs:
        if name != "check_difficulty":
            continue
        difficulty_runs += 1
        try:
            failed = json.loads(txt).get("pass") is False
        except Exception:
            failed = txt.strip().upper().startswith("FAIL")  # legacy fallback
        difficulty_fails += int(failed)

    # Regeneration = a content-generation tool called more than once, i.e. the
    # agent didn't accept its first draft. This is the self-correction signal.
    vocab_calls = sum(1 for t in actual if t == "generate_vocabulary")
    question_calls = sum(1 for t in actual if t == "generate_practice_questions")
    regenerated = vocab_calls > 1 or question_calls > 1

    # Task success:
    #   pack expected -> a pack was actually produced
    #   pack NOT expected -> agent correctly did NOT produce a pack (flag/refuse)
    if scenario["expect_pack"]:
        task_success = produced_pack
    else:
        task_success = not produced_pack

    tool_order_ok = is_subsequence(scenario["expected_tools"], actual)

    return {
        "actual_tools": actual,
        "task_success": task_success,
        "tool_order_ok": tool_order_ok,
        "produced_pack": produced_pack,
        "any_error_surfaced": any_error,
        "difficulty_runs": difficulty_runs,
        "difficulty_fails": difficulty_fails,
        "regenerated": regenerated,
    }


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY in your environment first.")

    OUT_DIR.mkdir(exist_ok=True)

    # Optional CLI filter: `python run_eval.py 6_` runs only matching scenarios.
    selectors = sys.argv[1:]
    scenarios = SCENARIOS
    if selectors:
        scenarios = [s for s in SCENARIOS if any(sel in s["name"] for sel in selectors)]
        if not scenarios:
            raise SystemExit(f"No scenarios match {selectors}")

    print(
        f"Planner: {PLANNER_MODEL} (billed) | "
        f"Generation/judge: {OLLAMA_MODEL} (local, free)"
    )
    print(f"Langfuse tracing: {'on' if langfuse_enabled() else 'off'}")
    print(f"Saving trajectories to: {OUT_DIR}")
    print(f"Running {len(scenarios)} of {len(SCENARIOS)} scenarios.\n")

    summary = []
    total_billable_in = total_billable_out = 0

    for sc in scenarios:
        print(f"Running {sc['name']} ...", flush=True)
        with get_usage_metadata_callback() as usage_cb:
            try:
                final = run(sc["topic"], sc["level"])
                messages = final["messages"]
                flagged_final = final.get("flagged_terms", [])
                error = None
            except Exception as e:  # recursion limit, API error, etc.
                messages = []
                flagged_final = []
                error = f"{type(e).__name__}: {e}"

        # Usage is keyed by model name. Only Claude (planner) calls are billed;
        # local Ollama calls, if they report usage at all, are free.
        billable_in = billable_out = 0
        for model_name, u in usage_cb.usage_metadata.items():
            if model_name.startswith("claude"):
                billable_in += u.get("input_tokens", 0)
                billable_out += u.get("output_tokens", 0)
        total_billable_in += billable_in
        total_billable_out += billable_out
        cost = billable_in * PRICE_IN + billable_out * PRICE_OUT

        if error:
            result = {"error": error, "task_success": False}
            traj = f"=== Scenario: {sc['name']} ===\nERROR: {error}\n"
        else:
            result = score(sc, messages)
            result["flagged_terms_final"] = flagged_final
            traj = render_trajectory(sc, messages)
            traj += f"\n--- Accumulated flagged terms: {flagged_final} ---\n"

        traj += (
            f"--- Billable Claude tokens: in={billable_in} out={billable_out} "
            f"(~${cost:.5f}) ---\n"
        )
        (OUT_DIR / f"{sc['name']}.txt").write_text(traj, encoding="utf-8")

        row = {"name": sc["name"], "cost_usd": round(cost, 5), **result}
        summary.append(row)
        print(f"  -> {json.dumps(row)}")

    total_cost = total_billable_in * PRICE_IN + total_billable_out * PRICE_OUT

    # Aggregate scorecard.
    scored = [r for r in summary if "error" not in r]
    print("\n================ SCORECARD ================")
    for r in summary:
        if "error" in r:
            print(f"{r['name']:24} ERROR: {r['error']}")
            continue
        ts = "PASS" if r["task_success"] else "FAIL"
        to = "ok" if r["tool_order_ok"] else "off"
        regen = "yes" if r.get("regenerated") else "no"
        print(
            f"{r['name']:24} task={ts:4} tools={to:3} "
            f"pack={r['produced_pack']!s:5} "
            f"diff_checks={r['difficulty_runs']} diff_fails={r['difficulty_fails']} "
            f"regen={regen:3} ${r['cost_usd']}"
        )
    if scored:
        succ = sum(1 for r in scored if r["task_success"])
        order = sum(1 for r in scored if r["tool_order_ok"])
        caught = sum(r["difficulty_fails"] for r in scored)
        print("-------------------------------------------")
        print(f"Task success:     {succ}/{len(scored)}")
        print(f"Tool order ok:    {order}/{len(scored)}")
        print(f"Off-level catches (check_difficulty FAILs): {caught}")
    print(
        f"Total billable Claude tokens: in={total_billable_in} "
        f"out={total_billable_out}  ~${total_cost:.5f}"
    )

    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "planner_model": PLANNER_MODEL,
                "local_model": OLLAMA_MODEL,
                "total_cost_usd": round(total_cost, 5),
                "runs": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {OUT_DIR / 'summary.json'} and per-scenario trajectories.")

    flush_langfuse()  # send any pending traces before exit


if __name__ == "__main__":
    main()
