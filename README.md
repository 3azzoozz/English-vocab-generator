# Lesson Prep Agent

A LangGraph agent that turns a **topic + CEFR level** (e.g. "food vocabulary, A2")
into a complete English vocabulary lesson pack. The agent decides its own steps.

## Architecture

A **hybrid** setup, to balance cost and reliability:

- **Planner** (decides which tool to call) → **Claude Haiku 4.5**, because
  tool-calling needs to be reliable.
- **Generation & difficulty-judging** (the heavy lifting inside the tools) →
  a **local Ollama model** (`qwen2.5:3b`), which is free and private.

Given a topic and level, the agent autonomously:

1. **Generates a vocabulary list** for the topic at the right level
2. **Generates practice questions** using that vocabulary
3. **Checks difficulty** — does the content match the level?
4. **Regenerates if off**, then **assembles** the final lesson pack

### The four tools (one job each)

| Tool | Purpose |
|---|---|
| `generate_vocabulary(topic, level, count)` | Word list + short definitions |
| `generate_practice_questions(topic, level, vocab_list)` | Practice questions using the vocab |
| `check_difficulty(text, target_level)` | Returns structured `{"pass", "flagged_terms", "reason"}`; accumulates flagged terms into state |
| `assemble_lesson_pack(vocab, questions)` | Formats the deliverable (pure Python); always appends a **Note for teacher** when terms were flagged |

The **agent node** chooses which tool to call from the message state. The
**routing function** just checks: did the last message include a tool call?
If yes → run it; if no → the agent is done.

### Guaranteed teacher disclosure

`check_difficulty` writes every above-level term it flags into a `flagged_terms`
list in graph state, via a de-duplicating reducer, so flags **accumulate across
retries**. `assemble_lesson_pack` reads that list from **injected state**
(`InjectedState`), not from an LLM-supplied argument — the field is hidden from
the model's tool schema. So the "Note for teacher" section is rendered by the
tool itself; the LLM cannot omit or suppress it.

## Project structure

```
.
├── my_agent/                 # all project code
│   ├── __init__.py
│   ├── agent.py              # builds the graph; exposes `graph` and `run()`
│   └── utils/
│       ├── __init__.py
│       ├── state.py          # LessonState + the flagged-terms reducer
│       ├── tools.py          # the 4 tools (+ local Ollama model)
│       └── nodes.py          # agent_node, route (+ Claude planner)
├── run_eval.py               # 6-scenario eval harness + scorecard
├── test_check_difficulty.py  # isolation test for the difficulty judge
├── langgraph.json            # LangGraph config (points to my_agent/agent.py:graph)
├── requirements.txt
├── .env.example              # copy to .env and fill in your key
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

**1. Claude (planner).** Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
# then edit .env:  ANTHROPIC_API_KEY=sk-ant-...
```

**2. Ollama (generation/judging).** Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:3b
```

Ollama must be running when you run the agent.

**3. Langfuse (optional observability).** Add your Langfuse keys to `.env`:

```bash
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

If these are unset, the agent simply runs untraced — no errors.

## Observability

When Langfuse keys are present, every `run()` is traced to Langfuse: the planner
(Claude) calls and the tools' local Ollama calls both appear, nested under one
trace per lesson pack, tagged with the topic and CEFR level. The eval harness
prints `Langfuse tracing: on/off` and flushes traces on exit.

The wiring lives in [`my_agent/utils/observability.py`](my_agent/utils/observability.py)
and degrades gracefully — a missing SDK or bad key disables tracing rather than
breaking the run. `LANGFUSE_BASE_URL` (or the SDK's native `LANGFUSE_HOST`) both
work.

## Run

One lesson pack:

```bash
python -m my_agent.agent
```

The eval (6 scenarios, trajectories + scorecard):

```bash
python run_eval.py            # all scenarios
python run_eval.py 6_         # just the difficulty-trap scenario
python run_eval.py 1_ 2_      # scenarios 1 and 2
```

The difficulty judge in isolation (local, no cost):

```bash
python test_check_difficulty.py
```

### LangGraph Studio / dev server

Because the repo follows the standard LangGraph layout with `langgraph.json`,
you can also run it in the LangGraph dev server:

```bash
pip install "langgraph-cli[inmem]"
langgraph dev
```

## Eval

`run_eval.py` runs 6 scenarios, each with its **expected tool-call sequence
declared before running**:

| # | Scenario | Expectation |
|---|---|---|
| 1 | food vocabulary, A2 | build a pack |
| 2 | travel & transport, B1 | build a pack |
| 3 | business idioms, C1 | build a pack |
| 4 | animal vocabulary, **Z9** (invalid level) | **flag / refuse** — no pack |
| 5 | quantum field theory terms, **A1** (topic too advanced) | **flag** the mismatch |
| 6 | legal & courtroom vocabulary, **A2** (difficulty trap) | draft too hard → FAIL → **regenerate** → pack |

It scores **task success**, **tool-call order**, and **off-level catches**
(`check_difficulty` FAILs — the best signal the agent isn't just going through
the motions), writes per-scenario trajectories to `eval_runs/`, and reports
billable Claude tokens (Ollama calls are free).

## Cost

Only the **Claude planner** calls are billed (Haiku 4.5: $1 / $5 per M tokens
in/out). All generation and judging run locally on Ollama at no cost, so a full
eval run is a fraction of a cent in API spend.
