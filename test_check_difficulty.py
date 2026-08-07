"""Isolation test for the check_difficulty tool.

Feeds it text whose real level is deliberately mismatched to the label it's
told to check against, plus one genuinely-matched control. Confirms it returns
FAIL on the bad ones and PASS on the good one.

Judging runs on the local Ollama model, so this needs Ollama running (no API key
and no cost). Start it with: ollama pull qwen2.5:3b
"""

from __future__ import annotations

from my_agent.utils.tools import assess_difficulty

# (label, text, target_level_to_check_against, expected_verdict)
CASES = [
    (
        "C2 academic prose labelled A1",
        "The epistemological ramifications of quantum decoherence necessitate a "
        "fundamental reconsideration of the ontological presuppositions "
        "underpinning classical determinism.",
        "A1",
        "FAIL",
    ),
    (
        "A1 kindergarten text labelled C1",
        "I have a cat. The cat is black. I like my cat. My cat is happy. "
        "It is a good cat. The cat can run.",
        "C1",
        "FAIL",
    ),
    (
        "B2/C1 business prose labelled A2",
        "The negotiations culminated in an unprecedented consensus, "
        "notwithstanding the stakeholders' divergent priorities and deeply "
        "entrenched ideological positions.",
        "A2",
        "FAIL",
    ),
    (
        "CONTROL: genuine A2 vocab labelled A2",
        "1. hungry - when you want to eat.\n"
        "2. breakfast - the food you eat in the morning.\n"
        "3. delicious - food that tastes very good.\n"
        "4. thirsty - when you want to drink water.",
        "A2",
        "PASS",
    ),
]


def main() -> None:
    passed = 0
    for label, text, level, expected in CASES:
        result = assess_difficulty(text, level)  # {"pass", "flagged_terms", "reason"}
        got = "PASS" if result["pass"] else "FAIL"
        ok = got == expected
        passed += ok
        print(f"[{'OK ' if ok else 'XX '}] {label}")
        print(f"       checked against: {level} | expected {expected} | got {got}")
        print(f"       flagged_terms: {result['flagged_terms']}")
        print(f"       reason: {result['reason'][:140]}")
        print()

    print(f"Result: {passed}/{len(CASES)} correct  (judged locally on Ollama, no cost).")


if __name__ == "__main__":
    main()
