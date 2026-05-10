"""
test_traces.py

Runs the agent against the 10 sample conversation traces and
checks that our responses are reasonable.

This isn't a full automated eval — it's a sanity check to run
locally before deploying. You can see what the agent says and
catch obvious failures like wrong URLs or hallucinations.

Run with:  python test_traces.py
"""

import json
import time

# We test the agent module directly, not via HTTP
from agent import chat


def run_trace(name: str, turns: list[tuple[str, dict | None]]):
    """
    Simulate a conversation turn by turn.

    turns is a list of (user_message, expected_info) pairs.
    expected_info can check things like whether recommendations
    should be empty or not.
    """
    print(f"\n{'='*60}")
    print(f"TRACE: {name}")
    print('='*60)

    messages = []
    for i, (user_msg, expected) in enumerate(turns, 1):
        messages.append({"role": "user", "content": user_msg})

        t0 = time.time()
        result = chat(messages)
        elapsed = time.time() - t0

        print(f"\n[Turn {i}] User: {user_msg[:80]}")
        print(f"[Turn {i}] Reply: {result['reply'][:150]}")
        print(f"[Turn {i}] Recommendations: {len(result['recommendations'])} items")
        print(f"[Turn {i}] EOC: {result['end_of_conversation']} | Time: {elapsed:.1f}s")

        if result['recommendations']:
            for r in result['recommendations']:
                print(f"          - {r['name']} ({r['test_type']}) {r['url'][:60]}")

        # basic assertions
        if expected:
            if expected.get("recs_empty"):
                assert result["recommendations"] == [], \
                    f"Expected no recommendations on turn {i} but got {len(result['recommendations'])}"
            if expected.get("recs_not_empty"):
                assert result["recommendations"], \
                    f"Expected recommendations on turn {i} but got none"
            if expected.get("eoc_true"):
                assert result["end_of_conversation"], \
                    f"Expected end_of_conversation=true on turn {i}"

        # add assistant message to history for next turn
        messages.append({"role": "assistant", "content": result["reply"]})

        # stop if eoc
        if result["end_of_conversation"]:
            break

    print(f"\n[Done] Final recommendations: {len(result['recommendations'])}")
    return result


def test_vague_query():
    """Turn 1 with a vague query should NOT give recommendations."""
    run_trace("Vague query — should clarify first", [
        ("I need an assessment", {"recs_empty": True}),
        ("We're hiring a data scientist, mid-level", {"recs_not_empty": True}),
    ])


def test_java_developer():
    """C9-style: JD provided, agent asks one clarification, then recommends."""
    run_trace("Java developer with JD", [
        ("Hiring a Java developer who works with stakeholders", {"recs_empty": True}),
        ("Mid-level, around 4 years experience", {"recs_not_empty": True}),
    ])


def test_refine():
    """Adding a test type mid-conversation should update, not restart."""
    run_trace("Refine shortlist mid-conversation", [
        ("We need tests for a Python backend engineer, senior level", {}),
        ("Actually, also add a personality test", {}),
        ("That looks good", {"eoc_true": True}),
    ])


def test_offtopic_refusal():
    """Off-topic questions should be refused."""
    run_trace("Off-topic refusal", [
        ("What salary should I offer a Java developer?", {"recs_empty": True}),
    ])


def test_legal_refusal():
    """Legal questions should be refused (C7 pattern)."""
    run_trace("Legal refusal", [
        ("Are we legally required under HIPAA to test all staff?", {"recs_empty": True}),
    ])


def test_compare():
    """Comparison question should answer from catalog, re-show shortlist."""
    run_trace("Compare two assessments (C5 pattern)", [
        ("Hiring sales reps, need personality and skills assessment", {}),
        ("What is the difference between OPQ32r and the GSA?", {}),
        ("Got it, keep the original shortlist", {"eoc_true": True}),
    ])


if __name__ == "__main__":
    print("Running agent tests against sample conversation patterns...")
    print("(These require GEMINI_API_KEY to be set)\n")

    test_vague_query()
    test_java_developer()
    test_refine()
    test_offtopic_refusal()
    test_legal_refusal()
    test_compare()

    print("\n\nAll traces completed.")
