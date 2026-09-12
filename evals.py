"""Retrieval evaluation for the Medicare assistant.

Each case names a phrase that must appear in the retrieved passages.
Tests the retrieval layer only, so it runs without model access.
"""

from agent import search_medicare_handbook

CASES = [
    # Specific facts - a topical-but-wrong passage won't contain these
    ("What is the standard Part B premium in 2026?", "202.90"),
    ("How long is the Initial Enrollment Period?", "7-month"),
    ("What is the Part A late enrollment penalty percentage?", "10%"),
    ("How long is the Medigap open enrollment period?", "6-month"),
    ("What is the insulin cost cap?", "$35"),
    ("When does General Enrollment run?", "January 1"),
    ("How many days of skilled nursing care before a new benefit period?", "60 days"),
    ("How many blood glucose screenings per year?", "2"),

    # Distractor-prone - Part A content often outranks Part B
    ("What does Part B cover?", "outpatient"),
    ("What does Part A cover?", "inpatient"),

    # Cross-reference targets - tests whether page-referenced content is reachable
    ("What help is available paying Medicare premiums?", "Medicare Savings"),
]

passed = 0
for question, expected in CASES:
    passages = search_medicare_handbook(question)
    hit = expected.lower() in passages.lower()
    passed += hit
    print(f"{'PASS' if hit else 'FAIL'}  {question}")
    if not hit:
        print(f"       expected to find: {expected!r}")

print(f"\n{passed}/{len(CASES)} passed")