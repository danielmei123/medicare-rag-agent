"""Verify the Knowledge Base retrieval tool works, without the model."""

from agent import search_medicare_handbook

QUESTIONS = [
    "What are the Medicare enrollment periods when I turn 65?",
    "What does Medicare Part B cover?",
    "How much is the Part B premium?",
    "What is a Medicare Advantage plan?",
]

for q in QUESTIONS:
    print("=" * 70)
    print("Q:", q)
    print("=" * 70)
    print(search_medicare_handbook(q)[:1200])
    print()