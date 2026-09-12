"""Medicare information assistant.

A RAG agent over the official Medicare & You 2026 handbook,
backed by an Amazon Bedrock Knowledge Base.
"""

import boto3
from strands import Agent, tool

KB_ID = "3GQOPHDIBZ"
REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"

bedrock = boto3.client("bedrock-agent-runtime", region_name=REGION)


@tool
def search_medicare_handbook(query: str) -> str:
    """Search the official Medicare & You 2026 handbook.

    Use this for any question about Medicare coverage, enrollment
    periods, costs, or plan types. Always use this before answering.

    Args:
        query: The search phrase to look up in the handbook.
    """
    try:
        response = bedrock.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "managedSearchConfiguration": {"numberOfResults": 5}
            },
        )
    except bedrock.exceptions.ClientError as exc:
        print(f"RETRIEVE FAILED: {type(exc).__name__}: {exc}")
        raise

    results = []
    for i, item in enumerate(response.get("retrievalResults", []), 1):
        text = item.get("content", {}).get("text", "")
        score = item.get("score")
        header = (
            f"[Passage {i}]"
            if score is None
            else f"[Passage {i} | relevance {score:.3f}]"
        )
        results.append(f"{header}\n{text}")

    if not results:
        return "No relevant passages found in the handbook."

    return "\n\n".join(results)


SYSTEM_PROMPT = """You are a Medicare information assistant for AARP members.

Rules:
- Answer ONLY from the Medicare & You 2026 handbook. Always call
  search_medicare_handbook before answering a Medicare question.
- If the handbook does not cover it, say so plainly. Do not guess.
- Never give personalized medical, financial, or enrollment advice
  for a specific individual's situation. Explain the general rules
  and direct them to 1-800-MEDICARE or a licensed advisor.
- Cite which passage your answer came from.
- You answer Medicare questions only. For anything outside the
  handbook's scope, decline and redirect - do not answer from
  general knowledge, even if you know the answer.
"""

agent = Agent(
    model=MODEL_ID,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_medicare_handbook],
)


if __name__ == "__main__":
    print("Medicare assistant ready. Ctrl+C to quit.\n")
    while True:
        try:
            question = input("You: ")
            if not question.strip():
                continue
            print()
            agent(question)
            print("\n" + "-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nBye.")
            break