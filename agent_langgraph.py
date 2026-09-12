"""Medicare information assistant - LangGraph implementation.

Same agent as agent.py, built as an explicit state graph rather than
a framework-managed loop. The two nodes (model, tools) and the
conditional edge between them are the agent loop, written out.
"""

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

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
    except bedrock.exceptions.ValidationException:
        # Older botocore versions only accept the vector search shape.
        try:
            response = bedrock.retrieve(
                knowledgeBaseId=KB_ID,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": 5}
                },
            )
        except bedrock.exceptions.ValidationException as exc:
            print(f"RETRIEVE FAILED: {type(exc).__name__}: {exc}")
            return "Handbook search is temporarily unavailable."

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

TOOLS = [search_medicare_handbook]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

model = ChatBedrockConverse(model=MODEL_ID, region_name=REGION).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    """Node: ask the model what to do next."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [model.invoke(messages)]}


def call_tools(state: MessagesState) -> dict:
    """Node: execute whatever tools the model asked for."""
    last = state["messages"][-1]
    outputs = []
    for call in last.tool_calls:
        result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
        outputs.append(
            ToolMessage(content=str(result), tool_call_id=call["id"])
        )
    return {"messages": outputs}


def should_continue(state: MessagesState) -> str:
    """Edge: route back to tools if the model asked for one, else finish."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", call_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", should_continue, ["tools", END])
builder.add_edge("tools", "model")

graph = builder.compile()


def ask(question: str) -> str:
    """Run one question through the graph and return the final answer."""
    result = graph.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Medicare assistant (LangGraph). Ctrl+C to quit.\n")
    while True:
        try:
            q = input("You: ")
            if not q.strip():
                continue
            print("\n" + ask(q) + "\n" + "-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nBye.")
            break