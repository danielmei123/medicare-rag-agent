"""Medicare assistant - multi-agent orchestration.

Demonstrates two patterns the single-agent version does not need:

  agents as tools   Each specialist is a fully formed agent, exposed to
                    the orchestrator as a callable tool. The orchestrator
                    does not know or care that a tool is itself an agent.

  orchestrator      A router node classifies the question and delegates
                    to exactly one specialist. It answers nothing itself.

The specialists are not three copies of the same thing. Each has a
different capability, which is what makes the separation worth its cost:

  eligibility   retrieval plus a deterministic date calculator, so
                enrollment windows are computed rather than reasoned about
  costs         retrieval with strict output rules - exact figures only,
                never an estimate, since wrong numbers do real harm here
  coverage      plain retrieval over a broad question space

For most questions the single-agent version in agent.py performs just as
well and answers faster. This exists to demonstrate the pattern and to
handle the enrollment-date case properly.
"""

import calendar
from typing import Literal

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

KB_ID = "3GQOPHDIBZ"
REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"

bedrock = boto3.client("bedrock-agent-runtime", region_name=REGION)
model = ChatBedrockConverse(model=MODEL_ID, region_name=REGION)


# ---------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------


@tool
def search_medicare_handbook(query: str) -> str:
    """Search the official Medicare & You 2026 handbook.

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

    passages = []
    for i, item in enumerate(response.get("retrievalResults", []), 1):
        text = item.get("content", {}).get("text", "")
        score = item.get("score")
        header = (
            f"[Passage {i}]"
            if score is None
            else f"[Passage {i} | relevance {score:.3f}]"
        )
        passages.append(f"{header}\n{text}")

    if not passages:
        return "No relevant passages found in the handbook."

    return "\n\n".join(passages)


@tool
def calculate_enrollment_window(birth_month: int, birth_day: int) -> str:
    """Compute the exact Initial Enrollment Period for someone turning 65.

    The window is seven months: three before the birthday month, the
    birthday month itself, and three after. A birthday on the first of
    the month shifts the whole window one month earlier.

    Args:
        birth_month: Month of birth as a number, 1 to 12.
        birth_day: Day of the month, 1 to 31.
    """
    if not 1 <= birth_month <= 12:
        return "Birth month must be between 1 and 12."
    if not 1 <= birth_day <= 31:
        return "Birth day must be between 1 and 31."

    # A first-of-month birthday is treated as the previous month for
    # Medicare purposes, which shifts the entire window back by one.
    offset = -1 if birth_day == 1 else 0
    anchor = birth_month + offset

    def month_name(n: int) -> str:
        return calendar.month_name[((n - 1) % 12) + 1]

    start = anchor - 3
    end = anchor + 3
    months = [month_name(m) for m in range(start, end + 1)]

    early = ", ".join(months[:3])
    late = ", ".join(months[4:])

    lines = [
        f"Initial Enrollment Period: {months[0]} through {months[-1]} "
        f"(7 months).",
        f"  Three months before: {early}",
        f"  Birthday month: {months[3]}",
        f"  Three months after: {late}",
        "",
        "Coverage start dates:",
        f"  Signing up in {early} - coverage begins the first day of "
        f"{months[3]}.",
    ]

    if birth_day == 1:
        lines.append(
            "  (Birthday on the 1st, so coverage starts the first day of "
            "the month before the birthday month.)"
        )

    lines.append(
        f"  Signing up in {months[3]} or later - coverage begins the "
        f"first day of the month after signing up."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Specialist agents
#
# Each is a small loop: call the model, run any tools it asks for, repeat
# until it stops asking. Wrapped as a tool below.
# ---------------------------------------------------------------------


SHARED_RULES = """
Answer only from the Medicare & You 2026 handbook. If the handbook does
not cover it, say so plainly - do not guess. Never give personalized
medical, financial, or enrollment advice for a specific individual's
situation; explain the general rules and direct them to 1-800-MEDICARE
or a licensed advisor. Cite which passage your answer came from.
"""

ELIGIBILITY_PROMPT = f"""You are a Medicare eligibility specialist.

You handle questions about who qualifies for Medicare, when they can
enroll, enrollment periods, and late enrollment penalties.

When a question involves someone's birthday, call
calculate_enrollment_window to get the exact dates rather than working
them out yourself. The calculation has edge cases - a birthday on the
first of the month shifts the whole window - and a tool gets those right
every time.
{SHARED_RULES}"""

COSTS_PROMPT = f"""You are a Medicare costs specialist.

You handle questions about premiums, deductibles, copayments,
coinsurance, and penalty amounts.

Give exact figures from the handbook. If the handbook does not state a
figure, say so rather than estimating or giving a range - a wrong number
in this domain causes real harm. Always name the year a figure applies
to, since amounts change annually.
{SHARED_RULES}"""

COVERAGE_PROMPT = f"""You are a Medicare coverage specialist.

You handle questions about what Medicare covers: services, equipment,
prescriptions, preventive care, and the differences between Parts A, B,
C and D.

Be precise about which Part covers what. The handbook's sections on
Part A and Part B sit close together and are easy to confuse.
{SHARED_RULES}"""


def _run_specialist(prompt: str, tools: list, question: str) -> str:
    """Run one specialist agent to completion and return its answer."""
    bound = model.bind_tools(tools)
    by_name = {t.name: t for t in tools}
    messages = [SystemMessage(content=prompt), HumanMessage(content=question)]

    # Bounded rather than while True: a specialist that keeps calling
    # tools is misbehaving, and should not loop forever in a web request.
    for _ in range(5):
        reply = bound.invoke(messages)
        messages.append(reply)

        if not getattr(reply, "tool_calls", None):
            return reply.content

        for call in reply.tool_calls:
            result = by_name[call["name"]].invoke(call["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    return "Unable to complete the lookup within the allowed steps."


@tool
def ask_eligibility_specialist(question: str) -> str:
    """Delegate a question about eligibility, enrollment periods, or late
    enrollment penalties to the eligibility specialist.

    Args:
        question: The user's question, passed through unchanged.
    """
    return _run_specialist(
        ELIGIBILITY_PROMPT,
        [search_medicare_handbook, calculate_enrollment_window],
        question,
    )


@tool
def ask_costs_specialist(question: str) -> str:
    """Delegate a question about premiums, deductibles, copays, or
    penalty amounts to the costs specialist.

    Args:
        question: The user's question, passed through unchanged.
    """
    return _run_specialist(COSTS_PROMPT, [search_medicare_handbook], question)


@tool
def ask_coverage_specialist(question: str) -> str:
    """Delegate a question about what Medicare covers, or about the
    differences between Parts A, B, C and D, to the coverage specialist.

    Args:
        question: The user's question, passed through unchanged.
    """
    return _run_specialist(COVERAGE_PROMPT, [search_medicare_handbook], question)


SPECIALISTS = [
    ask_eligibility_specialist,
    ask_costs_specialist,
    ask_coverage_specialist,
]
SPECIALISTS_BY_NAME = {t.name: t for t in SPECIALISTS}


# ---------------------------------------------------------------------
# Orchestrator graph
# ---------------------------------------------------------------------


ORCHESTRATOR_PROMPT = """You are the coordinator for a Medicare
information service. You do not answer questions yourself.

Your job is to pick the right specialist and pass the question to them:

  ask_eligibility_specialist  who qualifies, when to enroll, enrollment
                              periods, late enrollment penalties
  ask_costs_specialist        premiums, deductibles, copays, penalty
                              amounts, anything with a dollar figure
  ask_coverage_specialist     what is covered, plan types, differences
                              between Parts A, B, C and D

Pass the question through unchanged. Do not rephrase it.

If a question spans two areas, call both specialists and combine what
they return. If a question is not about Medicare at all, decline politely
and explain what this service covers - do not call any specialist, and do
not answer from general knowledge even if you know the answer.

Never recommend a specific plan or give advice tailored to one person's
circumstances. Direct those to 1-800-MEDICARE or a licensed advisor.

When returning a specialist's answer, keep its citations intact.
"""

orchestrator_model = model.bind_tools(SPECIALISTS)


def route(state: MessagesState) -> dict:
    """Node: the orchestrator decides which specialist should handle this."""
    messages = [SystemMessage(content=ORCHESTRATOR_PROMPT)] + state["messages"]
    return {"messages": [orchestrator_model.invoke(messages)]}


def delegate(state: MessagesState) -> dict:
    """Node: run whichever specialists the orchestrator selected."""
    last = state["messages"][-1]
    outputs = []
    for call in last.tool_calls:
        print(f"  -> delegating to {call['name']}")
        answer = SPECIALISTS_BY_NAME[call["name"]].invoke(call["args"])
        outputs.append(
            ToolMessage(content=str(answer), tool_call_id=call["id"])
        )
    return {"messages": outputs}


def should_delegate(state: MessagesState) -> Literal["delegate", "__end__"]:
    """Edge: delegate if a specialist was chosen, otherwise finish.

    An out-of-scope question ends here without touching a specialist,
    which is both correct and cheaper.
    """
    last = state["messages"][-1]
    return "delegate" if getattr(last, "tool_calls", None) else END


builder = StateGraph(MessagesState)
builder.add_node("orchestrator", route)
builder.add_node("delegate", delegate)
builder.add_edge(START, "orchestrator")
builder.add_conditional_edges("orchestrator", should_delegate)
builder.add_edge("delegate", "orchestrator")

graph = builder.compile()


def ask(question: str) -> str:
    """Run one question through the orchestrator and return the answer."""
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Medicare assistant (multi-agent). Ctrl+C to quit.\n")
    while True:
        try:
            q = input("You: ")
            if not q.strip():
                continue
            print()
            print(ask(q))
            print("\n" + "-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nBye.")
            break
