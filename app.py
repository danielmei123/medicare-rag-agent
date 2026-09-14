"""Web wrapper around the Medicare agent.

Serves a single-page front end and two endpoints. /ask runs the single
Strands agent; /ask-orchestrated routes the question through the
LangGraph orchestrator to a specialist. The same file runs locally under
uvicorn, on Lambda via Mangum, and in a Kubernetes pod; the DEPLOYMENT
environment variable is what distinguishes them on screen.

Each response carries its token usage and an estimated cost, because in a
RAG system the retrieved passages dominate the input and that is the lever
that actually moves the bill.
"""

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from strands import Agent

from agent import MODEL_ID, SYSTEM_PROMPT, search_medicare_handbook
from agent_orchestrator import graph as orchestrator_graph

app = FastAPI(title="Medicare Assistant")

DEPLOYMENT = os.environ.get("DEPLOYMENT", "Local (uvicorn)")

# Published on-demand rates for Claude Sonnet on Bedrock, in USD per
# million tokens. Kept as configuration rather than hardcoded because
# they change; override with env vars rather than editing code.
INPUT_USD_PER_MTOK = float(os.environ.get("INPUT_USD_PER_MTOK", "3.00"))
OUTPUT_USD_PER_MTOK = float(os.environ.get("OUTPUT_USD_PER_MTOK", "15.00"))

# Claude Sonnet's context window. Shown as a percentage so the headroom
# is visible: a RAG agent that retrieves five passages per call and loops
# for tool use can grow its context faster than it looks like it should.
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "200000"))


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Set the response headers the ZAP baseline scan found missing.

    FastAPI sets none of these by default. The first DAST run reported
    three Medium findings - no CSP, no anti-clickjacking header, and a
    CDN script without integrity - plus several Low ones for the
    cross-origin policy headers.
    """
    response = await call_next(request)

    # 'unsafe-inline' for scripts is a real weakening: the page has an
    # inline <script> block, and a strict policy would break it. Moving
    # that code to a served file would let this be dropped. Recorded as
    # an accepted finding in .zap/rules.tsv.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "base-uri 'self'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=()"
    )
    return response


class Question(BaseModel):
    question: str


def _extract_usage(result) -> dict:
    """Pull token counts out of whatever shape the framework returns.

    Deliberately defensive. Usage lives in different places across
    framework versions, and losing the answer because the accounting
    changed shape would be a poor trade. Returns zeros if nothing is
    found, and the UI hides the panel in that case.
    """
    candidates = []

    metrics = getattr(result, "metrics", None)
    if metrics is not None:
        candidates.append(getattr(metrics, "accumulated_usage", None))
        candidates.append(getattr(metrics, "usage", None))
    candidates.append(getattr(result, "usage", None))

    for usage in candidates:
        if usage is None:
            continue
        if not isinstance(usage, dict):
            usage = getattr(usage, "__dict__", None)
        if not isinstance(usage, dict):
            continue

        # Bedrock uses inputTokens/outputTokens; some wrappers use
        # snake_case or the OpenAI-style prompt/completion naming.
        input_tokens = (
            usage.get("inputTokens")
            or usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens = (
            usage.get("outputTokens")
            or usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        if input_tokens or output_tokens:
            return {
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
            }

    return {"input_tokens": 0, "output_tokens": 0}


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Estimated cost of one exchange at the configured rates."""
    return (
        input_tokens / 1_000_000 * INPUT_USD_PER_MTOK
        + output_tokens / 1_000_000 * OUTPUT_USD_PER_MTOK
    )


def _text(content) -> str:
    """Flatten Converse-style content blocks to a plain string.

    LangChain's Bedrock Converse wrapper returns content as a list of
    blocks rather than a bare string, unlike the Strands path.
    """
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def _usage_payload(input_tokens: int, output_tokens: int) -> dict:
    """Build the usage block returned alongside every answer."""
    total = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total,
        "estimated_cost_usd": round(_cost_usd(input_tokens, output_tokens), 5),
        "context_window": CONTEXT_WINDOW,
        "context_used_pct": round(total / CONTEXT_WINDOW * 100, 2),
    }


@app.post("/ask")
def ask(q: Question):
    # Fresh agent per request - no shared conversation state between users.
    agent = Agent(
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_medicare_handbook],
    )
    result = agent(q.question)

    usage = _extract_usage(result)

    return {
        "answer": str(result),
        "routed_to": [],
        "usage": _usage_payload(usage["input_tokens"], usage["output_tokens"]),
    }


@app.post("/ask-orchestrated")
def ask_orchestrated(q: Question):
    """Route the question through the multi-agent orchestrator.

    Slower than /ask by design: the orchestrator makes its own model call
    to choose a specialist before the specialist does any work.

    The token counts here cover the orchestrator's own calls only. A
    specialist runs its loop inside a tool, outside the graph state, so
    its usage never reaches these messages - the true cost of this path
    is higher than the figure shown.
    """
    result = orchestrator_graph.invoke(
        {"messages": [HumanMessage(content=q.question)]}
    )
    messages = result["messages"]

    routed = []
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            routed.append(call["name"])
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    return {
        "answer": _text(messages[-1].content),
        "routed_to": routed,
        "usage": _usage_payload(input_tokens, output_tokens),
    }


@app.get("/health")
def health():
    return {"status": "ok", "deployment": DEPLOYMENT}


# Pinned to an exact version with a subresource integrity hash. The
# unpinned URL the first version used meant a compromised CDN could have
# served arbitrary JavaScript; the browser now refuses anything whose
# hash does not match. Hash computed from the published npm artifact.
PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Medicare Assistant</title>
<script
  src="https://cdn.jsdelivr.net/npm/marked@16.4.1/lib/marked.umd.js"
  integrity="sha384-vR7TM/dokKkSOM5kqHVxdEfrVBXCpkhkl4hU++wkChheZe7/s629Wvv+vG94uKd4"
  crossorigin="anonymous"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px;
         margin: 40px auto 80px; padding: 0 20px; line-height: 1.6; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .sub { color: #666; font-size: 0.9rem; margin-bottom: 24px; }
  textarea { width: 100%; padding: 12px; font-size: 1rem;
             font-family: inherit; border: 1px solid #ccc;
             border-radius: 6px; box-sizing: border-box; }
  button { margin-top: 10px; padding: 10px 20px; font-size: 1rem;
           background: #0b5cad; color: white; border: none;
           border-radius: 6px; cursor: pointer; }
  button:disabled { background: #999; cursor: default; }
  .mode { margin-left: 14px; font-size: 0.9rem; color: #555; }
  #out { margin-top: 28px; }
  #out h2 { font-size: 1.1rem; margin-top: 24px; }
  #out table { border-collapse: collapse; margin: 12px 0; }
  #out th, #out td { border: 1px solid #ddd; padding: 6px 10px;
                     text-align: left; }
  .ex { color: #0b5cad; cursor: pointer; text-decoration: underline; }
  #usage { margin-top: 24px; padding: 10px 14px; background: #fafafa;
           border: 1px solid #e2e2e2; border-radius: 6px;
           font-size: 0.82rem; color: #555; display: none; }
  #usage b { color: #24486e; }
  .note { color: #888; }
  /* Pinned to the bottom of the viewport so it is visible no matter how
     long the answer runs - the three deployments look identical
     otherwise. */
  .deployment-footer {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #eef3f9; border-top: 1px solid #c5d6e8;
    padding: 8px 20px; font-size: 0.8rem; color: #24486e;
    text-align: center; font-weight: 600; letter-spacing: 0.02em;
  }
</style>
</head>
<body>
<h1>Medicare Assistant</h1>
<div class="sub">
  Answers from the official CMS <em>Medicare &amp; You 2026</em> handbook.
  General information only &mdash; not personal advice.
</div>

<textarea id="q" rows="3" placeholder="Ask a Medicare question..."></textarea>
<div>
  <button id="go" onclick="ask()">Ask</button>
  <label class="mode">
    <input type="checkbox" id="multi"> Multi-agent orchestration
  </label>
</div>

<div class="sub" style="margin-top:16px">
  Try: <span class="ex" onclick="fill(this)">I'm turning 65 in March. What are my enrollment options?</span>
</div>

<div id="out"></div>
<div id="usage"></div>

<div class="deployment-footer">Serving from: __DEPLOYMENT__</div>

<script>
function fill(el) { document.getElementById('q').value = el.textContent.trim(); }

function render(text) {
  // The UMD build exposes either marked.parse or marked itself,
  // depending on version. Fall back to plain text if neither is there -
  // an integrity mismatch would leave the script unloaded.
  if (window.marked && typeof window.marked.parse === 'function') {
    return window.marked.parse(text);
  }
  if (typeof window.marked === 'function') {
    return window.marked(text);
  }
  return null;
}

function showUsage(u, seconds, routed) {
  const el = document.getElementById('usage');
  if (!u || !u.total_tokens) { el.style.display = 'none'; return; }
  let html =
    '<b>' + u.input_tokens.toLocaleString() + '</b> input + ' +
    '<b>' + u.output_tokens.toLocaleString() + '</b> output = ' +
    '<b>' + u.total_tokens.toLocaleString() + '</b> tokens &middot; ' +
    'est. <b>$' + u.estimated_cost_usd.toFixed(5) + '</b> &middot; ' +
    u.context_used_pct + '% of the ' +
    (u.context_window / 1000) + 'k context window &middot; ' +
    seconds.toFixed(1) + 's';
  if (routed && routed.length) {
    html = '<b>Routed to:</b> ' + routed.join(', ') +
           ' <span class="note">(orchestrator tokens only &mdash; ' +
           'specialist usage runs inside a tool and is not counted)' +
           '</span><br>' + html;
  }
  el.innerHTML = html;
  el.style.display = 'block';
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('go');
  const out = document.getElementById('out');
  const multi = document.getElementById('multi').checked;
  btn.disabled = true;
  document.getElementById('usage').style.display = 'none';
  out.textContent = multi
    ? 'Routing to a specialist...'
    : 'Searching the handbook...';
  const started = performance.now();
  try {
    const r = await fetch(multi ? '/ask-orchestrated' : '/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await r.json();
    const html = render(data.answer);
    if (html === null) {
      out.style.whiteSpace = 'pre-wrap';
      out.textContent = data.answer;
    } else {
      out.style.whiteSpace = 'normal';
      out.innerHTML = html;
    }
    showUsage(data.usage, (performance.now() - started) / 1000,
              data.routed_to);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
  btn.disabled = false;
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.replace("__DEPLOYMENT__", DEPLOYMENT)
