"""Web wrapper around the Medicare agent.

Serves a single-page front end and one /ask endpoint. The same file runs
locally under uvicorn, on Lambda via Mangum, and in a Kubernetes pod; the
DEPLOYMENT environment variable is what distinguishes them on screen.
"""

import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from strands import Agent

from agent import MODEL_ID, SYSTEM_PROMPT, search_medicare_handbook

app = FastAPI(title="Medicare Assistant")

DEPLOYMENT = os.environ.get("DEPLOYMENT", "Local (uvicorn)")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Set the response headers the ZAP baseline scan found missing.

    FastAPI sets none of these by default. The first DAST run reported
    three Medium findings - no CSP, no anti-clickjacking header, and a
    CDN script without integrity - plus several Low ones for the
    cross-origin policy headers. This closes all of them except the
    inline-script allowance noted below.
    """
    response = await call_next(request)

    # 'unsafe-inline' for scripts is a real weakening: the page has an
    # inline <script> block, and a strict policy would break it. Moving
    # that code to a served file would let this be dropped. Styles are
    # inline for the same reason.
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


@app.post("/ask")
def ask(q: Question):
    # Fresh agent per request - no shared conversation state between users.
    agent = Agent(
        model=MODEL_ID,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_medicare_handbook],
    )
    result = agent(q.question)
    return {"answer": str(result)}


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
  #out { margin-top: 28px; }
  #out h2 { font-size: 1.1rem; margin-top: 24px; }
  #out table { border-collapse: collapse; margin: 12px 0; }
  #out th, #out td { border: 1px solid #ddd; padding: 6px 10px;
                     text-align: left; }
  .ex { color: #0b5cad; cursor: pointer; text-decoration: underline; }
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
<button id="go" onclick="ask()">Ask</button>

<div class="sub" style="margin-top:16px">
  Try: <span class="ex" onclick="fill(this)">I'm turning 65 in March. What are my enrollment options?</span>
</div>

<div id="out"></div>

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

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('go');
  const out = document.getElementById('out');
  btn.disabled = true;
  out.textContent = 'Searching the handbook...';
  try {
    const r = await fetch('/ask', {
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
