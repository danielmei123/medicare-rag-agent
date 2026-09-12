# Medicare Assistant

A retrieval-augmented question-answering agent over the official CMS
*Medicare & You 2026* handbook. Members ask general Medicare questions;
the agent retrieves relevant handbook passages and answers with citations,
declining anything that would amount to personal advice.

**Live demo:** https://ywjdj7lgudrhd5uaoiqf6o4ktq0xfxti.lambda-url.us-east-1.on.aws/

---

## Why this, and why this way

The hard problem in a member-facing Medicare assistant is not fluency —
it is being wrong with confidence. Enrollment windows, premium amounts and
penalty rules are specific, dated and consequential, and a model answering
from general knowledge will produce plausible text that is quietly out of
date. Everything here is built around that constraint: the agent answers
only from the handbook, cites what it used, and says so plainly when the
handbook does not cover something.

The second constraint is liability. Recommending a specific plan to a
specific person is plan-steering, and a member-facing product cannot do it.
The agent refuses those questions and redirects to 1-800-MEDICARE or a SHIP
counselor.

## Architecture

```
Browser
   |
   v
Lambda Function URL  (public, no auth)
   |
   v
Container image  (ECR)  -->  FastAPI + Mangum
   |
   v
Strands agent  -->  Claude Sonnet 4.6 on Bedrock
   |
   v
search_medicare_handbook tool
   |
   v
Bedrock Knowledge Base  (managed, 128-page handbook)
```

| Component | Choice | Reason |
|---|---|---|
| Agent framework | Strands | Native Bedrock integration; the tool-calling loop is the point, not hand-rolled orchestration |
| Model | `us.anthropic.claude-sonnet-4-6` | Cross-region inference profile; on-demand throughput is not supported for the bare model ID |
| Retrieval | Bedrock Knowledge Base, managed storage | No vector store to operate for a prototype |
| Hosting | Lambda container image | 75+ dependencies exceed the zip limit; scales to zero between requests |
| Web layer | FastAPI + Mangum | Mangum adapts the Lambda event format to ASGI, so the same app runs locally under uvicorn |

## Repository contents

| File | Purpose |
|---|---|
| `agent.py` | The agent: retrieval tool, system prompt, agent construction |
| `app.py` | FastAPI wrapper and the single-page front end |
| `lambda_handler.py` | Lambda entry point via Mangum |
| `Dockerfile` | Container image on the AWS Lambda Python 3.12 base |
| `test_retrieve.py` | Exercises retrieval alone, without model access |
| `get_kb.py`, `check_models.py` | Small boto3 utilities used during setup |

## Running locally

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. Requires AWS credentials with Bedrock access
(`aws configure`) and a Knowledge Base ID set in `agent.py`.

## Deploying

```bash
docker build -t medicare-agent .
docker tag medicare-agent:latest <account>.dkr.ecr.us-east-1.amazonaws.com/medicare-agent:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/medicare-agent:latest
```

Then deploy the new image in the Lambda console. Lambda needs 1024 MB memory
and a 60-second timeout — the defaults of 128 MB and 3 seconds both fail.

## Guardrails

The agent is constrained by its system prompt to:

- answer only from the handbook, and always search before answering
- say plainly when the handbook does not cover something
- refuse personalized medical, financial or enrollment advice
- cite the passage an answer came from

These were tested adversarially rather than assumed. Asked which plan a
specific person with diabetes should choose, the agent refused twice —
before and after searching — and redirected to SHIP counseling while still
explaining what Medicare generally covers for diabetes.

An earlier version leaked on out-of-scope questions: asked for the capital
of France, it answered from general knowledge. The prompt was tightened and
the agent now declines without even invoking the retrieval tool. The leak
and the fix are both in the commit history.

**This is a prompt-level guardrail, not a control.** It depends on the model
choosing to comply, and there is nothing outside the model enforcing it. For
a production member-facing system the right answer is an Amazon Bedrock
Guardrail screening input and output independently — a control a compliance
team can inspect and audit, which a prompt instruction is not.

## Known limitations

**Chunking is at service defaults and is visibly imperfect.** Querying the
Part B premium returns a passage that is mostly *Part A* content with the
Part B answer at the tail — the chunk boundary landed mid-topic. Retrieval
still succeeds, but a tuned chunking strategy would improve precision.

**The agent cannot follow the handbook's internal cross-references.**
Several passages say things like "go to page 93 for more information."
Those pages are in the Knowledge Base, but nothing connects the reference
to the target. Resolving them would need either a preprocessing pass that
inlines cross-referenced content, or a second retrieval hop.

**No conversation memory.** Each request constructs a fresh agent, so
follow-up questions do not carry context. This is deliberate for a
multi-user web demo — shared state across users would be worse — but a real
product needs per-session memory.

**Cold starts.** The first request after an idle period takes roughly 8–10
seconds to initialize the container. Provisioned concurrency would fix this
at the cost of paying for idle capacity.

**One document.** Only the CMS handbook is indexed. AARP's own Medicare
Supplement material would be the obvious next source.

## Notes from the build

A few things that were not obvious from the documentation:

- **Bedrock Agents (classic) is closed to new accounts** as of July 2026,
  which is what pushed this toward a code-first framework. The result is
  better — a real repository rather than console configuration.
- **The Model access console page has been retired.** Anthropic models now
  require a one-time use-case submission through the model catalog instead.
- **Managed Knowledge Bases take `vectorSearchConfiguration`, not
  `managedSearchConfiguration`** — but which one is accepted depends on the
  botocore version. The local virtual environment and the container had
  drifted to different versions and disagreed about the parameter name. The
  fix was pinning `boto3` and `botocore` explicitly.
- **That last bug was invisible** until the retrieval tool was changed to log
  and re-raise rather than silently fall back. The original exception
  handling swallowed the root cause and produced a graceful, useless
  apology instead.

## Cost

Ingestion is free. Running cost is query volume plus vector storage —
single-digit dollars for a demo. Lambda costs nothing while idle. A monthly
budget alarm is configured at $15.
