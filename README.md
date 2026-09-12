# Medicare Assistant

A retrieval-augmented question-answering agent over the official CMS
*Medicare & You 2026* handbook. Members ask general Medicare questions;
the agent retrieves relevant handbook passages and answers with citations,
declining anything that would amount to personal advice.

**Live demo:** https://q54itqvpoyfyvcjyvwhdpoa3cm0pbcow.lambda-url.us-east-1.on.aws/

The same agent is deployed three ways — a hand-built Lambda, a Lambda
created entirely from CloudFormation, and a Kubernetes deployment on EKS —
and load tested to compare them. Findings are under Load testing below.

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
Lambda Function URL  ->  live alias  ->  published version
   |
   v
Container image (ECR)  ->  FastAPI + Mangum
   |
   v
Agent  ->  Claude Sonnet 4.6 on Bedrock
   |
   v
search_medicare_handbook tool
   |
   v
Bedrock Knowledge Base (managed, 128-page handbook)
```

| Component | Choice | Reason |
|---|---|---|
| Agent framework | Strands (deployed), LangGraph (comparison and orchestration) | See Two frameworks below |
| Model | `us.anthropic.claude-sonnet-4-6` | Cross-region inference profile; on-demand throughput is not supported for the bare model ID |
| Retrieval | Bedrock Knowledge Base, managed storage | No vector store to operate for a prototype |
| Hosting | Lambda container image; also EKS | 75+ dependencies exceed the zip limit; scales to zero between requests |
| Web layer | FastAPI + Mangum | Mangum adapts the Lambda event format to ASGI, so the same app runs locally under uvicorn, on Lambda, and in a pod |

## Repository contents

| File | Purpose |
|---|---|
| `agent.py` | The agent: retrieval tool, system prompt, agent construction |
| `agent_langgraph.py` | The same agent as an explicit LangGraph state graph |
| `agent_orchestrator.py` | Multi-agent version: orchestrator routing to specialist agents |
| `app.py` | FastAPI wrapper and the single-page front end |
| `lambda_handler.py` | Lambda entry point via Mangum |
| `Dockerfile` | Lambda container image |
| `Dockerfile.k8s` | Kubernetes image — plain Python base, serves HTTP directly |
| `infrastructure.yaml` | CloudFormation: function, role, version, alias, URL, alarms, dashboard |
| `k8s-deployment.yaml` | Kubernetes Deployment and LoadBalancer Service |
| `.github/workflows/deploy.yml` | CI/CD: static analysis, evals, build, blue/green deploy |
| `evals.py` | Retrieval eval suite, 11 cases |
| `loadtest.py` | Concurrent load test harness |
| `test_retrieve.py` | Exercises retrieval alone, without model access |

## Two frameworks

The deployed agent uses **Strands**, AWS's agent framework. The same agent
is also implemented in **LangGraph** (`agent_langgraph.py`) as an explicit
state graph: two nodes, model and tools, with a conditional edge that
routes back to the model after each tool call. That graph *is* the agent
loop, written out.

The practical difference showed up in error handling. When retrieval failed,
Strands caught the exception and degraded gracefully; LangGraph let it
propagate and kill the whole graph until the tool was changed to return a
message instead of raising. LangGraph gives you the loop as data you can
inspect and modify, and the cost is that failure handling inside nodes is
your responsibility.

For a single-tool agent, Strands is the simpler fit. LangGraph earns its
complexity when the graph needs branching — which is what the orchestrator
below uses it for.

## Multi-agent orchestration

`agent_orchestrator.py` implements two patterns the single-agent version
does not need: specialists exposed as tools, and an orchestrator that
routes rather than answers.

The specialists differ in capability rather than just in prompt. The
eligibility specialist has a deterministic date calculator alongside
retrieval, so enrollment windows are computed rather than reasoned about —
including the rule that a first-of-month birthday shifts the whole
seven-month window one month earlier. The costs specialist has stricter
output rules, since a wrong figure does real harm in that domain. Coverage
is plain retrieval over a broad question space.

Out-of-scope questions are declined at the orchestrator without invoking a
specialist, which is both correct and cheaper.

**The pattern has a real cost.** Asked a follow-up about HSA contributions —
content the eligibility specialist had just cited from the handbook — the
orchestrator classified it as out of scope and declined, where the single
agent would have searched and found it. Adding a router adds a place to be
wrong, and the router gates everything downstream. With no conversation
memory, it also cannot see that a specialist raised the topic a moment
earlier.

For most questions the single-agent version answers just as well and
faster. The orchestrator exists to handle the enrollment-date case properly
and to demonstrate the pattern.

## Deployment

Three paths, all producing the same behaviour.

**1. Lambda, via CI/CD.** Push to `main` triggers static analysis
(`ruff`, `bandit`), then the retrieval evals against the live knowledge
base, then a container build and a blue/green deploy: publish a version,
shift 10% of traffic to it, smoke test the live URL, promote to 100%. If
the smoke test fails, the promotion never runs and 90% of traffic stays on
the previous version. Images are tagged with the commit SHA, so every
deployment traces to a commit. CI authenticates via GitHub OIDC — no
long-lived AWS credentials are stored.

**2. CloudFormation.** `infrastructure.yaml` creates the function, its
execution role, a published version, the `live` alias, a public function
URL, three CloudWatch alarms and a dashboard. The knowledge base and ECR
repository are parameters rather than resources: the knowledge base holds
ingested data that should outlive any single stack, and the image must
exist before a container-based function can start.

```bash
aws cloudformation create-stack \
  --stack-name medicare-agent-cfn \
  --template-body file://infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=ImageUri,ParameterValue=<account>.dkr.ecr.us-east-1.amazonaws.com/medicare-agent:latest \
               ParameterKey=KnowledgeBaseId,ParameterValue=<kb-id>
```

**3. Kubernetes on EKS.** Two replicas behind a LoadBalancer Service, with
liveness and readiness probes against `/health` and CPU and memory limits
set. Pods get Bedrock access through IRSA — a service account bound to an
IAM role via the cluster's OIDC provider — so no credentials are baked into
the image or passed as environment variables.

```bash
eksctl create cluster --name medicare-agent --region us-east-1 \
  --node-type t3.small --nodes 2 --managed --with-oidc

eksctl create iamserviceaccount --name medicare-agent-sa \
  --namespace default --cluster medicare-agent --region us-east-1 \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess --approve

kubectl apply -f k8s-deployment.yaml
```

## Load testing

`loadtest.py` fires concurrent requests at any of the endpoints and reports
latency distribution, throughput, and failure causes. Results:

| Test | Lambda | EKS (2 pods) |
|---|---|---|
| 30 requests, concurrency 10 | 30/30 succeeded, p95 14.6s, 0.85 req/s | 30/30 succeeded, p95 14.9s, 0.81 req/s |
| 50 requests, concurrency 25 | **10/50 succeeded**, 40 rejected `ConcurrentInvocationLimitExceeded` | **50/50 succeeded**, p95 17.5s, 1.58 req/s |

Three things came out of this.

**At moderate load the compute platform is irrelevant.** Lambda and EKS
performed within 4% of each other. Both spend roughly nine seconds waiting
on retrieval and model generation; where that wait happens makes no
measurable difference.

**They fail completely differently.** At concurrency 25 Lambda hit the
account concurrent-execution limit and refused 80% of requests with HTTP
429 in about 40ms. EKS served every request, absorbing the load by getting
slower — median rose from 9.5s to 13.2s. For a member-facing chatbot,
graceful degradation is almost certainly preferable to a fast rejection.
The Lambda fix is a service quota increase, not a code change.

**Cold starts dominate the tail when traffic is sparse.** The first test at
concurrency 2 showed a p95 of 21.8s; the same endpoint under heavier, warmer
load showed 14.6s.

## Monitoring

The CloudFormation stack creates a CloudWatch dashboard (traffic and
failures, p50/p95/max latency, concurrent executions) and three alarms:

- **Errors** — more than 3 in a five-minute window
- **Throttles** — any at all, since load testing showed these begin at
  roughly 10 concurrent requests on the default account limit
- **p95 latency** — above 25s for two consecutive periods, against a
  measured warm baseline of about 15s

The thresholds come from the load test rather than from defaults.

## Guardrails

The agent is constrained by its system prompt to answer only from the
handbook, always search before answering, say plainly when the handbook
does not cover something, refuse personalized medical, financial or
enrollment advice, and cite the passage an answer came from.

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

## Evaluation

`evals.py` runs 11 retrieval cases covering specific figures (the 2026 Part B
premium, the insulin cost cap, the late enrollment penalty percentage),
distractor-prone queries where Part A content competes with Part B, and a
cross-reference target. All 11 pass.

An earlier version of this suite passed 12/12 on its first run, which was a
sign the cases were too easy rather than that retrieval was perfect — asking
"what is Medigap?" and checking that the word "Medigap" appears tests very
little. The current cases were written so they could fail.

**This tests retrieval, not answer quality.** It proves the right passage
comes back; it does not prove the model then uses it correctly. That has
been verified by hand on a handful of questions, not systematically. An
LLM-as-judge eval scoring generated answers against expected content would
close the gap.

## Known limitations

**Chunking is at service defaults and is visibly imperfect.** Querying the
Part B premium returns a passage that is mostly *Part A* content with the
Part B answer at the tail — the chunk boundary landed mid-topic.

**The agent cannot follow the handbook's internal cross-references.**
Several passages say things like "go to page 93 for more information."
Those pages are in the knowledge base, but nothing connects the reference
to the target. Resolving them would need a preprocessing pass that inlines
cross-referenced content, or a second retrieval hop.

**No conversation memory.** Each request constructs a fresh agent, so
follow-up questions do not carry context. This is deliberate for a
multi-user web demo — shared state across users would be worse — but a real
product needs per-session memory, and it is what causes the orchestrator
routing miss described above.

**No response streaming.** Answers appear all at once after 10-15 seconds.
Streaming would not reduce total latency but would substantially change how
long it feels.

**Cold starts.** The first request after an idle period takes roughly 8-10
seconds to initialize the container. Provisioned concurrency would fix this
at the cost of paying for idle capacity.

**One document.** Only the CMS handbook is indexed. AARP's own Medicare
Supplement material would be the obvious next source.

## Running locally

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000. Requires AWS credentials with Bedrock access
and a knowledge base ID set in `agent.py`. For the LangGraph and
orchestrator versions, `pip install -r requirements-dev.txt` and run
`python agent_langgraph.py` or `python agent_orchestrator.py`.

## Notes from the build

A few things that were not obvious from the documentation:

- **Bedrock Agents (classic) is closed to new accounts** as of July 2026,
  which is what pushed this toward a code-first framework. The result is
  better — a real repository rather than console configuration.
- **App Runner closed to new customers** in April 2026, which is what
  pushed the deployment to Lambda container images.
- **The Model access console page has been retired.** Anthropic models now
  require a one-time use-case submission through the model catalog.
- **Managed knowledge bases take `managedSearchConfiguration`, not
  `vectorSearchConfiguration`** — but which one is accepted depends on the
  botocore version. The local virtual environment and the container had
  drifted to different versions and disagreed about the parameter name.
  The fix was pinning `boto3` and `botocore` explicitly.
- **That last bug was invisible** until the retrieval tool was changed to
  log and re-raise rather than silently fall back. The original exception
  handling swallowed the root cause and produced a graceful, useless
  apology instead.
- **GitHub changed its OIDC subject claim format in July 2026.** New
  repositories emit an immutable claim embedding numeric org and repo IDs,
  so a trust policy matching the legacy `repo:org/name:*` form fails. The
  policy here matches both forms.
- **Function URLs have needed both `lambda:InvokeFunctionUrl` and
  `lambda:InvokeFunction` since October 2025.** The console adds both
  automatically; CloudFormation does not. A hand-built function worked
  where the same thing expressed as code returned 403.

## Cost

Ingestion is free. Running cost is query volume plus vector storage —
single-digit dollars for a demo. Lambda costs nothing while idle; the EKS
cluster costs roughly $3.50/day whether or not it serves traffic. A monthly
budget alarm is configured at $15.
