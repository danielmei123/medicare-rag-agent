"""Load test the deployed Medicare agent.

Fires concurrent requests at a live endpoint and reports latency
distribution, throughput, and whether failures came from throttling.
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_URL = (
    "https://q54itqvpoyfyvcjyvwhdpoa3cm0pbcow.lambda-url.us-east-1.on.aws/ask"
)

QUESTIONS = [
    "What are the Medicare enrollment periods when I turn 65?",
    "How much is the Part B premium?",
    "What is a Medicare Advantage plan?",
    "Does Medicare cover dental care?",
    "What is Medigap and when can I buy it?",
    "What does Part D cover?",
    "What is the Part A late enrollment penalty?",
    "Does Medicare cover diabetes supplies?",
]


def one_request(url: str, i: int) -> dict:
    question = QUESTIONS[i % len(QUESTIONS)]
    start = time.perf_counter()
    try:
        r = requests.post(url, json={"question": question}, timeout=120)
        elapsed = time.perf_counter() - start
        body = r.text[:300]
        throttled = "throttl" in body.lower()
        return {
            "ok": r.status_code == 200 and not throttled,
            "status": r.status_code,
            "elapsed": elapsed,
            "throttled": throttled,
            "body": "" if (r.status_code == 200 and not throttled) else body,
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "status": 0,
            "elapsed": time.perf_counter() - start,
            "throttled": False,
            "body": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    print(f"{args.requests} requests at concurrency {args.concurrency}")
    print(f"Target: {args.url}\n")
    started = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args.url, i) for i in range(args.requests)
        ]
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            print("." if res["ok"] else "X", end="", flush=True)

    wall = time.perf_counter() - started
    latencies = sorted(r["elapsed"] for r in results)
    ok = [r for r in results if r["ok"]]
    throttled = [r for r in results if r["throttled"]]
    failed = [r for r in results if not r["ok"] and not r["throttled"]]

    def pct(p: float) -> float:
        return latencies[min(int(len(latencies) * p), len(latencies) - 1)]

    print("\n")
    print(f"Wall clock:     {wall:.1f}s")
    print(f"Throughput:     {len(results) / wall:.2f} req/s")
    print(f"Succeeded:      {len(ok)}/{len(results)}")
    print(f"Throttled:      {len(throttled)}")
    print(f"Other failures: {len(failed)}")
    print()
    print(f"min:    {latencies[0]:.2f}s")
    print(f"median: {statistics.median(latencies):.2f}s")
    print(f"p95:    {pct(0.95):.2f}s")
    print(f"max:    {latencies[-1]:.2f}s")

    for r in (throttled + failed)[:3]:
        print(f"\n  [{r['status']}] {r['body'][:200]}")


if __name__ == "__main__":
    main()