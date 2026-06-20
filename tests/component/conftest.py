import json
import os

results: list[dict] = []

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmark_results", "sq4_results.json")


def pytest_sessionfinish(session, exitstatus):
    if not results:
        return
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
