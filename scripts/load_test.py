"""Send varied requests to the API so dashboards show data."""
import argparse
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import setup_logging  # noqa: E402

logger = logging.getLogger("load_test")


def random_patient(rng: random.Random) -> dict:
    """Generate a plausible random patient."""
    return {
        "age": rng.randint(29, 77), "sex": rng.randint(0, 1),
        "cp": rng.randint(1, 4), "trestbps": rng.randint(94, 200),
        "chol": rng.randint(126, 564), "fbs": rng.randint(0, 1),
        "restecg": rng.randint(0, 2), "thalach": rng.randint(71, 202),
        "exang": rng.randint(0, 1), "oldpeak": round(rng.uniform(0, 6.2), 1),
        "slope": rng.randint(1, 3), "ca": rng.randint(0, 3),
        "thal": rng.choice([3, 6, 7]),
    }


def run(url: str, n: int, invalid_ratio: float, seed: int, delay: float) -> Counter:
    """Fire ``n`` requests; a fraction are deliberately invalid (422)."""
    rng = random.Random(seed)
    stats: Counter = Counter()
    session = requests.Session()
    for i in range(n):
        body = random_patient(rng)
        if rng.random() < invalid_ratio:
            body["age"] = 999
        try:
            resp = session.post(f"{url}/predict", json=body, timeout=10)
            stats[resp.status_code] += 1
            if resp.ok:
                stats[resp.json()["label"]] += 1
        except requests.RequestException as exc:
            stats["error"] += 1
            logger.warning("Request %d failed: %s", i, exc)
        if i % 3 == 0:
            session.get(f"{url}/health", timeout=10)
        time.sleep(delay)
    return stats


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--invalid-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()
    setup_logging()
    stats = run(args.url, args.n, args.invalid_ratio, args.seed, args.delay)
    logger.info("Done: %s", dict(stats))
    return 0 if stats["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
