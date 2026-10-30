"""Delete rented Runpod machines that no local `remote-test` is tracking any more.

remote-test deletes its machine when it ends, but a killed process (a session restart, a lost
worker) cannot; its machine keeps billing and nothing can collect its results. This sweeps
machines named ``oneapi-gputest-*`` that are either older than --max-age minutes or exist
while no `oneapi-agent remote-test` or `museport gpu-broker` process runs here. Other pods
are never touched.

usage: python3 scripts/sweep-pods.py [--dry-run] [--max-age 160]
Needs RUNPOD_API_KEY; prints no secrets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import urllib.request

API = "https://rest.runpod.io/v1/pods"


def call(method: str, url: str) -> object:
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
        "User-Agent": "museport-sweep-pods/1"})  # the default Python agent is refused (403)
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    return json.loads(body) if body else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-age", type=float, default=160,
                    help="minutes (remote-test's cap + slack)")
    a = ap.parse_args()
    # remote-test runs as its own process, or inside a GPU broker (museport gpu-broker)
    live = subprocess.run(["pgrep", "-f", "oneapi-agent remote-test|museport gpu-broker"],
                          capture_output=True, text=True).stdout.split()
    now = dt.datetime.now(dt.UTC)
    pods = call("GET", API) or []
    for p in pods:  # type: ignore[attr-defined]
        if not str(p.get("name", "")).startswith("oneapi-gputest-"):
            continue
        started = p.get("lastStartedAt") or p.get("createdAt") or ""
        try:
            age = (now - dt.datetime.fromisoformat(started.replace(" UTC", "").replace(
                " +0000", "+00:00"))).total_seconds() / 60
        except ValueError:
            age = float("inf")
        orphan = not live or age > a.max_age
        print(f"{p['id']} {p['name']} {p.get('desiredStatus')} ${p.get('costPerHr')}/h "
              f"{age:.0f} min: {'DELETE' if orphan else 'keep (a remote-test is running)'}")
        if orphan and not a.dry_run:
            call("DELETE", f"{API}/{p['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
