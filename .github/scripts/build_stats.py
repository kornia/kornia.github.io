#!/usr/bin/env python3
"""Rebuild stats.json from the GitHub and pypistats APIs.

Carry-forward is the core rule: a value is only replaced when a fetch actually
succeeds and returns a plausible number. Anything else - a 429, an HTML error
body, a network blip, a negative or non-integer payload - leaves the previous
value untouched. That keeps a transient upstream failure from blanking numbers
on the live site.

generated_at is only bumped when a value actually changed, so an unchanged run
leaves stats.json byte-identical and the workflow has nothing to commit.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATS_PATH = os.path.join(REPO_ROOT, "stats.json")

# Order here is the order written to stats.json.
REPOS = [
    "kornia/kornia",
    "kornia/kornia-rs",
    "kornia/bubbaloop",
    "kornia/kornia-slam",
    "kornia/vision-rt",
    "kornia/sensor-rt",
    "kornia/tutorials",
    "kornia/kornia-examples",
    "kornia/limbus",
]

PACKAGES = ["kornia"]

TIMEOUT = 30


def get_json(url, headers=None):
    """GET and parse JSON, or return None. Never raises.

    pypistats answers a rate limit with 429 *and an HTML body*, so a bare
    json.loads would explode or, worse, be caught somewhere that then writes a
    null. Both the status and the parse are checked here.
    """
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            if response.status != 200:
                print(f"  ! {url} -> HTTP {response.status}", file=sys.stderr)
                return None
            body = response.read()
    except urllib.error.HTTPError as exc:
        print(f"  ! {url} -> HTTP {exc.code}", file=sys.stderr)
        return None
    except Exception as exc:  # timeout, DNS, TLS, connection reset
        print(f"  ! {url} -> {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        print(f"  ! {url} -> non-JSON body ({exc})", file=sys.stderr)
        return None


def as_count(value):
    """Return value if it is a sane non-negative integer count, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def load_previous():
    try:
        with open(STATS_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print("no existing stats.json; starting fresh", file=sys.stderr)
        return {}
    except ValueError as exc:
        print(f"existing stats.json is not valid JSON ({exc}); starting fresh", file=sys.stderr)
        return {}


def previous_stars(previous, repo):
    return as_count((previous.get("repos", {}).get(repo) or {}).get("stars"))


def previous_downloads(previous, package):
    entry = previous.get("packages", {}).get(package) or {}
    return as_count(entry.get("downloads_last_month"))


def fetch_stars(repo):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "kornia.github.io-stats",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = get_json(f"https://api.github.com/repos/{repo}", headers)
    if not isinstance(data, dict):
        return None
    return as_count(data.get("stargazers_count"))


def fetch_downloads(package):
    data = get_json(
        f"https://pypistats.org/api/packages/{package}/recent?period=month",
        {"User-Agent": "kornia.github.io-stats"},
    )
    if not isinstance(data, dict):
        return None
    return as_count((data.get("data") or {}).get("last_month"))


def main():
    previous = load_previous()
    changed = []
    missing = []

    repos = OrderedDict()
    for repo in REPOS:
        old = previous_stars(previous, repo)
        new = fetch_stars(repo)
        if new is None:
            if old is None:
                missing.append(f"{repo} stars")
                print(f"  - {repo}: no value and no previous value; omitted")
                continue
            new = old
            print(f"  = {repo}: fetch failed, keeping {old}")
        elif new != old:
            changed.append(f"{repo} stars {old} -> {new}")
            print(f"  * {repo}: {old} -> {new}")
        else:
            print(f"  = {repo}: {new}")
        repos[repo] = {"stars": new}

    packages = OrderedDict()
    for package in PACKAGES:
        old = previous_downloads(previous, package)
        new = fetch_downloads(package)
        if new is None:
            if old is None:
                missing.append(f"{package} downloads")
                print(f"  - {package}: no value and no previous value; omitted")
                continue
            new = old
            print(f"  = {package} downloads: fetch failed, keeping {old}")
        elif new != old:
            changed.append(f"{package} downloads {old} -> {new}")
            print(f"  * {package} downloads: {old} -> {new}")
        else:
            print(f"  = {package} downloads: {new}")
        packages[package] = {"downloads_last_month": new}

    if not repos and not packages:
        # Every fetch failed and there was nothing to fall back on. Leave the
        # file exactly as it is rather than writing an empty one.
        print("no values available at all; leaving stats.json untouched", file=sys.stderr)
        return 1

    # Only move the timestamp when a number moved, so an unchanged run leaves
    # the file byte-identical and produces no commit.
    generated_at = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if changed
        else previous.get("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    )

    stats = OrderedDict(
        [("generated_at", generated_at), ("repos", repos), ("packages", packages)]
    )
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")

    print(f"\n{len(changed)} value(s) changed, {len(missing)} unavailable.")
    if missing:
        print("unavailable: " + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
