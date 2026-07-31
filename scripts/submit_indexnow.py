#!/usr/bin/env python3
"""Ping IndexNow with this project's recently changed URLs after a publish.

Runs as the last step of `make publish` (also available as `make indexnow`).
IndexNow instantly notifies Bing, Naver, Seznam and Yandex, which share
submissions with other participating engines - and those indexes feed the
retrieval layers of AI search products, so this covers "LLM discoverability".

Key management - there is nothing to generate per project:
  An IndexNow key authorizes an entire HOST, not a path. Every project site
  lives under the same host as the hub (migue-rc.github.io), so the single
  key file the hub serves at its root -- https://migue-rc.github.io/indexnow-key.txt
  -- already authorizes every project URL. This script fetches that key at
  publish time. IndexNow keys are public by design, so fetching is fine, and
  rotating the key is a one-file edit on the hub that every project inherits
  automatically.

What it does:
  1. Reads this project's site-url from _quarto.yml (stdlib only, no deps).
  2. Fetches the shared key from the hub.
  3. Fetches this project's own sitemap and collects URLs whose <lastmod>
     is within --since-days (default 1).
  4. Submits them in one batch, citing the hub's key location.

Network failures never break the publish: the script warns and exits 0.

Usage:
    python3 scripts/submit_indexnow.py [--since-days N] [--dry-run]
"""

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
HUB = "https://migue-rc.github.io"
KEY_URL = f"{HUB}/indexnow-key.txt"
ENDPOINT = "https://api.indexnow.org/indexnow"


def site_url() -> str:
    text = (ROOT / "_quarto.yml").read_text()
    match = re.search(r"^\s*site-url:\s*[\"']?([^\"'\s]+)", text, re.M)
    if not match:
        sys.exit("ERROR: site-url not set in _quarto.yml (required for IndexNow).")
    url = match.group(1).rstrip("/")
    if "PROJECT-NAME" in url:
        sys.exit("ERROR: replace the PROJECT-NAME placeholder in _quarto.yml first.")
    return url


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def changed_urls(sitemap_xml: str, cutoff: datetime) -> list[str]:
    urls = []
    for entry in re.findall(r"<url>(.*?)</url>", sitemap_xml, re.S):
        loc = re.search(r"<loc>(.*?)</loc>", entry)
        lastmod = re.search(r"<lastmod>(.*?)</lastmod>", entry)
        if not loc:
            continue
        if lastmod:
            try:
                modified = datetime.fromisoformat(lastmod.group(1).replace("Z", "+00:00"))
            except ValueError:
                modified = None
            if modified is not None and modified < cutoff:
                continue
        urls.append(loc.group(1).strip())
    return urls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-days", type=int, default=1,
                        help="submit URLs modified within this many days (default 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted without submitting")
    args = parser.parse_args()

    url = site_url()
    host = urlparse(url).netloc
    sitemap = f"{url}/sitemap.xml"
    print(f"==> Project site: {url}")

    print(f"==> Fetching shared IndexNow key from {KEY_URL}")
    try:
        key = fetch(KEY_URL).strip()
    except Exception as error:  # noqa: BLE001 - never break the publish
        print(f"==> WARNING could not fetch key ({error}); skipping IndexNow")
        return
    if not re.fullmatch(r"[0-9a-zA-Z-]{8,128}", key):
        print("==> WARNING hub key looks invalid; skipping IndexNow")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.since_days)
    print(f"==> Checking {sitemap} for URLs changed in the last {args.since_days} day(s)")
    try:
        xml = fetch(sitemap)
    except Exception as error:  # noqa: BLE001
        print(f"==> WARNING could not fetch sitemap ({error}); skipping IndexNow")
        return

    batch = sorted(set(changed_urls(xml, cutoff)))
    if not batch:
        print("==> Nothing changed in the window, nothing to submit")
        return
    print(f"==> {len(batch)} URL(s) to submit:")
    for entry in batch:
        print(f"    {entry}")
    if args.dry_run:
        print("==> DRY RUN: skipping submission")
        return

    payload = json.dumps({
        "host": host,
        "key": key,
        "keyLocation": KEY_URL,
        "urlList": batch,
    }).encode()
    request = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"})
    print(f"==> Submitting batch to {ENDPOINT}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"==> Done: {len(batch)} URL(s) accepted, HTTP {response.status}")
    except urllib.error.HTTPError as error:
        print(f"==> WARNING submission rejected: HTTP {error.code} {error.read().decode()[:200]}")
    except Exception as error:  # noqa: BLE001
        print(f"==> WARNING submission failed: {error}")


if __name__ == "__main__":
    main()
