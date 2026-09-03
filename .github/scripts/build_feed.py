#!/usr/bin/env python3
"""Write feed.xml (RSS 2.0) from news.json so the news section is subscribable.

news.json is the single source: its items carry a human date ("July 22, 2026"), a title, HTML
content, tags and a link. The feed keeps the newest 30 items. The file is only rewritten when its
content changed, so an unchanged run produces no commit.
"""

import json
import os
import sys
from datetime import datetime, timezone
from email.utils import format_datetime
from xml.sax.saxutils import escape

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWS_PATH = os.path.join(REPO_ROOT, "news.json")
FEED_PATH = os.path.join(REPO_ROOT, "feed.xml")
SITE = "https://www.kornia.org/"
MAX_ITEMS = 30


def parse_date(text):
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def main():
    with open(NEWS_PATH, encoding="utf-8") as handle:
        news = json.load(handle)
    items = []
    for item in news.get("items", []):
        when = parse_date(item.get("date", ""))
        if when is None or not item.get("title"):
            print(f"  ! skipping item without a parseable date/title: {item.get('title')!r}", file=sys.stderr)
            continue
        items.append((when, item))
    items.sort(key=lambda pair: pair[0], reverse=True)
    items = items[:MAX_ITEMS]

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
        "<channel>",
        "<title>Kornia news</title>",
        f"<link>{SITE}#news</link>",
        "<description>Releases, talks and milestones across the Kornia ecosystem</description>",
        "<language>en</language>",
        f'<atom:link href="{SITE}feed.xml" rel="self" type="application/rss+xml"/>',
    ]
    if items:
        out.append(f"<lastBuildDate>{format_datetime(items[0][0])}</lastBuildDate>")
    for when, item in items:
        link = item.get("link") or f"{SITE}#news"
        guid = f"{SITE}news/{when.date().isoformat()}/{escape(item['title'])}"
        out += [
            "<item>",
            f"<title>{escape(item['title'])}</title>",
            f"<link>{escape(link)}</link>",
            f'<guid isPermaLink="false">{guid}</guid>',
            f"<pubDate>{format_datetime(when)}</pubDate>",
        ]
        for tag in item.get("tags", []):
            if tag != "all":
                out.append(f"<category>{escape(tag)}</category>")
        out += [f"<description>{escape(item.get('content', ''))}</description>", "</item>"]
    out += ["</channel>", "</rss>", ""]
    text = "\n".join(out)

    try:
        with open(FEED_PATH, encoding="utf-8") as handle:
            unchanged = handle.read() == text
    except FileNotFoundError:
        unchanged = False
    if unchanged:
        print(f"feed.xml: {len(items)} items (unchanged)")
        return 0
    with open(FEED_PATH, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"feed.xml: {len(items)} items (updated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
