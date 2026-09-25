"""Collect public r/nba comments into a candidate pool (no login, no usernames stored).

Run this on your own computer: Reddit blocks unauthenticated requests from many cloud
IPs, including Colab.

    python scripts/collect_reddit.py --auto                # discover threads automatically
    python scripts/collect_reddit.py --threads threads.txt # or use your own list of thread URLs

Output: data/candidates.csv and data/collection_log.json (drop counts, used to check
the "label >= 90% of substantive posts" requirement).
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "takemeter-class-project/1.0 (educational; read-only)"}
BASE = "https://www.reddit.com"

# (query, source_type) pairs used by --auto. source_type lets us over-sample
# the threads where analysis actually lives.
AUTO_QUERIES = [
    ('flair:"Post Game Thread"', "postgame"),
    ('title:"Game Thread"', "game"),
    ('title:"Daily Discussion"', "daily"),
    ("[OC]", "oc"),
    ("[Serious]", "serious"),
]


def get_json(url, params=None, tries=4):
    for i in range(tries):
        r = requests.get(url, headers=UA, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Rate-limited too many times: {url}")


def search_threads(query, t="month", limit=10):
    data = get_json(f"{BASE}/r/nba/search.json",
                    {"q": query, "restrict_sr": 1, "sort": "top", "t": t, "limit": limit})
    return [c["data"]["permalink"] for c in data["data"]["children"]]


def walk(children, out):
    for c in children:
        if c.get("kind") != "t1":
            continue
        d = c["data"]
        out.append(d)
        replies = d.get("replies")
        if isinstance(replies, dict):
            walk(replies["data"]["children"], out)


def fetch_comments(permalink, limit=500):
    url = BASE + permalink.rstrip("/") + ".json"
    data = get_json(url, {"limit": limit, "sort": "top"})
    comments = []
    walk(data[1]["data"]["children"], comments)
    title = data[0]["data"]["children"][0]["data"]["title"]
    return title, comments


URL_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")


def exclusion_reason(text, author):
    """Return why a comment is excluded (not a take), or None if it should be kept."""
    if text in ("[deleted]", "[removed]") or not text.strip():
        return "deleted"
    if author in ("AutoModerator", "nba-bot", "[deleted]") or "I am a bot" in text:
        return "bot_or_mod"
    if URL_ONLY.match(text):
        return "link_only"
    words = text.split()
    if len(words) < 4:
        return "too_short"
    if text.strip().endswith("?") and len(words) <= 20 and text.count(".") == 0:
        return "pure_question"
    return None


def clean(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links -> anchor text
    text = re.sub(r"^&gt;.*$", "", text, flags=re.M)       # drop quoted replies
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="discover threads via r/nba search")
    ap.add_argument("--threads", help="text file with one r/nba thread URL per line")
    ap.add_argument("--per-thread", type=int, default=40, help="max comments kept per thread")
    ap.add_argument("--out", default="data/candidates.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    threads = []  # (permalink, source_type)
    if args.threads:
        for line in Path(args.threads).read_text().splitlines():
            line = line.strip()
            if line:
                threads.append((re.sub(r"^https?://(www|old)\.reddit\.com", "", line), "manual"))
    if args.auto:
        for q, st in AUTO_QUERIES:
            for p in search_threads(q):
                threads.append((p, st))
            time.sleep(2)
    if not threads:
        ap.error("pass --auto and/or --threads")

    rows, drops, seen = [], {}, set()
    for permalink, st in threads:
        try:
            title, comments = fetch_comments(permalink)
        except Exception as e:  # keep going if one thread fails
            print(f"skip {permalink}: {e}")
            continue
        kept = []
        for c in comments:
            text = clean(c.get("body", ""))
            reason = exclusion_reason(text, c.get("author", ""))
            if reason:
                drops[reason] = drops.get(reason, 0) + 1
                continue
            key = re.sub(r"\W+", "", text.lower())
            if key in seen:
                drops["duplicate"] = drops.get("duplicate", 0) + 1
                continue
            seen.add(key)
            kept.append({"id": c["id"], "text": text, "source_thread": title,
                         "source_type": st, "word_count": len(text.split())})
        # Keep the longest comments too, not only a random slice, so analysis isn't starved
        kept.sort(key=lambda r: -r["word_count"])
        long_part = kept[: args.per_thread // 4]
        rest = kept[args.per_thread // 4:]
        random.shuffle(rest)
        rows += long_part + rest[: args.per_thread - len(long_part)]
        print(f"{st:9s} {len(kept):4d} usable -> kept {min(len(kept), args.per_thread):3d} | {title[:70]}")
        time.sleep(2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sample(frac=1, random_state=args.seed)
    df.to_csv(args.out, index=False)
    log = {"threads": len(threads), "candidates": len(df), "dropped": drops}
    Path(args.out).with_name("collection_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
