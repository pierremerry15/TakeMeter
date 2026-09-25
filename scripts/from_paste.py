"""Build data/candidates.csv from comments you copy-paste by hand (no Reddit API needed).

Use this when Reddit blocks the automatic collector. The assignment allows manual
collection ("copy-paste into a spreadsheet").

1. Create data/pasted.txt and paste comments into it, ONE COMMENT PER LINE.
   (If a comment has several paragraphs, join them onto one line.)
2. Before each thread's comments, add a line starting with "# " naming the thread and
   its type, so the README can say where the data came from, e.g.:

       # postgame | Post Game Thread: Knicks def. Celtics 112-104
       LETS GOOOO BRUNSON
       Brunson is a top 5 player in the league now, not even debatable
       # oc | [OC] Every team's corner-three defense since the deadline
       The corner 3 thing is real: they were 3rd before the injury and 26th since...

   Types: postgame, game, daily, oc, serious (or anything you like).
3. Run:  python3 scripts/from_paste.py
   It applies the same filters as collect_reddit.py (drops <4 words, pure questions,
   links, duplicates) and writes data/candidates.csv for review_labels.py.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_reddit import clean, exclusion_reason  # noqa: E402


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pasted.txt")
    out = src.with_name("candidates.csv")
    rows, drops, seen = [], {}, set()
    thread, stype = "unknown", "manual"
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            head = line[2:]
            stype, thread = ([p.strip() for p in head.split("|", 1)] + [head])[:2] if "|" in head else ("manual", head)
            continue
        text = clean(line)
        reason = exclusion_reason(text, "")
        if reason:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        key = re.sub(r"\W+", "", text.lower())
        if key in seen:
            drops["duplicate"] = drops.get("duplicate", 0) + 1
            continue
        seen.add(key)
        rows.append({"id": f"m{len(rows):04d}", "text": text, "source_thread": thread,
                     "source_type": stype, "word_count": len(text.split())})
    pd.DataFrame(rows).to_csv(out, index=False)
    log = {"threads": len({r["source_thread"] for r in rows}), "candidates": len(rows),
           "dropped": drops, "method": "manual copy-paste"}
    out.with_name("collection_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
