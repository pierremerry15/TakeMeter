"""Terminal labeling tool. You label FIRST; the LLM pre-label (if any) is only revealed
afterward, so you label the post instead of rubber-stamping the pre-label.

    python scripts/review_labels.py --target 240

Keys:  a = analysis   h = hot_take   r = reaction   x = exclude (not a take)
       s = skip for now   q = save & quit
After labeling you can type a note (press Enter to skip). Notes on hard cases feed the
README's "difficult to label" section.

Progress is saved to data/takemeter_nba.csv after every post, so you can quit and resume.
"""
import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takemeter_labels import DEFINITIONS, LABELS  # noqa: E402

KEYS = {"a": "analysis", "h": "hot_take", "r": "reaction", "x": "excluded"}
COLS = ["id", "text", "label", "notes", "source_thread", "source_type",
        "prelabel", "prelabel_model", "label_changed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/candidates.csv")
    ap.add_argument("--out", default="data/takemeter_nba.csv")
    ap.add_argument("--target", type=int, default=240, help="stop after this many labeled (non-excluded)")
    args = ap.parse_args()

    cand = pd.read_csv(args.candidates).fillna("")
    out_path = Path(args.out)
    done = pd.read_csv(out_path).fillna("") if out_path.exists() else pd.DataFrame(columns=COLS)
    done_ids = set(done["id"].astype(str))

    print("\nDefinitions:")
    for l, d in DEFINITIONS.items():
        print(textwrap.fill(f"  {l}: {d}", 100, subsequent_indent="    "))

    for _, row in cand.iterrows():
        if str(row["id"]) in done_ids:
            continue
        labeled = done[done["label"].isin(LABELS)]
        if len(labeled) >= args.target:
            print(f"\nReached target of {args.target}.")
            break
        counts = labeled["label"].value_counts().reindex(LABELS, fill_value=0)
        print("\n" + "=" * 100)
        print(f"[{len(labeled)}/{args.target}]  " + "  ".join(f"{l}={n}" for l, n in counts.items())
              + f"   | {row.get('source_type', '')}: {str(row.get('source_thread', ''))[:50]}")
        print(textwrap.fill(row["text"], 100))
        key = ""
        while key not in list(KEYS) + ["s", "q"]:
            key = input("[a]nalysis [h]ot_take [r]eaction e[x]clude [s]kip [q]uit > ").strip().lower()
        if key == "q":
            break
        if key == "s":
            continue
        label = KEYS[key]
        pre = str(row.get("prelabel", ""))
        if pre and pre != "unparsed" and label != "excluded" and pre != label:
            print(f"   pre-label was: {pre}  (you said {label})")
            if input("   switch to pre-label? [y/N] > ").strip().lower() == "y":
                label = pre
        note = input("   note (Enter to skip) > ").strip()
        rec = {c: row.get(c, "") for c in COLS}
        rec.update(label=label, notes=note,
                   label_changed=bool(pre and pre != "unparsed" and label != "excluded" and pre != label))
        done = pd.concat([done, pd.DataFrame([rec])], ignore_index=True)
        done[COLS].to_csv(out_path, index=False)

    labeled = done[done["label"].isin(LABELS)]
    print("\nFinal distribution:")
    print(labeled["label"].value_counts().to_string())
    print(f"excluded: {(done['label'] == 'excluded').sum()}")
    top = labeled["label"].value_counts(normalize=True).max() if len(labeled) else 0
    if top > 0.70:
        print("WARNING: one label is over 70% - collect more of the minority labels.")


if __name__ == "__main__":
    main()
