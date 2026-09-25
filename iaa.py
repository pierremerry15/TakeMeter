"""Stretch: inter-annotator reliability.

1. Make a blind sheet for a second annotator (no labels, no notes):
       python scripts/iaa.py make-sheet --n 40
   -> data/iaa_sheet.csv  (they fill in the `label` column using planning.md only)
2. Score it:
       python scripts/iaa.py score --second data/iaa_sheet.csv
   -> prints agreement, Cohen's kappa, and every disagreement; writes outputs/iaa_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takemeter_labels import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-sheet", "score"])
    ap.add_argument("--data", default="data/takemeter_nba.csv")
    ap.add_argument("--second", default="data/iaa_sheet.csv")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    mine = pd.read_csv(args.data, dtype={"id": str}).fillna("")
    mine = mine[mine["label"].isin(LABELS)]

    if args.cmd == "make-sheet":
        s = mine.sample(args.n, random_state=7)[["id", "text"]].assign(label="")
        s.to_csv(args.second, index=False)
        print(f"wrote {args.second} - give it to your second annotator with planning.md")
        return

    other = pd.read_csv(args.second, dtype={"id": str}).fillna("")
    m = mine[["id", "text", "label"]].merge(other[["id", "label"]], on="id", suffixes=("_me", "_other"))
    m = m[m["label_other"].isin(LABELS)]
    agree = float((m["label_me"] == m["label_other"]).mean())
    kappa = float(cohen_kappa_score(m["label_me"], m["label_other"], labels=LABELS))
    cm = confusion_matrix(m["label_me"], m["label_other"], labels=LABELS).tolist()
    dis = m[m["label_me"] != m["label_other"]]
    print(f"n={len(m)}  agreement={agree:.1%}  Cohen's kappa={kappa:.3f}")
    print("\nDisagreements:")
    for _, r in dis.iterrows():
        print(f"- me={r['label_me']:9s} other={r['label_other']:9s} | {r['text'][:120]}")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/iaa_results.json").write_text(json.dumps({
        "n": len(m), "agreement": agree, "cohen_kappa": kappa, "confusion_me_rows_other_cols": cm,
        "disagreements": dis[["id", "text", "label_me", "label_other"]].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
