"""Optional: pre-label candidates with an LLM so review goes faster.

Uses a DIFFERENT model from the baseline (Llama 4 Scout) on purpose. If the baseline
model pre-labeled the data, gold labels would lean toward its opinions and inflate the
baseline's score.

    export GROQ_API_KEY=...
    python scripts/prelabel_groq.py --in data/candidates.csv --n 300

Writes `prelabel` and `prelabel_model` columns back into the same CSV.
Every pre-label MUST be reviewed with scripts/review_labels.py.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takemeter_labels import build_prompt, parse_label  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def groq_label(text, model, key, tries=5):
    for i in range(tries):
        r = requests.post(GROQ_URL, timeout=60,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "temperature": 0, "max_tokens": 5,
                                "messages": [{"role": "user", "content": build_prompt(text)}]})
        if r.status_code == 429:
            time.sleep(float(r.headers.get("retry-after", 2 * (i + 1))))
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="path", default="data/candidates.csv")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--n", type=int, default=300, help="how many rows to pre-label")
    args = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY") or sys.exit("Set GROQ_API_KEY")

    df = pd.read_csv(args.path)
    for col in ("prelabel", "prelabel_model"):
        if col not in df:
            df[col] = ""
    df[["prelabel", "prelabel_model"]] = df[["prelabel", "prelabel_model"]].fillna("").astype(str)
    todo = df.index[df["prelabel"] == ""][: args.n]
    for k, i in enumerate(todo, 1):
        lab = parse_label(groq_label(df.at[i, "text"], args.model, key)) or "unparsed"
        df.at[i, "prelabel"], df.at[i, "prelabel_model"] = lab, args.model
        if k % 25 == 0:
            df.to_csv(args.path, index=False)
            print(f"{k}/{len(todo)}")
    df.to_csv(args.path, index=False)
    print(df["prelabel"].value_counts())


if __name__ == "__main__":
    main()
