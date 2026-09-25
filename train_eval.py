"""TakeMeter: split -> zero-shot Groq baseline -> fine-tune DistilBERT -> evaluate -> export.

This does the same job as the course starter notebook (Sections 1-6) in one script, so
the whole thing can be re-run with one command. In Colab (T4 GPU):

    from google.colab import userdata; import os
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
    !python train_eval.py --data data/takemeter_nba.csv

Outputs (in outputs/):
    splits.csv                  which id is in train/val/test (the test set is locked)
    baseline_predictions.csv    cached Groq responses (re-used on re-runs, never re-queried)
    test_predictions.csv        fine-tuned predictions + probabilities on the test set
    evaluation_results.json     every number the README reports
    confusion_matrix.png        fine-tuned confusion matrix (supplementary; README has the table)
    model/                      saved fine-tuned model for app.py
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from takemeter_labels import ID2LABEL, LABEL2ID, LABELS, build_prompt, parse_label

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# ----------------------------------------------------------------------------- utils
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def norm(t):
    return re.sub(r"\W+", "", str(t).lower())


def metrics_block(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "per_class": {l: {k: rep[l][k] for k in ("precision", "recall", "f1-score", "support")}
                      for l in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


# ----------------------------------------------------------------------------- 1-2: data + split
def load_and_split(path, seed, out):
    df = pd.read_csv(path).fillna("")
    df = df[df["label"].isin(LABELS)].copy()
    df["id"] = df["id"].astype(str)
    # Drop exact/near-exact duplicate texts BEFORE splitting so none can leak across splits
    before = len(df)
    df["_n"] = df["text"].map(norm)
    df = df.drop_duplicates("_n").drop(columns="_n")
    dupes = before - len(df)

    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=seed)
    split = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    split[["id", "label", "split"]].to_csv(out / "splits.csv", index=False)
    print(f"rows={len(df)} (dropped {dupes} duplicates)  train={len(train)} val={len(val)} test={len(test)}")
    for name, d in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s} " + "  ".join(f"{l}={(d['label'] == l).sum()}" for l in LABELS))
    return df, train, val, test, dupes


# ----------------------------------------------------------------------------- 5: baseline
def run_baseline(test, model, out, mode):
    cache = out / "baseline_predictions.csv"
    if cache.exists():
        b = pd.read_csv(cache, dtype={"id": str}).fillna("")
        if set(b["id"]) == set(test["id"]):
            print(f"baseline: re-using cached {cache}")
            return b
    if mode == "skip":
        return None
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("baseline: GROQ_API_KEY not set -> skipping baseline")
        return None
    rows = []
    for _, r in test.iterrows():
        raw = ""
        for i in range(6):
            resp = requests.post(GROQ_URL, timeout=60, headers={"Authorization": f"Bearer {key}"},
                                 json={"model": model, "temperature": 0, "max_tokens": 5,
                                       "messages": [{"role": "user", "content": build_prompt(r["text"])}]})
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("retry-after", 2 * (i + 1))))
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            break
        rows.append({"id": r["id"], "text": r["text"], "label": r["label"],
                     "raw_response": raw, "pred": parse_label(raw) or "UNPARSEABLE"})
        time.sleep(0.3)
    b = pd.DataFrame(rows)
    b.to_csv(cache, index=False)
    return b


# ----------------------------------------------------------------------------- 3: fine-tune
def encode(tok, df, max_len):
    enc = tok(list(df["text"]), truncation=True, max_length=max_len)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "labels": LABEL2ID[l]} for i, l in enumerate(df["label"])]


def collate(tok):
    def f(batch):
        # copy, don't pop: the dataset items are re-used every epoch
        labels = torch.tensor([b["labels"] for b in batch])
        feats = [{k: v for k, v in b.items() if k != "labels"} for b in batch]
        padded = tok.pad(feats, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return f


@torch.no_grad()
def predict(model, tok, texts, max_len, device, bs=32):
    model.eval()
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, len(LABELS)))


def fine_tune(train, val, args, device, out):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True).to(device)
    dl = DataLoader(encode(tok, train, args.max_len), batch_size=args.batch_size,
                    shuffle=True, collate_fn=collate(tok))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    history, best_f1, best_state = [], -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
        vp = predict(model, tok, list(val["text"]), args.max_len, device)
        vpred = [ID2LABEL[i] for i in vp.argmax(1)]
        vf1 = f1_score(val["label"], vpred, labels=LABELS, average="macro", zero_division=0)
        vacc = accuracy_score(val["label"], vpred)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "val_accuracy": vacc, "val_macro_f1": vf1})
        print(f"epoch {ep}: train_loss={np.mean(losses):.4f} val_acc={vacc:.3f} val_macro_f1={vf1:.3f}")
        if vf1 > best_f1:  # keep the best epoch by validation macro-F1 (never by test)
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")
    return model, tok, history, best_epoch


# ----------------------------------------------------------------------------- stretch analyses
def calibration(conf, correct):
    bins = [(0.0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]
    table, ece, n = [], 0.0, len(conf)
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        k = int(m.sum())
        acc = float(correct[m].mean()) if k else None
        avg = float(conf[m].mean()) if k else None
        if k:
            ece += k / n * abs(acc - avg)
        table.append({"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": k, "accuracy": acc, "avg_confidence": avg})
    return {"bins": table, "ece": ece}


def error_patterns(pred_df):
    d = pred_df.copy()
    d["words"] = d["text"].str.split().str.len()
    d["length_bucket"] = pd.cut(d["words"], [0, 15, 40, 10**6], labels=["<=15 words", "16-40 words", ">40 words"])
    d["has_number"] = d["text"].str.contains(r"\d")
    d["wrong"] = d["label"] != d["pred"]

    def grp(col):
        g = d.groupby(col, observed=True)["wrong"].agg(["count", "sum"])
        return [{"group": str(i), "n": int(r["count"]), "errors": int(r["sum"]),
                 "error_rate": float(r["sum"] / r["count"])} for i, r in g.iterrows()]

    pairs = d[d["wrong"]].groupby(["label", "pred"]).size().sort_values(ascending=False)
    # Hypothesis from planning.md: short posts with a number get pushed toward analysis
    short_num = d[(d["words"] <= 15) & d["has_number"]]
    return {
        "by_length": grp("length_bucket"),
        "by_has_number": grp("has_number"),
        "confusion_pairs": [{"true": t, "pred": p, "count": int(c)} for (t, p), c in pairs.items()],
        "short_with_number": {"n": len(short_num),
                              "predicted_analysis": int((short_num["pred"] == "analysis").sum()),
                              "true_analysis": int((short_num["label"] == "analysis").sum())},
    }


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS); ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    mx = max(max(r) for r in cm) or 1
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > mx / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/takemeter_nba.csv")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline", choices=["groq", "skip"], default="groq")
    ap.add_argument("--baseline-model", default="meta-llama/llama-4-scout-17b-16e-instruct")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    df, train, val, test, dupes = load_and_split(args.data, args.seed, out)

    # Baseline runs on the locked test set BEFORE the fine-tuned model is evaluated
    base = run_baseline(test, args.baseline_model, out, args.baseline)
    baseline_res = None
    if base is not None:
        unparse = int((base["pred"] == "UNPARSEABLE").sum())
        baseline_res = metrics_block(base["label"], base["pred"])
        baseline_res.update(model=args.baseline_model, unparseable=unparse,
                            unparseable_rate=unparse / len(base))
        print(f"baseline: acc={baseline_res['accuracy']:.3f} macro_f1={baseline_res['macro_f1']:.3f} "
              f"unparseable={unparse}")

    t0 = time.time()
    model, tok, history, best_epoch = fine_tune(train, val, args, device, out)
    train_secs = time.time() - t0

    probs = predict(model, tok, list(test["text"]), args.max_len, device)
    pred = [ID2LABEL[i] for i in probs.argmax(1)]
    conf = probs.max(1)
    pred_df = test[["id", "text", "label"]].copy()
    pred_df["pred"], pred_df["confidence"] = pred, conf
    for i, l in enumerate(LABELS):
        pred_df[f"p_{l}"] = probs[:, i]
    if base is not None:
        pred_df = pred_df.merge(base[["id", "pred"]].rename(columns={"pred": "baseline_pred"}), on="id", how="left")
    pred_df.to_csv(out / "test_predictions.csv", index=False)

    ft = metrics_block(pred_df["label"], pred_df["pred"])
    plot_cm(ft["confusion_matrix"], "Fine-tuned DistilBERT (test)", out / "confusion_matrix.png")
    if baseline_res:
        plot_cm(baseline_res["confusion_matrix"], "Zero-shot Llama 4 Scout (test)", out / "confusion_matrix_baseline.png")

    correct = (pred_df["label"] == pred_df["pred"]).to_numpy()
    results = {
        "labels": LABELS,
        "data": {"path": args.data, "n": len(df), "duplicates_dropped": dupes,
                 "distribution": df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
                 "split_sizes": {"train": len(train), "val": len(val), "test": len(test)}},
        "training": {"base_model": args.model, "epochs": args.epochs, "best_epoch": best_epoch,
                     "learning_rate": args.lr, "batch_size": args.batch_size, "max_len": args.max_len,
                     "weight_decay": 0.01, "warmup_ratio": 0.1, "seed": args.seed, "device": device,
                     "train_seconds": round(train_secs, 1), "history": history},
        "fine_tuned": ft,
        "baseline": baseline_res,
        "calibration": calibration(conf, correct),
        "error_patterns": error_patterns(pred_df),
    }
    (out / "evaluation_results.json").write_text(json.dumps(results, indent=2, default=float))

    print("\n=== TEST RESULTS ===")
    hdr = f"{'':10s}{'fine-tuned':>12s}" + (f"{'baseline':>12s}" if baseline_res else "")
    print(hdr)
    for k in ("accuracy", "macro_f1"):
        print(f"{k:10s}{ft[k]:12.3f}" + (f"{baseline_res[k]:12.3f}" if baseline_res else ""))
    for l in LABELS:
        print(f"F1 {l:7s}{ft['per_class'][l]['f1-score']:12.3f}"
              + (f"{baseline_res['per_class'][l]['f1-score']:12.3f}" if baseline_res else ""))
    if ft["accuracy"] > 0.95:
        print("WARNING: >95% accuracy on a subjective task - check for leakage / too-easy labels.")
    print(f"\nWrote {out}/evaluation_results.json, test_predictions.csv, confusion_matrix.png, model/")


if __name__ == "__main__":
    main()
