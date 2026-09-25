"""Fill every <!-- AUTO:name --> ... <!-- /AUTO:name --> block in README.md from real outputs.

    python scripts/make_report.py

Reads outputs/evaluation_results.json, outputs/test_predictions.csv, data/takemeter_nba.csv,
and (if present) data/collection_log.json and outputs/iaa_results.json. Re-running is safe:
only the AUTO blocks are replaced, so your hand-written analysis is never touched.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from takemeter_labels import LABELS  # noqa: E402


def esc(t, n=160):
    t = str(t).replace("|", "\\|").replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def f3(x):
    return "n/a" if x is None else f"{x:.2f}"


def cm_table(cm):
    lines = ["| true ↓ / predicted → | " + " | ".join(f"`{l}`" for l in LABELS) + " | total |",
             "|---|" + "---:|" * (len(LABELS) + 1)]
    for l, row in zip(LABELS, cm):
        cells = [f"**{v}**" if i == LABELS.index(l) else str(v) for i, v in enumerate(row)]
        lines.append(f"| `{l}` | " + " | ".join(cells) + f" | {sum(row)} |")
    return "\n".join(lines)


def build(res, preds, data):
    ft, bl = res["fine_tuned"], res["baseline"]
    blocks = {}

    # --- data distribution
    labeled = data[data["label"].isin(LABELS)]
    dist = res["data"]["distribution"]  # after duplicate removal = what the model saw
    n = sum(dist.values())
    rows = ["| Label | Count | Share |", "|---|---:|---:|"]
    for l in LABELS:
        c = int(dist[l])
        rows.append(f"| `{l}` | {c} | {c / n:.1%} |")
    rows.append(f"| **Total labeled** | **{n}** | 100% |")
    extra = []
    excl = int((data["label"] == "excluded").sum())
    log_p = ROOT / "data" / "collection_log.json"
    if log_p.exists():
        log = json.loads(log_p.read_text())
        dropped = sum(log["dropped"].values())
        extra.append(f"- Collection: {log['threads']} threads → {log['candidates']} candidate comments "
                     f"after automatically dropping {dropped} non-takes/duplicates "
                     f"({', '.join(f'{k}: {v}' for k, v in log['dropped'].items())}).")
    if excl or n:
        extra.append(f"- During manual review, **{excl}** of {n + excl} substantive-looking comments "
                     f"({excl / max(n + excl, 1):.1%}) were marked `excluded` because they fit no label, "
                     f"so the taxonomy covered {n / max(n + excl, 1):.1%} (requirement: ≥ 90%).")
    sz = res["data"]["split_sizes"]
    extra.append(f"- Stratified split (seed {res['training']['seed']}): train {sz['train']} / "
                 f"val {sz['val']} / test {sz['test']}; duplicate texts dropped before splitting: "
                 f"{res['data']['duplicates_dropped']}.")
    blocks["distribution"] = "\n".join(rows) + "\n\n" + "\n".join(extra)

    # --- pre-label stats (AI usage disclosure)
    if "prelabel" in data and (data["prelabel"].astype(str).isin(LABELS)).any():
        pl = labeled[labeled["prelabel"].astype(str).isin(LABELS)]
        changed = pl[pl["label"] != pl["prelabel"]]
        r = [f"- {len(pl)} of {n} labeled examples had an LLM pre-label "
             f"(`{pl['prelabel_model'].iloc[0]}`); I overrode **{len(changed)}** ({len(changed) / max(len(pl), 1):.1%}).",
             "", "| Pre-label → my label | Count |", "|---|---:|"]
        for (a, b), c in changed.groupby(["prelabel", "label"]).size().sort_values(ascending=False).items():
            r.append(f"| `{a}` → `{b}` | {c} |")
        blocks["prelabel"] = "\n".join(r)
    else:
        blocks["prelabel"] = "- No LLM pre-labeling was used; every example was labeled by hand."

    # --- training
    t = res["training"]
    r = ["| Setting | Value |", "|---|---|",
         f"| Base model | `{t['base_model']}` |",
         f"| Epochs (max) / best epoch kept | {t['epochs']} / {t['best_epoch']} (chosen by validation macro-F1) |",
         f"| Learning rate | {t['learning_rate']:g} (linear decay, {t['warmup_ratio']:.0%} warmup) |",
         f"| Batch size | {t['batch_size']} |",
         f"| Max sequence length | {t['max_len']} tokens |",
         f"| Weight decay | {t['weight_decay']} |",
         f"| Hardware / time | {t['device']} / {t['train_seconds']} s |",
         "", "| Epoch | Train loss | Val accuracy | Val macro-F1 |", "|---:|---:|---:|---:|"]
    for h in t["history"]:
        r.append(f"| {h['epoch']} | {h['train_loss']:.3f} | {h['val_accuracy']:.3f} | {h['val_macro_f1']:.3f} |")
    blocks["training"] = "\n".join(r)

    # --- overall metrics
    r = ["| Metric (test set) | Zero-shot baseline | Fine-tuned DistilBERT | Δ |", "|---|---:|---:|---:|"]
    for k, name in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        b = bl[k] if bl else None
        d = f"{ft[k] - b:+.2f}" if bl else "n/a"
        r.append(f"| {name} | {f3(b)} | {ft[k]:.2f} | {d} |")
    if bl:
        r.append(f"| Unparseable baseline responses | {bl['unparseable']} ({bl['unparseable_rate']:.0%}) | – | |")
    r += ["", "**Per-class metrics**", "",
          "| Label | Baseline P | Baseline R | Baseline F1 | Fine-tuned P | Fine-tuned R | Fine-tuned F1 | Support |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for l in LABELS:
        f = ft["per_class"][l]
        b = bl["per_class"][l] if bl else {}
        r.append(f"| `{l}` | {f3(b.get('precision'))} | {f3(b.get('recall'))} | {f3(b.get('f1-score'))} | "
                 f"{f['precision']:.2f} | {f['recall']:.2f} | {f['f1-score']:.2f} | {int(f['support'])} |")
    blocks["metrics"] = "\n".join(r)

    blocks["confusion_ft"] = cm_table(ft["confusion_matrix"])
    blocks["confusion_base"] = cm_table(bl["confusion_matrix"]) if bl else "_Baseline not run._"

    # --- success criteria check (thresholds fixed in planning.md §6)
    checks = [
        ("Macro-F1 ≥ 0.70", ft["macro_f1"] >= 0.70, f"{ft['macro_f1']:.2f}"),
        ("Every class F1 ≥ 0.60", min(ft["per_class"][l]["f1-score"] for l in LABELS) >= 0.60,
         ", ".join(f"{l} {ft['per_class'][l]['f1-score']:.2f}" for l in LABELS)),
        ("`analysis` precision ≥ 0.75", ft["per_class"]["analysis"]["precision"] >= 0.75,
         f"{ft['per_class']['analysis']['precision']:.2f}"),
    ]
    if bl:
        checks.append(("Beats baseline by ≥ 0.05 macro-F1", ft["macro_f1"] - bl["macro_f1"] >= 0.05,
                       f"{ft['macro_f1'] - bl['macro_f1']:+.2f}"))
    r = ["| Criterion (set before training) | Result | Met? |", "|---|---:|:---:|"]
    r += [f"| {c} | {v} | {'✅' if ok else '❌'} |" for c, ok, v in checks]
    blocks["success"] = "\n".join(r)

    # --- wrong predictions (raw material for the 3 analyzed failures)
    wrong = preds[preds["label"] != preds["pred"]].sort_values("confidence", ascending=False)
    has_b = "baseline_pred" in preds
    r = ["| # | Post | True | Predicted | Confidence |" + (" Baseline |" if has_b else ""),
         "|---:|---|---|---|---:|" + ("---|" if has_b else "")]
    for i, (_, w) in enumerate(wrong.iterrows(), 1):
        r.append(f"| {i} | {esc(w['text'], 220)} | `{w['label']}` | `{w['pred']}` | {w['confidence']:.0%} |"
                 + (f" `{w['baseline_pred']}` |" if has_b else ""))
    blocks["wrong"] = "\n".join(r) if len(wrong) else "_No wrong predictions on the test set._"

    # --- sample classifications: one confident correct per label + the most confident mistakes
    right = preds[preds["label"] == preds["pred"]].sort_values("confidence", ascending=False)
    samp = pd.concat([right[right["label"] == l].head(1) for l in LABELS] + [wrong.head(2)])
    r = ["| Post | Predicted label | Confidence | Gold label | Correct? |", "|---|---|---:|---|:---:|"]
    for _, s in samp.iterrows():
        r.append(f"| {esc(s['text'], 200)} | `{s['pred']}` | {s['confidence']:.1%} | `{s['label']}` | "
                 f"{'✅' if s['pred'] == s['label'] else '❌'} |")
    blocks["samples"] = "\n".join(r)

    # --- calibration
    c = res["calibration"]
    r = ["| Confidence bin | n | Accuracy | Avg. confidence |", "|---|---:|---:|---:|"]
    r += [f"| {b['bin']} | {b['n']} | {pct(b['accuracy'])} | {pct(b['avg_confidence'])} |" for b in c["bins"]]
    r.append(f"\nExpected calibration error (ECE): **{c['ece']:.3f}**")
    blocks["calibration"] = "\n".join(r)

    # --- error patterns
    e = res["error_patterns"]
    r = ["| Slice | n | Errors | Error rate |", "|---|---:|---:|---:|"]
    r += [f"| {g['group']} | {g['n']} | {g['errors']} | {g['error_rate']:.0%} |" for g in e["by_length"]]
    r += [f"| contains a number = {g['group']} | {g['n']} | {g['errors']} | {g['error_rate']:.0%} |"
          for g in e["by_has_number"]]
    r += ["", "| Confusion (true → predicted) | Count |", "|---|---:|"]
    r += [f"| `{p['true']}` → `{p['pred']}` | {p['count']} |" for p in e["confusion_pairs"]]
    s = e["short_with_number"]
    r.append(f"\nPre-registered hypothesis check, *short posts (≤15 words) containing a number*: "
             f"{s['n']} in test, {s['predicted_analysis']} predicted `analysis`, {s['true_analysis']} actually `analysis`.")
    blocks["errors"] = "\n".join(r)

    # --- IAA
    iaa_p = ROOT / "outputs" / "iaa_results.json"
    if iaa_p.exists():
        i = json.loads(iaa_p.read_text())
        blocks["iaa"] = (f"n = {i['n']} · simple agreement **{i['agreement']:.1%}** · "
                         f"Cohen's κ **{i['cohen_kappa']:.2f}**\n\n" + cm_table(i["confusion_me_rows_other_cols"])
                         .replace("true ↓ / predicted →", "me ↓ / second annotator →"))
    else:
        blocks["iaa"] = "_Not run yet: `python scripts/iaa.py make-sheet`, then `score`._"
    return blocks


def main():
    res = json.loads((ROOT / "outputs" / "evaluation_results.json").read_text())
    preds = pd.read_csv(ROOT / "outputs" / "test_predictions.csv")
    data = pd.read_csv(ROOT / res["data"]["path"] if not Path(res["data"]["path"]).is_absolute()
                       else res["data"]["path"]).fillna("")
    blocks = build(res, preds, data)
    readme_p = ROOT / "README.md"
    readme = readme_p.read_text()
    for name, body in blocks.items():
        pat = re.compile(rf"(<!-- AUTO:{name} -->)(.*?)(<!-- /AUTO:{name} -->)", re.S)
        if pat.search(readme):
            readme = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", readme)
        else:
            print(f"(README has no AUTO:{name} block - skipped)")
    readme_p.write_text(readme)
    print(f"README.md updated: {', '.join(blocks)}")


if __name__ == "__main__":
    main()
