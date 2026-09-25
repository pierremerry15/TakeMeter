# TakeMeter: classifying take quality on r/nba

A fine-tuned DistilBERT classifier that labels r/nba comments as **`analysis`**, **`hot_take`**, or **`reaction`**, compared against a zero-shot Llama 4 Scout baseline on the same held-out test set.

📹 **Demo video (3–5 min):** _link here_
📄 **Design doc:** [`planning.md`](planning.md) · **Dataset:** [`data/takemeter_nba.csv`](data/takemeter_nba.csv) · **Raw results:** [`outputs/evaluation_results.json`](outputs/evaluation_results.json)

> Tables marked *auto-generated* are written by `scripts/make_report.py` from the actual run outputs, so every number in this README traces back to `outputs/`. Sections marked ✏️ are analysis that has to be written from the real results after the run.

---

## 1. Community choice and reasoning

**Community:** r/nba (about 15M members). Comments were taken from Game Threads, Post-Game Threads, Daily Discussion Threads, and `[OC]` / `[Serious]` discussion posts.

**Why r/nba.** The take-quality distinction is already part of how the sub talks about itself. Users reply "stats or GTFO" to unsupported claims, call out "game thread takes", and upvote film breakdowns as "actual analysis". So the labels model a boundary that community members already enforce, not one invented for this project. The same game produces all three kinds of post ("LETS GOOO", "Ant > prime Wade", a 200-word breakdown of the Wolves' switching scheme), so topic words can't separate the labels. The model has to pick up *how* a post argues. Volume is effectively unlimited and everything is public.

**What the tool would be for.** Surfacing real analysis inside a 5,000-comment post-game thread (e.g. an "Analysis" flair suggestion or a pinned "best breakdowns" digest), where mislabeling a hot take as analysis is the costly error.

---

## 2. Label taxonomy

| Label | Definition |
|---|---|
| **`analysis`** | The post makes a claim and supports it with specific, checkable evidence (statistics, historical comparison, or concrete tactical/film observation), so the reasoning would still stand if you deleted the opinion framing. |
| **`hot_take`** | The post asserts a bold, general claim (about a player's ability, legacy, a team's future, a trade, or an award) without evidence that actually supports it; any stat present is decorative or cherry-picked. |
| **`reaction`** | The post is an immediate emotional response to a specific moment or game (celebration, frustration, disbelief, a joke about a play) that makes no general claim reaching beyond that moment. |

**Examples** (✏️ swap in two real posts per label from the dataset; the ones below are the illustrative examples from `planning.md`):

- `analysis`
  1. "People keep saying the Knicks' defense collapsed, but it's really just the corner three. They were 3rd in opponent corner 3PA before Mitchell Robinson went down and 26th since…"
  2. "Tatum's playoff efficiency drop isn't about 'clutch'. His rim attempt rate goes from about 30% to about 22% in the playoffs because teams load up on his drives…"
- `hot_take`
  1. "Ant is already a better player than prime Wade and it's not even close."
  2. "The Suns are the worst-run franchise in the league. Every move they made the last two years has been a disaster."
- `reaction`
  1. "LETS GOOOOO WHAT A SHOT 😭😭"
  2. "I can't believe we blew a 20 point lead again. I'm going to bed."

**Decision rules for the boundaries** (full reasoning in `planning.md` §3):

1. **Deletion test (`analysis` vs `hot_take`).** Delete the opinion words. If the remaining evidence is specific and would on its own lead a reader toward the claim, it's `analysis`. One cherry-picked stat is `hot_take`.
2. **Scope test (`hot_take` vs `reaction`).** A claim that reaches beyond tonight's game ("Embiid *always* disappears") is `hot_take`, however emotional. A post that stays inside the moment ("Embiid was awful tonight") is `reaction`.
3. **Sarcasm.** Label what it does: a joke about a play is `reaction`; sarcasm carrying a general claim is `hot_take`.

---

## 3. Dataset

### Collection

`scripts/collect_reddit.py` pulls comments from public r/nba threads through Reddit's public `.json` endpoints (no login; usernames are not stored). It searches for Post-Game Threads, Game Threads, Daily Discussions, `[OC]`, and `[Serious]` posts from the past month. It keeps up to 40 comments per thread, a quarter of them the longest in the thread so that `analysis` isn't starved. It automatically drops deleted, bot/mod, link-only, pure-question, and under-4-word comments, and removes duplicates.

### Labeling process

1. **Pre-labeling (disclosed).** `scripts/prelabel_groq.py` pre-labeled candidates with Groq `llama-3.3-70b-versatile`. This is deliberately **not** the baseline model, so the gold labels don't lean toward the baseline's opinions.
2. **Blind review.** `scripts/review_labels.py` shows each post **with the pre-label hidden**. I enter my own label first, and the pre-label is revealed only afterwards, so disagreements are my call, not a rubber stamp. Posts that fit no label are marked `excluded`, which measures how much of the community the taxonomy actually covers.
3. **Notes.** Every post that took real thought got a note in the `notes` column, naming the competing labels and the rule that decided it.

### Label distribution *(auto-generated)*

<!-- AUTO:distribution -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:distribution -->

### Three difficult examples and what I decided

✏️ Pick three real posts from the `notes` column. Suggested format:

> **Post:** "…"
> **Could be:** `hot_take` or `analysis`
> **Decided:** `hot_take`, by the deletion test: without "overrated", all that's left is one shooting split from a 5-game series, which doesn't establish the claim.
> **Did it change the rules?** …

1. …
2. …
3. …

---

## 4. Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (66M parameters). A new 3-way classification head is trained on top, and all weights are updated (full fine-tuning).
- **Split:** stratified 70/15/15 (seed 42). Duplicate texts are removed *before* splitting so nothing leaks into the test set. Split membership is saved to `outputs/splits.csv`.
- **Training:** AdamW, linear LR decay with 10% warmup, gradient clipping at 1.0. The model is evaluated on the validation set after every epoch, and the **epoch with the best validation macro-F1 is kept**. The test set is never used for any choice.

### Key hyperparameter decision: 5 epochs at LR 3e-5 with best-epoch selection (vs. the notebook default of 3 epochs at 2e-5)

With about 168 training examples and batch size 16, one epoch is only about 11 optimizer steps. The default 3 epochs at 2e-5 gives about 33 updates total, and 10% of those are warmup. On a dataset this small that tends to *underfit*: the new classification head starts random and needs enough updates to separate three classes. I raised the budget to 5 epochs and nudged the learning rate to 3e-5. Because more epochs on tiny data risks *overfitting*, I added best-epoch checkpointing on validation macro-F1: the extra epochs are kept only if they actually help on held-out data. The per-epoch table below shows whether that was needed.

*(auto-generated)*

<!-- AUTO:training -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:training -->

---

## 5. Baseline: zero-shot Llama 4 Scout

- **Model:** Groq `meta-llama/llama-4-scout-17b-16e-instruct`, `temperature=0`, `max_tokens=5`, no examples (zero-shot).
- **Same test set:** the baseline classifies exactly the test rows listed in `outputs/splits.csv`. It was run **before** the fine-tuned model was evaluated. Responses are cached in `outputs/baseline_predictions.csv`, so re-runs never re-query or cherry-pick.
- **Parsing:** the response is lower-cased and stripped. If it contains exactly one label name, that's the prediction. Otherwise it's counted as `UNPARSEABLE` and scored as wrong.
- **Prompt** (built from the same definitions as `planning.md`, in `takemeter_labels.py`):

```text
You are classifying comments from the r/nba subreddit by HOW they make their point.

Labels:
- analysis: The post makes a claim and supports it with specific, checkable evidence (statistics, historical comparison, or concrete tactical/film observation), so the reasoning would still stand if you deleted the opinion framing.
- hot_take: The post asserts a bold, general claim (about a player's ability, legacy, a team's future, a trade, or an award) without evidence that actually supports it; any stat present is decorative or cherry-picked.
- reaction: The post is an immediate emotional response to a specific moment or game (celebration, frustration, disbelief, a joke about a play) that makes no general claim reaching beyond that moment.

Decision rules for borderline posts:
1. Deletion test (analysis vs hot_take): remove the opinion words ("overrated", "washed",
   "not even close"). If the remaining evidence is specific and would on its own lead a
   reasonable reader toward the claim, it is analysis. If the evidence is a single
   cherry-picked number, vague ("his numbers are bad"), or decorative, it is hot_take.
2. Scope test (reaction vs hot_take): if the post makes a claim that reaches beyond the
   specific moment (a general statement about a player, team, or legacy), it is hot_take,
   however emotional. If it stays inside the moment (this shot, this game, this call),
   it is reaction.
3. Sarcasm/jokes: label what the post does. A joke about a specific play or call is
   reaction; sarcasm that encodes a general claim is hot_take.

Comment:
"""{text}"""

Respond with exactly one word, the label name: analysis, hot_take, or reaction. No punctuation, no explanation.
```

The baseline gets the *full* decision rules, not just label names. That makes it a strong baseline, so if fine-tuning wins, it wins against an LLM that had the same instructions as the human annotator.

---

## 6. Evaluation report

### 6.1 Overall and per-class results *(auto-generated)*

<!-- AUTO:metrics -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:metrics -->

### 6.2 Confusion matrices *(auto-generated; rows = true, columns = predicted, bold = correct)*

**Fine-tuned DistilBERT** (supplementary image: [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png))

<!-- AUTO:confusion_ft -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:confusion_ft -->

**Zero-shot baseline**

<!-- AUTO:confusion_base -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:confusion_base -->

### 6.3 Did it meet the success criteria from `planning.md`? *(auto-generated)*

<!-- AUTO:success -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:success -->

**Caveat on test-set size:** with about 36 test examples, one example is about 2.8 accuracy points, and the 95% interval on accuracy is roughly ±15 points. Gaps under about 10 points between the models are "no clear difference".

✏️ **Summary (2–4 sentences):** Which model won, by how much, and on which class? Did the baseline struggle where you predicted during Milestone 4?

### 6.4 All wrong predictions from the fine-tuned model *(auto-generated, most confident first)*

<!-- AUTO:wrong -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:wrong -->

### 6.5 Three failures analyzed

✏️ For three rows from 6.4, answer the four guiding questions. Choose at least one high-confidence error, because a confident mistake says the most about what the model learned. Format:

> **Post:** "…" · **True:** `hot_take` · **Predicted:** `analysis` (91%)
> **Which boundary:** analysis ↔ hot_take, the same direction as the largest off-diagonal cell above.
> **Why it's hard:** the post contains two percentages, but they're decorative…
> **Labeling or data problem?** I re-checked the similar posts in train: 5 of 7 short posts with a stat were labeled `hot_take`, so the labels are consistent. The model is overweighting digits.
> **Fix:** add 20–30 more "one-stat hot takes" to training so the model sees digits without the analysis structure.

1. …
2. …
3. …

**AI-assisted pattern search** (planned in `planning.md` §7c): ✏️ What Claude suggested when given the wrong predictions, which suggestions the counts in §8.2 confirmed, and which you discarded.

### 6.6 Sample classifications *(auto-generated; one high-confidence correct prediction per class plus the two most confident mistakes)*

<!-- AUTO:samples -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:samples -->

✏️ **Why one correct prediction is reasonable:** pick one ✅ row and explain in a sentence which feature of the text matches the label definition (e.g. "no claim beyond the play itself, all-caps and emoji, and it names a single shot → `reaction` by the scope test").

---

## 7. Reflection: what the model learned vs. what I intended

✏️ Write this after reading §6 and §8.2. It's about the gap between the *definitions* and the *decision boundary*, not a list of errors. Questions to answer:

- **Intended:** `analysis` = evidence that *supports* the claim (the deletion test). **Learned?** Did it learn "support", or surface proxies like digits, length, or words like "per game" and "since"? (Check §8.2: error rate on short posts with numbers.)
- **Intended:** `reaction` vs `hot_take` = scope beyond the moment. **Learned?** Or did it learn "caps and emoji = reaction", which would misfire on calm reactions and all-caps hot takes?
- **What did it overfit to?** Player names that appeared mostly under one label? Thread type?
- **What did it miss entirely?** Sarcasm? Long hot takes that *sound* analytical?
- **One concrete change** to the labels or the data that would close the biggest gap.

---

## 8. Stretch features

### 8.1 Confidence calibration *(auto-generated)*

<!-- AUTO:calibration -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:calibration -->

✏️ Does accuracy go up with confidence? Is a ≥90% prediction actually right more often than a 60–80% one? Note bin sizes: with about 36 test examples, a bin of 4 is anecdotal. Would a confidence threshold make the "Analysis flair bot" from §1 safe to deploy?

### 8.2 Error pattern analysis *(auto-generated)*

<!-- AUTO:errors -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:errors -->

✏️ State the systematic pattern in one sentence (e.g. "short posts with a number are predicted `analysis` X of Y times but are `analysis` only Z times"). Say whether it confirms or refutes the hypothesis written in `planning.md` §8 before training.

### 8.3 Deployed interface

`app.py` is a Gradio web app that takes a comment and shows the predicted label with the probability for every class.

```bash
pip install -r requirements.txt
python app.py                         # http://127.0.0.1:7860
python app.py --share                 # public link (use in Colab)
python app.py "Jokic is the most skilled big man ever, no debate"   # terminal mode
```

### 8.4 Inter-annotator reliability *(auto-generated)*

<!-- AUTO:iaa -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:iaa -->

✏️ Which rule produced the most disagreement? Was it the deletion test or the scope test?

---

## 9. Spec reflection

✏️ **One way the spec helped:** e.g. fixing the success thresholds and the "pre-label with a different model than the baseline" rule *before* collecting data meant … (be specific about a moment it changed a decision).

✏️ **One way implementation diverged, and why:** e.g. a decision rule you had to extend during annotation (log it in `planning.md` §8), a label that came in under target and needed targeted collection, or a hyperparameter you changed after seeing the validation curve.

---

## 10. AI usage

1. **Repository scaffolding and planning draft (Claude).** I directed Claude to draft `planning.md` from the assignment spec and to write the pipeline scripts (collection, pre-labeling, blind review tool, training/evaluation, report generator, Gradio app). ✏️ *What I changed or overrode:* …
2. **Label stress-testing (Claude).** I gave it the definitions and edge cases and asked for 10 boundary posts. ✏️ *Result:* I placed X/10 cleanly. The ones I couldn't led to …
3. **Annotation assistance (Groq `llama-3.3-70b-versatile`).** Pre-labels were hidden during review and shown only after I entered my own label. *(auto-generated stats:)*

<!-- AUTO:prelabel -->
_Run `python scripts/make_report.py` after training to fill this in._
<!-- /AUTO:prelabel -->

4. **Failure analysis (Claude).** See §6.5. ✏️ Which suggested patterns held up against the §8.2 counts, and which were discarded.

---

## Reproducing this

```bash
pip install -r requirements.txt

# 1. Collect (run on your own computer; Reddit blocks many cloud IPs)
python scripts/collect_reddit.py --auto

# 2. (optional) pre-label with a NON-baseline model, then label everything yourself
export GROQ_API_KEY=...
python scripts/prelabel_groq.py --n 320
python scripts/review_labels.py --target 240

# 3. Train + baseline + evaluate (Colab T4: about 5 min; the notebook does this for you)
python train_eval.py --data data/takemeter_nba.csv

# 4. Fill every auto-generated table in this README
python scripts/make_report.py
```

Or open [`TakeMeter_Colab.ipynb`](TakeMeter_Colab.ipynb) in Colab, which runs steps 3–4 and the demo app.

```
├── planning.md                 design doc (graded separately)
├── README.md                   this report
├── takemeter_labels.py         labels, definitions, prompt (shared by every script)
├── train_eval.py               split → baseline → fine-tune → evaluate → export
├── app.py                      Gradio interface (stretch)
├── TakeMeter_Colab.ipynb       one-click Colab runner
├── scripts/
│   ├── collect_reddit.py       public r/nba comment collection
│   ├── prelabel_groq.py        optional LLM pre-labeling
│   ├── review_labels.py        blind labeling tool
│   ├── make_report.py          fills README tables from outputs/
│   └── iaa.py                  inter-annotator agreement (stretch)
├── data/takemeter_nba.csv      the labeled dataset
└── outputs/                    evaluation_results.json, confusion_matrix.png, predictions
```
