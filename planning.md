# TakeMeter — planning.md

*Written before data collection. Updated before stretch features (see §8).*

---

## 1. Community: r/nba

**Choice.** r/nba (about 15M members), mainly the comment sections of **Game Threads, Post-Game Threads, the Daily Discussion Thread, and discussion posts** (e.g. "[Serious] Is Jokić's peak better than Duncan's?").

**Why it fits a classification task.**

- **The distinction is native to the community.** Regulars already sort each other's comments into these buckets: "source?" / "stats or GTFO" for unsupported claims, "this is actually a good breakdown" for film or stat posts, and "game thread comments" as shorthand for pure emotion. The sub has a whole vocabulary for bad takes ("hot take", "take of the day", "overreaction Monday"). A classifier that separates these is modeling a boundary the members already police.
- **Quality varies enormously within one topic.** The same game produces "LETS GOOOO", "Ant is already better than prime Wade", and a 200-word breakdown of how the Wolves switched every screen in the fourth. Topic words alone ("Ant", "Wolves") can't separate the labels, so the model has to learn something about *how* the post argues, not just *what* it's about.
- **Volume and access.** Thousands of public comments a day, all readable without logging in, so 200+ examples is easy to reach.
- **Known risk.** Game threads are mostly reaction, so a random sample would be heavily imbalanced. The collection plan (§4) handles this with targeted sourcing.

**2–3 sentence summary.** TakeMeter classifies r/nba comments by *how* they make their point: `analysis` (the claim is supported by specific, checkable evidence), `hot_take` (a bold, general claim that is asserted rather than argued), or `reaction` (an in-the-moment emotional response to a specific event). The distinction matters on r/nba because the sub constantly fights about take quality, and a tool that can surface real analysis from a 5,000-comment post-game thread, or tell an overreaction apart from a claim, is something users there already ask for.

---

## 2. Labels

Three labels. Each is defined by **what the post does**, not by its topic or its tone.

### `analysis`
**Definition:** The post makes a claim and supports it with specific, checkable evidence (statistics, historical comparison, or concrete tactical/film observation), so the reasoning would still stand if you deleted the opinion framing.

- *Example 1:* "People keep saying the Knicks' defense collapsed, but it's really just the corner three. They were 3rd in opponent corner 3PA before Mitchell Robinson went down and 26th since. The switching scheme leaves the weak-side corner open every time Brunson is the low man."
- *Example 2:* "Tatum's playoff efficiency drop isn't about 'clutch'. His rim attempt rate goes from about 30% in the regular season to about 22% in the playoffs because teams load up on his drives, and that alone accounts for most of the TS% gap."

### `hot_take`
**Definition:** The post asserts a bold, general claim (about a player's ability, legacy, a team's future, a trade, or an award) without evidence that actually supports it; any stat present is decorative or cherry-picked.

- *Example 1:* "Ant is already a better player than prime Wade and it's not even close. People just aren't ready to admit it."
- *Example 2:* "The Suns are the worst-run franchise in the league. Every move they made the last two years has been a disaster."

### `reaction`
**Definition:** The post is an immediate emotional response to a specific moment or game (celebration, frustration, disbelief, a joke about a play) that makes no general claim reaching beyond that moment.

- *Example 1:* "LETS GOOOOO WHAT A SHOT 😭😭"
- *Example 2:* "I can't believe we blew a 20 point lead again. I'm going to bed."

*(The examples above are illustrative posts written in r/nba style. After collection they get replaced with real posts from `data/takemeter_nba.csv`, and the README cites real ones.)*

### Exclusion rule (not a label)
Some comments aren't takes at all: pure questions ("what channel is the game on?"), mod/bot comments, links with no text, non-English, and fragments under 4 words that carry no evaluation ("lol", "this"). These are **dropped during collection, not labeled**, and the drop count is recorded. That makes the taxonomy's coverage measurable: if more than 10% of *substantive* comments can't be placed in one of the three labels, the taxonomy fails the "exhaustive enough" requirement and gets revised.

---

## 3. Hard edge cases and decision rules

### Edge case A (the hardest): the one-stat take — `hot_take` vs `analysis`
> "Luka is overrated. He's shooting 31% from three in the playoffs."

It cites a real, checkable number, but the claim ("overrated") is far bigger than what one shooting split can support, and the stat is picked for effect.

**Decision rule, the deletion test:** Remove the opinion words ("overrated", "not even close", "washed"). If the remaining evidence would, *on its own*, lead a reasonable reader toward the claim, and it is specific (a number, a named play or scheme, a sourced comparison), label it `analysis`. If the evidence is a single number that doesn't establish the claim, is vague ("his numbers are bad"), or is clearly cherry-picked, label it `hot_take`.
→ The Luka post is `hot_take`. A post that pairs the shooting number with a mechanism (e.g. shot volume, defensive attention, a comparison to his own baseline) is `analysis`.

### Edge case B: the event-triggered claim — `reaction` vs `hot_take`
> "Embiid disappears in the playoffs AGAIN. Never trusting this guy."

It's clearly emotional and triggered by tonight's game, but it makes a general claim ("disappears in the playoffs" as a pattern).

**Decision rule, the scope test:** If the post makes a claim that reaches **beyond the specific moment** (a general statement about a player, team, or legacy), it is `hot_take`, however emotional. If it stays **inside the moment** (this shot, this game, this call), it is `reaction`.
→ `hot_take`. "Embiid was awful tonight, 4-for-19, brutal" stays in the moment → `reaction`.

### Edge case C: sarcasm and jokes
> "Refs really wanted that series to go 7 huh"

Label what the post *does*. A joke about a specific call or game is `reaction`. Sarcasm that encodes a general claim ("Great, another max contract for a guy who can't guard a chair") is `hot_take`.

### During annotation
Any post that takes more than a few seconds of thought gets a note in the `notes` column: which labels it could be, which rule decided it. Those notes become the README's "difficult examples" section. If one rule turns out to be triggered constantly, or applied inconsistently, I'll revise it, then **re-check every earlier label under the revised rule** and log the change in §8.

---

## 4. Data collection plan

- **Where:** public r/nba threads, read through Reddit's public `.json` endpoints (no login) with `scripts/collect_reddit.py`, which I run on my laptop because Reddit blocks many cloud IPs. Sources:
  1. Game Threads and Post-Game Threads from about 10 different games (to spread across teams and fanbases) — mostly `reaction` and `hot_take`.
  2. Daily Discussion Threads and top discussion posts of the week — mostly `hot_take`.
  3. `[OC]` stat posts and `[Serious]` discussion threads, plus long comments (≥ 60 words) — the main source of `analysis`.
- **Pool:** collect about 600 candidate comments, drop exclusions (§2), dedupe, then sample for labeling.
- **Target:** 240 labeled examples, **about 80 per label** (so no label is under 30%, comfortably above the 20% hint). The test split will then be about 36 examples, about 12 per class.
- **If a label is underrepresented after 240:** `analysis` is the likely one. I'll do targeted collection from `[OC]`/`[Serious]` threads and the longest comments in post-game threads until it reaches at least 25%. I will **not** duplicate or paraphrase examples to balance classes: duplicates would leak across the train/test split and inflate test scores. Collection stops when every label has at least 60 examples.
- **Format:** one CSV, `data/takemeter_nba.csv`, with columns `id, text, label, notes, source_thread, prelabel, prelabel_model, label_changed`. No usernames are stored.

---

## 5. Evaluation metrics and why

| Metric | Why it's needed for *this* task |
|---|---|
| **Macro-F1 (primary)** | Classes are roughly balanced by design, but the task is about all three distinctions. Macro-F1 weights each class equally, so the model can't hide a failure on `analysis` behind a good `reaction` score. |
| **Per-class precision / recall / F1** | The errors have different costs. If TakeMeter is used to *surface* analysis in a thread, a `hot_take` passed off as analysis (low `analysis` precision) is the worst outcome, because it rewards exactly what the tool is meant to filter. Missing a real analysis post (low recall) is a smaller cost. |
| **Confusion matrix** | The interesting question is *which boundary* fails. I expect `analysis ↔ hot_take` (edge case A) and `hot_take ↔ reaction` (edge case B) to fail differently. The matrix shows the direction of each error. |
| **Accuracy** | Reported because it's required and easy to read, but secondary. On about 36 test examples, each example is about 2.8 points of accuracy. |
| **Baseline delta** | Same metrics for the zero-shot Llama 4 Scout baseline on the **identical** test set, so I can tell whether fine-tuning added anything a prompt couldn't. |

**Small-test-set caveat:** with about 36 test examples, a 95% interval on accuracy is roughly ±15 points. A fine-tuned-vs-baseline gap smaller than about 10 points is reported as "no clear difference", not as a win.

---

## 6. Definition of success

Measured on the held-out test set, with the thresholds fixed now, before seeing any results:

1. **Macro-F1 ≥ 0.70** for the fine-tuned model.
2. **Every class F1 ≥ 0.60**, so no boundary is left unlearned.
3. **`analysis` precision ≥ 0.75**, the deployment-critical number from §5.
4. **The fine-tuned model beats the zero-shot baseline by ≥ 0.05 macro-F1.** If it doesn't, fine-tuning didn't justify itself, and that becomes the headline finding.

**"Good enough to deploy" in a real community tool** (e.g. a bot that suggests an "Analysis" flair or pins the top analysis comment in post-game threads): `analysis` precision ≥ 0.85 with a confidence threshold, where the bot abstains below the threshold. A wrong "Analysis" badge would cost the tool its credibility with r/nba users fast, so precision matters more than coverage.

**Red flag:** accuracy > 95% triggers a leakage check (duplicate or near-duplicate texts across splits) before anything gets reported.

---

## 7. AI Tool Plan

### 7a. Label stress-testing — **yes**
- **Tool:** Claude.
- **Prompt:** give it the three definitions plus edge cases A–C, and ask for 10 r/nba-style comments that sit exactly on the `analysis/hot_take` and `hot_take/reaction` boundaries.
- **Pass criterion:** I can label at least 8/10 with one of my decision rules in under 10 seconds each. Every post I can't place cleanly leads to a rule revision, logged in §8.

### 7b. Annotation assistance — **yes, pre-labeling with full review**
- **Tool:** Groq, `llama-3.3-70b-versatile`. This is **deliberately not** the baseline model (Llama 4 Scout). If the baseline model pre-labeled the data, the test set's gold labels would lean toward the baseline's own opinions and inflate the baseline score.
- **Process:** `scripts/prelabel_groq.py` writes a `prelabel` column. Then `scripts/review_labels.py` shows me each post **with the pre-label hidden until I've entered my own label**, so I'm labeling, not rubber-stamping. It records `label_changed` when I disagree with the pre-label.
- **Tracking and disclosure:** `prelabel`, `prelabel_model`, and `label_changed` stay in the final CSV. The README AI-usage section reports how many pre-labels I overrode, broken down by class.

### 7c. Failure analysis — **yes**
- **Tool:** Claude.
- **Input:** `outputs/test_predictions.csv` filtered to the wrong predictions (text, gold, predicted, confidence).
- **Ask:** find patterns across length, presence of numbers, sarcasm, emotional intensity, and specific label pairs.
- **Verification:** I don't accept any pattern until I've counted it myself. `train_eval.py` already computes error rate by length bucket and by has-number / no-number. Any pattern I can't confirm with a count gets reported as "suggested but not supported".

---

## 8. Change log and stretch-feature plan

*(Updated before starting stretch features.)*

| Stretch feature | Plan |
|---|---|
| **Confidence calibration** | Bin test predictions by max softmax probability (<0.6, 0.6–0.8, 0.8–0.9, ≥0.9) and report accuracy per bin plus ECE. Success = accuracy rises with confidence. With about 36 examples, bins will be small, so I report counts alongside rates. |
| **Error pattern analysis** | Error rate by length bucket (≤15 / 16–40 / >40 words), by contains-a-number, and by confusion pair. Hypothesis written in advance: **short posts containing a number get pushed toward `analysis`**, because the model learns "digits ⇒ analysis" instead of the deletion test. |
| **Deployed interface** | A Gradio app (`app.py`) that loads the saved model and shows the label plus all three class probabilities. It runs locally or in Colab with a public share link. |
| **Inter-annotator reliability** | A second person labels 40 random examples using only this document. I report Cohen's kappa (`scripts/iaa.py`) and read every disagreement to see which decision rule caused it. |

**Rule changes during annotation:** *(fill in as they happen — e.g. "after example #57, extended edge case B to cover 'never trusting X again' style posts → re-checked #1–56, 3 labels changed.")*
