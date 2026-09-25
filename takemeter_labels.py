"""Single source of truth for the TakeMeter label set, definitions, and LLM prompts.

The same definitions are used by the pre-labeler, the Groq baseline, and the README,
so the baseline is prompted with exactly the definitions written in planning.md.
"""

LABELS = ["analysis", "hot_take", "reaction"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

DEFINITIONS = {
    "analysis": (
        "The post makes a claim and supports it with specific, checkable evidence "
        "(statistics, historical comparison, or concrete tactical/film observation), "
        "so the reasoning would still stand if you deleted the opinion framing."
    ),
    "hot_take": (
        "The post asserts a bold, general claim (about a player's ability, legacy, a "
        "team's future, a trade, or an award) without evidence that actually supports "
        "it; any stat present is decorative or cherry-picked."
    ),
    "reaction": (
        "The post is an immediate emotional response to a specific moment or game "
        "(celebration, frustration, disbelief, a joke about a play) that makes no "
        "general claim reaching beyond that moment."
    ),
}

DECISION_RULES = """\
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
   reaction; sarcasm that encodes a general claim is hot_take."""


def build_prompt(text: str) -> str:
    defs = "\n".join(f"- {l}: {d}" for l, d in DEFINITIONS.items())
    return f"""You are classifying comments from the r/nba subreddit by HOW they make their point.

Labels:
{defs}

{DECISION_RULES}

Comment:
\"\"\"{text}\"\"\"

Respond with exactly one word, the label name: analysis, hot_take, or reaction. No punctuation, no explanation."""


def parse_label(response: str):
    """Map a raw LLM response to a label, or None if unparseable."""
    if not response:
        return None
    r = response.strip().lower().strip(" .`'\"*")
    r = r.replace("hot take", "hot_take").replace("hot-take", "hot_take")
    if r in LABEL2ID:
        return r
    # Accept a response that contains exactly one label name
    hits = [l for l in LABELS if l in r]
    return hits[0] if len(hits) == 1 else None
