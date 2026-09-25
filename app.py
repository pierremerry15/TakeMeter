"""TakeMeter interface (stretch feature: deployed interface).

    python app.py                       # web UI at http://127.0.0.1:7860
    python app.py --share               # public link (use this in Colab)
    python app.py "LeBron is washed"    # one-off classification in the terminal

Loads the fine-tuned model saved by train_eval.py in outputs/model.
"""
import argparse
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "outputs/model"
DESCRIPTIONS = {
    "analysis": "claim backed by specific, checkable evidence",
    "hot_take": "bold general claim, asserted not argued",
    "reaction": "in-the-moment emotional response",
}

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()


@torch.no_grad()
def classify(text: str) -> dict:
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    probs = torch.softmax(model(**enc).logits, -1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    if args.text:
        probs = classify(" ".join(args.text))
        top = max(probs, key=probs.get)
        print(f"{top}  (confidence {probs[top]:.1%})")
        for l, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            print(f"  {l:9s} {p:6.1%}  - {DESCRIPTIONS.get(l, '')}")
        return

    import gradio as gr

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=5, label="r/nba comment",
                          placeholder="Paste a comment from a game thread or discussion post..."),
        outputs=gr.Label(num_top_classes=3, label="TakeMeter"),
        title="TakeMeter: r/nba take classifier",
        description="Fine-tuned DistilBERT. analysis = evidence-backed claim · "
                    "hot_take = bold claim without real support · reaction = in-the-moment emotion.",
        examples=[
            ["LETS GOOOOO WHAT A SHOT"],
            ["Ant is already better than prime Wade and it's not even close."],
            ["The Knicks' drop is mostly corner threes: 3rd in opponent corner 3PA before the "
             "injury, 26th since, because the weak-side corner is open whenever Brunson is the low man."],
        ],
        flagging_mode="never",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    sys.exit(main())
