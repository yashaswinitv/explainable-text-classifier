import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    toks = tok(args.text, return_tensors="pt", truncation=True, max_length=192)
    with torch.no_grad():
        logits = mdl(**toks).logits
    probs = logits.softmax(-1).tolist()[0]
    print({"probs": probs})
