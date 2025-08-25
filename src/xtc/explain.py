import argparse
import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import cfg

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    model.eval()

    def f(texts):
        toks = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**toks).logits
        return logits.softmax(-1).numpy()

    explainer = shap.Explainer(f, shap.maskers.Text(tokenizer))
    shap_values = explainer([args.text])
    shap.plots.text(shap_values[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    main(parser.parse_args())
