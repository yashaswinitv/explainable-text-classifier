import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from .config import cfg
from .data import get_datasets

def main(args):
    ckpt = Path(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)

    # quick eval loop
    from torch.utils.data import DataLoader
    import torch

    ds, labels = get_datasets(tokenizer)
    loader = DataLoader(ds["test"], batch_size=32)
    preds, gold = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            preds.append(out.logits.argmax(-1).cpu().numpy())
            gold.append(batch["labels"].cpu().numpy())
    preds = np.concatenate(preds)
    gold = np.concatenate(gold)

    print(classification_report(gold, preds, target_names=labels))
    print("Confusion Matrix:\n", confusion_matrix(gold, preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    main(parser.parse_args())
