from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict
from .config import cfg

_label_names = ["World", "Sports", "Business", "Sci/Tech"]

def get_datasets(tokenizer=None):
    ds = load_dataset("ag_news")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)

    def tok(example: Dict):
        return tokenizer(example["text"], truncation=True, max_length=cfg.max_length)

    ds = ds.map(tok, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds, _label_names
