from transformers import AutoModelForSequenceClassification
from .config import cfg

def build_model(num_labels: int = 4):
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )
