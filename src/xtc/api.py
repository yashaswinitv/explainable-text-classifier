from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from .config import cfg

app = FastAPI(title="Explainable Text Classifier")

class Inp(BaseModel):
    text: str

@app.on_event("startup")
def load():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.output_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(cfg.output_dir))

@app.post("/predict")
def predict(inp: Inp):
    toks = tokenizer(inp.text, return_tensors="pt", truncation=True, max_length=cfg.max_length)
    with torch.no_grad():
        logits = model(**toks).logits
    probs = logits.softmax(-1).tolist()[0]
    return {"probs": probs}

@app.post("/explain")
def explain(inp: Inp):
    # lightweight placeholder; for full SHAP, call xtc.explain
    return {"note": "Use the CLI xtc.explain for rich plots."}
