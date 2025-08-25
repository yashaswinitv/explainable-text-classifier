# Explainable Text Classifier (DistilBERT + SHAP)

[![GitHub: yashaswinitv](https://img.shields.io/badge/GitHub-yashaswinitv-blue?logo=github)](https://github.com/yashaswinitv)

**Author:** Yashaswini Talakadu Vijaykumar  
**Email:** yashaswinivijaykumar07@gmail.com  
**Bio:** Master’s student in Data Science at Queen Mary University of London; interests include NLP, scientometrics, explainable AI, and MLOps.  
**GitHub:** [yashaswinitv](https://github.com/yashaswinitv)

A tidy, GitHub‑ready NLP project: fine‑tunes DistilBERT on the AG News dataset, reports metrics, and explains predictions with SHAP. Includes FastAPI serving, Docker, tests, and CI.

## Features
- **Data**: pulls `ag_news` from Hugging Face Datasets
- **Model**: DistilBERT fine‑tuning with 🤗 Transformers `Trainer`
- **Metrics**: accuracy, precision/recall/F1, confusion matrix
- **Explainability**: SHAP for token‑level attributions
- **Serving**: FastAPI endpoint with `/predict` and `/explain`
- **DevX**: `pyproject.toml`, pre‑commit, Makefile, CI, Docker

## Quickstart
```bash
# 1) Setup (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # or: pip install -e .
pre-commit install

# 2) Train
python -m xtc.train --model distilbert-base-uncased --epochs 1 --batch_size 16

# 3) Evaluate
python -m xtc.evaluate --ckpt runs/latest

# 4) Explain a text
python -m xtc.explain --ckpt runs/latest --text "Apple announces new iPhone with satellite features"

# 5) Serve API
uvicorn xtc.api:app --host 0.0.0.0 --port 8000
# then: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"NASA launches a new telescope"}'
```

## Project Commands
```bash
make format        # run black, isort
make lint          # run ruff
make test          # run pytest
make train         # default training run
make docker-build  # build container
make docker-run    # run API in Docker
```

## Results
- Baseline (1 epoch, DistilBERT): ~92–94% accuracy (varies by seed)
- Full run (3 epochs): typically 94–96%

## Repo Structure
```
explainable-text-classifier/
├─ README.md
├─ LICENSE
├─ pyproject.toml
├─ requirements.txt
├─ .gitignore
├─ .pre-commit-config.yaml
├─ Makefile
├─ Dockerfile
├─ .github/workflows/ci.yml
├─ src/
│  └─ xtc/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ model.py
│     ├─ train.py
│     ├─ evaluate.py
│     ├─ explain.py
│     └─ api.py
├─ notebooks/
│  └─ 01_eda_and_baseline.ipynb  (optional placeholder)
├─ tests/
│  ├─ test_data.py
│  ├─ test_model.py
│  └─ test_api.py
└─ scripts/
   ├─ download_data.py
   └─ predict_cli.py
```
