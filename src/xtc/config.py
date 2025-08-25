from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    output_dir: Path = Path("runs/latest")
    seed: int = 42
    max_length: int = 192
    batch_size: int = 16
    epochs: int = 1

cfg = Config()
