"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import LABELS

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_split(split: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a split ('train' | 'val' | 'test') with columns text, label."""
    df = pd.read_csv(data_dir / f"{split}.csv", encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"])
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels in {split}: {unknown}")
    return df.reset_index(drop=True)
