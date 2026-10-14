"""Fixtures de tests : données synthétiques reproductibles (graine fixe)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def demo_df() -> pd.DataFrame:
    """25 lignes, 5 études, signal connu : pic de DS vers 80 °C, temps long, 1,5 g."""
    rng = np.random.default_rng(42)
    n = 25
    df = pd.DataFrame({
        "source": [f"etude_{i // 5 + 1}" for i in range(n)],
        "temperature_C": rng.uniform(40, 120, n).round(1),
        "temps_h": rng.uniform(0.5, 8, n).round(2),
        "quantite_g": rng.uniform(0.5, 3, n).round(2),
        "solvant": rng.choice(["eau", "DMSO"], n),
    })
    ds = (0.9
          * np.exp(-((df.temperature_C - 80) / 30) ** 2)
          * (1 - np.exp(-df.temps_h / 2))
          * np.exp(-((df.quantite_g - 1.5) / 1.2) ** 2))
    df["DS"] = (ds + rng.normal(0, 0.02, n)).clip(0).round(3)
    return df
