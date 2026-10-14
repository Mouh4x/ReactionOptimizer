"""Génère un Excel de démonstration et vérifie la chaîne de chargement."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from reactionoptimizer.data_loader import load_excel, save_dataset, validate
from reactionoptimizer.schema import detect_schema

# --- Données fictives ---
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

# On simule un Excel « sale » : virgules décimales
df_excel = df.copy()
df_excel["temps_h"] = df_excel["temps_h"].astype(str).str.replace(".", ",", regex=False)

path = ROOT / "data" / "raw" / "demo.xlsx"
df_excel.to_excel(path, index=False)

# --- Test du chargement ---
data = load_excel(path)
schema = detect_schema(data, target="DS")
print(schema)
print()
for w in validate(data, schema):
    print("⚠", w)
print()
print(data.dtypes)
print("Sauvegardé :", save_dataset(data, ROOT / "data" / "datasets"))
