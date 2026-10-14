"""Entraîne le meilleur modèle puis recherche les conditions optimales."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from reactionoptimizer.data_loader import list_datasets, load_dataset, load_table
from reactionoptimizer.evaluation import fit_best
from reactionoptimizer.optimizer import optimize
from reactionoptimizer.preprocessing import prepare_xy
from reactionoptimizer.schema import detect_schema, find_group_column

pd.set_option("display.width", 140)

saved = list_datasets(ROOT / "data" / "datasets")
if saved:
    latest = saved[0]
    df = load_dataset(latest)
else:
    latest = ROOT / "data" / "examples" / "exemple_ds.csv"
    df = load_table(latest, latest.name)
print("Jeu de données :", latest.name)

schema = detect_schema(df, target="DS")
X, y = prepare_xy(df, schema)

group_col = find_group_column(df)
groups = None
if group_col is not None:
    groups = (df.dropna(subset=[schema.target])[group_col]
              .fillna("inconnu").astype(str).reset_index(drop=True))

fit = fit_best(X, y, schema, groups=groups)
print("Modèle retenu :", fit.best_name)

res = optimize(fit.model, X, y, schema)

print("\nConditions optimales prédites :")
print(res.optimum.round(3).to_string(index=False))

print("\nProchaines expériences suggérées :")
print(res.suggestions.round(3).to_string(index=False))

print()
for note in res.notes:
    print("⚠", note)

print("\nPlage explorée (issue des données) :")
for k, (lo, hi) in res.bounds.items():
    print(f"  {k}: {lo:g} – {hi:g}")
