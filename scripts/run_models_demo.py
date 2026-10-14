"""Compare les modèles sur le dernier jeu de données (ou l'exemple fourni)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from reactionoptimizer.data_loader import list_datasets, load_dataset, load_table
from reactionoptimizer.evaluation import fit_best
from reactionoptimizer.preprocessing import prepare_xy
from reactionoptimizer.schema import detect_schema, find_group_column

pd.set_option("display.width", 120)

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
    print("Validation par étude :", group_col, f"({groups.nunique()} études)")

result = fit_best(X, y, schema, groups=groups)
print()
print(result.results.round(4).to_string(index=False))
print("\nMeilleur modèle :", result.best_name)
for w in result.warnings:
    print("⚠", w)

check = X.head(5).copy()
check["DS_reel"] = y.head(5).round(3)
check["DS_predit"] = result.model.predict(X.head(5)).round(3)
print("\nVérification sur 5 lignes :")
print(check.to_string(index=False))
