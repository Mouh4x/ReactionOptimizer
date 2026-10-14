"""Tests applicatifs : imports, parcours principal, export."""
import importlib
from pathlib import Path

import pandas as pd
import pytest

from reactionoptimizer.evaluation import fit_best
from reactionoptimizer.optimizer import compute_bounds, optimize
from reactionoptimizer.preprocessing import prepare_xy
from reactionoptimizer.reporting import results_to_excel, template_excel
from reactionoptimizer.schema import detect_schema, find_group_column

MODULES = [
    "reactionoptimizer.config",
    "reactionoptimizer.constraints",
    "reactionoptimizer.data_loader",
    "reactionoptimizer.evaluation",
    "reactionoptimizer.models",
    "reactionoptimizer.optimizer",
    "reactionoptimizer.preprocessing",
    "reactionoptimizer.reporting",
    "reactionoptimizer.schema",
]


def test_imports_valides():
    for name in MODULES:
        importlib.import_module(name)


def test_parcours_complet_et_export(demo_df, tmp_path):
    """Parcours principal : schéma -> modèles -> optimisation -> export Excel."""
    schema = detect_schema(demo_df, target="DS")
    assert schema.target == "DS"
    assert "solvant" in schema.categorical
    assert find_group_column(demo_df) == "source"

    X, y = prepare_xy(demo_df, schema)
    groups = demo_df.dropna(subset=["DS"])["source"].astype(str).reset_index(drop=True)
    fit = fit_best(X, y, schema, groups=groups)
    bounds = compute_bounds(X, schema)
    res = optimize(fit.model, X, y, schema, bounds=bounds, maximize=True)
    assert len(res.suggestions) >= 1

    blob = results_to_excel({
        "Optimum": res.optimum,
        "Experiences_suggerees": res.suggestions,
        "Comparaison_modeles": fit.results,
        "Donnees": demo_df,
    })
    out = tmp_path / "export.xlsx"
    out.write_bytes(blob)
    lu = pd.read_excel(out, sheet_name=None)
    assert set(lu) == {"Optimum", "Experiences_suggerees", "Comparaison_modeles", "Donnees"}


def test_template_excel_chargeable():
    blob = template_excel()
    lu = pd.read_excel(pd.io.common.BytesIO(blob), sheet_name=None)
    assert "Donnees" in lu and "Instructions" in lu


def test_exemple_fourni_chargeable():
    from reactionoptimizer.data_loader import load_table
    root = Path(__file__).resolve().parent.parent
    path = root / "data" / "examples" / "exemple_ds.csv"
    df = load_table(path, path.name)
    assert "DS" in df.columns and len(df) >= 20
    assert df["DS"].min() >= 0  # cohérent avec la contrainte déclarée
