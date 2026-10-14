"""Tests : entraînement, validation (par étude), sélection honnête du modèle."""
import numpy as np
import pandas as pd
import pytest

from reactionoptimizer.evaluation import _baseline_predictions, evaluate_models, fit_best
from reactionoptimizer.models import BASELINE, get_candidate_models
from reactionoptimizer.preprocessing import prepare_xy
from reactionoptimizer.schema import detect_schema, find_group_column


def test_entrainement_sur_donnees_valides(demo_df):
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    fit = fit_best(X, y, schema)
    assert len(fit.results) == len(get_candidate_models(schema))
    assert BASELINE in fit.results["modele"].tolist()
    assert fit.results["RMSE"].notna().any()


def test_validation_par_etude_sans_fuite(demo_df):
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    col = find_group_column(demo_df)
    assert col == "source"
    groups = demo_df[col].astype(str)
    fit = fit_best(X, y, schema, groups=groups)
    assert "leave-one-study-out" in fit.cv_method


def test_reference_honnete_par_etude():
    """La référence prédit la médiane des AUTRES études, jamais la sienne."""
    y = pd.Series([0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
    groups = pd.Series(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    pred = _baseline_predictions(y, groups)
    assert pred[0] == pytest.approx(7.5)   # médiane des études b et c
    assert pred[3] == pytest.approx(2.5)   # médiane des études a et c


def test_moins_de_5_lignes_refuse(demo_df):
    df = demo_df.head(4)
    schema = detect_schema(df, target="DS")
    X, y = prepare_xy(df, schema)
    with pytest.raises(ValueError, match="au moins 5"):
        fit_best(X, y, schema)


def test_bruit_pur_reference_retenue():
    """Sans signal, aucun modèle ne doit être couronné à la place de la référence."""
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "source": [f"e{i // 5}" for i in range(n)],
        "x1": rng.uniform(0, 1, n),
        "x2": rng.uniform(0, 1, n),
        "DS": rng.normal(0.5, 0.1, n),
    })
    schema = detect_schema(df, target="DS")
    X, y = prepare_xy(df, schema)
    fit = fit_best(X, y, schema, groups=df["source"])
    assert fit.best_name == BASELINE
    assert any("référence" in n_ for n_ in fit.warnings)


def test_un_modele_en_echec_ne_bloque_pas_les_autres(demo_df):
    class ModeleCasse:
        def fit(self, X, y):
            raise RuntimeError("panne volontaire")

        def predict(self, X):
            raise RuntimeError("panne volontaire")

        def get_params(self, deep=True):
            return {}

        def set_params(self, **kw):
            return self

    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    models = {"cassé": ModeleCasse()}
    models.update(get_candidate_models(schema))
    results, _ = evaluate_models(X, y, models)
    casse = results[results["modele"] == "cassé"].iloc[0]
    assert casse["erreur"] != ""
    assert results[results["modele"] != "cassé"]["RMSE"].notna().all()


def test_valeurs_manquantes_gerees(demo_df):
    df = demo_df.copy()
    df.loc[0, "temperature_C"] = np.nan
    df.loc[1, "solvant"] = None
    schema = detect_schema(df, target="DS")
    X, y = prepare_xy(df, schema)
    fit = fit_best(X, y, schema)
    assert fit.model.predict(X.head(3)) is not None
