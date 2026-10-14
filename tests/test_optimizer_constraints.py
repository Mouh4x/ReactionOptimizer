"""Tests : optimisation, contraintes, DS négatif, extrapolation, reproductibilité."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from reactionoptimizer import constraints
from reactionoptimizer.evaluation import fit_best
from reactionoptimizer.models import get_candidate_models
from reactionoptimizer.optimizer import compute_bounds, find_optimum, optimize
from reactionoptimizer.preprocessing import build_preprocessor, prepare_xy
from reactionoptimizer.schema import detect_schema


@pytest.fixture
def fitted(demo_df):
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    fit = fit_best(X, y, schema)
    return fit, X, y, schema


def test_minimisation_et_maximisation(fitted):
    fit, X, y, schema = fitted
    rmax = optimize(fit.model, X, y, schema, maximize=True)
    rmin = optimize(fit.model, X, y, schema, maximize=False)
    assert rmax.optimum["DS_predit"].iloc[0] >= rmin.optimum["DS_predit"].iloc[0]


def test_optimum_dans_les_bornes(fitted):
    fit, X, y, schema = fitted
    res = optimize(fit.model, X, y, schema)
    for col, (lo, hi) in res.bounds.items():
        assert lo - 1e-9 <= res.optimum[col].iloc[0] <= hi + 1e-9
    # variables catégorielles : modalité réellement observée
    assert res.optimum["solvant"].iloc[0] in set(X["solvant"].unique())


def test_reproductibilite(fitted):
    fit, X, y, schema = fitted
    a = optimize(fit.model, X, y, schema).optimum["DS_predit"].iloc[0]
    b = optimize(fit.model, X, y, schema).optimum["DS_predit"].iloc[0]
    assert a == pytest.approx(b)


def test_ds_negatif_ramene_a_zero_et_signale(demo_df):
    """Modèle prédisant une valeur négative : la grandeur DS étant déclarée
    non négative (constraints.py), la valeur affichée est bornée et signalée,
    jamais silencieusement modifiée."""
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    model = Pipeline([("prep", build_preprocessor(schema)),
                      ("reg", DummyRegressor(strategy="constant", constant=-0.5))])
    model.fit(X, y)
    res = optimize(model, X, y, schema, maximize=False)
    assert res.optimum["DS_predit"].iloc[0] == pytest.approx(0.0)
    assert res.optimum["DS_predit_brut"].iloc[0] == pytest.approx(-0.5)  # preuve conservée
    assert any("borne" in n_ for n_ in res.notes)


def test_aucun_candidat_admissible_erreur_explicite(demo_df, monkeypatch):
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    model = get_candidate_models(schema)["Linéaire (Ridge)"].fit(X, y)
    monkeypatch.setattr("reactionoptimizer.optimizer.custom_filter",
                        lambda cand: pd.Series(False, index=cand.index))
    with pytest.raises(ValueError, match="Aucun candidat admissible"):
        optimize(model, X, y, schema)


def test_contrainte_personnalisee_respectee(demo_df, monkeypatch):
    """Une règle déclarée dans custom_filter écarte réellement les candidats."""
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    model = get_candidate_models(schema)["Linéaire (Ridge)"].fit(X, y)

    def sans_eau(cand):
        return cand["solvant"] != "eau"

    monkeypatch.setattr("reactionoptimizer.optimizer.custom_filter", sans_eau)
    res = optimize(model, X, y, schema)
    assert res.optimum["solvant"].iloc[0] != "eau"
    assert (res.suggestions["solvant"] != "eau").all()


def test_suggestions_excluent_experiences_existantes(demo_df):
    """Un modèle constant ne doit pas reproposer une expérience déjà réalisée."""
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    model = Pipeline([("prep", build_preprocessor(schema)),
                      ("reg", DummyRegressor(strategy="mean"))])
    model.fit(X, y)
    res = optimize(model, X, y, schema)
    existing = set(zip(X["temperature_C"].round(4), X["solvant"].astype(str)))
    for _, row in res.suggestions.iterrows():
        assert (round(row["temperature_C"], 4), str(row["solvant"])) not in existing


def test_elargissement_plage_signale_extrapolation(demo_df):
    schema = detect_schema(demo_df, target="DS")
    X, y = prepare_xy(demo_df, schema)
    bounds = compute_bounds(X, schema, expand=0.2)
    for col in schema.numeric:
        lo, hi = bounds[col]
        assert lo < float(X[col].min()) and hi > float(X[col].max())


def test_bornes_cibles_declarees():
    assert constraints.target_lower_bound("DS") == 0.0
    assert constraints.target_lower_bound("ds") == 0.0  # insensible à la casse
    assert constraints.target_lower_bound("autre") is None
