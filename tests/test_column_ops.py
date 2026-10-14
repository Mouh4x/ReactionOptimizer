"""Tests : gestion dynamique des colonnes (ajout, suppression, renommage,
retypage) et cohérence des références (cible, colonnes ignorées)."""
import numpy as np
import pandas as pd
import pytest

from reactionoptimizer.column_ops import (
    COLUMN_TYPES,
    add_column,
    remove_columns,
    rename_column,
    retype_column,
    update_references,
)
from reactionoptimizer.data_loader import load_table
from reactionoptimizer.evaluation import fit_best
from reactionoptimizer.optimizer import optimize
from reactionoptimizer.preprocessing import prepare_xy
from reactionoptimizer.schema import detect_schema


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["e1", "e1", "e2", "e2", "e3", "e3"],
        "temperature_C": [60.0, 80.0, 90.0, 70.0, 100.0, 85.0],
        "solvant": ["eau", "DMSO", "eau", "DMSO", "eau", "DMSO"],
        "DS": [0.2, 0.5, 0.4, 0.3, 0.6, 0.55],
    })


# ---------------- Ajout ----------------
def test_ajout_colonne(base_df):
    change = add_column(base_df, "  catalyseur  ")
    assert "catalyseur" in change.df.columns  # nom normalisé
    assert change.df["catalyseur"].isna().all()
    assert len(change.df) == len(base_df)          # lignes conservées
    assert change.df["DS"].equals(base_df["DS"])   # données intactes


def test_ajout_colonne_existante_refuse(base_df):
    with pytest.raises(ValueError, match="existe déjà"):
        add_column(base_df, "DS")


def test_ajout_colonne_vide_refuse(base_df):
    with pytest.raises(ValueError):
        add_column(base_df, "   ")


# ---------------- Suppression ----------------
def test_suppression_colonne_simple(base_df):
    change = remove_columns(base_df, ["solvant"])
    assert "solvant" not in change.df.columns
    assert len(change.df) == len(base_df)
    assert change.df["DS"].equals(base_df["DS"])


def test_suppression_cible_signalee_et_invalidee(base_df):
    change = remove_columns(base_df, ["DS"], target="DS", ignore=[])
    assert change.removed_target
    assert any("cible" in m.lower() for m in change.messages)
    new_target, _ = update_references(change, "DS", [])
    assert new_target is None  # sélection invalidée proprement


def test_suppression_colonnes_ignorees_suivie(base_df):
    change = remove_columns(base_df, ["source"], target="DS", ignore=["source"])
    assert change.removed_ignored == ["source"]
    _, new_ignore = update_references(change, "DS", ["source"])
    assert new_ignore == []


def test_suppression_de_toutes_les_colonnes_refusee(base_df):
    with pytest.raises(ValueError, match="toutes les colonnes"):
        remove_columns(base_df, list(base_df.columns))


def test_suppression_colonne_inexistante_refusee(base_df):
    with pytest.raises(ValueError):
        remove_columns(base_df, ["n existe pas"])


def test_suppression_variable_feature_puis_analyse_ok(demo_df):
    """Supprimer une variable utilisée par le modèle ne casse rien : le schéma
    est redétecté sur le nouveau tableau et la chaîne complète fonctionne."""
    change = remove_columns(demo_df, ["solvant"], target="DS", ignore=[])
    df = change.df
    schema = detect_schema(df, target="DS")
    assert "solvant" not in schema.features
    X, y = prepare_xy(df, schema)
    fit = fit_best(X, y, schema)
    res = optimize(fit.model, X, y, schema)
    assert len(res.suggestions) >= 1
    assert "solvant" not in res.optimum.columns


# ---------------- Renommage ----------------
def test_renommage_colonne(base_df):
    change = rename_column(base_df, "DS", "rendement", target="DS", ignore=["source"])
    assert "rendement" in change.df.columns and "DS" not in change.df.columns
    assert change.df["rendement"].equals(base_df["DS"])  # valeurs conservées
    new_target, new_ignore = update_references(change, "DS", ["source"])
    assert new_target == "rendement"      # la cible suit le renommage
    assert new_ignore == ["source"]


def test_renommage_vers_nom_existant_refuse(base_df):
    with pytest.raises(ValueError, match="existe déjà"):
        rename_column(base_df, "DS", "solvant")


def test_renommage_colonne_inconnue_refuse(base_df):
    with pytest.raises(ValueError, match="introuvable"):
        rename_column(base_df, "fantome", "x")


# ---------------- Retypage ----------------
def test_retypage_texte_vers_nombre_virgules():
    df = pd.DataFrame({"DS": ["1,5", "2.5", "abc", None]})
    change = retype_column(df, "DS", "number")
    assert pd.api.types.is_numeric_dtype(change.df["DS"])
    assert change.df["DS"].iloc[0] == 1.5
    assert pd.isna(change.df["DS"].iloc[2])   # invalide vidée, pas 0
    assert any("1 valeur" in m for m in change.messages)


def test_retypage_vers_booleen():
    df = pd.DataFrame({"actif": ["oui", "Non", "1", "0", "vrai", "peut-être", None]})
    change = retype_column(df, "actif", "boolean")
    vals = change.df["actif"].tolist()
    assert vals[:5] == [True, False, True, False, True]
    assert pd.isna(vals[5]) and pd.isna(vals[6])
    assert any("vidée" in m for m in change.messages)


def test_retypage_booleen_vers_nombre():
    df = pd.DataFrame({"actif": pd.array([True, False, pd.NA], dtype="boolean")})
    change = retype_column(df, "actif", "number")
    assert change.df["actif"].tolist()[:2] == [1.0, 0.0]
    assert pd.isna(change.df["actif"].iloc[2])


def test_retypage_nombre_vers_texte():
    df = pd.DataFrame({"DS": [0.5, 0.6]})
    change = retype_column(df, "DS", "text")
    assert pd.api.types.is_string_dtype(change.df["DS"])
    assert change.df["DS"].iloc[0] == "0.5"


def test_retypage_auto_detecte_nombre():
    df = pd.DataFrame({"x": ["1", "2", "3", "4"]})
    change = retype_column(df, "x", "auto")
    assert pd.api.types.is_numeric_dtype(change.df["x"])


def test_retypage_type_inconnu_refuse(base_df):
    with pytest.raises(ValueError, match="Type inconnu"):
        retype_column(base_df, "DS", "complexe")


def test_types_proposes_complets():
    assert set(COLUMN_TYPES) == {"auto", "number", "text", "boolean"}


# ---------------- Références croisées ----------------
def test_update_references_renommage_cible(base_df):
    change = rename_column(base_df, "DS", "score")
    new_target, _ = update_references(change, "DS", [])
    assert new_target == "score"


def test_update_references_cible_devenue_ignoree_retiree(base_df):
    """Une ignorée renommée vers le nom de la cible ne peut pas rester ignorée."""
    change = rename_column(base_df, "source", "DS2")
    target, ignore = update_references(change, "DS2", ["source"])
    assert target == "DS2" and ignore == []


# ---------------- Import Excel multi-feuilles ----------------
def test_import_excel_choix_feuille(tmp_path):
    path = tmp_path / "multi.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"a": [1.0], "DS": [0.5]}).to_excel(w, sheet_name="F1", index=False)
        pd.DataFrame({"b": [2.0], "DS": [0.7]}).to_excel(w, sheet_name="F2", index=False)
    df = load_table(path, path.name, sheet_name="F2")
    assert list(df.columns) == ["b", "DS"]
    df1 = load_table(path, path.name)  # feuille par défaut = première
    assert list(df1.columns) == ["a", "DS"]


def test_import_excel_colonnes_arbitraires(tmp_path):
    """Aucune liste prédéfinie : n'importe quels noms de colonnes sont reconnus."""
    path = tmp_path / "libre.xlsx"
    pd.DataFrame({"pression_bar": [1.0, 2.0], "catalyseur": ["Pd", "Pt"],
                  "rendement": [0.4, 0.8]}).to_excel(path, index=False)
    df = load_table(path, path.name)
    assert list(df.columns) == ["pression_bar", "catalyseur", "rendement"]
    schema = detect_schema(df, target="rendement")
    assert schema.numeric == ["pression_bar"]
    assert schema.categorical == ["catalyseur"]


# ---------------- Chaîne complète après changement de cible ----------------
def test_minimisation_sur_autre_cible(demo_df):
    """Cible et sens d'optimisation réellement pris en compte par l'algorithme."""
    df = demo_df.rename(columns={"DS": "score"})
    schema = detect_schema(df, target="score")
    X, y = prepare_xy(df, schema)
    fit = fit_best(X, y, schema)
    rmax = optimize(fit.model, X, y, schema, maximize=True)
    rmin = optimize(fit.model, X, y, schema, maximize=False)
    assert "score_predit" in rmax.optimum.columns
    assert rmax.optimum["score_predit"].iloc[0] >= rmin.optimum["score_predit"].iloc[0]
