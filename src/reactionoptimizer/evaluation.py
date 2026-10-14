"""Comparaison des modèles par validation croisée et choix du meilleur.

Principes :
- la validation se fait **par étude** (leave-one-study-out) dès qu'une colonne
  d'identification exploitable existe : les expériences sœurs d'une même étude
  ne sont jamais réparties entre entraînement et validation ;
- la référence est honnête : en validation par étude, elle prédit la **médiane
  des autres études** (et non la moyenne de toutes les lignes, études de
  validation incluses — ce qui serait une forme de fuite et un handicap
  artificiel pour la référence) ;
- si aucun modèle ne bat la référence, c'est la référence qui est retenue et
  le logiciel le dit explicitement au lieu de couronner un modèle complexe ;
- un R² négatif est affiché tel quel : le modèle fait alors moins bien que la
  prédiction de référence sur les données évaluées.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    cross_val_predict,
)

from .models import BASELINE, get_candidate_models
from .schema import Schema


@dataclass
class FitResult:
    best_name: str
    model: object
    results: pd.DataFrame
    warnings: list[str]
    cv_method: str = ""


def _cv_strategy(n: int, groups: pd.Series | None):
    """Validation par étude si une colonne source exploitable existe.

    Retirer toute une étude à la fois empêche le modèle de « tricher » grâce
    aux expériences sœurs issues du même article : le score est plus honnête.
    Avec moins de 3 études, retour à une validation ligne par ligne, signalée.
    """
    if groups is not None and groups.nunique() >= 3:
        k = int(groups.nunique())
        return LeaveOneGroupOut(), groups, f"par étude (leave-one-study-out, {k} études)"
    if n <= 40:
        return LeaveOneOut(), None, "leave-one-out (ligne par ligne)"
    return KFold(n_splits=5, shuffle=True, random_state=0), None, "KFold (5 plis)"


def _baseline_predictions(y: pd.Series, groups: pd.Series | None) -> np.ndarray:
    """Prédictions de la référence, sans fuite entre études.

    Avec groupes : médiane des AUTRES études pour chaque ligne. Sans groupes :
    médiane globale (équivalent en moyenne du leave-one-out pour une référence).
    """
    if groups is not None:
        g = groups.to_numpy()
        pred = np.empty(len(y), dtype=float)
        for grp in pd.unique(g):
            mask = g == grp
            pred[mask] = float(y[~mask].median())  # médiane des AUTRES études
        return pred
    return np.full(len(y), float(y.median()))


def evaluate_models(X: pd.DataFrame, y: pd.Series, models: dict,
                    groups: pd.Series | None = None):
    cv, cv_groups, cv_name = _cv_strategy(len(y), groups)
    rows = []
    for name, model in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == BASELINE:
                    pred = _baseline_predictions(y, cv_groups)
                else:
                    pred = cross_val_predict(clone(model), X, y, cv=cv,
                                             groups=cv_groups)
            rows.append({
                "modele": name,
                "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "MAE": float(mean_absolute_error(y, pred)),
                "R2": float(r2_score(y, pred)),
                "erreur": "",
            })
        except Exception as exc:  # un modèle qui échoue ne bloque pas les autres
            rows.append({"modele": name, "RMSE": np.nan, "MAE": np.nan,
                         "R2": np.nan, "erreur": str(exc)[:80]})
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return results, cv_name


def fit_best(X: pd.DataFrame, y: pd.Series, schema: Schema,
             groups: pd.Series | None = None) -> FitResult:
    if len(y) < 5:
        raise ValueError("Il faut au moins 5 lignes avec une valeur de cible.")

    models = get_candidate_models(schema)
    results, cv_name = evaluate_models(X, y, models, groups)
    candidates = results[(results["modele"] != BASELINE) & results["RMSE"].notna()]
    if candidates.empty:
        raise RuntimeError("Aucun modèle n'a pu être entraîné.")

    notes: list[str] = []
    best = candidates.iloc[0]
    baseline_row = results[results["modele"] == BASELINE]
    baseline_rmse = (float(baseline_row["RMSE"].iloc[0])
                     if not baseline_row.empty and baseline_row["RMSE"].notna().iloc[0]
                     else None)

    if baseline_rmse is not None and best["RMSE"] >= baseline_rmse:
        # Aucun modèle ne fait mieux que la référence : on retient la référence
        # au lieu de couronner un modèle complexe sans justification.
        best_name = BASELINE
        best_r2 = float(baseline_row["R2"].iloc[0])
        notes.append(
            "Aucun modèle ne fait mieux que la référence (moyenne/médiane par étude) "
            "en validation : c'est elle qui est retenue. Les conditions « optimales » "
            "ci-dessous ne sont qu'indicatives — collectez davantage de données."
        )
    else:
        best_name = best["modele"]
        best_r2 = float(best["R2"])

    best_model = clone(models[best_name]).fit(X, y)

    if best_r2 <= 0:
        notes.append("Aucun modèle ne fait mieux que la simple moyenne : "
                     "données insuffisantes ou variables importantes manquantes.")
    elif best_r2 < 0.5:
        notes.append(f"Le meilleur modèle n'explique qu'une part limitée des "
                     f"résultats (R² = {best_r2:.2f}) : à utiliser avec prudence.")
    if groups is None or groups.nunique() < 3:
        notes.append("Validation ligne par ligne : si vos lignes proviennent de "
                     "plusieurs études, ajoutez une colonne 'source' pour une "
                     "validation plus honnête (leave-one-study-out).")
    failed = results[results["erreur"] != ""]
    if not failed.empty:
        notes.append("Modèles en échec sur ce jeu de données : "
                     + ", ".join(failed["modele"]) + " (voir colonne 'erreur').")
    return FitResult(best_name, best_model, results, notes, cv_name)
