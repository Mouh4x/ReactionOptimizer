"""Recherche des conditions optimales et suggestion des prochaines expériences.

Méthode : échantillonnage de Sobol (reproductible, graine fixe) dans les bornes
déduites des données, affinage local L-BFGS-B par jeu de modalités, incertitude
par bootstrap du modèle retenu. Les contraintes déclarées (constraints.py) sont
vérifiées AVANT la présentation des résultats.

Un optimum trouvé ici est une PRÉDICTION de modèle dans un domaine couvert par
peu de données : ce n'est jamais une preuve de faisabilité expérimentale.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.base import clone

from .constraints import check_bounds, custom_filter, target_lower_bound, target_upper_bound
from .schema import Schema

MAX_COMBOS = 200  # nombre maximum de combinaisons de catégories explorées


@dataclass
class OptimizationResult:
    optimum: pd.DataFrame
    suggestions: pd.DataFrame
    bounds: dict
    notes: list[str] = field(default_factory=list)


# ---------- Plages de recherche ----------
def compute_bounds(X: pd.DataFrame, schema: Schema, expand: float = 0.0) -> dict:
    """Bornes = plage observée dans les données (expand=0.1 l'élargit de 10 %).

    Ce sont des bornes DÉDUITES DES DONNÉES, pas des limites chimiques : au-delà,
    le modèle extrapole. Les contraintes physiques éventuelles se déclarent dans
    constraints.py.
    """
    bounds = {}
    for col in schema.numeric:
        lo, hi = float(X[col].min()), float(X[col].max())
        pad = (hi - lo) * expand
        bounds[col] = (lo - pad, hi + pad)
    return bounds


def _category_combos(X: pd.DataFrame, schema: Schema) -> list[dict]:
    if not schema.categorical:
        return [{}]
    levels = [sorted(X[c].dropna().unique().tolist(), key=str) for c in schema.categorical]
    if any(len(lv) == 0 for lv in levels):
        raise ValueError("Une variable catégorielle n'a aucune valeur renseignée.")
    combos = [dict(zip(schema.categorical, v)) for v in itertools.product(*levels)]
    if len(combos) > MAX_COMBOS:
        raise ValueError(
            f"Trop de combinaisons de catégories ({len(combos)}) : "
            "retirez ou regroupez des variables catégorielles."
        )
    return combos


def _sample_candidates(X, schema, bounds, budget, seed) -> pd.DataFrame:
    """Points candidats répartis uniformément (séquence de Sobol, graine fixe)."""
    combos = _category_combos(X, schema)
    cols = schema.numeric
    frames = []
    for i, combo in enumerate(combos):
        if cols:
            # Budget réparti entre les combinaisons, arrondi à une puissance de 2.
            per_combo = max(1, budget // len(combos))
            n = min(2 ** 13, 2 ** max(6, int(np.ceil(np.log2(per_combo)))))
            lo = np.array([bounds[c][0] for c in cols])
            hi = np.array([bounds[c][1] for c in cols])
            u = qmc.Sobol(d=len(cols), scramble=True, seed=seed + i).random(n)
            frame = pd.DataFrame(qmc.scale(u, lo, hi), columns=cols)
        else:
            frame = pd.DataFrame(index=[0])
        for c, v in combo.items():
            frame[c] = v
        frames.append(frame[schema.features])
    return pd.concat(frames, ignore_index=True)


def _row_key(row: pd.Series, schema: Schema) -> tuple:
    """Clé de déduplication : numériques arrondies (1e-4) + modalités exactes."""
    num = tuple(round(float(row[c]), 4) for c in schema.numeric)
    cat = tuple(str(row[c]) for c in schema.categorical)
    return num + cat


# ---------- Optimum ----------
def _refine(model, start: pd.DataFrame, schema: Schema, bounds: dict, sign: float):
    """Affine localement un point de départ (optimisation bornée L-BFGS-B)."""
    cols = schema.numeric
    if not cols:
        return start, float(model.predict(start)[0])
    frame = start.copy()

    def objective(x):
        for c, v in zip(cols, x):
            frame.loc[0, c] = v
        return -sign * float(model.predict(frame)[0])

    x0 = start.loc[0, cols].to_numpy(dtype=float)
    res = minimize(objective, x0, method="L-BFGS-B", bounds=[bounds[c] for c in cols])
    for c, v in zip(cols, res.x):
        frame.loc[0, c] = v
    return frame, float(model.predict(frame)[0])


def find_optimum(model, cand: pd.DataFrame, schema: Schema, bounds: dict,
                 maximize: bool = True, n_starts: int = 5) -> pd.DataFrame:
    if cand.empty:
        raise ValueError(
            "Aucun candidat admissible : les contraintes (bornes, règles de "
            "constraints.py) éliminent tout le domaine de recherche."
        )
    sign = 1.0 if maximize else -1.0
    pred = model.predict(cand)
    starts = np.argsort(-sign * pred)[:n_starts]
    best_frame, best_score = None, -np.inf
    for idx in starts:
        start = cand.iloc[[idx]].reset_index(drop=True)
        frame, value = _refine(model, start, schema, bounds, sign)
        if sign * value > best_score:
            best_frame, best_score = frame, sign * value
    return best_frame


# ---------- Incertitude (bootstrap) ----------
def fit_ensemble(model, X: pd.DataFrame, y: pd.Series, n_models: int = 30, seed: int = 0) -> list:
    """Réentraîne le modèle sur des rééchantillonnages bootstrap des données."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ensemble = []
    for _ in range(n_models):
        idx = rng.integers(0, n, n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ensemble.append(clone(model).fit(X.iloc[idx], y.iloc[idx]))
        except Exception:
            continue
    return ensemble


def _std(ensemble: list, frame: pd.DataFrame) -> np.ndarray:
    if len(ensemble) < 5:
        return np.zeros(len(frame))
    preds = np.column_stack([m.predict(frame) for m in ensemble])
    return preds.std(axis=1)


# ---------- Prochaines expériences ----------
def _select_diverse(score, cand, schema, bounds, n, min_dist, exclude: set | None = None):
    """Choisit les n meilleurs candidats éloignés les uns des autres.

    Les candidats identiques (à 1e-4 près sur les numériques) à une expérience
    déjà réalisée sont écartés : inutile de proposer de refaire l'existant.
    """
    if schema.numeric:
        num = np.column_stack([
            (cand[c].to_numpy(dtype=float) - bounds[c][0])
            / ((bounds[c][1] - bounds[c][0]) or 1.0)
            for c in schema.numeric
        ])
    else:
        num = np.zeros((len(cand), 0))
    cat = cand[schema.categorical].to_numpy(dtype=object) if schema.categorical else None

    def dist(i, j):
        d = float(np.sqrt(((num[i] - num[j]) ** 2).sum()))
        if cat is not None:
            d += float((cat[i] != cat[j]).sum())
        return d

    chosen: list[int] = []
    n_excluded = 0
    for idx in np.argsort(-score):
        if exclude and _row_key(cand.iloc[idx], schema) in exclude:
            n_excluded += 1
            continue
        if all(dist(idx, j) >= min_dist for j in chosen):
            chosen.append(int(idx))
            if len(chosen) == n:
                break
    return chosen, n_excluded


def _apply_target_bounds(frame: pd.DataFrame, col: str, target: str,
                         notes: list[str], label: str) -> None:
    """Applique les bornes déclarées (constraints.py) aux prédictions AFFICHÉES.

    Le modèle n'est pas modifié : une valeur hors borne est une extrapolation,
    elle est ramenée à la borne pour la présentation ET signalée, jamais
    masquée silencieusement.
    """
    if col not in frame:
        return
    lo, hi = target_lower_bound(target), target_upper_bound(target)
    if lo is None and hi is None:
        return
    vals = frame[col].astype(float)
    out = ((vals < lo) if lo is not None else pd.Series(False, index=frame.index)) | \
          ((vals > hi) if hi is not None else pd.Series(False, index=frame.index))
    if out.any():
        desc = []
        if lo is not None:
            desc.append(f"≥ {lo:g}")
        if hi is not None:
            desc.append(f"≤ {hi:g}")
        notes.append(
            f"{label} : {int(out.sum())} prédiction(s) brute(s) hors de la plage "
            f"physique déclarée pour '{target}' ({' et '.join(desc)}) — artefact "
            "d'extrapolation du modèle, valeur(s) ramenée(s) à la borne "
            "(contrainte déclarée dans constraints.py, modèle lui-même inchangé)."
        )
        frame[col] = vals.clip(lower=lo, upper=hi)


# ---------- Fonction principale ----------
def optimize(model, X: pd.DataFrame, y: pd.Series, schema: Schema,
             bounds: dict | None = None, maximize: bool = True,
             n_suggestions: int = 3, kappa: float = 1.5, seed: int = 0) -> OptimizationResult:
    bounds = bounds or compute_bounds(X, schema)
    sign = 1.0 if maximize else -1.0
    target = schema.target
    notes: list[str] = []

    cand = _sample_candidates(X, schema, bounds, budget=8192, seed=seed)

    # Contraintes vérifiées AVANT toute présentation : bornes des variables et
    # règles personnalisées de compatibilité (constraints.py).
    admissible = check_bounds(cand, bounds) & custom_filter(cand)
    n_rejected = int((~admissible).sum())
    if n_rejected:
        notes.append(f"{n_rejected} candidat(s) écarté(s) par les contraintes "
                     "(bornes ou règles de constraints.py).")
    cand = cand[admissible].reset_index(drop=True)

    # Déduplication des points quasi identiques (Sobol peut en générer).
    keys = cand.apply(_row_key, axis=1, schema=schema)
    cand = cand[~keys.duplicated()].reset_index(drop=True)

    best = find_optimum(model, cand, schema, bounds, maximize)
    ensemble = fit_ensemble(model, X, y, seed=seed)

    optimum = best.copy()
    optimum[f"{target}_predit_brut"] = model.predict(best)
    optimum["incertitude"] = _std(ensemble, best)

    # Score d'exploration : bonne valeur prédite + forte incertitude
    pred = model.predict(cand)
    std = _std(ensemble, cand)
    existing = {_row_key(X.iloc[i], schema) for i in range(len(X))}
    chosen, n_seen = _select_diverse(sign * pred + kappa * std, cand, schema, bounds,
                                     n_suggestions, min_dist=0.25, exclude=existing)
    if n_seen:
        notes.append(f"{n_seen} candidat(s) identique(s) à une expérience déjà "
                     "réalisée ont été écartés des suggestions.")
    suggestions = cand.iloc[chosen].reset_index(drop=True)
    suggestions[f"{target}_predit_brut"] = pred[chosen]
    suggestions["incertitude"] = std[chosen]

    pred_col = f"{target}_predit"
    optimum[pred_col] = optimum[f"{target}_predit_brut"]
    suggestions[pred_col] = suggestions[f"{target}_predit_brut"]
    _apply_target_bounds(optimum, pred_col, target, notes, "Optimum")
    _apply_target_bounds(suggestions, pred_col, target, notes, "Suggestions")

    for c in schema.numeric:
        lo, hi = bounds[c]
        v, span = float(optimum.loc[0, c]), hi - lo
        if span > 0 and (v - lo < 0.02 * span or hi - v < 0.02 * span):
            notes.append(
                f"L'optimum pour '{c}' est au bord de la plage étudiée ({lo:g} – {hi:g}) : "
                "il pourrait se trouver au-delà. À vérifier expérimentalement, avec prudence."
            )
    if not ensemble:
        notes.append("Incertitude non calculée (trop peu de données).")
    notes.append(
        "Rappel : l'optimum est une prédiction de modèle dans un domaine couvert "
        "par peu de données — une hypothèse à tester, pas un résultat établi."
    )
    return OptimizationResult(optimum, suggestions, bounds, notes)
