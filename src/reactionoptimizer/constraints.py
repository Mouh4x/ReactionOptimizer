"""Contraintes expérimentales explicites appliquées aux résultats de l'optimiseur.

Lisez ceci avant toute modification — ce que ce module EST et N'EST PAS :

1. Les contraintes définies ici sont des **déclarations de l'utilisateur du
   projet** (hypothèses de modélisation documentées), pas des constantes
   physico-chimiques inventées. N'ajoutez une contrainte que si vous pouvez
   la justifier (définition de la grandeur, protocole, source fiable).
2. Les bornes de recherche par défaut proviennent des **plages observées dans
   les données** (voir ``optimizer.compute_bounds``) : une plage observée
   n'est PAS une limite chimique absolue, seulement le domaine où le modèle
   a des données.
3. Les contraintes de **compatibilité entre variables** (ex. température
   maximale admissible pour un solvant donné) ne sont PAS implémentées :
   elles doivent être fournies par l'utilisateur après validation
   scientifique, puis codées dans :func:`custom_filter`.

Pour ajouter une contrainte sans comprendre tout le code :
- borne sur la CIBLE    -> ajoutez une entrée dans TARGET_LOWER_BOUNDS /
                           TARGET_UPPER_BOUNDS ;
- borne sur une VARIABLE -> ajustez la plage à l'import (les bornes suivent
                           les données) ou codez-la dans custom_filter ;
- règle entre variables -> complétez custom_filter (exemple commenté).
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Bornes physiques / par définition sur les CIBLES.
#
# DS : dans ce projet, DS est un score expérimental non négatif par
# construction — toutes les valeurs observées sont dans [0, 1] et aucune
# définition signée n'est documentée dans le code ni les données. Un DS prédit
# négatif est donc un artefact du modèle (extrapolation hors du domaine des
# données), PAS un résultat chimique : la valeur AFFICHÉE est ramenée à la
# borne, avec un avertissement. Le modèle prédictif lui-même n'est pas modifié
# (les métriques de validation restent calculées sans contrainte, pour ne pas
# masquer ses erreurs).
#
# Si votre définition de DS admet réellement des valeurs négatives, supprimez
# l'entrée ci-dessous : les valeurs négatives seront alors conservées et
# affichées telles quelles.
# ---------------------------------------------------------------------------
TARGET_LOWER_BOUNDS: dict[str, float] = {"DS": 0.0}
TARGET_UPPER_BOUNDS: dict[str, float] = {}  # ex. {"DS": 1.0} si borne haute justifiée


def _lookup(table: dict[str, float], target: str) -> float | None:
    key = str(target).strip().lower()
    for name, value in table.items():
        if name.lower() == key:
            return value
    return None


def target_lower_bound(target: str) -> float | None:
    """Borne inférieure déclarée pour la cible (None si aucune)."""
    return _lookup(TARGET_LOWER_BOUNDS, target)


def target_upper_bound(target: str) -> float | None:
    """Borne supérieure déclarée pour la cible (None si aucune)."""
    return _lookup(TARGET_UPPER_BOUNDS, target)


def check_bounds(candidates: pd.DataFrame, bounds: dict) -> pd.Series:
    """Masque des candidats respectant toutes les bornes des variables.

    `bounds` vient de optimizer.compute_bounds : {colonne: (min, max)}.
    """
    mask = pd.Series(True, index=candidates.index)
    for col, (lo, hi) in bounds.items():
        if col in candidates.columns:
            mask &= candidates[col].between(lo, hi)
    return mask


def custom_filter(candidates: pd.DataFrame) -> pd.Series:
    """Règles de compatibilité entre variables — à compléter par l'utilisateur.

    Retourne un masque booléen : True = candidat admissible. Par défaut, tout
    est admissible (aucune règle définie). Exemple à adapter UNIQUEMENT si une
    source fiable le justifie :

        # Température maximale documentée pour l'eau (exemple fictif) :
        mask = pd.Series(True, index=candidates.index)
        if {"solvant", "temperature_C"} <= set(candidates.columns):
            mask &= ~((candidates["solvant"] == "eau")
                      & (candidates["temperature_C"] > 100))
        return mask
    """
    return pd.Series(True, index=candidates.index)
