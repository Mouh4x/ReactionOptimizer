"""Détection automatique du rôle de chaque colonne (cible, variables, ignorées)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api.types import is_numeric_dtype

# Colonnes de traçabilité : jamais utilisées comme variables du modèle.
ID_LIKE = {
    "id", "id_etude", "source", "etude", "doi", "ref", "reference",
    "references", "auteur", "auteurs", "note", "notes", "commentaire",
    "commentaires", "date", "lien", "url",
}


@dataclass
class Schema:
    target: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


def detect_schema(df: pd.DataFrame, target: str, ignore: list | None = None) -> Schema:
    """Classe chaque colonne : numérique, catégorielle ou ignorée."""
    ignore = set(ignore or [])
    numeric, categorical, ignored = [], [], []
    for col in df.columns:
        if col == target:
            continue
        if col in ignore or str(col).strip().lower() in ID_LIKE:
            ignored.append(col)
        elif is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return Schema(target=target, numeric=numeric, categorical=categorical,
                  ignored=ignored)


def find_group_column(df: pd.DataFrame) -> str | None:
    """Trouve la colonne identifiant l'étude d'origine (validation par étude)."""
    candidats = {"source", "etude", "id_etude", "doi", "ref", "reference"}
    for col in df.columns:
        if str(col).strip().lower() in candidats:
            s = df[col]
            if s.notna().sum() >= 3 and s.nunique(dropna=True) >= 3:
                return col
    return None
