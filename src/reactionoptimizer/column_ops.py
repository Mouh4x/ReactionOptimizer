"""Opérations dynamiques sur les colonnes du jeu de données.

Toutes les modifications de structure (ajout, suppression, renommage, retypage)
passent par ce module : c'est le point unique qui garantit que les références
externes (cible sélectionnée, colonnes ignorées) restent cohérentes avec le
tableau après chaque changement.

Aucune colonne n'est « indispensable » en soi — y compris « solvant », qui n'est
qu'une variable catégorielle d'exemple : le logiciel fonctionne avec n'importe
quel ensemble de colonnes dès lors qu'il reste au moins une colonne numérique
exploitable comme cible. Les protections ci-dessous empêchent uniquement les
états incohérents (tableau sans colonne, nom dupliqué, valeur convertie
silencieusement en une valeur fausse) ; elles n'interdisent jamais une
structure légitime.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .data_loader import normalize_column_name

# Types proposés à l'utilisateur (clé interne -> libellé affiché).
COLUMN_TYPES = {
    "auto": "Automatique (détection)",
    "number": "Nombre (entier ou décimal)",
    "text": "Texte / catégorie",
    "boolean": "Booléen (oui/non, vrai/faux, 0/1)",
}

BOOL_TRUE = {"1", "oui", "yes", "true", "vrai", "o", "y", "t"}
BOOL_FALSE = {"0", "non", "no", "false", "faux", "n", "f"}


@dataclass
class ColumnChange:
    """Résultat d'une opération : nouveau tableau + messages + effets constatés."""
    df: pd.DataFrame
    messages: list[str] = field(default_factory=list)
    removed_target: bool = False                   # la cible a été supprimée
    removed_ignored: list[str] = field(default_factory=list)  # ignorées supprimées
    renamed: dict[str, str] = field(default_factory=dict)     # ancien -> nouveau


# ---------------------------------------------------------------------------
# Ajout / suppression / renommage
# ---------------------------------------------------------------------------
def add_column(df: pd.DataFrame, name) -> ColumnChange:
    """Ajoute une colonne vide. Le nom est normalisé (BOM, espaces, NFC)."""
    name = normalize_column_name(name)
    if not name:
        raise ValueError("Donnez un nom à la colonne.")
    if name in df.columns:
        raise ValueError(f"La colonne « {name} » existe déjà.")
    new = df.copy()
    new[name] = pd.Series([None] * len(df), dtype="object")
    return ColumnChange(new, [f"Colonne « {name} » ajoutée (valeurs vides)."])


def remove_columns(df: pd.DataFrame, columns, target: str | None = None,
                   ignore: list[str] | None = None) -> ColumnChange:
    """Supprime des colonnes sans casser les références (cible, ignorées).

    La cible n'est PAS protégée artificiellement : sa suppression est permise,
    signalée, et ``update_references`` invalide proprement la sélection pour
    que l'utilisateur en choisisse une nouvelle.
    """
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError("Aucune des colonnes demandées n'existe dans le tableau.")
    if len(columns) >= len(df.columns):
        raise ValueError("Impossible de supprimer toutes les colonnes du tableau.")
    ignore = list(ignore or [])
    removed_target = bool(target) and target in columns
    removed_ignored = [c for c in columns if c in ignore]
    messages = ["Colonne(s) supprimée(s) : " + ", ".join(f"« {c} »" for c in columns) + "."]
    if removed_target:
        messages.append(
            f"La cible « {target} » a été supprimée : la sélection est invalidée, "
            "choisissez une nouvelle colonne à optimiser."
        )
    if removed_ignored:
        messages.append(
            "Retirées de la liste des colonnes ignorées : "
            + ", ".join(f"« {c} »" for c in removed_ignored) + "."
        )
    new = df.drop(columns=columns)
    return ColumnChange(new, messages, removed_target=removed_target,
                        removed_ignored=removed_ignored)


def rename_column(df: pd.DataFrame, old: str, new_name,
                  target: str | None = None,
                  ignore: list[str] | None = None) -> ColumnChange:
    """Renomme une colonne ; les références (cible, ignorées) suivent."""
    new_name = normalize_column_name(new_name)
    if old not in df.columns:
        raise ValueError(f"Colonne « {old} » introuvable.")
    if not new_name:
        raise ValueError("Donnez un nouveau nom à la colonne.")
    if new_name == old:
        return ColumnChange(df.copy(), ["Nom inchangé."])
    if new_name in df.columns:
        raise ValueError(f"La colonne « {new_name} » existe déjà.")
    new = df.rename(columns={old: new_name})
    messages = [f"Colonne « {old} » renommée en « {new_name} »."]
    if target == old:
        messages.append(f"La cible suit le renommage : « {new_name} ».")
    if ignore and old in ignore:
        messages.append("La liste des colonnes ignorées suit le renommage.")
    return ColumnChange(new, messages, renamed={old: new_name})


# ---------------------------------------------------------------------------
# Retypage
# ---------------------------------------------------------------------------
def _to_boolean(series: pd.Series) -> pd.Series:
    """Convertit vers le booléen nullable pandas ; non reconnu -> NA (jamais False arbitraire)."""
    if is_bool_dtype(series):
        return series.astype("boolean")

    def parse(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return pd.NA
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            if float(v) == 1.0:
                return True
            if float(v) == 0.0:
                return False
            return pd.NA
        t = str(v).strip().lower()
        if t in BOOL_TRUE:
            return True
        if t in BOOL_FALSE:
            return False
        return pd.NA

    return series.map(parse).astype("boolean")


def _to_number(series: pd.Series) -> pd.Series:
    """Convertit vers numérique (virgules décimales acceptées) ; non convertible -> NaN."""
    if is_bool_dtype(series):
        # True -> 1.0, False -> 0.0, NA préservé (conversion explicite, documentée).
        return series.astype("Float64")
    if is_numeric_dtype(series):
        return series
    text = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(text.where(series.notna()), errors="coerce")


def retype_column(df: pd.DataFrame, column: str, target_type: str) -> ColumnChange:
    """Change le type d'une colonne, sans conversion silencieuse trompeuse.

    Les valeurs incompatibles sont vidées (NaN/NA) ET comptées dans un message :
    elles ne sont jamais remplacées par une valeur arbitraire (0, False...).
    """
    if column not in df.columns:
        raise ValueError(f"Colonne « {column} » introuvable.")
    if target_type not in COLUMN_TYPES:
        raise ValueError(f"Type inconnu : « {target_type} » "
                         f"(attendu : {', '.join(COLUMN_TYPES)}).")
    new = df.copy()
    s = new[column]
    messages: list[str] = []

    if target_type == "auto":
        # Réutilise la règle d'inférence centralisée (>= 80 % convertibles).
        from .data_loader import _to_numeric_if_possible
        conv = _to_numeric_if_possible(s)
        if is_numeric_dtype(conv) and not is_numeric_dtype(s):
            messages.append(f"Colonne « {column} » détectée comme numérique.")
        elif not is_numeric_dtype(conv):
            bools = _to_boolean(s)
            if bools.notna().sum() == s.notna().sum() and s.notna().sum() > 0:
                conv = bools
                messages.append(f"Colonne « {column} » détectée comme booléenne.")
            else:
                messages.append(f"Colonne « {column} » conservée en texte/catégorie.")
        new[column] = conv
    elif target_type == "number":
        conv = _to_number(s)
        lost = int((s.notna() & conv.isna()).sum())
        if lost:
            messages.append(
                f"{lost} valeur(s) non numérique(s) vidée(s) dans « {column} » "
                "(elles ne sont pas remplacées par 0)."
            )
        messages.append(f"Colonne « {column} » convertie en nombre.")
        new[column] = conv
    elif target_type == "boolean":
        conv = _to_boolean(s)
        lost = int((s.notna() & conv.isna()).sum())
        if lost:
            messages.append(
                f"{lost} valeur(s) non reconnue(s) comme booléen vidée(s) dans "
                f"« {column} » (attendu : oui/non, vrai/faux, 0/1)."
            )
        messages.append(f"Colonne « {column} » convertie en booléen.")
        new[column] = conv
    else:  # text
        new[column] = s.astype("string")
        messages.append(f"Colonne « {column} » convertie en texte.")

    return ColumnChange(new, messages)


# ---------------------------------------------------------------------------
# Cohérence des références externes
# ---------------------------------------------------------------------------
def update_references(change: ColumnChange, target: str | None,
                      ignore: list[str] | None) -> tuple[str | None, list[str]]:
    """Recalcule (cible, colonnes ignorées) après une opération sur les colonnes.

    - renommage : les références suivent automatiquement le nouveau nom ;
    - suppression : une cible supprimée devient ``None`` (à re-sélectionner),
      les ignorées supprimées disparaissent de la liste.
    """
    new_target = target
    if target is not None:
        if target in change.renamed:
            new_target = change.renamed[target]
        elif target not in change.df.columns:
            new_target = None
    new_ignore: list[str] = []
    for c in (ignore or []):
        c2 = change.renamed.get(c, c)
        if c2 in change.df.columns and c2 != new_target and c2 not in new_ignore:
            new_ignore.append(c2)
    return new_target, new_ignore
