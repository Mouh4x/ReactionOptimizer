"""Lecture, nettoyage, validation, fusion et sauvegarde des jeux de données.

Toutes les voies d'importation (Excel, CSV, saisie manuelle, fusion de tableaux,
rechargement d'une sauvegarde) passent par :func:`clean_dataframe` : le nettoyage
des en-têtes (BOM, espaces, doublons) est centralisé ici, pas dispersé dans les
pages Streamlit.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schema import Schema

# Encodages essayés après utf-8-sig / utf-8 pour les CSV.
_ENCODINGS = ("cp1252", "latin-1")


def normalize_column_name(name) -> str:
    """Normalise un nom de colonne sans altérer les noms légitimes.

    - supprime les BOM (﻿), même répétés ou en milieu de nom, et les
      espaces de largeur nulle ;
    - normalise l'Unicode (NFC) ;
    - supprime les espaces en début/fin et compacte les espaces multiples.
    """
    text = unicodedata.normalize("NFC", str(name))
    text = text.replace("﻿", "").replace("​", "")
    return re.sub(r"\s+", " ", text).strip()


def _deduplicate_columns(columns: list[str]) -> list[str]:
    """Renomme les doublons exacts en 'nom.1', 'nom.2'... (convention pandas)."""
    seen: dict[str, int] = {}
    result = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            result.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            result.append(col)
    return result


def _to_numeric_if_possible(series: pd.Series, threshold: float = 0.8) -> pd.Series:
    """Convertit '1,5' ou '1.5' en nombre si au moins 80 % des valeurs s'y prêtent."""
    if is_numeric_dtype(series):
        return series
    text = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    converted = pd.to_numeric(text.where(series.notna()), errors="coerce")
    ratio = converted.notna().sum() / max(series.notna().sum(), 1)
    return converted if ratio >= threshold else series


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage unique appliqué à toutes les sources de données."""
    df = df.copy()
    df.columns = _deduplicate_columns([normalize_column_name(c) for c in df.columns])
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for col in df.columns:
        df[col] = _to_numeric_if_possible(df[col])
    return df.reset_index(drop=True)


def read_csv_safely(file) -> pd.DataFrame:
    """Lit un CSV en gérant BOM (même répété), séparateur et encodages courants.

    utf-8-sig supprime le BOM de début de fichier ; les BOM restants (répétés
    ou collés à un en-tête) sont supprimés ensuite par clean_dataframe.
    """
    def _try(encoding: str) -> pd.DataFrame:
        if hasattr(file, "seek"):
            file.seek(0)
        return pd.read_csv(file, sep=None, engine="python", encoding=encoding)

    for encoding in ("utf-8-sig",) + _ENCODINGS:
        try:
            return _try(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Encodage du CSV non reconnu : enregistrez-le en UTF-8.")


def load_excel(file, sheet_name=0) -> pd.DataFrame:
    """`file` peut être un chemin ou un fichier uploadé dans Streamlit."""
    return clean_dataframe(pd.read_excel(file, sheet_name=sheet_name))


def load_csv(file) -> pd.DataFrame:
    """Charge un CSV (chemin ou fichier uploadé) puis le nettoie."""
    return clean_dataframe(read_csv_safely(file))


def load_table(file, filename: str = "", sheet_name=0) -> pd.DataFrame:
    """Point d'entrée unique d'importation : Excel ou CSV selon l'extension.

    `sheet_name` est transmis à :func:`load_excel` (ignoré pour les CSV) :
    la page d'importation peut ainsi proposer le choix de la feuille.
    """
    if str(filename).lower().endswith(".csv"):
        return load_csv(file)
    return load_excel(file, sheet_name=sheet_name)


def load_dataset(path: Path) -> pd.DataFrame:
    """Recharge une sauvegarde CSV en appliquant le même nettoyage qu'à l'import."""
    return clean_dataframe(read_csv_safely(path))


def merge_datasets(existing: pd.DataFrame, new: pd.DataFrame):
    """Ajoute les lignes de `new` à `existing` après normalisation des deux schémas.

    Règle de correspondance : les en-têtes sont normalisés (BOM, espaces,
    Unicode) AVANT comparaison, donc « ﻿source » et « source » désignent la
    même colonne et ne sont jamais dupliquées. Deux colonnes de noms
    normalisés différents restent des colonnes distinctes.

    Retourne (tableau fusionné, avertissements, nouvelles_colonnes) :
    - les colonnes absentes d'un côté donnent des valeurs vides (NaN) de
      l'autre, avec un avertissement explicite ;
    - les types sont harmonisés après concaténation (virgules décimales...).
    """
    a = clean_dataframe(existing)
    b = clean_dataframe(new)
    warnings: list[str] = []
    new_cols = [c for c in b.columns if c not in a.columns]
    missing_in_new = [c for c in a.columns if c not in b.columns]
    if new_cols:
        warnings.append(
            "Nouvelles colonnes (absentes du tableau actuel) : " + ", ".join(new_cols)
            + ". Les anciennes lignes auront des valeurs vides pour ces colonnes."
        )
    if missing_in_new:
        warnings.append(
            "Colonnes absentes du fichier importé : " + ", ".join(missing_in_new)
            + ". Les nouvelles lignes auront des valeurs vides pour ces colonnes."
        )
    merged = pd.concat([a, b], ignore_index=True)
    for col in merged.columns:
        merged[col] = _to_numeric_if_possible(merged[col])
    return merged.reset_index(drop=True), warnings, new_cols


def validate(df: pd.DataFrame, schema: Schema) -> list[str]:
    """Retourne la liste des avertissements lisibles (rien n'est masqué)."""
    warnings: list[str] = []
    n_valid = int(df[schema.target].notna().sum())
    k = len(schema.features)
    needed = max(10, 5 * k)

    if n_valid < len(df):
        warnings.append(
            f"{len(df) - n_valid} ligne(s) sans valeur pour '{schema.target}' : "
            "elles seront ignorées."
        )
    if df[schema.target].nunique(dropna=True) <= 1:
        warnings.append(
            f"La cible '{schema.target}' est constante : aucun modèle ne peut "
            "apprendre quoi que ce soit."
        )
    if n_valid < needed:
        warnings.append(
            f"Seulement {n_valid} ligne(s) exploitable(s) pour {k} variable(s) : "
            f"résultats peu fiables (visez au moins {needed})."
        )
    missing = df[schema.features].isna().sum()
    for col, m in missing[missing > 0].items():
        warnings.append(f"Colonne '{col}' : {m} valeur(s) manquante(s).")
    for col in schema.numeric:
        if df[col].nunique(dropna=True) <= 1:
            warnings.append(
                f"Colonne '{col}' constante : elle n'apporte aucune information au modèle."
            )
    for col in schema.categorical:
        n_mod = df[col].nunique(dropna=True)
        if n_mod <= 1:
            warnings.append(
                f"Colonne '{col}' n'a qu'une seule modalité : aucun effet mesurable."
            )
        elif n_mod > max(20, len(df) // 2):
            warnings.append(
                f"Colonne '{col}' a presque autant de modalités ({n_mod}) que de "
                "lignes : s'agit-il d'un identifiant ? Elle devrait peut-être être "
                "ignorée (page Import / Modèle)."
            )
    if schema.features and df.duplicated(subset=schema.features + [schema.target]).any():
        warnings.append("Des lignes en double ont été détectées.")
    if schema.ignored:
        warnings.append(f"Colonnes ignorées par le modèle : {', '.join(schema.ignored)}.")
    return warnings


def save_dataset(df: pd.DataFrame, directory: Path, path: Path | None = None) -> Path:
    """Sauvegarde le jeu de données.

    - `path=None` : crée un nouveau fichier horodaté (nouveau jeu de données) ;
    - `path` fourni : met à jour CE fichier en place (ajout/édition d'un jeu
      existant) — évite la multiplication des fichiers à chaque enregistrement.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if path is None:
        path = directory / f"dataset_{datetime.now():%Y%m%d_%H%M%S}.csv"
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def prune_datasets(directory: Path, keep: int = 20) -> list[Path]:
    """Ne conserve que les `keep` sauvegardes les plus récentes.

    Sécurité : seuls les fichiers correspondant exactement au motif
    ``dataset_AAAAMMJJ_HHMMSS.csv`` créé par save_dataset sont concernés ;
    tout autre fichier du dossier est laissé intact.
    """
    removed: list[Path] = []
    files = list_datasets(directory)
    for f in files[max(keep, 1):]:
        f.unlink()
        removed.append(f)
    return removed


def list_datasets(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("dataset_*.csv"), reverse=True)
