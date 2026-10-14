"""Tests : importation CSV/Excel, nettoyage des en-têtes, fusion, persistance."""
import io
from pathlib import Path

import pandas as pd
import pytest

from reactionoptimizer.data_loader import (
    clean_dataframe,
    load_dataset,
    load_excel,
    load_table,
    merge_datasets,
    normalize_column_name,
    prune_datasets,
    read_csv_safely,
    save_dataset,
)

BASE = "source,temperature_C,temps_h,quantite_g,solvant,DS\n" \
       "etude_1,80,2,1.5,eau,0.5\netude_2,90,3,2,DMSO,0.4\n"


def test_normalise_bom_simple_et_repete():
    assert normalize_column_name("﻿source") == "source"
    assert normalize_column_name("﻿﻿source") == "source"
    assert normalize_column_name("  temperature_C  ") == "temperature_C"
    assert normalize_column_name("DS") == "DS"


def test_csv_bom_repete_en_premiere_colonne():
    raw = ("﻿" * 3 + BASE).encode("utf-8")
    df = read_csv_safely(io.BytesIO(raw))
    df = clean_dataframe(df)
    assert "source" in df.columns
    assert not any("﻿" in c for c in df.columns)


def test_csv_encodage_latin1():
    raw = "source;solvant;DS\netude_1;écoulé;0.5\n".encode("latin-1")
    df = clean_dataframe(read_csv_safely(io.BytesIO(raw)))
    assert list(df.columns) == ["source", "solvant", "DS"]


def test_virgules_decimales_converties():
    df = clean_dataframe(pd.DataFrame({"DS": ["1,5", "2,5", "3,0"], "note": list("abc")}))
    assert pd.api.types.is_numeric_dtype(df["DS"])
    assert not pd.api.types.is_numeric_dtype(df["note"])


def test_colonnes_dupliquees_renommees():
    df = clean_dataframe(pd.DataFrame([[1, 2]], columns=["DS", "DS"]))
    assert list(df.columns) == ["DS", "DS.1"]


def test_excel_import(tmp_path):
    path = tmp_path / "t.xlsx"
    pd.DataFrame({"source": ["e1"], "DS": [0.3]}).to_excel(path, index=False)
    df = load_excel(path)
    assert list(df.columns) == ["source", "DS"]


def test_fusion_sans_dupliquer_source():
    """Le cas d'origine : fusionner un tableau dont « source » porte un BOM."""
    a = pd.DataFrame({"source": ["e1"], "temperature_C": [80.0], "DS": [0.5]})
    b = pd.DataFrame({"﻿source": ["e2"], "temperature_C": [90.0], "DS": [0.4]})
    merged, warnings, new_cols = merge_datasets(a, b)
    assert list(merged.columns).count("source") == 1
    assert len(merged) == 2
    assert new_cols == []


def test_fusion_colonnes_reordonnees():
    a = pd.DataFrame({"source": ["e1"], "DS": [0.5], "temperature_C": [80.0]})
    b = pd.DataFrame({"temperature_C": [90.0], "DS": [0.4], "source": ["e2"]})
    merged, _, _ = merge_datasets(a, b)
    assert merged.loc[1, "source"] == "e2"
    assert merged.loc[1, "temperature_C"] == 90.0


def test_fusion_schema_different_signale():
    a = pd.DataFrame({"source": ["e1"], "DS": [0.5]})
    b = pd.DataFrame({"source": ["e2"], "DS": [0.4], "catalyseur": ["Pd"]})
    merged, warnings, new_cols = merge_datasets(a, b)
    assert new_cols == ["catalyseur"]
    assert warnings  # incompatibilité explicitement signalée
    assert pd.isna(merged.loc[0, "catalyseur"])


def test_sauvegarde_et_rechargement_sans_bom_residuel(tmp_path):
    df = pd.DataFrame({"source": ["e1"], "DS": [0.5]})
    path = save_dataset(df, tmp_path)
    assert list(load_dataset(path).columns) == ["source", "DS"]


def test_mise_a_jour_en_place_pas_de_fichier_en_plus(tmp_path):
    a = pd.DataFrame({"source": ["e1"], "DS": [0.5]})
    path = save_dataset(a, tmp_path)
    b = pd.DataFrame({"source": ["e2"], "DS": [0.4]})
    merged, _, _ = merge_datasets(a, b)
    save_dataset(merged, tmp_path, path=path)
    assert len(list(tmp_path.glob("dataset_*.csv"))) == 1
    assert len(load_dataset(path)) == 2


def test_purge_ne_touche_que_les_fichiers_generes(tmp_path):
    for i in range(5):
        (tmp_path / f"dataset_2026101{i}_120000.csv").write_text("DS\n0\n", encoding="utf-8")
    autre = tmp_path / "mes_notes.csv"
    autre.write_text("ne pas toucher", encoding="utf-8")
    removed = prune_datasets(tmp_path, keep=2)
    assert len(removed) == 3
    assert autre.exists()
    assert len(list(tmp_path.glob("dataset_*.csv"))) == 2


def test_load_table_dispatch(tmp_path):
    csv_path = tmp_path / "t.csv"
    csv_path.write_text(BASE, encoding="utf-8")
    df = load_table(csv_path, csv_path.name)
    assert len(df) == 2
