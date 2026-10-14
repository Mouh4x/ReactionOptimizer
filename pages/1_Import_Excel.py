import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

from reactionoptimizer.config import datasets_dir, load_settings
from reactionoptimizer.data_loader import (
    load_excel,
    load_table,
    merge_datasets,
    save_dataset,
    validate,
)
from reactionoptimizer.reporting import template_excel
from reactionoptimizer.schema import detect_schema
from reactionoptimizer.session import set_dataset

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.title("📥 Import d'un fichier Excel")

st.download_button("Télécharger le modèle Excel vide", data=template_excel(),
                   file_name="modele_reactionoptimizer.xlsx", mime=XLSX)

uploaded = st.file_uploader("Fichier Excel ou CSV", type=["xlsx", "xls", "csv"])

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            # Les colonnes sont détectées automatiquement, sans liste prédéfinie ;
            # seule la feuille est à choisir quand le classeur en contient plusieurs.
            book = pd.ExcelFile(uploaded)
            if not book.sheet_names:
                st.error("Classeur vide : aucune feuille trouvée.")
                st.stop()
            sheet = (st.selectbox("Feuille à importer", book.sheet_names)
                     if len(book.sheet_names) > 1 else book.sheet_names[0])
            df = load_excel(book, sheet_name=sheet)
        else:
            df = load_table(uploaded, uploaded.name)
    except Exception as exc:
        st.error(f"Lecture impossible : {exc}")
        st.stop()

    if df.empty or len(df.columns) == 0:
        st.error("Aucune donnée exploitable dans ce fichier (tableau vide). "
                 "Le jeu de données actuel n'a pas été modifié.")
        st.stop()

    st.subheader("Aperçu")
    st.dataframe(df)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        st.error("Aucune colonne numérique trouvée : impossible de choisir une cible. "
                 "Le jeu de données actuel n'a pas été modifié.")
        st.stop()

    default = load_settings()["default_target"].lower()
    idx = next((i for i, c in enumerate(numeric_cols) if c.lower() == default), 0)
    target = st.selectbox("Colonne à optimiser (cible)", numeric_cols, index=idx)
    ignore = st.multiselect("Colonnes à ignorer (identifiants, notes...)",
                            [c for c in df.columns if c != target])

    schema = detect_schema(df, target=target, ignore=ignore)
    c1, c2, c3 = st.columns(3)
    c1.metric("Variables numériques", len(schema.numeric))
    c1.caption(", ".join(schema.numeric) or "—")
    c2.metric("Variables catégorielles", len(schema.categorical))
    c2.caption(", ".join(schema.categorical) or "—")
    c3.metric("Colonnes ignorées", len(schema.ignored))
    c3.caption(", ".join(schema.ignored) or "—")

    for w in validate(df, schema):
        st.warning(w)

    has_current = "df" in st.session_state
    mode = st.radio("Que faire de ces données ?",
                    ["Remplacer le jeu de données actuel", "Ajouter au jeu de données actuel"],
                    disabled=not has_current)

    if st.button("✅ Utiliser ces données", type="primary"):
        if has_current and mode.startswith("Ajouter"):
            # Fusion avec normalisation des en-têtes des deux côtés : une même
            # colonne logique (ex. « ﻿source » et « source ») n'est jamais dupliquée.
            new, merge_warnings, _ = merge_datasets(st.session_state["df"], df)
            for w in merge_warnings:
                st.warning(w)
            # Ajout à un jeu existant : on met à jour SON fichier, pas de
            # nouveau fichier à chaque clic.
            existing_path = st.session_state.get("dataset_path")
            path = save_dataset(new, datasets_dir(),
                                path=Path(existing_path) if existing_path else None)
        else:
            new = df
            path = save_dataset(new, datasets_dir())
        set_dataset(new, target=target, ignore=ignore, path=path)
        st.success(f"{len(new)} lignes chargées (sauvegardé : {path.name}). "
                   "Passez à la page « Modèle et optimisation ».")
