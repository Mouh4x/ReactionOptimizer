import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

from reactionoptimizer.column_ops import (
    COLUMN_TYPES,
    add_column,
    remove_columns,
    rename_column,
    retype_column,
    update_references,
)
from reactionoptimizer.config import datasets_dir, load_settings
from reactionoptimizer.data_loader import clean_dataframe, save_dataset
from reactionoptimizer.session import set_dataset

st.title("✍️ Saisie manuelle")
st.write("Remplissez le tableau ligne par ligne (le « + » en bas ajoute une ligne). "
         "Les virgules décimales sont acceptées. Gérez les colonnes ci-dessous : "
         "ajout, suppression, renommage et type de données.")

settings = load_settings()
st.session_state.setdefault("manual_version", 0)

if "manual_df" not in st.session_state:
    if "df" in st.session_state:
        st.session_state["manual_df"] = st.session_state["df"].copy()
    else:
        cols = ["temperature_C", "temps_h", "quantite_g", "solvant", settings["default_target"]]
        st.session_state["manual_df"] = pd.DataFrame(
            {c: pd.Series(dtype="object") for c in cols}
        )

edited = st.data_editor(
    st.session_state["manual_df"],
    num_rows="dynamic",
    key=f"editor_{st.session_state['manual_version']}",
)

# Messages différés : affichés après le rerun déclenché par une opération.
for msg, level in st.session_state.pop("_colop_messages", []):
    (st.warning if level == "warning" else st.success)(msg)


def _apply(change):
    """Enregistre le résultat d'une opération sur les colonnes.

    Met à jour le tableau de travail ET les références de session (cible,
    colonnes ignorées) via update_references, puis réinitialise l'éditeur
    pour refléter la nouvelle structure.
    """
    target, ignore = update_references(
        change,
        st.session_state.get("target"),
        st.session_state.get("ignore", []),
    )
    st.session_state["target"] = target
    st.session_state["ignore"] = ignore
    st.session_state["manual_df"] = change.df
    st.session_state["_colop_messages"] = [
        (m, "warning") if "vidée" in m or "supprimée" in m else (m, "success")
        for m in change.messages
    ]
    st.session_state["manual_version"] += 1
    st.rerun()


with st.expander("🧩 Gérer les colonnes (ajouter, supprimer, renommer, changer le type)"):
    tab_add, tab_del, tab_ren, tab_typ = st.tabs(
        ["➕ Ajouter", "🗑️ Supprimer", "✏️ Renommer", "🔢 Type de données"])

    with tab_add:
        new_col = st.text_input("Nom de la nouvelle variable (ex. catalyseur, concentration_M)")
        if st.button("Ajouter la colonne"):
            try:
                _apply(add_column(edited, new_col))
            except ValueError as exc:
                st.warning(str(exc))

    with tab_del:
        to_drop = st.multiselect("Colonne(s) à supprimer", list(edited.columns))
        st.caption("Aucune colonne n'est verrouillée : la suppression est toujours "
                   "possible. Si la colonne supprimée était la cible ou une colonne "
                   "ignorée, la sélection est mise à jour automatiquement — vous "
                   "devrez choisir une nouvelle cible à la page « Modèle et "
                   "optimisation ». Impossible de supprimer toutes les colonnes.")
        if st.button("Supprimer la/les colonne(s)", disabled=not to_drop):
            try:
                _apply(remove_columns(edited, to_drop,
                                      target=st.session_state.get("target"),
                                      ignore=st.session_state.get("ignore", [])))
            except ValueError as exc:
                st.warning(str(exc))

    with tab_ren:
        if len(edited.columns):
            old = st.selectbox("Colonne à renommer", list(edited.columns))
            new_name = st.text_input("Nouveau nom")
            if st.button("Renommer"):
                try:
                    _apply(rename_column(edited, old, new_name,
                                         target=st.session_state.get("target"),
                                         ignore=st.session_state.get("ignore", [])))
                except ValueError as exc:
                    st.warning(str(exc))

    with tab_typ:
        if len(edited.columns):
            col_typ = st.selectbox("Colonne à convertir", list(edited.columns),
                                   key="retype_col")
            new_type = st.selectbox("Nouveau type", list(COLUMN_TYPES),
                                    format_func=lambda k: COLUMN_TYPES[k])
            st.caption("Les valeurs incompatibles sont vidées (jamais remplacées "
                       "par 0 ou « faux ») et leur nombre est signalé.")
            if st.button("Appliquer le type"):
                try:
                    _apply(retype_column(edited, col_typ, new_type))
                except ValueError as exc:
                    st.warning(str(exc))

if st.button("💾 Enregistrer ce tableau", type="primary"):
    cleaned = clean_dataframe(edited)
    if cleaned.empty:
        st.warning("Le tableau est vide.")
    else:
        # Édition du jeu de données courant : on met à jour son fichier ; sinon
        # on crée un nouveau fichier horodaté. set_dataset purge les références
        # (cible, ignorées) qui pointeraient vers des colonnes supprimées.
        existing_path = st.session_state.get("dataset_path")
        path = save_dataset(cleaned, datasets_dir(),
                            path=Path(existing_path) if existing_path else None)
        set_dataset(cleaned, path=path)
        st.success(f"{len(cleaned)} lignes enregistrées (sauvegardé : {path.name}). "
                   "Passez à la page « Modèle et optimisation ».")
