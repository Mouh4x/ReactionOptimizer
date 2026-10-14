import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

from reactionoptimizer.config import datasets_dir, load_settings
from reactionoptimizer.data_loader import clean_dataframe, save_dataset
from reactionoptimizer.session import set_dataset

st.title("✍️ Saisie manuelle")
st.write("Remplissez le tableau ligne par ligne (le « + » en bas ajoute une ligne). "
         "Les virgules décimales sont acceptées.")

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

with st.expander("➕ Ajouter une variable (colonne)"):
    new_col = st.text_input("Nom de la nouvelle variable (ex. catalyseur, concentration_M)")
    if st.button("Ajouter la colonne"):
        name = new_col.strip()
        if not name:
            st.warning("Donnez un nom à la variable.")
        elif name in edited.columns:
            st.warning("Cette colonne existe déjà.")
        else:
            st.session_state["manual_df"] = edited.assign(**{name: None})
            st.session_state["manual_version"] += 1
            st.rerun()

if st.button("💾 Enregistrer ce tableau", type="primary"):
    cleaned = clean_dataframe(edited)
    if cleaned.empty:
        st.warning("Le tableau est vide.")
    else:
        # Édition du jeu de données courant : on met à jour son fichier ; sinon
        # on crée un nouveau fichier horodaté.
        existing_path = st.session_state.get("dataset_path")
        path = save_dataset(cleaned, datasets_dir(),
                            path=Path(existing_path) if existing_path else None)
        set_dataset(cleaned, path=path)
        st.success(f"{len(cleaned)} lignes enregistrées (sauvegardé : {path.name}). "
                   "Passez à la page « Modèle et optimisation ».")
