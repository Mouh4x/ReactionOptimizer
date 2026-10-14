"""État partagé entre les pages Streamlit."""
from __future__ import annotations

import streamlit as st


def set_dataset(df, target=None, ignore=None, path=None) -> None:
    """Définit le jeu de données courant et réinitialise ce qui en dépend.

    `path` : fichier de sauvegarde associé. Les prochains enregistrements
    mettront à jour CE fichier au lieu d'en créer un nouveau à chaque clic.
    """
    st.session_state["df"] = df
    if target:
        st.session_state["target"] = target
    if ignore is not None:
        st.session_state["ignore"] = ignore
    if path is not None:
        st.session_state["dataset_path"] = str(path)
    elif "dataset_path" not in st.session_state:
        st.session_state["dataset_path"] = None
    st.session_state.pop("analysis", None)
    st.session_state.pop("manual_df", None)
    st.session_state["manual_version"] = st.session_state.get("manual_version", 0) + 1
