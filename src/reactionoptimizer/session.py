"""État partagé entre les pages Streamlit."""
from __future__ import annotations

import streamlit as st


def set_dataset(df, target=None, ignore=None, path=None) -> None:
    """Définit le jeu de données courant et réinitialise ce qui en dépend.

    `path` : fichier de sauvegarde associé. Les prochains enregistrements
    mettront à jour CE fichier au lieu d'en créer un nouveau à chaque clic.

    Les références (cible, colonnes ignorées) sont purgées des noms qui
    n'existent plus dans le tableau : une cible disparue devient None (l'page
    « Modèle et optimisation » invitera alors à en choisir une nouvelle) et
    une ignorée disparue est retirée de la liste — jamais de référence
    obsolète conservée silencieusement.
    """
    st.session_state["df"] = df
    cols = set(df.columns)
    if target:
        st.session_state["target"] = target if target in cols else None
    elif st.session_state.get("target") not in cols:
        st.session_state["target"] = None
    if ignore is not None:
        st.session_state["ignore"] = [c for c in ignore if c in cols]
    else:
        st.session_state["ignore"] = [
            c for c in st.session_state.get("ignore", []) if c in cols
        ]
    if path is not None:
        st.session_state["dataset_path"] = str(path)
    elif "dataset_path" not in st.session_state:
        st.session_state["dataset_path"] = None
    st.session_state.pop("analysis", None)
    st.session_state.pop("manual_df", None)
    st.session_state["manual_version"] = st.session_state.get("manual_version", 0) + 1
