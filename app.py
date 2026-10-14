import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from reactionoptimizer.config import datasets_dir
from reactionoptimizer.data_loader import list_datasets, load_dataset
from reactionoptimizer.session import set_dataset

st.set_page_config(page_title="ReactionOptimizer", page_icon="⚗️", layout="wide")

st.title("⚗️ ReactionOptimizer")
st.write(
    "Trouvez les meilleures conditions de réaction à partir de vos données d'études : "
    "le logiciel compare plusieurs modèles, retient le meilleur, prédit l'optimum "
    "et propose les prochaines expériences à tester."
)
st.markdown(
    """
**Comment ça marche**
1. **Importez** un Excel (page *Import Excel*) ou **saisissez** vos lignes (page *Saisie manuelle*).
2. Allez sur **Modèle et optimisation** et lancez l'analyse.
3. Testez les conditions proposées, puis ajoutez vos nouveaux résultats pour affiner le modèle.

> Les résultats sont des prédictions statistiques destinées à orienter vos
> prochaines expériences — ils ne remplacent pas la validation au laboratoire.
"""
)

with st.sidebar:
    st.header("Jeux de données sauvegardés")
    saved = list_datasets(datasets_dir())
    if saved:
        choice = st.selectbox("Fichier", saved, format_func=lambda p: p.name)
        if st.button("Charger ce jeu de données"):
            set_dataset(load_dataset(choice), path=choice)
            st.success("Jeu de données chargé.")
    else:
        st.caption("Aucun jeu de données sauvegardé pour l'instant.")

df = st.session_state.get("df")
if df is not None:
    st.subheader(f"Jeu de données actuel : {len(df)} lignes, {len(df.columns)} colonnes")
    st.dataframe(df)
else:
    st.info("Aucun jeu de données chargé pour l'instant.")
