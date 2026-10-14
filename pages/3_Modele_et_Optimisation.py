import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import streamlit as st

from reactionoptimizer.config import load_settings
from reactionoptimizer.data_loader import validate
from reactionoptimizer.evaluation import fit_best
from reactionoptimizer.optimizer import compute_bounds, optimize
from reactionoptimizer.preprocessing import prepare_xy
from reactionoptimizer.reporting import response_slices_figure, results_to_excel
from reactionoptimizer.schema import detect_schema, find_group_column

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.title("📊 Modèle et optimisation")

df = st.session_state.get("df")
if df is None or df.empty:
    st.info("Aucune donnée chargée : importez un Excel ou saisissez des lignes d'abord.")
    st.stop()

st.dataframe(df)

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("Aucune colonne numérique : impossible de choisir une cible.")
    st.stop()

default = st.session_state.get("target", load_settings()["default_target"]).lower()
idx = next((i for i, c in enumerate(numeric_cols) if c.lower() == default), 0)

c1, c2, c3 = st.columns(3)
target = c1.selectbox("Colonne à optimiser", numeric_cols, index=idx)
goal = c2.radio("Objectif", ["Maximiser", "Minimiser"], horizontal=True)
expand = c3.slider("Élargir la plage de recherche (%)", 0, 30, 0) / 100

others = [c for c in df.columns if c != target]
ignore_default = [c for c in st.session_state.get("ignore", []) if c in others]
ignore = st.multiselect("Colonnes ignorées", others, default=ignore_default)

schema = detect_schema(df, target=target, ignore=ignore)
X, y = prepare_xy(df, schema)

group_col = find_group_column(df)
groups = None
if group_col is not None:
    groups = (df.dropna(subset=[schema.target])[group_col]
              .fillna("inconnu").astype(str).reset_index(drop=True))
    st.caption(f"Validation par étude activée (colonne « {group_col} », "
               f"{groups.nunique()} études).")

st.caption(f"{len(y)} lignes exploitables · {len(schema.numeric)} variables numériques · "
           f"{len(schema.categorical)} catégorielles")
for w in validate(df, schema):
    st.warning(w)
if expand > 0:
    st.warning("Plage élargie au-delà des données : l'optimum devient une extrapolation, "
               "donc moins fiable.")

if st.button("🚀 Lancer l'analyse", type="primary"):
    with st.spinner("Comparaison des modèles et optimisation en cours..."):
        try:
            fit = fit_best(X, y, schema, groups=groups)
            bounds = compute_bounds(X, schema, expand=expand)
            res = optimize(fit.model, X, y, schema, bounds=bounds,
                           maximize=(goal == "Maximiser"))
        except Exception as exc:
            st.error(f"Analyse impossible : {exc}")
            st.stop()
    st.session_state["analysis"] = {"fit": fit, "res": res, "schema": schema}

analysis = st.session_state.get("analysis")
if analysis is None:
    st.stop()
if analysis["schema"].target != target:
    st.info("Les résultats affichés concernent une autre cible : relancez l'analyse.")
    st.stop()

fit, res, sc = analysis["fit"], analysis["res"], analysis["schema"]

st.header("1. Comparaison des modèles")
st.dataframe(fit.results.round(4))
st.success(f"Modèle retenu : {fit.best_name}")
if fit.cv_method:
    st.caption(f"Méthode de validation : {fit.cv_method}. Un R² négatif signifie que le "
               "modèle fait moins bien que la référence sur les données évaluées — il "
               "est affiché tel quel.")
for note in fit.warnings:
    st.warning(note)

st.header("2. Conditions optimales prédites")
st.dataframe(res.optimum.round(3))
st.caption("« _predit_brut » : sortie du modèle avant application des bornes physiques "
           "déclarées (constraints.py). « incertitude » : variation de la prédiction "
           "entre rééchantillonnages des données ; elle sous-estime l'erreur réelle — "
           "référez-vous au RMSE ci-dessus.")
for note in res.notes:
    st.warning(note)

st.header("3. Prochaines expériences suggérées")
st.dataframe(res.suggestions.round(3))
st.caption("Suggestions = bonne valeur prédite + incertitude élevée, éloignées les "
           "unes des autres et différentes des expériences déjà réalisées.")

st.header("4. Effet de chaque variable autour de l'optimum")
fig = response_slices_figure(fit.model, res.optimum, sc, res.bounds)
if fig is not None:
    st.plotly_chart(fig)

st.download_button(
    "⬇️ Exporter les résultats (Excel)",
    data=results_to_excel({
        "Optimum": res.optimum,
        "Experiences_suggerees": res.suggestions,
        "Comparaison_modeles": fit.results,
        "Donnees": df,
    }),
    file_name="resultats_reactionoptimizer.xlsx",
    mime=XLSX,
)
