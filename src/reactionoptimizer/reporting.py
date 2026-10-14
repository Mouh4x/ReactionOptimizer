"""Exports Excel et graphiques."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .schema import Schema


def results_to_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def template_excel() -> bytes:
    columns = ["source", "temperature_C", "temps_h", "quantite_g", "solvant", "DS"]
    instructions = pd.DataFrame({"Instructions": [
        "Onglet 'Donnees' : une ligne = une expérience, en-têtes sur la première ligne.",
        "Pour ajouter une variable, ajoutez simplement une colonne.",
        "Mettez l'unité dans le nom de la colonne (temperature_C, temps_h...).",
        "La colonne à optimiser (par défaut 'DS') doit être numérique.",
        "Colonnes source, id, doi, ref, note... : ignorées par le modèle (traçabilité).",
        "Texte (solvant, catalyseur...) = variable catégorielle ; nombres = variable numérique.",
        "Les virgules décimales (1,5) sont acceptées.",
        "DS est déclaré non négatif (borne 0) : voir src/reactionoptimizer/constraints.py.",
    ]})
    return results_to_excel({"Donnees": pd.DataFrame(columns=columns),
                             "Instructions": instructions})


def response_slices_figure(model, optimum: pd.DataFrame, schema: Schema,
                           bounds: dict, n_points: int = 60):
    """Une courbe par variable numérique, les autres étant fixées à l'optimum."""
    cols = schema.numeric
    if not cols:
        return None
    n_cols = min(3, len(cols))
    n_rows = -(-len(cols) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cols)
    base = optimum[schema.features].iloc[[0]].reset_index(drop=True)
    best_pred = float(model.predict(base)[0])

    for k, col in enumerate(cols):
        r, c = divmod(k, n_cols)
        lo, hi = bounds[col]
        grid = np.linspace(lo, hi, n_points)
        frame = pd.concat([base] * n_points, ignore_index=True)
        frame[col] = grid
        fig.add_trace(go.Scatter(x=grid, y=model.predict(frame), mode="lines",
                                 showlegend=False), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=[float(base.loc[0, col])], y=[best_pred],
                                 mode="markers", marker=dict(size=11, symbol="star"),
                                 showlegend=False), row=r + 1, col=c + 1)
        fig.update_xaxes(title_text=col, row=r + 1, col=c + 1)
    fig.update_yaxes(title_text=f"{schema.target} prédit", col=1)
    fig.update_layout(height=320 * n_rows, margin=dict(l=10, r=10, t=40, b=10))
    return fig
