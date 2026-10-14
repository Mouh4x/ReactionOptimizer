"""Modèles candidats comparés automatiquement (100 % scikit-learn, gratuit).

Justification de la présence de chaque modèle (petits jeux de données) :
- Référence (moyenne) : modèle de base ; si rien ne le bat, le logiciel le dit.
- Ridge linéaire / quadratique : hypothèses simples, régularisées, robustes
  quand n est faible ; le degré 2 couvre les optima « en cloche » fréquents
  en chimie (plans de type surface de réponse).
- SVR (RBF) : non-linéaire souple, efficace en petite dimension ; C et epsilon
  sont choisis par validation croisée interne (pas de réglage manuel).
- Processus gaussien : probabiliste, bien adapté aux très petits échantillons.
- Random Forest / Gradient Boosting : non-linéaires à base d'arbres, mais
  paramétrés ici de façon volontairement conservative (voir commentaire
  « Gradient Boosting ») pour ne pas mémoriser le bruit quand n est petit.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from .preprocessing import build_preprocessor
from .schema import Schema

BASELINE = "Référence (moyenne)"


def get_candidate_models(schema: Schema) -> dict[str, Pipeline]:
    def make(regressor, poly: bool = False) -> Pipeline:
        steps = [("prep", build_preprocessor(schema))]
        if poly:
            steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        steps.append(("reg", regressor))
        return Pipeline(steps)

    alphas = np.logspace(-3, 3, 13)
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 1e2), nu=2.5)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e0))
    )
    # SVR : C et epsilon choisis automatiquement par validation croisée interne.
    svr = GridSearchCV(
        SVR(kernel="rbf", gamma="scale"),
        param_grid={"C": [0.3, 1, 3, 10, 30, 100], "epsilon": [0.01, 0.05, 0.1]},
        cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    return {
        BASELINE: make(DummyRegressor(strategy="mean")),
        "Linéaire (Ridge)": make(RidgeCV(alphas=alphas)),
        "Quadratique (Ridge)": make(RidgeCV(alphas=alphas), poly=True),
        "SVR (noyau RBF)": make(svr),
        "Processus gaussien": make(
            GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=5,
                random_state=0,
            )
        ),
        # Arbres : conservateurs par choix. Sur ~25 lignes, des forêts profondes
        # ou un boosting agressif mémorisent le bruit et affichent des métriques
        # flatteuses mais non reproductibles.
        "Random Forest": make(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=0)),
        "Gradient Boosting": make(
            HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=7,
                min_samples_leaf=5, l2_regularization=1.0, random_state=0,
            )
        ),
    }
