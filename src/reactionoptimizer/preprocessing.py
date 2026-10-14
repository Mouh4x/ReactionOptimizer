"""Préparation des données pour les modèles."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import Schema


def build_preprocessor(schema: Schema) -> ColumnTransformer:
    """Prétraitement appliqué à l'intérieur de chaque pipeline de modèle.

    - numériques : imputation par la médiane puis standardisation (centrage /
      réduction), adaptée aux modèles sensibles à l'échelle (SVR, processus
      gaussien, ridge) comme aux arbres ;
    - catégorielles : imputation par la modalité la plus fréquente puis
      encodage one-hot (modalités inconnues ignorées à la prédiction).

    Tout est dans le Pipeline scikit-learn : les statistiques d'imputation et
    de normalisation sont ajustées uniquement sur les plis d'entraînement, ce
    qui empêche toute fuite de données vers la validation.
    """
    transformers = []
    if schema.numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            schema.numeric,
        ))
    if schema.categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            schema.categorical,
        ))
    if not transformers:
        raise ValueError("Aucune variable exploitable : vérifiez les colonnes du tableau.")
    return ColumnTransformer(transformers)


def prepare_xy(df: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, pd.Series]:
    """Garde uniquement les lignes où la cible est connue."""
    data = df.dropna(subset=[schema.target])
    X = data[schema.features].reset_index(drop=True)
    y = data[schema.target].astype(float).reset_index(drop=True)
    return X, y
