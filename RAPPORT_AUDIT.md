# Rapport d'audit — ReactionOptimizer

## Méthode

Extraction complète de l'archive (RAR5, extraction via `unar` — 7z ne supporte pas
le codec), lecture de 100 % des sources, inspection binaire des jeux de données,
exécution réelle des tests et de l'application dans un environnement Python 3
(Streamlit, scikit-learn) fraîchement installé.

## Problèmes constatés → causes racines → corrections

| # | Problème | Cause racine | Correction |
|---|---|---|---|
| 1 | Colonne `source` dupliquée (`﻿source`) après « Ajouter au tableau existant » | `pd.concat` brut dans la page, sans normalisation des en-têtes ; preuve conservée dans `dataset_20261014_123743.csv` (données dédoublonnées en 2 blocs) | Nouveau `data_loader.merge_datasets` : normalisation des DEUX tableaux avant comparaison/concaténation, correspondance explicite, colonnes absentes → NaN + avertissement, types ré-harmonisés |
| 2 | BOM répétés dans les en-têtes (`﻿﻿source`) | Nettoyage limité à `str.strip()`, qui ne retire pas le BOM ; CSV sauvegardés en `utf-8-sig` puis relus avec `read_csv` par défaut → accumulation | `normalize_column_name` (BOM même répétés, espaces, NFC) + lecture `utf-8-sig` avec repli cp1252/latin-1 + nettoyage **centralisé** appliqué à toutes les voies d'import (Excel, CSV, saisie, rechargement) |
| 3 | Fichier nouveau à chaque clic dans `data/datasets/` (5 CSV ≈ identiques, dont 2 duplicatas exacts) | `save_dataset` crée toujours un fichier horodaté, même pour une mise à jour | `save_dataset(path=...)` met à jour le fichier du jeu courant ; `session.py` mémorise le fichier associé ; `prune_datasets` (≥ 21 fichiers) ne touche que le motif `dataset_AAAAMMJJ_HHMMSS.csv` |
| 4 | Gradient Boosting trop souple pour ~25 lignes | `max_leaf_nodes=15`, pas de `min_samples_leaf` → mémorisation du bruit possible | Paramétrage conservateur justifié (`max_leaf_nodes=7`, `min_samples_leaf=5`, `max_iter=200`) ; validé par la même validation croisée que les autres modèles (aucun réglage « pour la métrique ») |
| 5 | Référence (moyenne) biaisée en validation par étude | `DummyRegressor` + leave-one-study-out : prédire la moyenne globale d'études déséquilibrées handicape artificiellement la référence | Référence = **médiane des autres études** pour chaque pli, sans fuite ; si rien ne la bat, elle est retenue et annoncée |
| 6 | DS négatif en minimisation | Prédiction non contrainte ; extrapolation du modèle hors du domaine. DS : score observé dans [0,1], aucune définition signée documentée → déclaré non négatif | `constraints.py` : borne 0 déclarée et documentée, appliquée à la **présentation** (valeur brute conservée dans `DS_predit_brut`, avertissement affiché) ; modèle et métriques inchangés ; test dédié. Si votre DS est signé : supprimer l'entrée (tout est expliqué dans le fichier) |
| 7 | Contraintes expérimentales : aucune règle documentée, rien entre l'optimiseur et l'affichage | — | `constraints.py` (bornes cibles, `check_bounds`, `custom_filter` à compléter, ex. solvant/température) vérifié **avant** présentation ; « aucun candidat admissible » → erreur explicite ; README §10 explique où modifier |
| 8 | Suggestions pouvant reproposer une expérience déjà réalisée | Pas de déduplication contre le jeu de données | Exclusion des candidats quasi identiques (numériques à 1e-4, modalités exactes) + dédoublonnage Sobol |
| 9 | Fichiers superflus | `*.bak` (dont `schema.py.bak`, `models.py.bak` vides), `src/reactionoptimizer.rar` (archive imbriquée vide), `__pycache__/`, `README.md` vide, `config/settings.yaml` vide | Nettoyés ; `.gitignore` ajouté ; `update_reactionoptimizer.py` remplace l'ancien `mise_a_jour_reactionoptimizer.py` |
| 10 | Aucun test automatisé | — | 27 tests pytest (données, modèles, optimisation, application), 100 % synthétiques et reproductibles |
| 11 | Incertitude non présentée comme telle | — | Bootstrap 30 réentraînements, affiché avec la réserve « sous-estime l'erreur réelle » ; optimum au bord et extrapolation signalés |

## Points vérifiés et sains

- `find_group_column` : présent et fonctionnel dans `schema.py` (l'erreur d'import
  signalée provenait d'une version antérieure ; l'actuel `mise_a_jour_reactionoptimizer.py`
  l'avait réintroduit) ;
- pas de fuite de données : tout le prétraitement est dans les `Pipeline` scikit-learn ;
- imports et chemins relatifs corrects depuis la racine ;
- `schema.py` détecte et ignore les colonnes constantes côté modèle via la validation
  (avertissements ajoutés pour colonnes constantes / quasi uniques).

## Validation exécutée (réelle)

- `python -m pytest tests/ -q` : **27 tests, 0 échec** ;
- démarrage Streamlit sans erreur (`streamlit run app.py`, vérification HTTP 200) ;
- chaîne complète sur l'exemple fourni : import → comparaison 7 modèles (validation
  leave-one-study-out, 5 études) → optimisation max/min → export Excel ;
- fusion BOM → aucune colonne `source` dupliquée (test dédié + scénario reconstitué
  du fichier `dataset_20261014_123743.csv`) ;
- réexécution de `update_reactionoptimizer.py` : idempotente (fichiers « inchangés »,
  tests relancés, aucune donnée supprimée).

Non exécuté : tests d'intégration navigateur (clics réels dans l'UI) — environnement
sans navigateur ; la logique couverte par ces clics est testée unitairement.
