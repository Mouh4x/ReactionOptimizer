# ⚗️ ReactionOptimizer

Application **Streamlit** d'aide à l'optimisation des conditions expérimentales d'une
réaction chimique lorsque les réactifs sont limités et que peu d'expériences sont
possibles. Elle exploite vos résultats existants pour comparer plusieurs modèles
prédictifs, estimer les réponses sur le domaine étudié, proposer des conditions
optimales et suggérer les prochaines expériences.

> **Principe directeur** : les résultats sont des *prédictions statistiques* destinées
> à **orienter** de futures expériences. Ils ne remplacent jamais la validation au
> laboratoire. Un optimum prédit est une hypothèse à tester, pas un résultat établi.

---

## 1. Problématique et besoin

Optimiser une réaction (température, temps, quantité, solvant…) exige de nombreuses
expériences. Quand les réactifs sont rares ou chers, chaque essai compte : il faut un
outil qui apprenne du peu de données disponibles, **dise honnêtement ce qu'il sait et
ne sait pas**, et propose les essais les plus instructifs.

## 2. Objectifs et périmètre

**Dans le périmètre** : importation Excel/CSV, saisie manuelle, fusion de tableaux,
analyse des données, comparaison de modèles, recherche de conditions optimales,
contraintes déclarées, avertissements, export Excel.

**Hors périmètre (volontairement)** : prétraitement avancé (descripteurs moléculaires…),
optimisation bayésienne sophistiquée, base de données. Le prétraitement reste celui
décrit en §8.

## 3. Fonctionnalités

- 📥 Import **Excel (.xlsx/.xls)** et **CSV** (BOM, encodages courants, virgules décimales) ;
- ✍️ Saisie manuelle par tableau éditable ;
- ➕ **Ajout à un tableau existant** avec correspondance normalisée des colonnes
  (une même colonne logique n'est jamais dupliquée, même avec des en-têtes abîmés) ;
- 🔍 Détection automatique des variables numériques / catégorielles / de traçabilité ;
- 🤖 Comparaison de **7 modèles** avec validation **par étude** (leave-one-study-out) ;
- 🎯 Recherche d'optimum (maximisation ou minimisation) avec contraintes ;
- 📈 Incertitude par bootstrap, signalement de l'extrapolation et des optimums au bord ;
- 📤 Export Excel des résultats ;
- 💾 Sauvegardes horodatées dans `data/datasets/`, mises à jour **en place** lors d'un
  ajout (pas de fichier nouveau à chaque clic).

## 4. Architecture

```
app.py / pages/            Interface Streamlit (aucune logique métier)
src/reactionoptimizer/
  config.py                Lecture de config/settings.yaml
  data_loader.py           Import, nettoyage centralisé, fusion, sauvegarde
  schema.py                Rôle des colonnes ; find_group_column (validation par étude)
  preprocessing.py         Pipeline scikit-learn (imputation, normalisation, one-hot)
  models.py                Les 7 modèles candidats
  evaluation.py            Validation croisée par étude, métriques, choix du modèle
  optimizer.py             Sobol + L-BFGS-B, incertitude bootstrap, suggestions
  constraints.py           Contraintes expérimentales déclarées (voir §10)
  reporting.py             Export Excel, graphiques de réponse
  session.py               État Streamlit partagé entre pages
scripts/                   Démonstrations en ligne de commande
tests/                     Suite pytest
config/settings.yaml       Cible par défaut, dossier des sauvegardes
data/examples/             Jeu de données d'exemple (synthétique)
data/datasets/             VOS sauvegardes locales (ignorées par git)
```

## 5. Flux de traitement

Import/saisie → `clean_dataframe` (nettoyage unique) → détection du schéma →
validation (avertissements) → comparaison des modèles par validation croisée →
entraînement du meilleur → recherche d'optimum sous contraintes → suggestions →
export.

## 6. Prétraitement réellement implémenté

- Nettoyage des en-têtes : BOM (même répétés), espaces, normalisation Unicode NFC,
  dédoublonnage (`nom.1`, `nom.2`) — **centralisé** dans `data_loader.clean_dataframe` ;
- Conversion « 1,5 » → 1.5 si ≥ 80 % des valeurs de la colonne sont convertibles ;
- Numériques : imputation médiane + standardisation ; catégorielles : imputation par
  la modalité fréquente + one-hot (modalités inconnues ignorées à la prédiction) ;
- Tout est dans le `Pipeline` scikit-learn : les statistiques sont ajustées **uniquement
  sur les plis d'entraînement** — pas de fuite vers la validation.

## 7. Modèles prédictifs et justification

| Modèle | Pourquoi il est là |
|---|---|
| Référence (moyenne/médiane) | Point de comparaison obligatoire ; retenue si rien ne la bat |
| Linéaire (Ridge) | Hypothèse simple, régularisée, robuste quand n est faible |
| Quadratique (Ridge) | Optima « en cloche » fréquents (surfaces de réponse) |
| SVR (RBF) | Non-linéaire souple ; C/ε par validation croisée interne |
| Processus gaussien | Probabiliste, adapté aux très petits échantillons |
| Random Forest | Non-linéaire à base d'arbres, `min_samples_leaf=2` |
| Gradient Boosting | Paramétrage **conservateur** (`max_leaf_nodes=7`, `min_samples_leaf=5`) pour ne pas mémoriser le bruit sur petits jeux de données |

## 8. Métriques et validation

- **Validation par étude** (leave-one-study-out) dès qu'une colonne `source`/`etude`/
  `doi`… exploitable existe (≥ 3 études) : les expériences sœurs ne sont jamais
  réparties entre entraînement et validation. Sinon leave-one-out (≤ 40 lignes) ou
  KFold(5) — le cas est toujours affiché.
- La référence prédit la **médiane des autres études** (sans fuite) ; en validation
  ligne par ligne, la médiane globale.
- Métriques : **RMSE, MAE, R²**. Un **R² négatif est affiché tel quel** : le modèle
  fait moins bien que la référence. Si aucun modèle ne bat la référence, **c'est elle
  qui est retenue** et le logiciel le dit.
- Un modèle qui échoue n'interrompt pas les autres (colonne `erreur` du tableau).

## 9. Optimisation et contraintes

- Échantillonnage de **Sobol** (graine fixe → reproductible) dans les bornes,
  affinage local **L-BFGS-B** par jeu de modalités catégorielles ;
- Bornes par défaut = **plages observées dans les données** (curseur « élargir » =
  extrapolation, signalée) ; une plage observée n'est pas une limite chimique ;
- Les contraintes déclarées (`constraints.py`) sont vérifiées **avant** présentation :
  candidats violant les bornes ou `custom_filter` sont écartés ; si aucun candidat
  admissible n'existe, une erreur explicite est levée ;
- Optimum **au bord** de la plage → avertissement ; suggestions diversifiées,
  **excluant** les expériences déjà réalisées ; incertitude par bootstrap (30
  réentraînements) — elle sous-estime l'erreur réelle : fiez-vous au RMSE.

## 10. La variable DS

`DS` est le score expérimental à optimiser (cible par défaut, `settings.yaml`).
**Dans ce projet**, d'après le code et les données : score continu, toutes valeurs
observées dans **[0, 1]**, aucune définition signée documentée → DS est déclaré
**non négatif** (`constraints.TARGET_LOWER_BOUNDS = {"DS": 0.0}`).

Conséquence : un **DS prédit négatif** est un artefact d'extrapolation du modèle, pas
un résultat chimique. La valeur *affichée* est ramenée à la borne **avec
avertissement**, et la valeur brute est conservée (colonne `DS_predit_brut`) — rien
n'est masqué, le modèle et les métriques ne sont pas modifiés.

⚠️ L'unité et la définition métier exactes de DS ne sont pas documentées dans le projet
d'origine : si votre DS admet réellement des valeurs négatives, supprimez l'entrée dans
`src/reactionoptimizer/constraints.py` (tout y est expliqué, y compris comment ajouter
une borne ou une règle entre variables — ex. compatibilité solvant/température — via
`custom_filter`).

## 11. Incertitude et limites

Le meilleur candidat prédit peut être fragile : mauvais ajustement, zone peu explorée,
optimum au bord. Le logiciel signale ces situations et ne présente jamais un optimum
numérique comme une vérité expérimentale.

## 12. Installation

Prérequis : **Python 3.10+** (Windows, Linux, macOS). Dépendances : 100 % libres et
gratuites (Streamlit, pandas, scikit-learn, scipy, openpyxl, plotly, pyyaml, pytest).

```bat
:: Windows
cd ReactionOptimizer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Linux / macOS
cd ReactionOptimizer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 13. Lancement

```bat
streamlit run app.py
```

L'application s'ouvre dans le navigateur (http://localhost:8501).

## 14. Mise à jour automatique

Le script `update_reactionoptimizer.py` (à la racine) applique les corrections,
nettoie les fichiers inutiles (sauvegardes `.bak`, caches, archives imbriquées) et
lance les tests. **Idempotent** : le relancer est sûr. Une sauvegarde horodatée
(`backup_YYYYMMDD_HHMMSS/`) est créée avant toute modification.

```bat
python update_reactionoptimizer.py
```

Aucune donnée utilisateur (`data/datasets/`) n'est supprimée.

## 15. Utilisation détaillée

1. **Import Excel** : téléchargez le modèle vide, remplissez l'onglet `Donnees`
   (1 ligne = 1 expérience, en-têtes en ligne 1), importez. Choisissez *Remplacer*
   ou *Ajouter au jeu de données actuel* (fusion sûre, schémas différents signalés).
2. **Saisie manuelle** : tableau éditable, « + » pour ajouter lignes/colonnes,
   virgules décimales acceptées.
3. **Modèle et optimisation** : choisissez la cible et l'objectif (maximiser /
   minimiser), lancez l'analyse, lisez la comparaison des modèles, l'optimum, les
   suggestions, les graphiques ; exportez en Excel.

Format d'entrée attendu :

```csv
source,temperature_C,temps_h,quantite_g,solvant,DS
etude_1,80,2,1.5,eau,0.42
etude_1,95,4,1.0,DMSO,0.31
```

Un exemple synthétique prêt à l'emploi : `data/examples/exemple_ds.csv`.

## 16. Erreurs fréquentes

| Message | Cause | Solution |
|---|---|---|
| « Encodage du CSV non reconnu » | encodage exotique | Enregistrez en UTF-8 |
| « Il faut au moins 5 lignes » | cible renseignée sur < 5 lignes | Ajoutez des mesures |
| « Aucun candidat admissible » | contraintes trop strictes | Revoyez `constraints.py` |
| « Trop de combinaisons de catégories » | > 200 combinaisons | Regroupez/retirez des variables |
| R² négatif | normal : le modèle fait moins bien que la référence | Plus de données / autres variables |

## 17. Tests

```bat
python -m pytest tests/ -q
```

Couverture : import CSV (BOM simple/répété, encodages), Excel, normalisation,
fusions (colonnes identiques/réordonnées/incompatibles, **absence de `source`
dupliquée**), persistance/déduplication, entraînement, petits jeux de données,
référence honnête, validation par étude, modèle en échec, minimisation/
maximisation, bornes, contraintes, **cas DS négatif**, extrapolation, aucun
candidat admissible, parcours complet et export. Données 100 % synthétiques,
graines fixes → reproductible.

## 18. Arborescence

Voir §4. `data/datasets/`, `__pycache__/`, `.venv/`, `*.bak`, `backup_*/` sont exclus
du dépôt (`.gitignore`).

## 19. Dépendances et licences

Toutes les dépendances sont open source (licences BSD/MIT/Apache — vérifiez celles
applicables à votre usage via `pip show <paquet>`). Versions dans `requirements.txt`.

## 20. Limites scientifiques

- Modèles statistiques sur **peu de données** : puissance prédictive limitée,
  métriques instables ;
- Aucune connaissance chimique n'est codée (hors contraintes que **vous** déclarez) :
  compatibilité solvant/conditions, sécurité, faisabilité restent à votre charge ;
- Les bornes par défaut suivent les données, pas la chimie ;
- L'incertitude bootstrap **sous-estime** l'erreur réelle.

## 21. Perspectives pertinentes

Descripteurs moléculaires, optimisation bayésienne à contraintes, validation externe
sur expériences nouvelles, gestion multi-cibles. (Hors périmètre actuel.)

## 22. Crédits et contribution

Projet développé dans un contexte de recherche ; contributions bienvenues via issues
et pull requests. Licence : à définir par le propriétaire du dépôt (aucune licence
n'était fournie dans le projet d'origine — ajoutez un fichier `LICENSE` avant
publication).
