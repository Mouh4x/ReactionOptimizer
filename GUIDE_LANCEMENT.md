# Guide de lancement — ReactionOptimizer

## De l'archive à l'application (5 minutes)

1. **Extraire** l'archive ZIP où vous voulez, puis ouvrir un terminal dans le dossier
   `ReactionOptimizer` (celui qui contient `app.py`).
2. **Créer l'environnement et installer** :

   ```bat
   :: Windows
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   ```bash
   # Linux / macOS
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Lancer** :

   ```bat
   streamlit run app.py
   ```

   Le navigateur s'ouvre sur http://localhost:8501.

4. **Essayer tout de suite** : page « Import Excel » → importez
   `data/examples/exemple_ds.csv` → « Utiliser ces données » → page
   « Modèle et optimisation » → « 🚀 Lancer l'analyse ».

## Mettre à jour / réparer une installation existante

Placez-vous à la racine du projet puis :

```bat
python update_reactionoptimizer.py
```

Le script vérifie la structure, sauvegarde (`backup_YYYYMMDD_HHMMSS/`), applique les
fichiers corrigés, nettoie les `.bak`/caches/archives superflues et lance les tests.
Relancez-le sans crainte : il est idempotent. Vos données (`data/datasets/`) ne sont
jamais supprimées.

## Lancer les tests

```bat
python -m pytest tests/ -q
```

## Démonstrations en ligne de commande (sans interface)

```bat
python scripts\run_demo.py            :: génère data/raw/demo.xlsx et teste l'import
python scripts\run_models_demo.py     :: compare les modèles
python scripts\run_optimizer_demo.py  :: recherche l'optimum
```
