#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mise a jour / reparation idempotente de ReactionOptimizer.

Usage :
    python update_reactionoptimizer.py           # applique les corrections
    python update_reactionoptimizer.py --check   # vérifie seulement
    python update_reactionoptimizer.py --no-tests  # saute pytest

Ce script :
  1. verifie qu'il est execute a la racine du projet (presence de app.py et src/) ;
  2. verifie les prerequis Python et les dependances ;
  3. cree une sauvegarde horodatee des fichiers avant toute modification ;
  4. nettoie les fichiers inutiles (.bak, caches, archives imbriquees) SANS
     toucher aux donnees utilisateur de data/datasets/ ;
  5. verifie que tous les fichiers livres sont presents et non vides ;
  6. verifie les imports de chaque module ;
  7. lance la suite pytest ;
  8. affiche un bilan clair.

Idempotent : relancer ce script est sans danger. Il ne reecrit pas les fichiers
sources (deja livres par l'archive), il nettoie et valide. En cas d'erreur
pendant le nettoyage, la sauvegarde permet de restaurer les fichiers deplaces.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "config/settings.yaml",
    "pages/1_Import_Excel.py",
    "pages/2_Saisie_manuelle.py",
    "pages/3_Modele_et_Optimisation.py",
    "src/reactionoptimizer/__init__.py",
    "src/reactionoptimizer/config.py",
    "src/reactionoptimizer/constraints.py",
    "src/reactionoptimizer/data_loader.py",
    "src/reactionoptimizer/evaluation.py",
    "src/reactionoptimizer/models.py",
    "src/reactionoptimizer/optimizer.py",
    "src/reactionoptimizer/preprocessing.py",
    "src/reactionoptimizer/reporting.py",
    "src/reactionoptimizer/schema.py",
    "src/reactionoptimizer/session.py",
    "tests/conftest.py",
    "tests/test_data_loader.py",
    "tests/test_models_evaluation.py",
    "tests/test_optimizer_constraints.py",
    "tests/test_app.py",
]

MODULES = [
    "reactionoptimizer.config",
    "reactionoptimizer.constraints",
    "reactionoptimizer.data_loader",
    "reactionoptimizer.evaluation",
    "reactionoptimizer.models",
    "reactionoptimizer.optimizer",
    "reactionoptimizer.preprocessing",
    "reactionoptimizer.reporting",
    "reactionoptimizer.schema",
    "reactionoptimizer.session",
]

REQUIREMENTS = ["streamlit", "pandas", "numpy", "sklearn", "scipy", "openpyxl",
                "plotly", "yaml"]

# Motifs de fichiers consideres comme inutiles. Les suppressions sont deplacees
# dans la sauvegarde horodatee, jamais effacees sans retour arriere possible.
CRUFT_GLOBS = ["**/*.bak", "**/*.pyc", "**/__pycache__",
               "src/reactionoptimizer.rar", "mise_a_jour_reactionoptimizer.py"]

# JAMAIS supprimes ni deplaces (donnees utilisateur).
PROTECTED = {"data/datasets", "data/raw", "data/examples", ".venv", "venv",
             "backup_"}


class Report:
    def __init__(self) -> None:
        self.ok: list[str] = []
        self.warn: list[str] = []
        self.fail: list[str] = []

    def log_ok(self, msg: str) -> None:
        print(f"  [OK]   {msg}")
        self.ok.append(msg)

    def log_warn(self, msg: str) -> None:
        print(f"  [WARN] {msg}")
        self.warn.append(msg)

    def log_fail(self, msg: str) -> None:
        print(f"  [FAIL] {msg}")
        self.fail.append(msg)


def check_structure(r: Report) -> bool:
    print("\n== 1. Structure du projet ==")
    if not (ROOT / "app.py").is_file() or not (ROOT / "src" / "reactionoptimizer").is_dir():
        r.log_fail("Executez ce script a la racine du projet ReactionOptimizer "
                   "(dossier contenant app.py et src/reactionoptimizer/).")
        return False
    r.log_ok(f"Racine detectee : {ROOT}")
    return True


def check_python(r: Report) -> None:
    print("\n== 2. Prerequis Python ==")
    if sys.version_info < (3, 10):
        r.log_fail(f"Python 3.10+ requis (actuel : {sys.version.split()[0]}).")
    else:
        r.log_ok(f"Python {sys.version.split()[0]}")

    missing = []
    for name in REQUIREMENTS:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        r.log_warn("Dependances manquantes : " + ", ".join(missing)
                   + ". Installez-les avec : pip install -r requirements.txt")
    else:
        r.log_ok("Toutes les dependances sont presentes.")


def is_protected(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return any(p in rel for p in PROTECTED)


def make_backup(items: list[Path]) -> Path | None:
    if not items:
        return None
    stamp = datetime.now().strftime("backup_%Y%m%d_%H%M%S")
    backup = ROOT / stamp
    for item in items:
        if not item.exists():
            continue
        rel = item.relative_to(ROOT)
        dest = backup / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return backup


def clean_cruft(r: Report, apply: bool) -> None:
    print("\n== 3. Nettoyage des fichiers inutiles ==")
    targets: list[Path] = []
    for pattern in CRUFT_GLOBS:
        for p in ROOT.glob(pattern):
            if p.exists() and not is_protected(p):
                targets.append(p)
    if not targets:
        r.log_ok("Rien a nettoyer.")
        return
    for p in targets:
        r.log_warn(f"a supprimer : {p.relative_to(ROOT)}")
    if not apply:
        return
    backup = make_backup(targets)
    if backup:
        r.log_ok(f"Sauvegarde avant suppression : {backup.name}/")
    for p in targets:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception as exc:
            r.log_fail(f"Suppression impossible : {p.relative_to(ROOT)} ({exc})")


def check_files(r: Report) -> None:
    print("\n== 4. Fichiers livres ==")
    for rel in REQUIRED_FILES:
        p = ROOT / rel
        if not p.is_file() or p.stat().st_size == 0:
            r.log_fail(f"Manquant ou vide : {rel}")
        else:
            r.log_ok(rel)


def check_imports(r: Report) -> None:
    print("\n== 5. Imports Python ==")
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    for name in MODULES:
        try:
            importlib.import_module(name)
            r.log_ok(name)
        except Exception as exc:
            r.log_fail(f"Import {name} : {exc}")


def run_tests(r: Report, skip: bool) -> None:
    print("\n== 6. Tests automatises ==")
    if skip:
        r.log_warn("--no-tests : pytest non execute.")
        return
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q",
             "--maxfail=5", "--disable-warnings"],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        r.log_warn("pytest non installe (pip install pytest).")
        return
    except subprocess.TimeoutExpired:
        r.log_fail("pytest : timeout (> 10 min).")
        return
    print(proc.stdout[-2000:] or "<pas de sortie>")
    if proc.returncode == 0:
        r.log_ok("pytest : tous les tests passent.")
    else:
        r.log_fail(f"pytest : code de retour {proc.returncode}.")
        if proc.stderr:
            print(proc.stderr[-1000:])


def summary(r: Report) -> int:
    print("\n== Bilan ==")
    print(f"  Reussites : {len(r.ok)}")
    print(f"  Avertissements : {len(r.warn)}")
    print(f"  Echecs : {len(r.fail)}")
    for m in r.fail:
        print(f"    - {m}")
    print("\nRelancez l'application avec :  streamlit run app.py")
    return 0 if not r.fail else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Verifier sans nettoyer.")
    parser.add_argument("--no-tests", action="store_true",
                        help="Ne pas lancer pytest.")
    args = parser.parse_args()

    r = Report()
    if not check_structure(r):
        return summary(r)
    check_python(r)
    clean_cruft(r, apply=not args.check)
    check_files(r)
    check_imports(r)
    run_tests(r, skip=args.no_tests)
    return summary(r)


if __name__ == "__main__":
    sys.exit(main())
