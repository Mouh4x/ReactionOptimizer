# Lecture de config/settings.yaml avec valeurs par défaut.
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_settings() -> dict:
    settings = {"default_target": "DS", "datasets_dir": "data/datasets"}
    path = ROOT / "config" / "settings.yaml"
    if path.exists():
        settings.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return settings


def datasets_dir() -> Path:
    return ROOT / load_settings()["datasets_dir"]
