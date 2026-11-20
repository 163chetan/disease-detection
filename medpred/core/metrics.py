"""Reads models/metrics.json (written by training/evaluate_models.py)."""
import json
import os

PATH = os.path.join("models", "metrics.json")


def get(key: str):
    """Returns the stored metrics dict for a model key, or None if it has not been evaluated."""
    try:
        with open(PATH, encoding="utf-8") as f:
            return json.load(f).get(key)
    except (OSError, ValueError):
        return None
