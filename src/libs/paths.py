"""Project paths, independent of the directory you launch from.

PROJECT_ROOT is the folder that contains src/, configs/, data/, trained_models/ ...
It is inferred from where this file lives (src/libs/paths.py -> two levels up)
and can be overridden with the PROJECT_ROOT environment variable.
Relative paths in the config are always relative to PROJECT_ROOT.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def project_path(p) -> Path:
    """Absolute path: kept as-is if already absolute, otherwise under PROJECT_ROOT."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def config_path(p) -> Path:
    """Config file: relative to the current directory if it exists there, otherwise to PROJECT_ROOT."""
    p = Path(p)
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p
