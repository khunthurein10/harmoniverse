"""
Load config/settings.yaml and resolve paths.

Every script and experiment calls load_settings() instead of hard-coding paths,
so paths and grid settings live in one file.
"""
from pathlib import Path

import yaml


def find_root(start=None):
    """Walk up from `start` until a folder containing config/settings.yaml is found."""
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "config" / "settings.yaml").exists():
            return folder
    raise FileNotFoundError("Could not find config/settings.yaml. Run from inside the project folder.")


def load_settings(root=None):
    """Return the settings dict, with every entry under `paths` turned into an absolute Path."""
    root = Path(root) if root else find_root()
    with open(root / "config" / "settings.yaml", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    settings["root"] = root
    settings["paths"] = {k: (root / v) for k, v in settings.get("paths", {}).items()}
    return settings
