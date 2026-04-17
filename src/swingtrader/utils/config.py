"""YAML config loading.

All user-tunable parameters live under ``<repo>/config/*.yaml``. This module is the only
canonical way to read them — tests and runtime code both go through ``load_config``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIG_DIR: Path = REPO_ROOT / "config"


@dataclass(frozen=True)
class Config:
    """Lightweight wrapper around a YAML-backed config dict.

    Item access reads keys (``cfg["key"]``); nested dicts are returned as-is.
    """

    name: str
    path: Path
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def load_config(name: str, config_dir: Path | None = None) -> Config:
    """Load a YAML config by short name (e.g. ``"universe"`` → ``config/universe.yaml``)."""
    config_dir = config_dir or CONFIG_DIR
    path = config_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config {name!r} must be a YAML mapping at top level, got {type(data).__name__}"
        )
    return Config(name=name, path=path, data=data)


def load_all_configs(config_dir: Path | None = None) -> dict[str, Config]:
    """Load every ``*.yaml`` file in the config directory, keyed by stem."""
    config_dir = config_dir or CONFIG_DIR
    configs: dict[str, Config] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        configs[path.stem] = load_config(path.stem, config_dir)
    return configs
