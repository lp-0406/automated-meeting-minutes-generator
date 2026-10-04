"""Configuration loading. Plain YAML -> nested dict with attribute access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with attribute access: cfg.whisper.model == cfg["whisper"]["model"]."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config({k: Config._wrap(v) for k, v in value.items()})
        return value

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else copy.deepcopy(v)) for k, v in self.items()}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Load the YAML config. `overrides` is a nested dict merged on top."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config._wrap(raw)


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(cfg: Config, key: str) -> Path:
    """Return an absolute path for cfg.paths[key], relative to the repo root."""
    p = Path(cfg.paths[key])
    return p if p.is_absolute() else REPO_ROOT / p
