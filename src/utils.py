"""Shared utilities: seeding, device/VRAM detection, model-size policy, logging, caching, config snapshots."""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import Config, REPO_ROOT, resolve_path

# --------------------------------------------------------------------------- seeding
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (each only if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- hardware
def get_hardware_info() -> dict:
    """Describe Python, PyTorch, GPU name and VRAM. Never raises if torch is missing."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda_available": False,
        "gpu_name": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            info["vram_total_gb"] = round(total / 1024**3, 2)
            info["vram_free_gb"] = round(free / 1024**3, 2)
    except ImportError:
        pass
    return info


def resolve_device(requested: str = "auto") -> str:
    """'auto' -> 'cuda' if available else 'cpu'. Explicit 'cuda' falls back to CPU with a warning."""
    cuda = get_hardware_info()["cuda_available"]
    if requested == "auto":
        return "cuda" if cuda else "cpu"
    if requested == "cuda" and not cuda:
        logging.getLogger(__name__).warning("device=cuda requested but CUDA is unavailable; using CPU")
        return "cpu"
    return requested


def select_models(cfg: Config, hardware: dict | None = None) -> dict:
    """Pick model names from config, downgrading on CPU / low VRAM when policy allows.

    Never upgrades beyond what the config asks for and never selects large models on its own.
    """
    hw = hardware or get_hardware_info()
    policy = cfg.resource_policy
    chosen = {
        "device": resolve_device(cfg.device),
        "whisper": cfg.whisper.model,
        "summarization": cfg.summarization.model,
        "classifier": cfg.classifier.model,
        "downgraded": False,
    }
    low_vram = hw["vram_total_gb"] is not None and hw["vram_total_gb"] < policy.low_vram_gb
    if policy.auto_downgrade and (chosen["device"] == "cpu" or low_vram):
        order = ["tiny", "base", "small", "medium", "large"]
        fb = policy.fallback.whisper
        if chosen["whisper"] in order and order.index(chosen["whisper"]) > order.index(fb):
            chosen["whisper"] = fb
            chosen["downgraded"] = True
        if "large" in chosen["summarization"]:
            chosen["summarization"] = policy.fallback.summarization
            chosen["downgraded"] = True
    return chosen


def free_gpu_memory() -> None:
    """Release cached GPU memory between stages (no-op without CUDA)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- logging
def setup_logging(cfg: Config | None = None, name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Console + file logging. File goes to outputs/logs/<name>_<timestamp>.log."""
    logger = logging.getLogger("mmg")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    log_dir = resolve_path(cfg, "logs") if cfg else REPO_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    """Context manager that logs elapsed seconds; yields a dict that will hold {'seconds': ...}."""
    result: dict = {}
    t0 = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = round(time.perf_counter() - t0, 3)
        (logger or logging.getLogger("mmg")).info("%s took %.2fs", label, result["seconds"])


# --------------------------------------------------------------------------- resumability
def should_skip(output_path: str | Path, force: bool = False) -> bool:
    """True if a non-empty output already exists (stage can be skipped)."""
    p = Path(output_path)
    return (not force) and p.exists() and p.stat().st_size > 0


def save_json(obj, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")           # write-then-rename: no half-written files on disconnect
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(p)
    return p


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def snapshot_config(cfg: Config, out_dir: str | Path, tag: str = "run") -> Path:
    """Save the config + hardware used for an experiment next to its outputs (reproducibility)."""
    payload = {"timestamp": datetime.now().isoformat(timespec="seconds"),
               "config": cfg.to_dict(), "hardware": get_hardware_info()}
    return save_json(payload, Path(out_dir) / f"config_snapshot_{tag}.json")
