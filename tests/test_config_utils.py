import json, random
from src.config import load_config, resolve_path
from src.utils import (get_hardware_info, resolve_device, select_models, set_seed,
                       should_skip, save_json, load_json, snapshot_config)


def test_config_loads_and_attribute_access():
    cfg = load_config()
    assert cfg.whisper.model == "small"
    assert cfg.classifier.labels[2] == "action"
    assert resolve_path(cfg, "metrics").name == "metrics"


def test_overrides_deep_merge():
    cfg = load_config(overrides={"whisper": {"model": "tiny"}})
    assert cfg.whisper.model == "tiny" and cfg.whisper.language == "en"


def test_set_seed_is_deterministic():
    set_seed(7); a = [random.random() for _ in range(3)]
    set_seed(7); b = [random.random() for _ in range(3)]
    assert a == b


def test_device_cpu_fallback_and_downgrade():
    assert resolve_device("cuda") in ("cuda", "cpu")
    cfg = load_config(overrides={"device": "cpu", "summarization": {"model": "facebook/bart-large-cnn"}})
    hw = {"cuda_available": False, "vram_total_gb": None}
    sel = select_models(cfg, hw)
    assert sel["device"] == "cpu" and sel["whisper"] == "base"
    assert "large" not in sel["summarization"] and sel["downgraded"] is True


def test_no_downgrade_when_disabled():
    cfg = load_config(overrides={"device": "cpu", "resource_policy": {"auto_downgrade": False}})
    assert select_models(cfg, {"cuda_available": False, "vram_total_gb": None})["whisper"] == "small"


def test_resume_helpers(tmp_path):
    p = tmp_path / "x" / "t.json"
    assert not should_skip(p)
    save_json({"a": 1}, p)
    assert should_skip(p) and not should_skip(p, force=True) and load_json(p) == {"a": 1}


def test_snapshot_config(tmp_path):
    out = snapshot_config(load_config(), tmp_path)
    assert "config" in json.loads(out.read_text()) and "hardware" in json.loads(out.read_text())


def test_hardware_info_keys():
    assert {"python", "cuda_available", "gpu_name", "vram_total_gb"} <= set(get_hardware_info())
