"""Phase-1 smoke test: environment, config, hardware, imports. Run: python scripts/smoke_test.py [--strict]

--strict makes missing heavy dependencies a FAIL (use it on Colab after installing requirements).
Without it, missing packages are reported as SKIP so the test also works in a bare environment.
"""
import argparse, importlib, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.utils import get_hardware_info, select_models, set_seed

PACKAGES = ["torch", "transformers", "datasets", "accelerate", "sentencepiece", "evaluate", "rouge_score",
            "bert_score", "sklearn", "pandas", "numpy", "spacy", "nltk", "jiwer", "whisper", "matplotlib", "yaml"]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--strict", action="store_true"); args = ap.parse_args()
    failures = 0; skipped = 0
    print(f"[1/5] Python {sys.version.split()[0]}")
    cfg = load_config(); set_seed(cfg.seed); print("[2/5] Config loaded, seed set: PASS")
    hw = get_hardware_info()
    print(f"[3/5] Hardware: torch={hw['torch']} cuda={hw['cuda_available']} gpu={hw['gpu_name']} "
          f"vram_total_gb={hw['vram_total_gb']}")
    sel = select_models(cfg, hw); print(f"[4/5] Model selection: {sel}")
    print("[5/5] Imports:")
    for pkg in PACKAGES:
        try:
            importlib.import_module(pkg); print(f"   PASS  {pkg}")
        except Exception as e:
            status = "FAIL" if args.strict else "SKIP (not installed)"
            failures += args.strict; skipped += not args.strict
            print(f"   {status}  {pkg}: {type(e).__name__}")
    try:
        import spacy; spacy.load("en_core_web_sm"); print("   PASS  spaCy model en_core_web_sm")
    except Exception:
        status = "FAIL" if args.strict else "SKIP (not installed)"; failures += args.strict; skipped += not args.strict
        print(f"   {status}  spaCy model en_core_web_sm (python -m spacy download en_core_web_sm)")
    verdict = "FAILED" if failures else ("PASSED (core only; %d optional checks skipped, re-run with --strict after installing requirements)" % skipped if skipped else "PASSED")
    print("SMOKE TEST:", verdict); return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
