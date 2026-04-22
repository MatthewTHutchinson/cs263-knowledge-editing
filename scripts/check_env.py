"""
Local environment sanity check — no GPU or model weights required.
Run this on any machine (Mac, GCP) to verify the conda env is healthy.

    conda activate cs263-project
    cd /path/to/cs263-knowledge-editing
    python scripts/check_env.py
"""

import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "external", "EasyEdit"))

REQUIRED_PACKAGES = [
    ("torch",               "2.9"),
    ("transformers",        "4.57"),
    ("datasets",            "1.18"),
    ("einops",              None),
    ("hydra",               None),
    ("omegaconf",           None),
    ("sentence_transformers", None),
    ("sklearn",             None),
    ("scipy",               None),
    ("numpy",               None),
    ("tqdm",                None),
]


def check_packages():
    print("=== Package versions ===")
    all_ok = True
    for pkg_name, min_ver in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            ok = True
            if min_ver and not ver.startswith(min_ver):
                ok = False
                all_ok = False
            status = "OK" if ok else f"WARN (expected {min_ver}.x)"
            print(f"  {pkg_name:<28} {ver:<15} {status}")
        except ImportError:
            print(f"  {pkg_name:<28} {'MISSING':<15} FAIL")
            all_ok = False
    return all_ok


def check_easyedit():
    print("\n=== EasyEdit imports ===")
    try:
        from easyeditor import ROMEHyperParams, BaseEditor, MEMITHyperParams, IKEHyperParams
        print("  ROMEHyperParams   OK")
        print("  MEMITHyperParams  OK")
        print("  IKEHyperParams    OK")
        print("  BaseEditor        OK")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def check_hparams():
    print("\n=== ROME config load ===")
    try:
        from easyeditor import ROMEHyperParams
        hp = ROMEHyperParams.from_hparams("configs/ROME/gpt2-xl")
        print(f"  alg_name:  {hp.alg_name}")
        print(f"  model:     {hp.model_name}")
        print(f"  layers:    {hp.layers}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def check_device():
    import torch
    print("\n=== Device ===")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    print(f"  MPS available:   {torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def main():
    pkg_ok    = check_packages()
    edit_ok   = check_easyedit()
    hparm_ok  = check_hparams()
    check_device()

    print("\n=== Summary ===")
    if pkg_ok and edit_ok and hparm_ok:
        print("  All checks passed. Environment is ready.")
        print("  Next: run smoke_test_rome.py on GCP T4 to test the full ROME edit pipeline.")
    else:
        print("  One or more checks failed. See above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
