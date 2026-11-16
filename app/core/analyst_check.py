from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYST_DIR = REPO_ROOT / "bench/analyst"


# an_common from bench/analyst, reusing the loaded copy when it's the same file.
def _load_common() -> ModuleType:
    path = ANALYST_DIR / "an_common.py"
    loaded = sys.modules.get("an_common")
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() == path:
        return loaded

    spec = importlib.util.spec_from_file_location("an_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["an_common"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("an_common", None)
        raise
    return module


an = _load_common()
rp = an.load_sibling("an_run_pipeline", "run_pipeline.py")
ck = an.load_sibling("an_checks", "checks.py")

# The copies run_pipeline and checks already loaded, so each class exists once.
sf, dd = ck.sf, ck.dd
pr, bc, fz = rp.pr, rp.bc, rp.fz

_model_mlx: ModuleType | None = None
_model_mlx_lock = threading.Lock()


# model_mlx, loaded on the first call so nothing touches MLX before the model is wanted.
def load_model_mlx() -> ModuleType:
    global _model_mlx
    with _model_mlx_lock:
        if _model_mlx is None:
            _model_mlx = an.load_sibling("an_model_mlx", "model_mlx.py")
        return _model_mlx
