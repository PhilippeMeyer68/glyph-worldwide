from pathlib import Path
import os

def project_root() -> Path:
    """
    Resolve project root reliably:
    1) GLYPH_ROOT env var (cluster / override)
    2) repo layout: src/glyphworld/paths.py → parents[2] = repo root
    """
    env = os.environ.get("GLYPH_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    return Path(__file__).resolve().parents[3]


ROOT = project_root()

DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ALPHABETS_DIR = PROCESSED_DIR / "alphabets"
DISTANCES_DIR = PROCESSED_DIR / "distances"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR    = ROOT / "results"
