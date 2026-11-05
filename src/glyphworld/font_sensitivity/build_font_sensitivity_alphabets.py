"""
Render the font-sensitivity scripts with alternative Noto fonts (Sans, Serif, Serif Italic).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from glyphworld.core.paths import DATA_DIR, PROCESSED_DIR
from glyphworld.data.alphabet_unicode import alphabet_unicode_map
from glyphworld.data.build_alphabet_data import create_glyph


# =====================================================================
# Scripts and fonts
# =====================================================================

SCRIPTS = [
    "Latin",
    "Greek",
    "Cyrillic",
    "Devanagari",
    "Georgian Asomtavruli",
    "Georgian Mkhedruli",
    "Grantha",
    "Hebrew",
    "Hiragana",
    "Katakana",
    "Kanbun",
    "Hangul Jamo",
    "Oriya",
]

FONT_DIR = DATA_DIR / "raw" / "fonts"
ALT_SERIF_DIR = DATA_DIR / "raw" / "alternative fonts" / "serif"

# Script-specific alternative serif fonts.
# The "sans" font is automatically taken from alphabet_unicode_map.
SERIF_FONT_PATHS = {
    "Latin": ALT_SERIF_DIR / "NotoSerif.ttf",
    "Greek": ALT_SERIF_DIR / "NotoSerif.ttf",
    "Cyrillic": ALT_SERIF_DIR / "NotoSerif.ttf",

    "Devanagari": ALT_SERIF_DIR / "NotoSerifDevanagari.ttf",
    "Georgian Asomtavruli": ALT_SERIF_DIR / "NotoSerifGeorgian.ttf",
    "Georgian Mkhedruli": ALT_SERIF_DIR / "NotoSerifGeorgian.ttf",
    "Grantha": ALT_SERIF_DIR / "NotoSerifGrantha.ttf",
    "Hebrew": ALT_SERIF_DIR / "NotoSerifHebrew.ttf",
    "Hiragana": ALT_SERIF_DIR / "NotoSerifJapanese.ttf",
    "Katakana": ALT_SERIF_DIR / "NotoSerifJapanese.ttf",
    "Kanbun": ALT_SERIF_DIR / "NotoSerifJapanese.ttf",
    "Hangul Jamo": ALT_SERIF_DIR / "NotoSerifKorean.ttf",
    "Oriya": ALT_SERIF_DIR / "NotoSerifOriya.ttf",
}

# Only available / meaningful for Latin, Greek, Cyrillic.
SERIF_ITALIC_FONT_PATHS = {
    "Latin": ALT_SERIF_DIR / "NotoSerif-Italic.ttf",
    "Greek": ALT_SERIF_DIR / "NotoSerif-Italic.ttf",
    "Cyrillic": ALT_SERIF_DIR / "NotoSerif-Italic.ttf",
}


# =====================================================================
# Rendering parameters
# =====================================================================

XSIZE = 105
YSIZE = 105
FONTSIZE = 50

OUT_DIR = PROCESSED_DIR / "font_sensitivity"
OUT_PATH = OUT_DIR / "alphabets_font_variants.pkl"


# =====================================================================
# Helpers
# =====================================================================

def get_default_sans_font_path(script: str) -> Path:
    """Return the default Noto Sans font path used in the main dataset."""
    _, font_name = alphabet_unicode_map[script]
    return FONT_DIR / f"{font_name}.ttf"


def get_font_paths_for_script(script: str) -> dict[str, Path]:
    """Return all available font variants for a script."""
    font_paths = {
        "sans": get_default_sans_font_path(script),
    }

    if script in SERIF_FONT_PATHS:
        font_paths["serif"] = SERIF_FONT_PATHS[script]

    if script in SERIF_ITALIC_FONT_PATHS:
        font_paths["serif_italic"] = SERIF_ITALIC_FONT_PATHS[script]

    # Keep only fonts that exist.
    available = {}
    for name, path in font_paths.items():
        if path.exists():
            available[name] = path
        else:
            print(f"[WARN] Missing font for {script}: {name} -> {path}")

    return available


def load_alphabet(script: str, font_path: Path) -> dict[str, np.ndarray]:
    """Render all glyphs of one script using a given font."""
    unicodes, _ = alphabet_unicode_map[script]

    glyphs = {}
    for idx, code in enumerate(unicodes):
        letter = chr(code)
        key = f"glyph_{idx:05d}_unicode_{code:05X}_{letter}"

        img = create_glyph(
            code,
            XSIZE,
            YSIZE,
            FONTSIZE,
            font_path,
        )

        glyphs[key] = img.astype(np.uint8, copy=False)

    return glyphs


def plot_script_fonts(
    script_name: str,
    script_data: dict[str, dict[str, np.ndarray]],
    max_glyphs: int = 40,
) -> None:
    """
    Plot one script across available fonts.

    Rows = fonts.
    Columns = glyphs.
    """

    preferred_order = ["sans", "serif", "serif_italic"]
    font_names = [f for f in preferred_order if f in script_data]

    if len(font_names) == 0:
        print(f"[WARN] No fonts available for {script_name}")
        return

    keys = list(script_data[font_names[0]].keys())[:max_glyphs]
    n_fonts = len(font_names)
    n_glyphs = len(keys)

    fig, axes = plt.subplots(
        n_fonts,
        n_glyphs,
        figsize=(max(8, n_glyphs * 0.45), max(2.5, n_fonts * 1.2)),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
        squeeze=False,
    )

    for row, font_name in enumerate(font_names):
        glyphs = script_data[font_name]

        for col, key in enumerate(keys):
            img = glyphs[key]
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].axis("off")

            if row == 0:
                letter = key.split("_")[-1]
                axes[row, col].set_title(letter, fontsize=7)

        axes[row, 0].set_ylabel(font_name, fontsize=10)

    plt.suptitle(script_name)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "config": {
            "scripts": SCRIPTS,
            "xsize": XSIZE,
            "ysize": YSIZE,
            "fontsize": FONTSIZE,
            "font_policy": (
                "Default sans font from alphabet_unicode_map; "
                "script-specific Noto Serif alternatives when available; "
                "NotoSerif-Italic only for Latin, Greek, Cyrillic."
            ),
        },
        "alphabets": {},
        "font_paths": {},
    }

    for script in SCRIPTS:
        print(f"\n[BUILD] {script}")

        font_paths = get_font_paths_for_script(script)
        data["font_paths"][script] = {k: str(v) for k, v in font_paths.items()}
        data["alphabets"][script] = {}

        for font_name, font_path in font_paths.items():
            print(f"  font={font_name:12s} | path={font_path}")
            glyphs = load_alphabet(script, font_path)
            data["alphabets"][script][font_name] = glyphs
            print(f"  n_glyphs={len(glyphs)}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"\n[SAVED] {OUT_PATH}")

    for script in SCRIPTS:
        plot_script_fonts(script, data["alphabets"][script])