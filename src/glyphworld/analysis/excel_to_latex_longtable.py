import math
import warnings
from pathlib import Path

import pandas as pd

from glyphworld.core.paths import ALPHABETS_DIR, DATA_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "tables"

DEFAULT_SCRIPTS_LOCATION = (
    DATA_DIR / "info" / "scripts_location_with_latlon_and_timerange.xlsx"
)

ALPHABETS_PKL_NAME = "alphabets_unicode.pkl"
DEFAULT_ALPHABETS_PKL = ALPHABETS_DIR / ALPHABETS_PKL_NAME

DEFAULT_OUTPUT_TEX = DEFAULT_OUTPUT_DIR / "dataset_details_table.tex"


# =====================================================================
# Table metadata
# =====================================================================

CAPTION = (
    "Geographical, temporal, and size metadata for all scripts included in "
    "the dataset. For each script, we report the number of glyph classes "
    "retained (\\#Glyphs), representative locations, and approximate attested "
    "time ranges (in centuries; negative values denote BCE). An asterisk (*) "
    "indicates scripts for which the glyph inventory was limited to 100 medoids."
)

LABEL = "tab:dataset_details"

MEDOID_LIMITED_SCRIPTS = {
    "Anatolian Hieroglyphs",
    "CJK",
    "Cuneiform",
    "Egyptian Hieroglyphs",
    "Linear A",
    "Linear B",
    "Tangut",
}


# =====================================================================
# Utilities
# =====================================================================

def latex_escape(text: object) -> str:
    """Escape LaTeX-sensitive characters."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    text = str(text)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_value(value: object) -> str:
    """Return a clean string for Excel values."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def normalize_script_name(name: object) -> str:
    """
    Normalize script names for robust matching between Excel and alphabet dict.
    """
    name = clean_value(name)
    name = name.replace("_", " ")
    name = name.replace("`", "'")
    name = " ".join(name.split())
    return name.lower()


def century_value(value: object) -> str:
    """Format century values as integers when possible."""
    if value is None or pd.isna(value):
        return "?"

    try:
        value = float(value)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except Exception:
        return str(value).strip()


def split_excel_list(value: object) -> list[str]:
    """
    Split semicolon-separated Excel cells.

    Example:
    "Rome; Athens; Cairo" -> ["Rome", "Athens", "Cairo"]
    """
    value = clean_value(value)

    if not value:
        return []

    return [item.strip() for item in value.split(";") if item.strip()]


def build_alphabet_lookup(alphabets_dic: dict) -> dict[str, object]:
    """Build a normalized lookup table for alphabet dictionary keys."""
    return {
        normalize_script_name(script_name): glyphs
        for script_name, glyphs in alphabets_dic.items()
    }


def get_glyph_count(script: str, alphabets_lookup: dict[str, object]) -> str:
    """
    Return the number of glyphs for a script from the alphabet dictionary.

    If the exact script name is not found, also tries:
    - "<script> Full"

    If still not found, returns '?'.
    """
    candidate_names = [
        script,
        f"{script} Full",
    ]

    for candidate in candidate_names:
        key = normalize_script_name(candidate)

        if key in alphabets_lookup:
            glyphs = alphabets_lookup[key]

            try:
                return str(len(glyphs))
            except TypeError:
                return "?"

    return "?"


def format_locations(row: pd.Series) -> str:
    """
    Format locations over several lines inside one LaTeX table cell.

    Requires:
    \\usepackage{makecell}

    Output example:
    \\makecell[l]{Rome (Italy) \\\\ Athens (Greece)}
    """
    cities = split_excel_list(row.get("Cities", ""))
    countries = split_excel_list(row.get("Countries", ""))

    locations = []

    max_len = max(len(cities), len(countries))

    for i in range(max_len):
        city = cities[i] if i < len(cities) else ""
        country = countries[i] if i < len(countries) else ""

        if city and country:
            locations.append(latex_escape(f"{city} ({country})"))
        elif city:
            locations.append(latex_escape(city))
        elif country:
            locations.append(latex_escape(country))

    if not locations:
        return ""

    return r"\makecell[l]{" + r" \\ ".join(locations) + "}"


def format_glyph_count(row: pd.Series, alphabets_lookup: dict[str, object]) -> str:
    """
    Format glyph count using alphabets_unicode.pkl.

    The asterisk is added for scripts whose glyph inventory was limited
    to 100 medoids downstream.
    """
    script = clean_value(row.get("Script", ""))

    glyph_count = get_glyph_count(script, alphabets_lookup)

    if script in MEDOID_LIMITED_SCRIPTS:
        glyph_count = f"{glyph_count}*"

    return glyph_count


def format_time_range(row: pd.Series) -> str:
    """Format begin/end centuries as a tuple."""
    begin = century_value(row.get("begin", None))
    end = century_value(row.get("end", None))

    return f"({begin}, {end})"


# =====================================================================
# Main table generation
# =====================================================================

def dataframe_to_latex_longtable(
    df: pd.DataFrame,
    alphabets_lookup: dict[str, object],
) -> str:
    """Convert the script metadata dataframe to a NeurIPS-style LaTeX longtable."""

    lines = []

    lines.append(r"\begin{longtable}{lclc}")
    lines.append(rf"\caption{{{CAPTION}}}\label{{{LABEL}}} \\")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Script name} & \textbf{\#Glyphs} & "
        r"\textbf{Locations} & \textbf{Time range} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Script name} & \textbf{\#Glyphs} & "
        r"\textbf{Locations} & \textbf{Time range} \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{r}{\emph{Continued on next page}} \\")
    lines.append(r"\endfoot")

    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    missing_glyph_counts = []

    for _, row in df.iterrows():
        script_raw = clean_value(row.get("Script", ""))

        script = latex_escape(script_raw)
        glyphs = latex_escape(format_glyph_count(row, alphabets_lookup))
        locations = format_locations(row)
        time_range = latex_escape(format_time_range(row))

        if glyphs.startswith("?"):
            missing_glyph_counts.append(script_raw)

        lines.append(
            f"{script} & {glyphs} & {locations} & {time_range} \\\\"
        )

    lines.append(r"\end{longtable}")

    if missing_glyph_counts:
        print("\nWarning: missing glyph counts for:")
        for script in missing_glyph_counts:
            print(f"  - {script}")

    return "\n".join(lines)


def main() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DEFAULT_SCRIPTS_LOCATION)

    alphabets_dic = load_glyph_dictionary(DEFAULT_ALPHABETS_PKL)
    alphabets_lookup = build_alphabet_lookup(alphabets_dic)

    # Keep a stable alphabetical order.
    if "Script" in df.columns:
        df = df.sort_values("Script").reset_index(drop=True)

    latex_table = dataframe_to_latex_longtable(
        df=df,
        alphabets_lookup=alphabets_lookup,
    )

    DEFAULT_OUTPUT_TEX.write_text(latex_table, encoding="utf-8")

    print(f"\nSaved LaTeX table to: {DEFAULT_OUTPUT_TEX}")


if __name__ == "__main__":
    main()