"""Load CarDD's image_info.xlsx and bin its descriptive columns (bins fixed in D-012)."""
import pandas as pd

# Bins fixed before any results were seen (D-012)
COLOUR_BINS = {"white": "light", "silver": "light", "champagne": "light",
               "black": "dark", "gray": "dark"}        # any other named colour -> "coloured"
ANGLE_BINS = {"front": "front", "side": "side", "rear": "rear"}   # "inside" and blank -> "other-unknown"


def _clean(value):
    """Lower-case, stripped text; '' for blank cells."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def colour_bin(value):
    v = _clean(value)
    if not v:
        return "unknown"
    return COLOUR_BINS.get(v, "coloured")


def angle_bin(value):
    return ANGLE_BINS.get(_clean(value), "other-unknown")


def load_image_info(path):
    """Read the spreadsheet, fix column names and add the three binned columns."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]   # 'complete or partial ' has a trailing space
    df["colour_bin"] = df["color"].map(colour_bin)
    df["angle_bin"] = df["shooting angle"].map(angle_bin)
    df["completeness"] = df["complete or partial"].map(_clean)
    return df
