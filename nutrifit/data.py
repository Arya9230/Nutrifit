"""Load and clean the food composition dataset.

The source table is a scan of a printed nutrition handbook, so values contain
artefacts: thousands separators ("1,373"), trace markers ("t", "t'"), ranges
("8-44"), footnote letters and sentinel values ("-1"). Everything is handled
explicitly here and a data-quality report is returned so nothing is silently
zeroed out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import NAME_FIXES

DATA_PATH = Path(__file__).resolve().parent / "data" / "food_data.csv"

NUMERIC_COLS = ["Grams", "Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"]
REQUIRED_COLS = ["Grams", "Calories", "Protein", "Fat", "Carbs"]
TRACE_TOKENS = {"t", "t'", "tr", "trace"}

# Dataset category -> role a food can play when composing a meal.
ROLE_BY_CATEGORY_PREFIX = {
    "Meat": "protein",
    "Fish": "protein",
    "Dairy": "dairy",
    "Breads": "carb",
    "Vegetables": "veg",
    "Fruits": "fruit",
    "Seeds": "nut",
}

# Rows that are technically in an eligible category but are not sensible meal
# components (desserts filed under dairy, cured fat, raw bulk quantities...).
EXCLUDE_KEYWORDS = (
    "ice cream", "cocoa", "custard", "cornstarch", "pudding", "shake",
    "salt pork", "lard", "fortified", "powdered", "evaporated", "condensed",
    "loaf", "dry", "flour", "cake", "pie", "doughnut", "cookie", "candy",
    "syrup", "sugar", "frosting", "gravy", "sauce", "pickle", "olive",
    "juice", "nectar", "lemonade", "limeade", "concentrate",
)


@dataclass
class DataQualityReport:
    rows_raw: int
    rows_clean: int
    rows_dropped_missing: int
    rows_dropped_inconsistent: int
    values_repaired: int
    rows_in_meal_pool: int

    def as_dict(self) -> dict:
        return asdict(self)


def parse_number(raw) -> float:
    """Parse one messy cell into a float, or NaN when it is unusable."""
    if raw is None:
        return np.nan
    s = str(raw).strip().lower().replace(",", "").replace(" ", "")
    if not s:
        return np.nan
    if s in TRACE_TOKENS:
        return 0.0
    s = s.rstrip("'")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", s)
    if m:  # a range such as "8-44" -> midpoint
        return (float(m.group(1)) + float(m.group(2))) / 2
    try:
        value = float(s)
    except ValueError:
        return np.nan
    return value if value >= 0 else np.nan  # -1 is a "not available" sentinel


def _is_repaired(raw) -> bool:
    """True if the cell needed more than a plain float() to be understood."""
    s = str(raw).strip().lower()
    return s in TRACE_TOKENS or "," in s or "'" in s or bool(re.fullmatch(r"\d+-\d+", s))


def _clean_name(name: str) -> str:
    name = re.sub(r",?\s*or\s*$", "", str(name).strip(), flags=re.I)
    name = name.strip(" ,()")
    return name[:1].upper() + name[1:]


def load_food_data(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, DataQualityReport]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows_raw = len(raw)

    repaired = 0
    df = raw.copy()
    for col in NUMERIC_COLS:
        repaired += int(raw[col].map(_is_repaired).sum())
        df[col] = raw[col].map(parse_number)

    # Missing fibre / saturated fat is treated as 0; missing core fields drop the row.
    df[["Sat.Fat", "Fiber"]] = df[["Sat.Fat", "Fiber"]].fillna(0.0)
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    df = df[(df["Grams"] > 0) & (df["Calories"] > 0)]
    dropped_missing = before - len(df)

    # Atwater check: 4*P + 4*C + 9*F should roughly equal the stated calories.
    est = 4 * df["Protein"] + 4 * df["Carbs"] + 9 * df["Fat"]
    rel_err = (est - df["Calories"]).abs() / df["Calories"]
    consistent = (rel_err <= 0.6) | (df["Calories"] < 40)
    dropped_inconsistent = int((~consistent).sum())
    df = df[consistent].copy()

    df["Food"] = df["Food"].map(_clean_name)
    df["Category"] = df["Category"].str.strip()
    df["Measure"] = df["Measure"].str.strip()
    df = df.drop_duplicates(subset=["Food", "Measure", "Category", "Calories"]).reset_index(drop=True)

    # Per-100 g nutrient density (the basis for clustering and scoring).
    for col in ["Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"]:
        df[f"{col}_100g"] = df[col] / df["Grams"] * 100

    df["Role"] = df["Category"].map(_role_for_category)
    name = df["Food"].str.lower()
    excluded = name.apply(lambda n: any(k in n for k in EXCLUDE_KEYWORDS))

    # Apply curated display names; a value of None removes the row from the pool.
    display, curated_drop = [], []
    for food, grams, measure in zip(df["Food"], df["Grams"].round().astype(int), df["Measure"]):
        fix_key = (food, grams, measure) if (food, grams, measure) in NAME_FIXES else (food, grams)
        if fix_key in NAME_FIXES:
            fixed = NAME_FIXES[fix_key]
            display.append(fixed if fixed else food)
            curated_drop.append(fixed is None)
        else:
            display.append(food)
            curated_drop.append(False)
    df["Display"] = display

    sensible = df["Grams"].between(15, 450) & df["Calories"].between(15, 700)
    df["InMealPool"] = df["Role"].notna() & ~excluded & sensible & ~pd.Series(curated_drop, index=df.index)
    bare = df["Measure"].str.fullmatch(r"\d+")
    unit = np.where(df["Measure"] == "1", " piece", " pieces")
    measure = df["Measure"].where(~bare, df["Measure"] + unit)
    df["Serving"] = measure + " (" + df["Grams"].round().astype(int).astype(str) + " g)"

    report = DataQualityReport(
        rows_raw=rows_raw,
        rows_clean=len(df),
        rows_dropped_missing=dropped_missing,
        rows_dropped_inconsistent=dropped_inconsistent,
        values_repaired=repaired,
        rows_in_meal_pool=int(df["InMealPool"].sum()),
    )
    return df, report


def _role_for_category(category: str):
    for prefix, role in ROLE_BY_CATEGORY_PREFIX.items():
        if category.startswith(prefix):
            return role
    return None
