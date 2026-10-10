import math

import pytest

from nutrifit.data import load_food_data, parse_number


@pytest.mark.parametrize("raw,expected", [
    ("1,373", 1373.0), ("  42 ", 42.0), ("t", 0.0), ("t'", 0.0), ("8-44", 26.0), ("3.5", 3.5),
])
def test_parse_number_ok(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("raw", ["", "-1", "a", None, "abc"])
def test_parse_number_unusable(raw):
    assert math.isnan(parse_number(raw))


def test_thousands_separator_regression():
    df, _ = load_food_data()
    fortified = df[df["Grams"] > 1000]
    assert (fortified["Calories"] > 1000).all()  # used to become 0


def test_cleaned_data_is_sane():
    df, report = load_food_data()
    assert report.rows_clean == len(df) < report.rows_raw
    for col in ["Calories", "Protein", "Fat", "Carbs", "Grams"]:
        assert df[col].notna().all() and (df[col] >= 0).all()
    pool = df[df["InMealPool"]]
    assert len(pool) >= 100
    assert not pool["Display"].str.lower().isin(["lima", "converted", "frozen"]).any()
