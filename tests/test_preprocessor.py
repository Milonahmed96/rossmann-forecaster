"""
tests/test_preprocessor.py
--------------------------
Unit tests for data/preprocessor.py
"""

import pytest
import pandas as pd
import numpy as np
from data.preprocessor import clean_data


def make_raw_df():
    """Create a minimal raw DataFrame that mimics the real merged dataset."""
    return pd.DataFrame({
        "Store":                    [1, 1, 1, 2, 2],
        "DayOfWeek":                [5, 5, 5, 5, 5],
        "Date":                     ["2015-07-31", "2015-07-30", "2015-07-29",
                                     "2015-07-31", "2015-07-30"],
        "Sales":                    [5263, 0, 4987, 6064, 5800],
        "Customers":                [555, 0, 476, 625, 610],
        "Open":                     [1, 0, 1, 1, 1],
        "Promo":                    [1, 1, 0, 1, 0],
        "StateHoliday":             ["0", "0", "0", "0", "0"],
        "SchoolHoliday":            [1, 1, 0, 0, 0],
        "StoreType":                ["c", "c", "c", "a", "a"],
        "Assortment":               ["a", "a", "a", "a", "a"],
        "CompetitionDistance":      [1270.0, 1270.0, 1270.0, np.nan, 570.0],
        "CompetitionOpenSinceMonth":[9.0, 9.0, 9.0, np.nan, 11.0],
        "CompetitionOpenSinceYear": [2008.0, 2008.0, 2008.0, np.nan, 2007.0],
        "Promo2":                   [0, 0, 0, 1, 1],
        "Promo2SinceWeek":          [np.nan, np.nan, np.nan, 13.0, 13.0],
        "Promo2SinceYear":          [np.nan, np.nan, np.nan, 2010.0, 2010.0],
        "PromoInterval":            [np.nan, np.nan, np.nan, "Jan,Apr,Jul,Oct",
                                     "Jan,Apr,Jul,Oct"],
    })


def test_clean_data_removes_closed_rows():
    """Rows where Open=0 or Sales=0 should be removed."""
    df = make_raw_df()
    clean = clean_data(df)
    # Only 1 row removed — the one with Open=0 (which also has Sales=0)
    assert len(clean) == 4
    assert (clean["Sales"] > 0).all()


def test_clean_data_drops_open_column():
    """Open column should be dropped after filtering."""
    df = make_raw_df()
    clean = clean_data(df)
    assert "Open" not in clean.columns


def test_clean_data_converts_date_to_datetime():
    """Date column should be datetime dtype after cleaning."""
    df = make_raw_df()
    clean = clean_data(df)
    assert pd.api.types.is_datetime64_any_dtype(clean["Date"])


def test_clean_data_fills_competition_distance():
    """CompetitionDistance NaNs should be filled with median."""
    df = make_raw_df()
    clean = clean_data(df)
    assert clean["CompetitionDistance"].isnull().sum() == 0


def test_clean_data_fills_promo_fields():
    """Promo2 related NaN fields should be filled with 0."""
    df = make_raw_df()
    clean = clean_data(df)
    assert clean["Promo2SinceWeek"].isnull().sum() == 0
    assert clean["Promo2SinceYear"].isnull().sum() == 0
    assert clean["CompetitionOpenSinceMonth"].isnull().sum() == 0
    assert clean["CompetitionOpenSinceYear"].isnull().sum() == 0


def test_clean_data_fills_promo_interval():
    """PromoInterval NaNs should be filled with string '0'."""
    df = make_raw_df()
    clean = clean_data(df)
    assert clean["PromoInterval"].isnull().sum() == 0


def test_clean_data_no_nulls_remaining():
    """No null values should remain after cleaning."""
    df = make_raw_df()
    clean = clean_data(df)
    assert clean.isnull().sum().sum() == 0


def test_clean_data_does_not_mutate_input():
    """Original DataFrame should not be modified."""
    df = make_raw_df()
    original_len = len(df)
    clean_data(df)
    assert len(df) == original_len