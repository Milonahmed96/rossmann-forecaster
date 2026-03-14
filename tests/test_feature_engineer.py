"""
tests/test_feature_engineer.py
-------------------------------
Unit tests for data/feature_engineer.py
"""

import pytest
import pandas as pd
import numpy as np
from data.feature_engineer import engineer_features


def make_clean_df():
    """
    Create a minimal clean DataFrame that mimics preprocessor output.
    Two stores, 10 days each — enough to test lag features properly.
    """
    stores = []
    for store_id in [1, 2]:
        for day in range(10):
            stores.append({
                "Store":                    store_id,
                "DayOfWeek":                day % 7,
                "Date":                     pd.Timestamp("2015-01-01") + pd.Timedelta(days=day),
                "Sales":                    5000 + day * 100,
                "Customers":                500 + day * 10,
                "Promo":                    1 if day % 2 == 0 else 0,
                "StateHoliday":             "0",
                "SchoolHoliday":            0,
                "StoreType":                "c",
                "Assortment":               "a",
                "CompetitionDistance":      1270.0,
                "CompetitionOpenSinceMonth":9.0,
                "CompetitionOpenSinceYear": 2008.0,
                "Promo2":                   1,
                "Promo2SinceWeek":          13.0,
                "Promo2SinceYear":          2010.0,
                "PromoInterval":            "Jan,Apr,Jul,Oct",
            })
    df = pd.DataFrame(stores)
    df["Promo"] = df["Promo"].astype("category")
    df["Promo2"] = df["Promo2"].astype("category")
    df["StoreType"] = df["StoreType"].astype("category")
    df["Assortment"] = df["Assortment"].astype("category")
    df["PromoInterval"] = df["PromoInterval"].astype("category")
    return df


def test_temporal_features_created():
    """Year, Month, Day, DayOfWeek, WeekOfYear, DayOfYear should be added."""
    df = make_clean_df()
    result = engineer_features(df)
    for col in ["Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "DayOfYear"]:
        assert col in result.columns, f"Missing column: {col}"


def test_competition_open_created():
    """CompetitionOpen should be created and source columns dropped."""
    df = make_clean_df()
    result = engineer_features(df)
    assert "CompetitionOpen" in result.columns
    assert "CompetitionOpenSinceMonth" not in result.columns
    assert "CompetitionOpenSinceYear" not in result.columns


def test_competition_open_non_negative():
    """CompetitionOpen should never be negative."""
    df = make_clean_df()
    result = engineer_features(df)
    assert (result["CompetitionOpen"] >= 0).all()


def test_is_holiday_created():
    """Is_Holiday should be created and source columns dropped."""
    df = make_clean_df()
    result = engineer_features(df)
    assert "Is_Holiday" in result.columns
    assert "StateHoliday" not in result.columns
    assert "SchoolHoliday" not in result.columns


def test_is_holiday_binary():
    """Is_Holiday should only contain 0 and 1."""
    df = make_clean_df()
    result = engineer_features(df)
    assert set(result["Is_Holiday"].unique()).issubset({0, 1})


def test_promo2_features_created():
    """Condition_of_Promo2 and Promo2Status should be created."""
    df = make_clean_df()
    result = engineer_features(df)
    assert "Condition_of_Promo2" in result.columns
    assert "Promo2Status" in result.columns


def test_promo2_status_non_negative():
    """Promo2Status should never be negative."""
    df = make_clean_df()
    result = engineer_features(df)
    assert (result["Promo2Status"] >= 0).all()


def test_lag_features_created():
    """Sales(Lag_7) and Sales(Rolling_Mean_7) should be created."""
    df = make_clean_df()
    result = engineer_features(df)
    assert "Sales(Lag_7)" in result.columns
    assert "Sales(Rolling_Mean_7)" in result.columns


def test_lag_features_no_nulls():
    """Lag features should have no NaN values after fillna(0)."""
    df = make_clean_df()
    result = engineer_features(df)
    assert result["Sales(Lag_7)"].isnull().sum() == 0
    assert result["Sales(Rolling_Mean_7)"].isnull().sum() == 0


def test_lag_features_grouped_by_store():
    """
    Lag features should not bleed across stores.
    Store 2's lag should not contain Store 1's sales values.
    """
    df = make_clean_df()
    result = engineer_features(df)

    store1 = result[result["Store"] == 1]
    store2 = result[result["Store"] == 2]

    # Both stores have same sales pattern in our test data
    # The lag features should be identical for both stores
    store1_lags = store1["Sales(Lag_7)"].values
    store2_lags = store2["Sales(Lag_7)"].values
    np.testing.assert_array_equal(store1_lags, store2_lags)


def test_does_not_mutate_input():
    """Original DataFrame should not be modified."""
    df = make_clean_df()
    original_cols = set(df.columns)
    engineer_features(df)
    assert set(df.columns) == original_cols