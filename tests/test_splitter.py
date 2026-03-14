"""
tests/test_splitter.py
----------------------
Unit tests for data/splitter.py
"""

import pytest
import numpy as np
import pandas as pd
from data.splitter import prepare_data


def make_engineered_df():
    """
    Create a minimal engineered DataFrame that mimics
    feature_engineer output. 3 stores x 70 days = 210 rows.
    Enough for an 8-week test split.
    """
    rows = []
    for store_id in [1, 2, 3]:
        for day in range(70):
            date = pd.Timestamp("2015-01-01") + pd.Timedelta(days=day)
            rows.append({
                "Store":               store_id,
                "Date":                date,
                "Sales":               5000 + day * 10,
                "Customers":           500,
                "Promo":               day % 2,
                "Promo2":              1,
                "StoreType":           "c",
                "Assortment":          "a",
                "PromoInterval":       "Jan,Apr,Jul,Oct",
                "CompetitionDistance": 1270.0,
                "Promo2SinceWeek":     13.0,
                "Promo2SinceYear":     2010.0,
                "Year":                date.year,
                "Month":               date.month,
                "Day":                 date.day,
                "DayOfWeek":           date.dayofweek,
                "WeekOfYear":          date.isocalendar()[1],
                "DayOfYear":           date.dayofyear,
                "CompetitionOpen":     80.0,
                "Is_Holiday":          0,
                "Condition_of_Promo2": 0,
                "Promo2Status":        60.0,
                "Sales(Lag_7)":        4900.0,
                "Sales(Rolling_Mean_7)": 4950.0,
            })
    return pd.DataFrame(rows)


def test_prepare_data_returns_dict():
    """Should return a dictionary with all required keys."""
    df = make_engineered_df()
    splits = prepare_data(df)
    required_keys = [
        "X_train_LE", "X_test_LE",
        "X_train_OHE", "X_test_OHE",
        "Y_train", "Y_test",
        "X_train_scaled", "X_test_scaled",
        "scaler_X",
    ]
    for key in required_keys:
        assert key in splits, f"Missing key: {key}"


def test_split_sizes():
    """Test set should be last 8 weeks x num_stores records."""
    df = make_engineered_df()
    splits = prepare_data(df)
    num_stores = 3
    expected_test = 8 * 7 * num_stores
    expected_train = len(df) - expected_test
    assert len(splits["X_test_LE"]) == expected_test
    assert len(splits["X_train_LE"]) == expected_train


def test_target_is_log_transformed():
    """Y values should be log1p transformed — all positive and less than raw Sales."""
    df = make_engineered_df()
    splits = prepare_data(df)
    assert (splits["Y_train"] > 0).all()
    assert splits["Y_train"].max() < df["Sales"].max()


def test_customers_and_date_not_in_features():
    """Customers and Date should be dropped from features."""
    df = make_engineered_df()
    splits = prepare_data(df)
    assert "Customers" not in splits["X_train_LE"].columns
    assert "Date" not in splits["X_train_LE"].columns
    assert "Sales" not in splits["X_train_LE"].columns


def test_scaled_features_in_range():
    """MinMax scaled features should be between 0 and 1."""
    df = make_engineered_df()
    splits = prepare_data(df)
    assert splits["X_train_scaled"].min() >= 0.0
    assert splits["X_train_scaled"].max() <= 1.0


def test_no_nulls_in_splits():
    """No NaN values should remain in any split."""
    df = make_engineered_df()
    splits = prepare_data(df)
    assert pd.DataFrame(splits["X_train_LE"]).isnull().sum().sum() == 0
    assert pd.DataFrame(splits["X_test_LE"]).isnull().sum().sum() == 0