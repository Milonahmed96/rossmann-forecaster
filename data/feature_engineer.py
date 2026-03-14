"""
data/feature_engineer.py
------------------------
Creates engineered features from the cleaned Rossmann DataFrame.

Responsibilities:
    - Extract temporal features from the Date column
    - Create promotional lifecycle features
    - Create lag and rolling mean features
    - Drop redundant columns

This module expects clean data from preprocessor.clean_data().
It does NOT load data or handle missing values.
"""

import calendar
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the cleaned Rossmann DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocessor.clean_data().
        Must contain: Date, Store, Sales, StateHoliday, SchoolHoliday,
        CompetitionOpenSinceMonth, CompetitionOpenSinceYear,
        Promo2, Promo2SinceWeek, Promo2SinceYear, PromoInterval.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features added and
        redundant columns removed.

    Examples
    --------
    >>> from data.loader import load_data
    >>> from data.preprocessor import clean_data
    >>> from data.feature_engineer import engineer_features
    >>> raw = load_data()
    >>> clean = clean_data(raw)
    >>> features = engineer_features(clean)
    """

    print(f"Starting feature engineering — input shape: {df.shape}")

    # Work on a copy — never mutate the input
    df = df.copy()

    # Sort by Store and Date — required for lag features to be correct
    # Without this sort, lag features pick up values from wrong stores/dates
    df = df.sort_values(by=["Store", "Date"]).reset_index(drop=True)

    # Run each feature group in order
    df = _add_temporal_features(df)
    df = _add_competition_features(df)
    df = _add_holiday_features(df)
    df = _add_promo2_features(df)
    df = _add_lag_features(df)

    # Convert remaining categorical columns now that all features exist
    df = _convert_categorical_dtypes(df)

    print(f"Feature engineering complete — output shape: {df.shape}")
    print(f"  Features created: {df.shape[1]} columns")

    return df


# ── Private helper functions ──────────────────────────────────────────────────

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date components from the Date column."""
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Day"]       = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["WeekOfYear"]= df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    print(f"  Added temporal features: Year, Month, Day, DayOfWeek, WeekOfYear, DayOfYear")
    return df


def _add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate how long competition has been open in months."""
    df["CompetitionOpen"] = (
        12 * (df["Year"] - df["CompetitionOpenSinceYear"]) +
        (df["Month"] - df["CompetitionOpenSinceMonth"])
    )
    # Negative values mean competition opened in the future — treat as 0
    df["CompetitionOpen"] = df["CompetitionOpen"].clip(lower=0)

    # Drop source columns — CompetitionOpen captures all the information
    df = df.drop(columns=["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"])
    print(f"  Added CompetitionOpen feature")
    return df


def _add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine StateHoliday and SchoolHoliday into a single Is_Holiday flag."""
    df["Is_Holiday"] = np.where(
        (df["StateHoliday"] != "0") | (df["SchoolHoliday"] == 1),
        1,
        0
    )
    # Drop source columns — Is_Holiday captures both
    df = df.drop(columns=["StateHoliday", "SchoolHoliday"])
    print(f"  Added Is_Holiday feature")
    return df


def _add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Promo2 activity and duration features."""

    # Month abbreviation map — used to check if Promo2 is active this month
    month_abbr = {i: calendar.month_abbr[i] for i in range(1, 13)}
    df["_MONTH_ABBR"] = df["Month"].map(month_abbr)

    # Condition_of_Promo2: 1 if Promo2 is active AND this month is in PromoInterval
    # PromoInterval contains strings like "Jan,Apr,Jul,Oct"
    df["Condition_of_Promo2"] = np.where(
        (df["Promo2"] == 1) &
        (df.apply(
            lambda row: str(row["_MONTH_ABBR"]) in str(row["PromoInterval"]),
            axis=1
        )),
        1,
        0
    )

    # Promo2Status: how many months Promo2 has been running
    df["Promo2Status"] = (
        12 * (df["Year"] - df["Promo2SinceYear"]) +
        ((df["WeekOfYear"] - df["Promo2SinceWeek"]) / 4.0)
    )
    # Negative values mean Promo2 not yet started — treat as 0
    df["Promo2Status"] = df["Promo2Status"].clip(lower=0)

    # Drop the temporary month abbreviation column
    df = df.drop(columns=["_MONTH_ABBR"])

    print(f"  Added Condition_of_Promo2 and Promo2Status features")
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 7-day lag and rolling mean features grouped by store.

    These are the most important features in the model (confirmed by SHAP).
    Grouping by Store ensures each store only looks at its own history.
    Shift(7) ensures no data leakage — we never use same-day or recent sales.

    NaN values for the first 7 days per store are filled with 0.
    This is acceptable because: the model treats 0 as 'no prior history'
    which is correct for a store's first 7 trading days in the dataset.
    """
    df["Sales(Lag_7)"] = (
        df.groupby("Store")["Sales"]
        .shift(7)
        .fillna(0)
    )

    df["Sales(Rolling_Mean_7)"] = (
        df.groupby("Store")["Sales"]
        .transform(lambda x: x.shift(7).rolling(7).mean())
        .fillna(0)
    )

    print(f"  Added Sales(Lag_7) and Sales(Rolling_Mean_7) features")
    return df


def _convert_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert remaining columns to category dtype."""
    categorical_cols = [
        "DayOfWeek", "Year", "Month", "Day", "DayOfYear",
        "Is_Holiday", "Condition_of_Promo2",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df