"""
data/preprocessor.py
--------------------
Cleans and prepares the raw Rossmann DataFrame for feature engineering.

Responsibilities:
    - Remove closed store days and zero-sales rows
    - Handle missing values with appropriate strategies
    - Convert Date column to datetime
    - Set correct data types for categorical columns

This module does NOT create new features.
That is the responsibility of feature_engineer.py.
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw merged Rossmann DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame from loader.load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ~844,338 rows.

    Examples
    --------
    >>> from data.loader import load_data
    >>> from data.preprocessor import clean_data
    >>> raw = load_data()
    >>> clean = clean_data(raw)
    """

    print(f"Starting preprocessing — input shape: {df.shape}")

    # Work on a copy — never mutate the input
    df = df.copy()

    # ── Step 1: Remove closed store days ─────────────────────────────────────
    # Stores that are closed have Sales=0 and Open=0.
    # These are not useful for forecasting — we only predict open days.
    rows_before = len(df)
    df = df[(df["Open"] != 0) & (df["Sales"] != 0)]
    rows_removed = rows_before - len(df)
    print(f"  Removed {rows_removed:,} closed store rows — {len(df):,} rows remaining")

    # Drop Open column — all remaining rows are open days, column is redundant
    df = df.drop(columns=["Open"])

    # ── Step 2: Convert Date to datetime ─────────────────────────────────────
    # Must happen before feature engineering which extracts year, month, day
    df["Date"] = pd.to_datetime(df["Date"])

    # ── Step 3: Handle missing values ────────────────────────────────────────
    # CompetitionDistance: use median — preserves distribution, avoids outlier bias
    competition_median = df["CompetitionDistance"].median()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(competition_median)
    print(f"  Filled CompetitionDistance NaNs with median: {competition_median:.0f}")

    # Promo2 fields: fill with 0 — NaN means Promo2 was never active
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)

    # PromoInterval: fill with '0' string — used later in string matching
    df["PromoInterval"] = df["PromoInterval"].fillna("0")

    # ── Step 4: Convert categorical columns ──────────────────────────────────
    # These columns have a fixed set of values — category dtype saves memory
    # and signals to models that these are discrete, not continuous
    categorical_cols = [
        "Store", "Promo", "Promo2",
        "StoreType", "Assortment", "PromoInterval",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Verify no missing values remain
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        null_cols = df.columns[df.isnull().any()].tolist()
        print(f"  WARNING: {remaining_nulls} null values remain in: {null_cols}")
    else:
        print("  No null values remaining")

    print(f"Preprocessing complete — output shape: {df.shape}")

    return df