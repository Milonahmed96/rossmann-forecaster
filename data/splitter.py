"""
data/splitter.py
----------------
Prepares features and target, encodes categorical columns,
and splits data into train and test sets.

Responsibilities:
    - Define X (features) and Y (log-transformed target)
    - Label encode for LightGBM and LSTM
    - One-hot encode for Ridge
    - Split on time — last 8 weeks as test set

This module expects engineered data from feature_engineer.engineer_features().
It does NOT train any models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def prepare_data(df: pd.DataFrame) -> dict:
    """
    Prepare features, encode, and split into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame from feature_engineer.engineer_features().

    Returns
    -------
    dict with keys:
        X_train_LE, X_test_LE   — Label encoded (for LightGBM and LSTM)
        X_train_OHE, X_test_OHE — One-hot encoded (for Ridge)
        Y_train, Y_test         — Log-transformed target
        X_train_scaled          — MinMax scaled LE (for LSTM)
        X_test_scaled           — MinMax scaled LE (for LSTM)
        scaler_X                — Fitted MinMaxScaler (for inverse transform)

    Examples
    --------
    >>> splits = prepare_data(features)
    >>> X_train_LE = splits["X_train_LE"]
    >>> Y_train = splits["Y_train"]
    """

    print("Preparing data for modelling ...")

    # ── Step 1: Define target and features ───────────────────────────────────
    # Log-transform target — reduces right skew, makes RMSPE more meaningful
    Y = np.log1p(df["Sales"])

    # Drop Sales (target), Customers (leakage), Date (already extracted)
    X = df.drop(columns=["Sales", "Customers", "Date"])

    # ── Step 2: Define categorical columns for encoding ──────────────────────
    categorical_cols = [
        "Store", "DayOfWeek", "Promo", "Promo2", "Is_Holiday",
        "StoreType", "Assortment", "PromoInterval",
        "Year", "Month", "Day", "DayOfYear",
    ]
    # Only encode columns that actually exist in X
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    # ── Step 3: Label Encoding (for LightGBM and LSTM) ───────────────────────
    X_LE = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_LE[col] = le.fit_transform(X_LE[col].astype(str))

    # ── Step 4: One-Hot Encoding (for Ridge) ─────────────────────────────────
    X_OHE = X.copy()
    X_OHE = pd.get_dummies(X_OHE, columns=categorical_cols, drop_first=True)

    # ── Step 5: Time-based train/test split ──────────────────────────────────
    # Use the final 8 weeks (56 days) as test set
    # Multiply by num_stores to get the correct number of records
    test_period_days = 8 * 7
    num_stores = df["Store"].nunique()
    split_records = test_period_days * num_stores
    split_index = X.shape[0] - split_records

    X_train_LE  = X_LE.iloc[:split_index].fillna(0)
    X_test_LE   = X_LE.iloc[split_index:].fillna(0)
    X_train_OHE = X_OHE.iloc[:split_index].fillna(0)
    X_test_OHE  = X_OHE.iloc[split_index:].fillna(0)
    Y_train     = Y.iloc[:split_index]
    Y_test      = Y.iloc[split_index:]

    print(f"  Training set: {len(X_train_LE):,} records")
    print(f"  Test set:     {len(X_test_LE):,} records")

    # ── Step 6: Scale OHE features for Ridge ─────────────────────────────────
    numerical_cols_OHE = X_OHE.select_dtypes(include=np.number).columns
    scaler_OHE = StandardScaler()
    X_train_OHE[numerical_cols_OHE] = scaler_OHE.fit_transform(
        X_train_OHE[numerical_cols_OHE]
    )
    X_test_OHE[numerical_cols_OHE] = scaler_OHE.transform(
        X_test_OHE[numerical_cols_OHE]
    )

    # ── Step 7: MinMax scale LE features for LSTM ────────────────────────────
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train_LE)
    X_test_scaled  = scaler_X.transform(X_test_LE)

    print("Data preparation complete.")

    # ── Step 8: Extract Store IDs for sequence windowing ─────────────────────
    store_ids_train = df["Store"].iloc[:split_index].values
    store_ids_test  = df["Store"].iloc[split_index:].values

    return {
        "X_train_LE":       X_train_LE,
        "X_test_LE":        X_test_LE,
        "X_train_OHE":      X_train_OHE,
        "X_test_OHE":       X_test_OHE,
        "Y_train":          Y_train,
        "Y_test":           Y_test,
        "X_train_scaled":   X_train_scaled,
        "X_test_scaled":    X_test_scaled,
        "scaler_X":         scaler_X,
        "store_ids_train":  store_ids_train,
        "store_ids_test":   store_ids_test,
    }