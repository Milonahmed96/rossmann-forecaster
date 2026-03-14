"""
data/loader.py
--------------
Loads raw Rossmann CSV files from disk and returns a merged DataFrame.

Responsibilities:
    - Read train.csv and store.csv from the data directory
    - Merge them on the Store column
    - Return the raw merged DataFrame

This module does NOT clean or transform data.
That is the responsibility of preprocessor.py.
"""

import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load and merge the Rossmann train and store datasets.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing train.csv and store.csv.
        If not provided, reads DATA_DIR from the .env file.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with 1,017,209 rows (before preprocessing).

    Raises
    ------
    FileNotFoundError
        If train.csv or store.csv cannot be found in data_dir.
    ValueError
        If data_dir is not provided and DATA_DIR is not set in .env.

    Examples
    --------
    >>> df = load_data()                          # uses .env
    >>> df = load_data("D:/My Data/Dataset")      # explicit path
    """

    # Use provided path or fall back to .env
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR")

    if data_dir is None:
        raise ValueError(
            "No data directory provided. "
            "Either pass data_dir as an argument or set DATA_DIR in your .env file."
        )

    # Build full file paths
    train_path = os.path.join(data_dir, "train.csv")
    store_path = os.path.join(data_dir, "store.csv")

    # Check files exist before trying to load them
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train.csv not found at: {train_path}\n"
            f"Check your DATA_DIR setting in .env"
        )

    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"store.csv not found at: {store_path}\n"
            f"Check your DATA_DIR setting in .env"
        )

    # Load the datasets
    print(f"Loading train.csv from {data_dir} ...")
    data_train = pd.read_csv(train_path, low_memory=False)
    print(f"  Loaded {len(data_train):,} rows from train.csv")

    print(f"Loading store.csv from {data_dir} ...")
    data_store = pd.read_csv(store_path, low_memory=False)
    print(f"  Loaded {len(data_store):,} rows from store.csv")

    # Merge on Store column — left join keeps all training records
    print("Merging datasets on Store column ...")
    df = pd.merge(data_train, data_store, on="Store", how="left")
    print(f"  Merged dataset: {len(df):,} rows x {df.shape[1]} columns")

    return df