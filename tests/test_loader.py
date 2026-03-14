"""
tests/test_loader.py
--------------------
Unit tests for data/loader.py
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.loader import load_data


def test_load_data_raises_if_no_data_dir():
    """Should raise ValueError when no data_dir given and .env not set."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("data.loader.load_dotenv"):
            with pytest.raises(ValueError, match="No data directory provided"):
                load_data()


def test_load_data_raises_if_train_missing(tmp_path):
    """Should raise FileNotFoundError when train.csv does not exist."""
    # tmp_path is an empty temporary directory — no CSV files there
    with pytest.raises(FileNotFoundError, match="train.csv not found"):
        load_data(data_dir=str(tmp_path))


def test_load_data_raises_if_store_missing(tmp_path):
    """Should raise FileNotFoundError when store.csv does not exist."""
    # Create train.csv but not store.csv
    (tmp_path / "train.csv").write_text("Store,Sales\n1,1000\n")
    with pytest.raises(FileNotFoundError, match="store.csv not found"):
        load_data(data_dir=str(tmp_path))


def test_load_data_returns_dataframe(tmp_path):
    """Should return a merged DataFrame when both files exist."""
    # Create minimal valid CSV files
    (tmp_path / "train.csv").write_text(
        "Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday\n"
        "1,5,2015-07-31,5263,555,1,1,0,1\n"
        "2,5,2015-07-31,6064,625,1,1,0,1\n"
    )
    (tmp_path / "store.csv").write_text(
        "Store,StoreType,Assortment,CompetitionDistance\n"
        "1,c,a,1270\n"
        "2,a,a,570\n"
    )

    df = load_data(data_dir=str(tmp_path))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Store" in df.columns
    assert "StoreType" in df.columns
    assert "Sales" in df.columns


def test_load_data_merge_is_left_join(tmp_path):
    """All training rows should be preserved after merge."""
    (tmp_path / "train.csv").write_text(
        "Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday\n"
        "1,5,2015-07-31,5263,555,1,1,0,1\n"
        "1,4,2015-07-30,5020,490,1,0,0,0\n"
        "1,3,2015-07-29,4987,476,1,0,0,0\n"
    )
    (tmp_path / "store.csv").write_text(
        "Store,StoreType,Assortment,CompetitionDistance\n"
        "1,c,a,1270\n"
    )

    df = load_data(data_dir=str(tmp_path))

    # All 3 training rows should be preserved
    assert len(df) == 3