import numpy as np
import pytest
import torch
from data.sequence_dataset import SequenceDataset


def make_fake_data(n_stores=3, days_per_store=20, n_features=5, window_size=7):
    """
    Build a tiny fake dataset for testing.
    Store IDs: [0,0,0,...,1,1,1,...,2,2,2,...]
    Features: random. Sales: random positive.
    """
    total_rows = n_stores * days_per_store
    X = np.random.randn(total_rows, n_features).astype(np.float32)
    Y = np.random.rand(total_rows).astype(np.float32)
    store_ids = np.repeat(np.arange(n_stores), days_per_store)
    return X, Y, store_ids, window_size


# ── Test 1: correct number of windows ──────────────────────────────
def test_length():
    X, Y, store_ids, ws = make_fake_data(n_stores=3, days_per_store=20, window_size=7)
    ds = SequenceDataset(X, Y, store_ids, window_size=ws)
    # Each store has 20 rows → 20-7 = 13 windows per store → 3×13 = 39 total
    assert len(ds) == 39, f"Expected 39, got {len(ds)}"


# ── Test 2: correct output shapes ──────────────────────────────────
def test_output_shapes():
    X, Y, store_ids, ws = make_fake_data(n_stores=2, days_per_store=15,
                                          n_features=5, window_size=7)
    ds = SequenceDataset(X, Y, store_ids, window_size=ws)
    x_window, y_target = ds[0]
    assert x_window.shape == torch.Size([7, 5]), f"X shape wrong: {x_window.shape}"
    assert y_target.shape == torch.Size([]),     f"Y should be scalar: {y_target.shape}"


# ── Test 3: no window crosses a store boundary ──────────────────────
def test_no_boundary_crossing():
    X, Y, store_ids, ws = make_fake_data(n_stores=4, days_per_store=25, window_size=7)
    ds = SequenceDataset(X, Y, store_ids, window_size=ws)

    for i in range(len(ds)):
        window_start, target_pos = ds.index[i]
        # All rows in the window must belong to the same store
        stores_in_window = store_ids[window_start : window_start + ws]
        assert len(set(stores_in_window)) == 1, \
            f"Window {i} crosses store boundary: {stores_in_window}"


# ── Test 4: target is the correct row ──────────────────────────────
def test_target_is_correct_row():
    X, Y, store_ids, ws = make_fake_data(n_stores=1, days_per_store=20, window_size=7)
    ds = SequenceDataset(X, Y, store_ids, window_size=ws)

    window_start, target_pos = ds.index[0]
    _, y_target = ds[0]
    expected = torch.FloatTensor([Y[target_pos]])
    assert torch.isclose(y_target, expected[0]), \
        f"Target mismatch: got {y_target}, expected {expected[0]}"


# ── Test 5: tiny store (fewer rows than window) is skipped ─────────
def test_tiny_store_skipped():
    # Store 0: 3 rows (too small for window=7)
    # Store 1: 20 rows (fine)
    X = np.random.randn(23, 4).astype(np.float32)
    Y = np.random.rand(23).astype(np.float32)
    store_ids = np.array([0]*3 + [1]*20)
    ds = SequenceDataset(X, Y, store_ids, window_size=7)
    # Only store 1 contributes: 20-7 = 13 windows
    assert len(ds) == 13, f"Expected 13, got {len(ds)}"
