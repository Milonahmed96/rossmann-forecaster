import numpy as np
from data.loader import load_data
from data.preprocessor import clean_data
from data.feature_engineer import engineer_features
from data.splitter import prepare_data
from models.pytorch_lstm import PyTorchLSTM

print("=" * 55)
print("SMOKE TEST — 5 stores, 3 epochs")
print("=" * 55)

print("\n[1/4] Loading data...")
df = load_data()

print("[2/4] Cleaning and engineering features...")
df = clean_data(df)
df = engineer_features(df)

print("[3/4] Preparing splits...")
data = prepare_data(df)

X_train   = data['X_train_scaled']
Y_train   = data['Y_train'].values
store_ids = data['store_ids_train']

five_stores = np.unique(store_ids)[:5]
mask        = np.isin(store_ids, five_stores)

X_small  = X_train[mask]
Y_small  = Y_train[mask]
ids_small = store_ids[mask]

print(f"    Rows for 5 stores: {len(X_small):,}")
print(f"    Features:          {X_small.shape[1]}")

print("\n[4/4] Training PyTorchLSTM (3 epochs)...")
model = PyTorchLSTM(input_size=X_small.shape[1])
history = model.fit(
    X_small, Y_small, ids_small,
    window_size=7,
    epochs=3,
    batch_size=128,
    patience=10,
    save_path='models/smoke_test_lstm.pt'
)

preds = model.predict(X_small, ids_small, window_size=7)
print(f"\nSample predictions (euros): {preds[:5].round(0)}")
print(f"Prediction range: euro{preds.min():.0f} to euro{preds.max():.0f}")
print("\nSmoke test PASSED — pipeline is working.")
