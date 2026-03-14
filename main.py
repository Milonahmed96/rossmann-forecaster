from data.loader import load_data
from data.preprocessor import clean_data
from data.feature_engineer import engineer_features

# Run the full data pipeline on real data
raw = load_data()
clean = clean_data(raw)
features = engineer_features(clean)

print("\n--- Pipeline Summary ---")
print(f"Raw data shape:      {raw.shape}")
print(f"Clean data shape:    {clean.shape}")
print(f"Final features:      {features.shape}")
print(f"\nColumns in final dataset:")
for col in features.columns:
    print(f"  {col}")