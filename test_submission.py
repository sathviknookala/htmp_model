"""
Test Submission Script - Validates models load correctly
"""
import sys
from pathlib import Path
import pickle

print("=" * 70)
print("TESTING SUBMISSION SCRIPT")
print("=" * 70)

MODEL_DIR = Path('models')

# Test 1: Check files exist
print("\n1. Checking files...")
required_files = [
    'gbm_model.txt',
    'gbm_model.txt.metadata',
    'xgb_model.json',
    'xgb_model.json.metadata',
    'feature_engineer.pkl',
    'allocation_strategy.pkl',
    'model_metadata.pkl'
]

all_exist = True
for file in required_files:
    exists = (MODEL_DIR / file).exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {file}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n✗ Some files missing!")
    sys.exit(1)

print("\n✓ All required files present")

# Test 2: Load LightGBM
print("\n2. Loading LightGBM...")
try:
    import lightgbm as lgb
    lgb_model = lgb.Booster(model_file=str(MODEL_DIR / 'gbm_model.txt'))
    print("   ✓ LightGBM loaded successfully")
except Exception as e:
    print(f"   ✗ LightGBM failed: {e}")
    sys.exit(1)

# Test 3: Load XGBoost
print("\n3. Loading XGBoost...")
try:
    import xgboost as xgb
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(MODEL_DIR / 'xgb_model.json'))
    print("   ✓ XGBoost loaded successfully")
except Exception as e:
    print(f"   ✗ XGBoost failed: {e}")
    sys.exit(1)

# Test 4: Load feature engineer
print("\n4. Loading feature engineer...")
try:
    with open(MODEL_DIR / 'feature_engineer.pkl', 'rb') as f:
        feature_engineer = pickle.load(f)
    print("   ✓ Feature engineer loaded")
except Exception as e:
    print(f"   ✗ Feature engineer failed: {e}")
    sys.exit(1)

# Test 5: Load model metadata
print("\n5. Loading model metadata...")
try:
    with open(MODEL_DIR / 'model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print(f"   ✓ Metadata loaded")
    print(f"   • Models: {metadata['models']}")
    print(f"   • Features: {metadata['num_features']}")
    print(f"   • Training samples: {metadata['train_samples']}")
except Exception as e:
    print(f"   ✗ Metadata failed: {e}")
    sys.exit(1)

# Test 6: Load allocation strategy
print("\n6. Loading allocation strategy...")
try:
    with open(MODEL_DIR / 'allocation_strategy.pkl', 'rb') as f:
        allocator = pickle.load(f)
    print("   ✓ Allocation strategy loaded")
except Exception as e:
    print(f"   ✗ Allocation strategy failed: {e}")
    sys.exit(1)

# Test 7: Test prediction pipeline
print("\n7. Testing prediction pipeline...")
try:
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    test_data = pd.DataFrame({col: [0.0] for col in metadata['feature_names']})
    test_data['market_forward_excess_returns'] = 0.0
    
    # Engineer features
    combined = pd.concat([test_data] * 10, ignore_index=True)
    features = feature_engineer.create_all_tabular_features(
        combined,
        target_col='market_forward_excess_returns'
    )
    
    # Get predictions
    X = features.iloc[-1:][metadata['feature_names']].fillna(0)
    
    lgb_pred = lgb_model.predict(X)[0]
    xgb_pred = xgb_model.predict(xgb.DMatrix(X))[0]
    
    ensemble_pred = (lgb_pred + xgb_pred) / 2
    
    print(f"   ✓ Pipeline works!")
    print(f"   • LightGBM prediction: {lgb_pred:.6f}")
    print(f"   • XGBoost prediction: {xgb_pred:.6f}")
    print(f"   • Ensemble prediction: {ensemble_pred:.6f}")
    
    # Test allocation
    allocation = allocator.volatility_scaled_allocation(
        ensemble_pred, 0.15
    )
    print(f"   • Allocation: {allocation:.4f}")
    
    if 0 <= allocation <= 2:
        print("   ✓ Allocation in valid range [0, 2]")
    else:
        print(f"   ✗ Allocation out of range: {allocation}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nYour submission script is ready!")
print("\nDataset path for Kaggle:")
print("  /kaggle/input/eshaanganguly-hull-tactical-models")
print("\n" + "=" * 70)

