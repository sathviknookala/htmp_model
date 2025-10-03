"""
Train Models and Prepare for Kaggle Upload
This script trains optimized models and prepares them for Kaggle submission
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering import FeatureEngineer
from src.models import (
    GradientBoostingPredictor, 
    AllocationStrategy,
    ModelEnsemble,
    FeatureSelector
)

# Configuration
TRAIN_ENSEMBLE = True  # Set False for faster single-model training
MAX_FEATURES = 100  # Number of features to select

print("=" * 80)
print("TRAINING MODELS FOR KAGGLE SUBMISSION")
print("=" * 80)

# Load data
data_dir = Path(__file__).parent
train_df = pd.read_csv(data_dir / 'train.csv')

print(f"\nLoaded training data: {train_df.shape}")

# Initialize feature engineer
feature_engineer = FeatureEngineer(lookback_window=20)

# Use clean data from date_id 7000 onwards
min_date_id = 7000
train_df_clean = train_df[train_df['date_id'] >= min_date_id].copy()
print(f"Using data from date_id {min_date_id}: {train_df_clean.shape[0]} samples")

# Prepare features
data_dict = feature_engineer.prepare_data(train_df_clean)
train_tabular = data_dict['train_tabular']
feature_names = data_dict['feature_names']
target_col = 'market_forward_excess_returns'

# Remove rows with NaN target
train_tabular_valid = train_tabular[train_tabular[target_col].notna()].copy()
print(f"Valid training samples: {train_tabular_valid.shape[0]}")
print(f"Initial features: {len(feature_names)}")

# ============================================================================
# FEATURE SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# Train quick model for importance
print("Training quick model for feature importance...")
quick_gbm = GradientBoostingPredictor()
quick_gbm.train_full(
    train_tabular_valid,
    feature_names,
    target_col=target_col,
    num_boost_round=200
)

# Select features
feature_selector = FeatureSelector(
    max_features=MAX_FEATURES,
    correlation_threshold=0.95
)

selected_features = feature_selector.comprehensive_selection(
    train_tabular_valid,
    feature_names,
    target_col,
    feature_importance=quick_gbm.feature_importance,
    use_correlation=True,
    use_variance=True,
    use_null_filter=True
)

print(f"\nSelected {len(selected_features)} features")
feature_names = selected_features

# ============================================================================
# TRAIN FINAL MODELS
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING FINAL MODELS")
print("=" * 80)

models = {}

# 1. LightGBM (primary model)
print("\n1. Training LightGBM (primary model)...")
gbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42
}

gbm_model = GradientBoostingPredictor(params=gbm_params)
gbm_model.train_full(
    train_tabular_valid,
    feature_names,
    target_col=target_col,
    num_boost_round=1000
)
models['lightgbm'] = gbm_model

# 2. XGBoost (if available)
if TRAIN_ENSEMBLE:
    try:
        from src.models import XGBoostPredictor
        print("\n2. Training XGBoost...")
        xgb_model = XGBoostPredictor()
        xgb_model.train_full(
            train_tabular_valid,
            feature_names,
            target_col=target_col,
            num_boost_round=1000
        )
        models['xgboost'] = xgb_model
        print("XGBoost trained successfully")
    except Exception as e:
        print(f"XGBoost not available: {e}")

    # 3. CatBoost (if available)
    try:
        from src.models import CatBoostPredictor
        print("\n3. Training CatBoost...")
        cat_model = CatBoostPredictor()
        cat_model.train_full(
            train_tabular_valid,
            feature_names,
            target_col=target_col,
            iterations=1000
        )
        models['catboost'] = cat_model
        print("CatBoost trained successfully")
    except Exception as e:
        print(f"CatBoost not available: {e}")

print(f"\nTrained {len(models)} models: {list(models.keys())}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

model_dir = data_dir / 'models'
model_dir.mkdir(exist_ok=True)

# Save all models
gbm_model.save(str(model_dir / 'gbm_model.txt'))
print(f"Saved: gbm_model.txt")

if 'xgboost' in models:
    models['xgboost'].save(str(model_dir / 'xgb_model.json'))
    print(f"Saved: xgb_model.json")

if 'catboost' in models:
    models['catboost'].save(str(model_dir / 'catboost_model.cbm'))
    print(f"Saved: catboost_model.cbm")

# Save feature engineer
with open(model_dir / 'feature_engineer.pkl', 'wb') as f:
    pickle.dump(feature_engineer, f)
print(f"Saved: feature_engineer.pkl")

# Save feature selector
feature_selector.save(str(model_dir / 'feature_selector.pkl'))
print(f"Saved: feature_selector.pkl")

# Save allocation strategy
allocation_strategy = AllocationStrategy(
    max_volatility_multiplier=1.2,
    lookback_window=252
)
with open(model_dir / 'allocation_strategy.pkl', 'wb') as f:
    pickle.dump(allocation_strategy, f)
print(f"Saved: allocation_strategy.pkl")

# Save ensemble config
if len(models) > 1:
    ensemble = ModelEnsemble(models=models)
    ensemble.save(str(model_dir / 'ensemble.pkl'))
    print(f"Saved: ensemble.pkl")

# Save model metadata
model_metadata = {
    'models': list(models.keys()),
    'feature_names': feature_names,
    'target_col': target_col,
    'num_features': len(feature_names),
    'train_samples': len(train_tabular_valid)
}
with open(model_dir / 'model_metadata.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)
print(f"Saved: model_metadata.pkl")

# ============================================================================
# CREATE KAGGLE DATASET METADATA
# ============================================================================
print("\n" + "=" * 80)
print("CREATING KAGGLE DATASET METADATA")
print("=" * 80)

dataset_metadata = {
    "title": "Hull Tactical Models",
    "id": "your-username/hull-tactical-models",  # Update with your Kaggle username
    "licenses": [{"name": "CC0-1.0"}],
    "resources": [
        {
            "path": "gbm_model.txt",
            "description": "LightGBM model"
        },
        {
            "path": "gbm_model.txt.metadata",
            "description": "LightGBM model metadata"
        },
        {
            "path": "feature_engineer.pkl",
            "description": "Feature engineering pipeline"
        },
        {
            "path": "feature_selector.pkl",
            "description": "Feature selector"
        },
        {
            "path": "allocation_strategy.pkl",
            "description": "Allocation strategy"
        },
        {
            "path": "model_metadata.pkl",
            "description": "Model configuration metadata"
        }
    ]
}

# Add optional files if they exist
if (model_dir / 'xgb_model.json').exists():
    dataset_metadata['resources'].extend([
        {"path": "xgb_model.json", "description": "XGBoost model"},
        {"path": "xgb_model.json.metadata", "description": "XGBoost metadata"}
    ])

if (model_dir / 'catboost_model.cbm').exists():
    dataset_metadata['resources'].extend([
        {"path": "catboost_model.cbm", "description": "CatBoost model"},
        {"path": "catboost_model.cbm.metadata", "description": "CatBoost metadata"}
    ])

if (model_dir / 'ensemble.pkl').exists():
    dataset_metadata['resources'].append(
        {"path": "ensemble.pkl", "description": "Ensemble configuration"}
    )

import json
with open(model_dir / 'dataset-metadata.json', 'w') as f:
    json.dump(dataset_metadata, f, indent=2)
print(f"Saved: dataset-metadata.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 80)

print(f"\nModels trained: {list(models.keys())}")
print(f"Features selected: {len(feature_names)}")
print(f"Training samples: {len(train_tabular_valid)}")

print("\nFiles ready for Kaggle upload:")
for resource in dataset_metadata['resources']:
    filepath = model_dir / resource['path']
    if filepath.exists():
        size = filepath.stat().st_size / 1024  # KB
        print(f"  - {resource['path']} ({size:.1f} KB)")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. Create Kaggle dataset:")
print("   cd models")
print("   kaggle datasets create")
print("\n2. Or update existing dataset:")
print("   kaggle datasets version -p models -m 'Updated with new models'")
print("\n3. Update kaggle_submission.py with your dataset path")
print("\n4. Test locally:")
print("   python kaggle_submission.py")
print("\n5. Submit to competition:")
print("   bash submit_kaggle.sh")
print("\n" + "=" * 80)

