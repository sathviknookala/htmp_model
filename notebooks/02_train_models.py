"""
Training Script for Hull Tactical Market Prediction
Trains both Gradient Boosting and Neural Network models
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.models import GradientBoostingPredictor, AllocationStrategy

print("=" * 80)
print("HULL TACTICAL MARKET PREDICTION - MODEL TRAINING")
print("=" * 80)

# Load data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')

print(f"\nLoaded training data: {train_df.shape}")

# Initialize feature engineer
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

feature_engineer = FeatureEngineer(lookback_window=20)

# Only use data where most features are available
# Based on EDA, E features complete from date_id 6969
# Let's use data from 7000 onwards to have clean features
min_date_id = 7000
train_df_clean = train_df[train_df['date_id'] >= min_date_id].copy()

print(f"Using data from date_id {min_date_id}: {train_df_clean.shape[0]} samples")

# Prepare features
data_dict = feature_engineer.prepare_data(train_df_clean)

train_tabular = data_dict['train_tabular']
feature_names = data_dict['feature_names']

print(f"Created {len(feature_names)} features")
print(f"Training samples after feature engineering: {train_tabular.shape[0]}")

# Remove rows with NaN target
target_col = 'market_forward_excess_returns'
train_tabular_valid = train_tabular[train_tabular[target_col].notna()].copy()

print(f"Valid training samples (non-null target): {train_tabular_valid.shape[0]}")

# ============================================================================
# GRADIENT BOOSTING MODEL
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING GRADIENT BOOSTING MODEL")
print("=" * 80)

# Initialize model
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

# Train with cross-validation
cv_results = gbm_model.train_with_cv(
    train_tabular_valid,
    feature_names,
    target_col=target_col,
    n_splits=5,
    num_boost_round=1000,
    early_stopping_rounds=50
)

print("\n" + "=" * 80)
print("GRADIENT BOOSTING CV RESULTS")
print("=" * 80)
print(f"Average RMSE: {cv_results['avg_rmse']:.6f}")
print(f"Average Correlation: {cv_results['avg_correlation']:.4f}")

# Train final model on all data
print("\n" + "=" * 80)
print("TRAINING FINAL GRADIENT BOOSTING MODEL")
print("=" * 80)

gbm_model.train_full(
    train_tabular_valid,
    feature_names,
    target_col=target_col,
    num_boost_round=1000
)

# Save model
model_dir = data_dir / 'models'
model_dir.mkdir(exist_ok=True)

gbm_model_path = model_dir / 'gbm_model.txt'
gbm_model.save(str(gbm_model_path))
print(f"\nSaved model to {gbm_model_path}")

# Save feature engineer
feature_engineer_path = model_dir / 'feature_engineer.pkl'
with open(feature_engineer_path, 'wb') as f:
    pickle.dump(feature_engineer, f)
print(f"Saved feature engineer to {feature_engineer_path}")

# ============================================================================
# NEURAL NETWORK MODEL (OPTIONAL - not implemented yet)
# ============================================================================

print("\n" + "=" * 80)
print("Neural network models (LSTM/GRU) can be added later")
print("For now, focusing on gradient boosting which is fast and effective")
print("=" * 80)

# ============================================================================
# TEST ALLOCATION STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("TESTING ALLOCATION STRATEGY")
print("=" * 80)

allocation_strategy = AllocationStrategy(
    max_volatility_multiplier=1.2,
    lookback_window=252
)

# Make predictions on recent data
recent_data = train_tabular_valid.tail(100)
predictions = gbm_model.predict(recent_data, feature_names)

print(f"Made {len(predictions)} predictions")
print(f"Prediction stats:")
print(f"  Mean: {predictions.mean():.6f}")
print(f"  Std: {predictions.std():.6f}")
print(f"  Min: {predictions.min():.6f}")
print(f"  Max: {predictions.max():.6f}")

# Get allocations
market_returns = train_df_clean['market_forward_excess_returns'].dropna()
risk_free_rates = train_df_clean['risk_free_rate'].iloc[-len(predictions):]

allocations = allocation_strategy.get_allocation_batch(
    predictions,
    market_returns,
    risk_free_rates,
    strategy='volatility_scaled'
)

print(f"\nAllocation stats:")
print(f"  Mean: {allocations.mean():.4f}")
print(f"  Std: {allocations.std():.4f}")
print(f"  Min: {allocations.min():.4f}")
print(f"  Max: {allocations.max():.4f}")
print(f"  % in range [0, 2]: {((allocations >= 0) & (allocations <= 2)).mean() * 100:.2f}%")

# Save allocation strategy
allocation_strategy_path = model_dir / 'allocation_strategy.pkl'
with open(allocation_strategy_path, 'wb') as f:
    pickle.dump(allocation_strategy, f)
print(f"\nSaved allocation strategy to {allocation_strategy_path}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nModels saved to:", model_dir)
print("\nNext steps:")
print("1. Create submission notebook using saved models")
print("2. Test on kaggle evaluation framework")
print("3. Submit to competition")

