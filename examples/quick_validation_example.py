"""
Quick Example: How to Validate Your Model for Overfitting
Run this to quickly check if your model is overfitting
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.models import (
    GradientBoostingPredictor,
    ValidationMetrics,
    OverfittingDetector,
    create_validation_report
)

print("Quick Model Validation Example")
print("=" * 60)

# 1. Load your data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')
train_df_clean = train_df[train_df['date_id'] >= 7000].copy()

# 2. Split into train and test (important!)
split_idx = int(len(train_df_clean) * 0.85)
train_data = train_df_clean.iloc[:split_idx]
test_data = train_df_clean.iloc[split_idx:]

print(f"Train: {len(train_data)} samples")
print(f"Test: {len(test_data)} samples")

# 3. Feature engineering
feature_engineer = FeatureEngineer(lookback_window=20)
train_dict = feature_engineer.prepare_data(train_data)
test_dict = feature_engineer.prepare_data(test_data)

train_tabular = train_dict['train_tabular']
test_tabular = test_dict['train_tabular']
feature_names = train_dict['feature_names']

target_col = 'market_forward_excess_returns'
train_tabular = train_tabular[train_tabular[target_col].notna()].copy()
test_tabular = test_tabular[test_tabular[target_col].notna()].copy()

print(f"Features: {len(feature_names)}")

# 4. Train your model
print("\nTraining model...")
model = GradientBoostingPredictor()
model.train_full(
    train_tabular,
    feature_names,
    target_col=target_col,
    num_boost_round=500
)

# 5. Make predictions
train_pred = model.predict(train_tabular, feature_names)
train_actual = train_tabular[target_col].values

test_pred = model.predict(test_tabular, feature_names)
test_actual = test_tabular[target_col].values

# 6. Calculate metrics
train_metrics = ValidationMetrics.all_metrics(train_actual, train_pred)
test_metrics = ValidationMetrics.all_metrics(test_actual, test_pred)

print("\nPerformance:")
print(f"Train: RMSE={train_metrics['rmse']:.6f}, Corr={train_metrics['correlation']:.4f}")
print(f"Test:  RMSE={test_metrics['rmse']:.6f}, Corr={test_metrics['correlation']:.4f}")

# 7. Check for overfitting
detector = OverfittingDetector()
diagnosis = detector.diagnose(
    train_metrics,
    test_metrics,
    len(feature_names),
    len(train_tabular)
)

print(f"\nOverfitting Score: {diagnosis['overfitting_score']}/10")
print(f"Severity: {diagnosis['severity']}")

if diagnosis['warnings']:
    print("\nWarnings:")
    for warning in diagnosis['warnings']:
        print(f"  - {warning}")

if diagnosis['recommendations']:
    print("\nRecommendations:")
    for rec in diagnosis['recommendations']:
        print(f"  - {rec}")

# 8. Create full report
print("\n" + "=" * 60)
report = create_validation_report(
    train_metrics,
    test_metrics,  # Use as validation
    test_metrics,
    len(feature_names),
    len(train_tabular)
)
print(report)

