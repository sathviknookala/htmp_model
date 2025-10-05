"""
Comprehensive Model Validation Script
Detects overfitting and provides actionable recommendations
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.models import GradientBoostingPredictor

print("=" * 80)
print("COMPREHENSIVE MODEL VALIDATION & OVERFITTING DETECTION")
print("=" * 80)

# Configuration
MIN_DATE_ID = 7000
N_CV_FOLDS = 5
HELD_OUT_TEST_RATIO = 0.15

# Load data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')

print(f"\nTotal data: {train_df.shape}")

# Filter to clean data
train_df_clean = train_df[train_df['date_id'] >= MIN_DATE_ID].copy()
print(f"Clean data (date_id >= {MIN_DATE_ID}): {train_df_clean.shape[0]} samples")

# Split into train and held-out test set
n_samples = len(train_df_clean)
split_idx = int(n_samples * (1 - HELD_OUT_TEST_RATIO))

train_data = train_df_clean.iloc[:split_idx]
test_data = train_df_clean.iloc[split_idx:]

print(f"\nTrain set: {len(train_data)} samples")
print(f"Held-out test set: {len(test_data)} samples")

# Feature engineering
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

feature_engineer = FeatureEngineer(lookback_window=20)
train_dict = feature_engineer.prepare_data(train_data)
test_dict = feature_engineer.prepare_data(test_data)

train_tabular = train_dict['train_tabular']
test_tabular = test_dict['train_tabular']
feature_names = train_dict['feature_names']

# Remove NaN targets
target_col = 'market_forward_excess_returns'
train_tabular = train_tabular[train_tabular[target_col].notna()].copy()
test_tabular = test_tabular[test_tabular[target_col].notna()].copy()

print(f"Features: {len(feature_names)}")
print(f"Train samples: {len(train_tabular)}")
print(f"Test samples: {len(test_tabular)}")

# ============================================================================
# VALIDATION 1: TIME SERIES CROSS-VALIDATION
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION 1: TIME SERIES CROSS-VALIDATION")
print("=" * 80)

# Test with default parameters
gbm_model = GradientBoostingPredictor()
cv_results = gbm_model.train_with_cv(
    train_tabular,
    feature_names,
    target_col=target_col,
    n_splits=N_CV_FOLDS,
    num_boost_round=1000,
    early_stopping_rounds=50
)

print("\nCross-Validation Results:")
for fold_result in cv_results['cv_scores']:
    print(f"Fold {fold_result['fold']}: RMSE={fold_result['rmse']:.6f}, Corr={fold_result['correlation']:.4f}")

avg_rmse = cv_results['avg_rmse']
avg_corr = cv_results['avg_correlation']
rmse_std = np.std([s['rmse'] for s in cv_results['cv_scores']])
corr_std = np.std([s['correlation'] for s in cv_results['cv_scores']])

print(f"\nAverage: RMSE={avg_rmse:.6f} ± {rmse_std:.6f}, Corr={avg_corr:.4f} ± {corr_std:.4f}")

# Check fold stability
if rmse_std / avg_rmse > 0.2:
    print("⚠️  WARNING: High variance across folds - model may be unstable!")
if corr_std > 0.1:
    print("⚠️  WARNING: Correlation varies significantly across folds!")

# ============================================================================
# VALIDATION 2: TRAIN VS VALIDATION PERFORMANCE
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION 2: OVERFITTING DETECTION (Train vs Validation)")
print("=" * 80)

# Train final model on training set
gbm_model.train_full(
    train_tabular,
    feature_names,
    target_col=target_col,
    num_boost_round=1000
)

# Evaluate on train set
train_pred = gbm_model.predict(train_tabular, feature_names)
train_actual = train_tabular[target_col].values

train_rmse = np.sqrt(np.mean((train_actual - train_pred) ** 2))
train_corr = np.corrcoef(train_actual, train_pred)[0, 1]
train_mae = np.mean(np.abs(train_actual - train_pred))

# Evaluate on held-out test set
test_pred = gbm_model.predict(test_tabular, feature_names)
test_actual = test_tabular[target_col].values

test_rmse = np.sqrt(np.mean((test_actual - test_pred) ** 2))
test_corr = np.corrcoef(test_actual, test_pred)[0, 1]
test_mae = np.mean(np.abs(test_actual - test_pred))

print("\nPerformance Metrics:")
print(f"{'Metric':<20} {'Train':<15} {'Test':<15} {'Gap':<15}")
print("-" * 65)
print(f"{'RMSE':<20} {train_rmse:<15.6f} {test_rmse:<15.6f} {test_rmse - train_rmse:<15.6f}")
print(f"{'Correlation':<20} {train_corr:<15.4f} {test_corr:<15.4f} {test_corr - train_corr:<15.4f}")
print(f"{'MAE':<20} {train_mae:<15.6f} {test_mae:<15.6f} {test_mae - train_mae:<15.6f}")

# Overfitting diagnosis
print("\n" + "=" * 80)
print("OVERFITTING DIAGNOSIS")
print("=" * 80)

rmse_gap = (test_rmse - train_rmse) / train_rmse
corr_gap = train_corr - test_corr

overfitting_score = 0
warnings_list = []
recommendations = []

# Check RMSE gap
if rmse_gap > 0.5:
    overfitting_score += 3
    warnings_list.append("SEVERE: Test RMSE is 50%+ higher than train RMSE")
    recommendations.append("Increase regularization (lambda_l1, lambda_l2)")
    recommendations.append("Reduce model complexity (num_leaves, max_depth)")
elif rmse_gap > 0.3:
    overfitting_score += 2
    warnings_list.append("MODERATE: Test RMSE is 30%+ higher than train RMSE")
    recommendations.append("Consider adding more regularization")
elif rmse_gap > 0.15:
    overfitting_score += 1
    warnings_list.append("MILD: Test RMSE is 15%+ higher than train RMSE")

# Check correlation gap
if corr_gap > 0.3:
    overfitting_score += 3
    warnings_list.append("SEVERE: Correlation drops by 0.3+ from train to test")
    recommendations.append("Use fewer features or feature selection")
elif corr_gap > 0.15:
    overfitting_score += 2
    warnings_list.append("MODERATE: Correlation drops by 0.15+ from train to test")
    recommendations.append("Consider ensemble methods to improve generalization")
elif corr_gap > 0.08:
    overfitting_score += 1
    warnings_list.append("MILD: Correlation drops by 0.08+ from train to test")

# Check training performance
if train_corr > 0.9:
    overfitting_score += 2
    warnings_list.append("CONCERN: Very high training correlation (>0.9)")
    recommendations.append("Model may be memorizing training data")

# Check feature to sample ratio
feature_sample_ratio = len(feature_names) / len(train_tabular)
if feature_sample_ratio > 0.1:
    overfitting_score += 2
    warnings_list.append(f"CONCERN: High feature-to-sample ratio ({feature_sample_ratio:.2%})")
    recommendations.append("Reduce features using feature selection")

print(f"\nOverfitting Score: {overfitting_score}/10")
print("0-2: Minimal overfitting")
print("3-5: Moderate overfitting")
print("6+: Severe overfitting")

if warnings_list:
    print("\n⚠️  WARNINGS:")
    for i, warning in enumerate(warnings_list, 1):
        print(f"{i}. {warning}")

if recommendations:
    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("\n✅ Model appears well-calibrated with minimal overfitting!")

# ============================================================================
# VALIDATION 3: PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION 3: PREDICTION DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\nPrediction Statistics:")
print(f"{'Dataset':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 63)
print(f"{'Train Actual':<15} {train_actual.mean():<12.6f} {train_actual.std():<12.6f} {train_actual.min():<12.6f} {train_actual.max():<12.6f}")
print(f"{'Train Pred':<15} {train_pred.mean():<12.6f} {train_pred.std():<12.6f} {train_pred.min():<12.6f} {train_pred.max():<12.6f}")
print(f"{'Test Actual':<15} {test_actual.mean():<12.6f} {test_actual.std():<12.6f} {test_actual.min():<12.6f} {test_actual.max():<12.6f}")
print(f"{'Test Pred':<15} {test_pred.mean():<12.6f} {test_pred.std():<12.6f} {test_pred.min():<12.6f} {test_pred.max():<12.6f}")

# Check for distribution shift
train_pred_std_ratio = train_pred.std() / train_actual.std()
test_pred_std_ratio = test_pred.std() / test_actual.std()

print(f"\nStd Dev Ratio (Pred/Actual):")
print(f"Train: {train_pred_std_ratio:.4f}")
print(f"Test: {test_pred_std_ratio:.4f}")

if train_pred_std_ratio < 0.7 or test_pred_std_ratio < 0.7:
    print("⚠️  WARNING: Predictions have lower variance than actuals (model is too conservative)")
elif train_pred_std_ratio > 1.3 or test_pred_std_ratio > 1.3:
    print("⚠️  WARNING: Predictions have higher variance than actuals (model is too aggressive)")
else:
    print("✅ Prediction variance is reasonable")

# ============================================================================
# VALIDATION 4: FEATURE IMPORTANCE STABILITY
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION 4: FEATURE IMPORTANCE STABILITY")
print("=" * 80)

# Get feature importance from CV models
all_importances = []
for model in cv_results['models']:
    importance = model.feature_importance(importance_type='gain')
    all_importances.append(importance)

importance_array = np.array(all_importances)
importance_mean = importance_array.mean(axis=0)
importance_std = importance_array.std(axis=0)

# Calculate coefficient of variation for each feature
cv_scores = importance_std / (importance_mean + 1e-10)

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': importance_mean,
    'importance_std': importance_std,
    'cv_score': cv_scores
}).sort_values('importance_mean', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# Check for unstable features
unstable_features = importance_df[importance_df['cv_score'] > 1.0]
if len(unstable_features) > 0:
    print(f"\n⚠️  WARNING: {len(unstable_features)} features have high instability (CV > 1.0)")
    print("These features may not generalize well:")
    print(unstable_features.head(10)[['feature', 'cv_score']].to_string(index=False))
else:
    print("\n✅ Feature importance is stable across CV folds")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY & ACTION ITEMS")
print("=" * 80)

print(f"\n📊 Performance Summary:")
print(f"  CV Performance: RMSE={avg_rmse:.6f}, Correlation={avg_corr:.4f}")
print(f"  Test Performance: RMSE={test_rmse:.6f}, Correlation={test_corr:.4f}")
print(f"  Overfitting Score: {overfitting_score}/10")

print(f"\n✅ Strengths:")
print(f"  - Using time series cross-validation")
print(f"  - Testing on held-out data")
print(f"  - Feature importance analysis")

if overfitting_score >= 6:
    print(f"\n🚨 CRITICAL: Severe overfitting detected!")
    print(f"   Immediate actions:")
    print(f"   1. Reduce features to <50 most important")
    print(f"   2. Increase regularization (lambda_l1=1.0, lambda_l2=1.0)")
    print(f"   3. Reduce num_leaves to 15-20")
    print(f"   4. Set max_depth to 5-6")
    print(f"   5. Use ensemble with diverse models")
elif overfitting_score >= 3:
    print(f"\n⚠️  MODERATE: Some overfitting present")
    print(f"   Recommended actions:")
    print(f"   1. Feature selection to reduce dimensionality")
    print(f"   2. Tune regularization parameters")
    print(f"   3. Consider ensemble methods")
else:
    print(f"\n✅ GOOD: Minimal overfitting - model is well-calibrated")
    print(f"   Optimization suggestions:")
    print(f"   1. Try hyperparameter tuning for better performance")
    print(f"   2. Experiment with ensemble methods")
    print(f"   3. Consider adding more diverse features")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

# Save validation report
report = {
    'cv_results': cv_results['cv_scores'],
    'train_metrics': {'rmse': train_rmse, 'correlation': train_corr, 'mae': train_mae},
    'test_metrics': {'rmse': test_rmse, 'correlation': test_corr, 'mae': test_mae},
    'overfitting_score': overfitting_score,
    'warnings': warnings_list,
    'recommendations': recommendations,
    'feature_importance': importance_df.to_dict()
}

report_path = data_dir / 'validation_report.pkl'
with open(report_path, 'wb') as f:
    pickle.dump(report, f)

print(f"\nValidation report saved to: {report_path}")

