"""
Optimize Model Hyperparameters for Best Generalization
Tests different regularization levels to find optimal balance
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.models import GradientBoostingPredictor
from src.models.validation_utils import ValidationMetrics, OverfittingDetector

print("=" * 80)
print("HYPERPARAMETER OPTIMIZATION FOR GENERALIZATION")
print("=" * 80)

# Configuration
MIN_DATE_ID = 7000
HELD_OUT_TEST_RATIO = 0.15
CV_FOLDS = 3

# Load data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')

# Prepare data
train_df_clean = train_df[train_df['date_id'] >= MIN_DATE_ID].copy()
n_samples = len(train_df_clean)
split_idx = int(n_samples * (1 - HELD_OUT_TEST_RATIO))

train_data = train_df_clean.iloc[:split_idx]
test_data = train_df_clean.iloc[split_idx:]

# Feature engineering
feature_engineer = FeatureEngineer(lookback_window=20)
train_dict = feature_engineer.prepare_data(train_data)
test_dict = feature_engineer.prepare_data(test_data)

train_tabular = train_dict['train_tabular']
test_tabular = test_dict['train_tabular']
feature_names = train_dict['feature_names']

target_col = 'market_forward_excess_returns'
train_tabular = train_tabular[train_tabular[target_col].notna()].copy()
test_tabular = test_tabular[test_tabular[target_col].notna()].copy()

print(f"\nTrain samples: {len(train_tabular)}")
print(f"Test samples: {len(test_tabular)}")
print(f"Features: {len(feature_names)}")

# ============================================================================
# TEST DIFFERENT CONFIGURATIONS
# ============================================================================

print("\n" + "=" * 80)
print("TESTING DIFFERENT REGULARIZATION CONFIGURATIONS")
print("=" * 80)

configurations = [
    {
        'name': 'Baseline (Current)',
        'params': {
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
    },
    {
        'name': 'Conservative (Anti-Overfit)',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 20,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'max_depth': 6,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'min_gain_to_split': 0.01,
            'verbose': -1,
            'seed': 42
        }
    },
    {
        'name': 'Moderate Regularization',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 25,
            'learning_rate': 0.04,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 30,
            'lambda_l1': 0.3,
            'lambda_l2': 0.3,
            'verbose': -1,
            'seed': 42
        }
    },
    {
        'name': 'High Capacity (Risk Overfit)',
        'params': {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.08,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 10,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'verbose': -1,
            'seed': 42
        }
    }
]

results = []
detector = OverfittingDetector()

for config in configurations:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")
    
    # Train with CV
    model = GradientBoostingPredictor(params=config['params'])
    
    # Quick CV to get validation performance
    X = train_tabular[feature_names].fillna(method='ffill').fillna(0)
    y = train_tabular[target_col]
    
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_cv = train_tabular.iloc[train_idx]
        X_val_cv = train_tabular.iloc[val_idx]
        
        # Train
        model_temp = GradientBoostingPredictor(params=config['params'])
        model_temp.train_full(
            X_train_cv,
            feature_names,
            target_col=target_col,
            num_boost_round=500
        )
        
        # Predict on validation
        y_val_pred = model_temp.predict(X_val_cv, feature_names)
        y_val_true = X_val_cv[target_col].values
        
        metrics = ValidationMetrics.all_metrics(y_val_true, y_val_pred)
        cv_scores.append(metrics)
    
    # Average CV scores
    avg_cv_metrics = {
        'rmse': np.mean([s['rmse'] for s in cv_scores]),
        'correlation': np.mean([s['correlation'] for s in cv_scores])
    }
    
    # Train on full training set
    model.train_full(
        train_tabular,
        feature_names,
        target_col=target_col,
        num_boost_round=500
    )
    
    # Evaluate on train
    train_pred = model.predict(train_tabular, feature_names)
    train_actual = train_tabular[target_col].values
    train_metrics = ValidationMetrics.all_metrics(train_actual, train_pred)
    
    # Evaluate on test
    test_pred = model.predict(test_tabular, feature_names)
    test_actual = test_tabular[target_col].values
    test_metrics = ValidationMetrics.all_metrics(test_actual, test_pred)
    
    # Detect overfitting
    diagnosis = detector.diagnose(
        train_metrics,
        test_metrics,
        len(feature_names),
        len(train_tabular)
    )
    
    # Store results
    result = {
        'name': config['name'],
        'params': config['params'],
        'cv_metrics': avg_cv_metrics,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'overfitting_score': diagnosis['overfitting_score'],
        'severity': diagnosis['severity']
    }
    results.append(result)
    
    # Print summary
    print(f"\nResults for {config['name']}:")
    print(f"  CV:    RMSE={avg_cv_metrics['rmse']:.6f}, Corr={avg_cv_metrics['correlation']:.4f}")
    print(f"  Train: RMSE={train_metrics['rmse']:.6f}, Corr={train_metrics['correlation']:.4f}")
    print(f"  Test:  RMSE={test_metrics['rmse']:.6f}, Corr={test_metrics['correlation']:.4f}")
    print(f"  Overfitting Score: {diagnosis['overfitting_score']}/10 ({diagnosis['severity']})")

# ============================================================================
# COMPARE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON OF ALL CONFIGURATIONS")
print("=" * 80)

# Create comparison table
comparison_df = pd.DataFrame([
    {
        'Configuration': r['name'],
        'CV_RMSE': r['cv_metrics']['rmse'],
        'CV_Corr': r['cv_metrics']['correlation'],
        'Test_RMSE': r['test_metrics']['rmse'],
        'Test_Corr': r['test_metrics']['correlation'],
        'Overfit_Score': r['overfitting_score'],
        'Severity': r['severity']
    }
    for r in results
])

print("\n" + comparison_df.to_string(index=False))

# Find best configuration
best_by_test_corr = comparison_df.loc[comparison_df['Test_Corr'].idxmax()]
best_by_overfit = comparison_df.loc[comparison_df['Overfit_Score'].idxmin()]

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print(f"\n🏆 Best Test Correlation:")
print(f"   Configuration: {best_by_test_corr['Configuration']}")
print(f"   Test Correlation: {best_by_test_corr['Test_Corr']:.4f}")
print(f"   Overfitting Score: {best_by_test_corr['Overfit_Score']}/10")

print(f"\n✅ Best Generalization (Least Overfitting):")
print(f"   Configuration: {best_by_overfit['Configuration']}")
print(f"   Test Correlation: {best_by_overfit['Test_Corr']:.4f}")
print(f"   Overfitting Score: {best_by_overfit['Overfit_Score']}/10")

# Calculate balance score (performance + generalization)
comparison_df['Balance_Score'] = (
    comparison_df['Test_Corr'] * 10 - comparison_df['Overfit_Score']
)
best_balance = comparison_df.loc[comparison_df['Balance_Score'].idxmax()]

print(f"\n⚖️  Best Balance (Performance + Generalization):")
print(f"   Configuration: {best_balance['Configuration']}")
print(f"   Test Correlation: {best_balance['Test_Corr']:.4f}")
print(f"   Overfitting Score: {best_balance['Overfit_Score']}/10")
print(f"   Balance Score: {best_balance['Balance_Score']:.2f}")

# Print optimal parameters
best_result = [r for r in results if r['name'] == best_balance['Configuration']][0]
print(f"\n💡 RECOMMENDED PARAMETERS:")
print(f"   Configuration: {best_balance['Configuration']}")
print("\nParameters to use:")
for key, value in best_result['params'].items():
    if key not in ['objective', 'metric', 'boosting_type', 'verbose', 'seed']:
        print(f"   {key}: {value}")

# Save results
results_path = data_dir / 'optimization_results.pkl'
with open(results_path, 'wb') as f:
    pickle.dump({
        'results': results,
        'comparison': comparison_df,
        'best_config': best_result
    }, f)

print(f"\n\nResults saved to: {results_path}")

# Save recommended configuration
recommended_config_path = data_dir / 'models' / 'recommended_params.pkl'
with open(recommended_config_path, 'wb') as f:
    pickle.dump(best_result['params'], f)

print(f"Recommended parameters saved to: {recommended_config_path}")

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Use recommended parameters for final training")
print("2. Consider ensemble with multiple configurations")
print("3. Monitor performance on Kaggle leaderboard")

