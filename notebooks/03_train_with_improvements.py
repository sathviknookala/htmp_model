"""
Advanced Training Script with All Improvements
Demonstrates: Neural Networks, Ensemble, Feature Selection, Hyperparameter Tuning
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.models import (
    GradientBoostingPredictor, 
    AllocationStrategy,
    ModelEnsemble,
    FeatureSelector,
    HyperparameterOptimizer,
    GridSearchOptimizer
)

# Optional imports
try:
    from src.models import LSTMPredictor, GRUPredictor
    NEURAL_NETS_AVAILABLE = True
except ImportError:
    NEURAL_NETS_AVAILABLE = False
    print("Warning: Neural networks not available (TensorFlow required)")

try:
    from src.models import XGBoostPredictor, CatBoostPredictor
    ALT_MODELS_AVAILABLE = True
except ImportError:
    ALT_MODELS_AVAILABLE = False
    print("Warning: Alternative models not available (XGBoost/CatBoost required)")


print("=" * 80)
print("ADVANCED MODEL TRAINING WITH IMPROVEMENTS")
print("=" * 80)

# Configuration
USE_FEATURE_SELECTION = True
USE_HYPERPARAMETER_TUNING = False  # Set to True for full optimization (slower)
USE_NEURAL_NETWORKS = NEURAL_NETS_AVAILABLE
USE_ALTERNATIVE_MODELS = ALT_MODELS_AVAILABLE
USE_ENSEMBLE = True

# Load data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')

print(f"\nLoaded training data: {train_df.shape}")

# Initialize feature engineer
print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

feature_engineer = FeatureEngineer(lookback_window=20)

# Use clean data
min_date_id = 7000
train_df_clean = train_df[train_df['date_id'] >= min_date_id].copy()
print(f"Using data from date_id {min_date_id}: {train_df_clean.shape[0]} samples")

# Prepare features
data_dict = feature_engineer.prepare_data(train_df_clean)
train_tabular = data_dict['train_tabular']
feature_names = data_dict['feature_names']

print(f"Created {len(feature_names)} features")

# Remove rows with NaN target
target_col = 'market_forward_excess_returns'
train_tabular_valid = train_tabular[train_tabular[target_col].notna()].copy()
print(f"Valid training samples: {train_tabular_valid.shape[0]}")

# ============================================================================
# FEATURE SELECTION
# ============================================================================

if USE_FEATURE_SELECTION:
    print("\n" + "=" * 80)
    print("AUTOMATED FEATURE SELECTION")
    print("=" * 80)
    
    feature_selector = FeatureSelector(
        max_features=100,  # Limit to top 100 features
        correlation_threshold=0.95
    )
    
    # First train a quick model to get feature importance
    print("\nTraining quick model for feature importance...")
    quick_gbm = GradientBoostingPredictor()
    quick_gbm.train_full(
        train_tabular_valid,
        feature_names,
        target_col=target_col,
        num_boost_round=200
    )
    
    # Perform comprehensive feature selection
    selected_features = feature_selector.comprehensive_selection(
        train_tabular_valid,
        feature_names,
        target_col,
        feature_importance=quick_gbm.feature_importance,
        use_correlation=True,
        use_variance=True,
        use_null_filter=True
    )
    
    print(f"\nSelected {len(selected_features)} features for final training")
    feature_names = selected_features
    
    # Save feature selector
    model_dir = data_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    feature_selector.save(str(model_dir / 'feature_selector.pkl'))

# ============================================================================
# HYPERPARAMETER TUNING (OPTIONAL)
# ============================================================================

best_params = None
if USE_HYPERPARAMETER_TUNING:
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING")
    print("=" * 80)
    
    try:
        optimizer = HyperparameterOptimizer(model_type='lightgbm')
        best_params = optimizer.optimize_lightgbm(
            train_tabular_valid,
            feature_names,
            target_col=target_col,
            n_iter=20,  # Reduce for faster testing
            cv_folds=3
        )
        
        # Save best params
        optimizer.save(str(model_dir / 'hyperparameter_optimizer.pkl'))
    except ImportError:
        print("Bayesian optimization not available. Using default parameters.")
        USE_HYPERPARAMETER_TUNING = False

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {}

# 1. Gradient Boosting (LightGBM)
print("\n1. Training LightGBM...")
if best_params:
    gbm_params = best_params.copy()
    gbm_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42
    })
else:
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

# Save LightGBM
gbm_model.save(str(model_dir / 'gbm_model.txt'))
print("Saved LightGBM model")

# 2. XGBoost (if available)
if USE_ALTERNATIVE_MODELS:
    print("\n2. Training XGBoost...")
    try:
        xgb_model = XGBoostPredictor()
        xgb_model.train_full(
            train_tabular_valid,
            feature_names,
            target_col=target_col,
            num_boost_round=1000
        )
        models['xgboost'] = xgb_model
        xgb_model.save(str(model_dir / 'xgb_model.json'))
        print("Saved XGBoost model")
    except Exception as e:
        print(f"XGBoost training failed: {e}")

    # 3. CatBoost (if available)
    print("\n3. Training CatBoost...")
    try:
        cat_model = CatBoostPredictor()
        cat_model.train_full(
            train_tabular_valid,
            feature_names,
            target_col=target_col,
            iterations=1000
        )
        models['catboost'] = cat_model
        cat_model.save(str(model_dir / 'catboost_model.cbm'))
        print("Saved CatBoost model")
    except Exception as e:
        print(f"CatBoost training failed: {e}")

# 4. Neural Networks (if available)
if USE_NEURAL_NETWORKS:
    print("\n4. Training Neural Networks...")
    
    # Prepare sequence data
    train_sequences, train_targets = data_dict['train_sequences']
    
    # Remove NaN targets
    valid_mask = ~np.isnan(train_targets)
    X_seq = train_sequences[valid_mask]
    y_seq = train_targets[valid_mask]
    
    # Split for validation
    split_idx = int(len(X_seq) * 0.8)
    X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
    y_train_seq, y_val_seq = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"   Sequence data: {X_train_seq.shape}")
    
    # LSTM
    try:
        print("   Training LSTM...")
        lstm_model = LSTMPredictor(
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
            lstm_units=[64, 32],
            dropout=0.2,
            learning_rate=0.001
        )
        lstm_model.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        lstm_metrics = lstm_model.evaluate(X_val_seq, y_val_seq)
        print(f"   LSTM Validation - RMSE: {lstm_metrics['rmse']:.6f}, Corr: {lstm_metrics['correlation']:.4f}")
        
        models['lstm'] = lstm_model
        lstm_model.save(str(model_dir / 'lstm_model'))
        print("   Saved LSTM model")
    except Exception as e:
        print(f"   LSTM training failed: {e}")
    
    # GRU
    try:
        print("   Training GRU...")
        gru_model = GRUPredictor(
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
            gru_units=[64, 32],
            dropout=0.2,
            learning_rate=0.001
        )
        gru_model.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate
        gru_metrics = gru_model.evaluate(X_val_seq, y_val_seq)
        print(f"   GRU Validation - RMSE: {gru_metrics['rmse']:.6f}, Corr: {gru_metrics['correlation']:.4f}")
        
        models['gru'] = gru_model
        gru_model.save(str(model_dir / 'gru_model'))
        print("   Saved GRU model")
    except Exception as e:
        print(f"   GRU training failed: {e}")

# ============================================================================
# CREATE ENSEMBLE
# ============================================================================

if USE_ENSEMBLE and len(models) > 1:
    print("\n" + "=" * 80)
    print("CREATING ENSEMBLE")
    print("=" * 80)
    
    ensemble = ModelEnsemble(models=models)
    
    # Test ensemble predictions on recent data
    recent_data = train_tabular_valid.tail(100)
    
    print(f"\nTesting ensemble on {len(recent_data)} recent samples...")
    ensemble_pred, all_preds = ensemble.predict_ensemble(
        recent_data,
        feature_names,
        strategy='confidence',
        return_all=True
    )
    
    print("\nIndividual model predictions (first 5):")
    for name, pred in all_preds.items():
        print(f"  {name}: {pred[:5]}")
    
    print(f"\nEnsemble predictions (first 5): {ensemble_pred[:5]}")
    
    # Calculate ensemble confidence
    confidence = ensemble.calculate_ensemble_confidence(all_preds)
    print(f"Ensemble confidence - Mean: {confidence.mean():.4f}, Std: {confidence.std():.4f}")
    
    # Save ensemble
    ensemble.save(str(model_dir / 'ensemble.pkl'))
    print("\nSaved ensemble configuration")

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING ARTIFACTS")
print("=" * 80)

# Save feature engineer
with open(model_dir / 'feature_engineer.pkl', 'wb') as f:
    pickle.dump(feature_engineer, f)
print(f"Saved feature engineer to {model_dir / 'feature_engineer.pkl'}")

# Save allocation strategy
allocation_strategy = AllocationStrategy(
    max_volatility_multiplier=1.2,
    lookback_window=252
)
with open(model_dir / 'allocation_strategy.pkl', 'wb') as f:
    pickle.dump(allocation_strategy, f)
print(f"Saved allocation strategy to {model_dir / 'allocation_strategy.pkl'}")

# Save model list
model_list = {
    'models': list(models.keys()),
    'feature_names': feature_names,
    'target_col': target_col
}
with open(model_dir / 'model_list.pkl', 'wb') as f:
    pickle.dump(model_list, f)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModels trained: {list(models.keys())}")
print(f"Models saved to: {model_dir}")
print(f"\nFeatures used: {len(feature_names)}")
print("\nNext steps:")
print("1. Test ensemble on validation data")
print("2. Use rolling window for adaptive retraining")
print("3. Create submission with ensemble predictions")

