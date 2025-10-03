"""
Simple Demo: Quick Examples of Each Improvement
Run this to see how each improvement works independently
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.models import GradientBoostingPredictor, AllocationStrategy

print("=" * 80)
print("SIMPLE IMPROVEMENTS DEMO")
print("=" * 80)

# Load sample data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')
train_df = train_df[train_df['date_id'] >= 7000].tail(1000)  # Use recent 1000 samples

print(f"\nLoaded {len(train_df)} samples")

# Prepare features
feature_engineer = FeatureEngineer()
data_dict = feature_engineer.prepare_data(train_df)
train_tabular = data_dict['train_tabular']
feature_names = data_dict['feature_names']
target_col = 'market_forward_excess_returns'

train_valid = train_tabular[train_tabular[target_col].notna()].copy()
print(f"Valid samples: {len(train_valid)}")
print(f"Features: {len(feature_names)}")

# ============================================================================
# 1. FEATURE SELECTION
# ============================================================================
print("\n" + "=" * 80)
print("1. FEATURE SELECTION")
print("=" * 80)

try:
    from src.models import FeatureSelector
    
    # Train quick model for importance
    quick_model = GradientBoostingPredictor()
    quick_model.train_full(train_valid, feature_names, target_col, num_boost_round=100)
    
    # Select features
    selector = FeatureSelector(max_features=50)
    selected_features = selector.comprehensive_selection(
        train_valid, feature_names, target_col,
        feature_importance=quick_model.feature_importance
    )
    
    print(f"\nReduced from {len(feature_names)} to {len(selected_features)} features")
    print(f"Top 10: {selected_features[:10]}")
    
    # Use selected features for remaining demos
    feature_names = selected_features
    
except Exception as e:
    print(f"Feature selection failed: {e}")

# ============================================================================
# 2. ALTERNATIVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("2. ALTERNATIVE MODELS")
print("=" * 80)

models = {}

# LightGBM (always available)
print("\nTraining LightGBM...")
gbm = GradientBoostingPredictor()
gbm.train_full(train_valid, feature_names, target_col, num_boost_round=200)
models['lightgbm'] = gbm
print("LightGBM trained")

# XGBoost
try:
    from src.models import XGBoostPredictor
    print("\nTraining XGBoost...")
    xgb = XGBoostPredictor()
    xgb.train_full(train_valid, feature_names, target_col, num_boost_round=200)
    models['xgboost'] = xgb
    print("XGBoost trained")
except Exception as e:
    print(f"XGBoost not available: {e}")

# CatBoost
try:
    from src.models import CatBoostPredictor
    print("\nTraining CatBoost...")
    cat = CatBoostPredictor()
    cat.train_full(train_valid, feature_names, target_col, iterations=200)
    models['catboost'] = cat
    print("CatBoost trained")
except Exception as e:
    print(f"CatBoost not available: {e}")

print(f"\nTrained {len(models)} models: {list(models.keys())}")

# ============================================================================
# 3. ENSEMBLE
# ============================================================================
print("\n" + "=" * 80)
print("3. ENSEMBLE")
print("=" * 80)

if len(models) > 1:
    try:
        from src.models import ModelEnsemble
        
        ensemble = ModelEnsemble(models=models)
        
        # Make predictions on recent data
        test_data = train_valid.tail(50)
        
        print("\nMaking predictions with ensemble...")
        ensemble_pred, all_preds = ensemble.predict_ensemble(
            test_data, feature_names, strategy='confidence', return_all=True
        )
        
        print("\nPredictions (first 5):")
        for name, pred in all_preds.items():
            print(f"  {name}: {pred[:5]}")
        print(f"  Ensemble: {ensemble_pred[:5]}")
        
        # Calculate confidence
        confidence = ensemble.calculate_ensemble_confidence(all_preds)
        print(f"\nConfidence scores - Mean: {confidence.mean():.3f}, Range: [{confidence.min():.3f}, {confidence.max():.3f}]")
        
    except Exception as e:
        print(f"Ensemble failed: {e}")
else:
    print("Need at least 2 models for ensemble")

# ============================================================================
# 4. CONFIDENCE-BASED ALLOCATION
# ============================================================================
print("\n" + "=" * 80)
print("4. CONFIDENCE-BASED ALLOCATION")
print("=" * 80)

try:
    allocator = AllocationStrategy(max_volatility_multiplier=1.2)
    
    # Example predictions with different confidence
    test_cases = [
        (0.002, 0.9, "High confidence, positive return"),
        (0.002, 0.3, "Low confidence, positive return"),
        (-0.002, 0.9, "High confidence, negative return"),
        (-0.002, 0.3, "Low confidence, negative return"),
    ]
    
    market_vol = 0.15
    
    print("\nAllocation examples:")
    print(f"{'Return':<10} {'Confidence':<12} {'Allocation':<12} {'Description':<40}")
    print("-" * 80)
    
    for ret, conf, desc in test_cases:
        alloc = allocator.confidence_based_allocation(ret, conf, market_vol)
        print(f"{ret:<10.4f} {conf:<12.2f} {alloc:<12.4f} {desc:<40}")
    
    print("\nKey insight: Higher confidence = more extreme positions (0 or 2)")
    print("             Lower confidence = closer to neutral (1)")
    
except Exception as e:
    print(f"Allocation demo failed: {e}")

# ============================================================================
# 5. NEURAL NETWORKS (if available)
# ============================================================================
print("\n" + "=" * 80)
print("5. NEURAL NETWORKS")
print("=" * 80)

try:
    from src.models import LSTMPredictor
    
    print("\nPreparing sequence data...")
    X_seq, y_seq = data_dict['train_sequences']
    
    # Remove NaN
    valid_mask = ~np.isnan(y_seq)
    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask]
    
    # Use less data for demo
    X_seq = X_seq[-500:]
    y_seq = y_seq[-500:]
    
    # Split
    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]
    
    print(f"Training LSTM on {len(X_train)} samples...")
    print("(This may take a minute...)")
    
    lstm = LSTMPredictor(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=[32, 16],
        dropout=0.2
    )
    
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,  # Few epochs for demo
        batch_size=32,
        verbose=0
    )
    
    metrics = lstm.evaluate(X_val, y_val)
    print(f"\nLSTM Results:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")
    
except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
except Exception as e:
    print(f"Neural network demo failed: {e}")

# ============================================================================
# 6. HYPERPARAMETER TUNING (demo only, not running)
# ============================================================================
print("\n" + "=" * 80)
print("6. HYPERPARAMETER TUNING")
print("=" * 80)

print("\nHyperparameter tuning example (not running in demo):")
print("""
from src.models import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(model_type='lightgbm')
best_params = optimizer.optimize_lightgbm(
    train_data=train_df,
    feature_cols=features,
    n_iter=20,  # Number of iterations
    cv_folds=3
)

# Use best params
model = GradientBoostingPredictor(params=best_params)
""")
print("\nNote: This is computationally expensive (takes 30-60 minutes)")

# ============================================================================
# 7. ROLLING WINDOW (demo only, not running)
# ============================================================================
print("\n" + "=" * 80)
print("7. ROLLING WINDOW RETRAINING")
print("=" * 80)

print("\nRolling window example (not running in demo):")
print("""
from src.models import RollingWindowTrainer

trainer = RollingWindowTrainer(
    model_factory=lambda: GradientBoostingPredictor(),
    feature_engineer=feature_engineer,
    retrain_frequency=50  # Retrain every 50 samples
)

results = trainer.predict_with_rolling_window(
    test_data=test_df,
    feature_cols=features,
    initial_train_data=train_df
)
""")
print("\nNote: Useful for live trading and adaptive learning")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DEMO COMPLETE - SUMMARY")
print("=" * 80)

print("\nWhat we demonstrated:")
print("1. Feature Selection: Reduced features from 200+ to 50")
print(f"2. Alternative Models: Trained {len(models)} models")
if len(models) > 1:
    print("3. Ensemble: Combined models with confidence scoring")
print("4. Confidence Allocation: Dynamic position sizing")
print("5. Neural Networks: LSTM for temporal patterns (if available)")
print("6. Hyperparameter Tuning: Bayesian optimization (example shown)")
print("7. Rolling Window: Adaptive retraining (example shown)")

print("\nNext steps:")
print("- Run notebooks/03_train_with_improvements.py for full training")
print("- Read IMPROVEMENTS_README.md for detailed documentation")
print("- Read QUICK_START_IMPROVEMENTS.md for quick reference")

print("\n" + "=" * 80)

