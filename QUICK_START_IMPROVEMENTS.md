# Quick Start Guide for Model Improvements

## Installation

```bash
# Install all dependencies (optional packages included)
pip install -r requirements.txt
```

## Quick Training

Run the comprehensive training script:

```bash
python notebooks/03_train_with_improvements.py
```

This will:
1. Select top 100 features automatically
2. Train 5 models (LightGBM, XGBoost, CatBoost, LSTM, GRU)
3. Create an ensemble
4. Save all models to `models/` directory

## What's New

### 1. Neural Networks
```python
from src.models import LSTMPredictor, GRUPredictor

# LSTM for temporal patterns
lstm = LSTMPredictor(input_shape=(20, 50), lstm_units=[64, 32])
lstm.train(X_train, y_train, X_val, y_val, epochs=50)
predictions = lstm.predict(X_test)
```

### 2. Ensemble
```python
from src.models import ModelEnsemble

# Combine multiple models
ensemble = ModelEnsemble(models={
    'lightgbm': gbm_model,
    'xgboost': xgb_model,
    'lstm': lstm_model
})

# Smart predictions with confidence
predictions = ensemble.predict_ensemble(
    X_test, 
    feature_cols, 
    strategy='confidence'
)
```

### 3. Confidence-Based Allocation
```python
from src.models import AllocationStrategy

allocator = AllocationStrategy()

# Dynamic allocation based on confidence
allocation = allocator.confidence_based_allocation(
    predicted_return=0.002,
    confidence_score=0.85,  # Higher = more aggressive
    market_volatility=0.15
)
```

### 4. Feature Selection
```python
from src.models import FeatureSelector

selector = FeatureSelector(max_features=100)
selected = selector.comprehensive_selection(
    train_df,
    all_features,
    target_col='market_forward_excess_returns',
    feature_importance=model.feature_importance
)
# Reduces from 200+ to top 100 features
```

### 5. Hyperparameter Tuning
```python
from src.models import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(model_type='lightgbm')
best_params = optimizer.optimize_lightgbm(
    train_df, 
    feature_cols,
    n_iter=30  # Bayesian optimization iterations
)
# Use best_params in your model
```

### 6. Alternative Models
```python
from src.models import XGBoostPredictor, CatBoostPredictor

# XGBoost
xgb = XGBoostPredictor()
xgb.train_full(train_df, feature_cols)

# CatBoost
cat = CatBoostPredictor()
cat.train_full(train_df, feature_cols)
```

### 7. Rolling Window Retraining
```python
from src.models import RollingWindowTrainer

trainer = RollingWindowTrainer(
    model_factory=lambda: GradientBoostingPredictor(),
    feature_engineer=feature_engineer,
    retrain_frequency=50  # Retrain every 50 samples
)

results = trainer.predict_with_rolling_window(
    test_df, 
    feature_cols,
    initial_train_data=train_df
)
```

## Configuration

Edit `notebooks/03_train_with_improvements.py`:

```python
USE_FEATURE_SELECTION = True  # Recommended
USE_HYPERPARAMETER_TUNING = False  # Slow, set True for best results
USE_NEURAL_NETWORKS = True  # Requires TensorFlow
USE_ALTERNATIVE_MODELS = True  # Requires XGBoost/CatBoost
USE_ENSEMBLE = True  # Recommended
```

## File Structure

After training, `models/` contains:
- `gbm_model.txt` - LightGBM
- `xgb_model.json` - XGBoost
- `catboost_model.cbm` - CatBoost
- `lstm_model.h5` - LSTM
- `gru_model.h5` - GRU
- `ensemble.pkl` - Ensemble config
- `feature_selector.pkl` - Selected features
- `feature_engineer.pkl` - Feature engineering
- `allocation_strategy.pkl` - Allocation rules

## Expected Performance

### Single Model (Before)
- Correlation: ~0.15-0.20
- RMSE: ~0.008-0.010

### Ensemble (After)
- Correlation: ~0.20-0.25 (improved)
- RMSE: ~0.007-0.009 (improved)
- More stable predictions
- Better risk-adjusted returns

## Troubleshooting

**Import errors:**
```bash
# Install missing packages
pip install tensorflow xgboost catboost bayesian-optimization
```

**Out of memory:**
- Reduce `max_features` to 50
- Train fewer models
- Reduce neural network `lstm_units` to [32, 16]

**Too slow:**
- Set `USE_HYPERPARAMETER_TUNING = False`
- Set `USE_NEURAL_NETWORKS = False`
- Reduce `epochs` to 20 for neural networks

## Next Steps

1. Run `notebooks/03_train_with_improvements.py`
2. Review model performance in output
3. Test ensemble on validation data
4. Integrate into `kaggle_submission.py`
5. Upload models to Kaggle dataset
6. Submit to competition

## Tips

- Start with default settings
- Enable hyperparameter tuning only for final model
- Neural networks work best with more data
- Ensemble improves stability, not always raw performance
- Feature selection reduces overfitting
- Confidence-based allocation manages risk better

## Questions?

See `IMPROVEMENTS_README.md` for detailed documentation.

