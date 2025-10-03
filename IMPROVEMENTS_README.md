# Model Improvements Implementation

This document describes the advanced improvements implemented for the Hull Tactical Market Prediction model.

## Overview

The following improvements have been implemented:

1. **Neural Networks (LSTM/GRU)** - Temporal pattern learning
2. **Ensemble Framework** - Combine multiple models
3. **Dynamic Allocation** - Confidence-based position sizing
4. **Feature Selection** - Automated importance analysis
5. **Hyperparameter Tuning** - Bayesian optimization
6. **Alternative Models** - XGBoost and CatBoost
7. **Rolling Window Retraining** - Adaptive model updates

## Installation

Install additional dependencies:

```bash
pip install tensorflow>=2.13.0
pip install xgboost>=2.0.0
pip install catboost>=1.2.0
pip install bayesian-optimization>=1.4.0
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## Module Documentation

### 1. Neural Networks (`src/models/neural_networks.py`)

**Classes:**
- `LSTMPredictor`: LSTM model for temporal sequence learning
- `GRUPredictor`: GRU model (lighter alternative to LSTM)

**Usage Example:**
```python
from src.models import LSTMPredictor

# Create LSTM model
lstm = LSTMPredictor(
    input_shape=(20, 50),  # (timesteps, features)
    lstm_units=[64, 32],
    dropout=0.2,
    learning_rate=0.001
)

# Train
lstm.train(X_train, y_train, X_val, y_val, epochs=50)

# Predict
predictions = lstm.predict(X_test)

# Save/Load
lstm.save('models/lstm_model')
lstm.load('models/lstm_model')
```

### 2. Ensemble Framework (`src/models/ensemble.py`)

**Classes:**
- `ModelEnsemble`: Combine multiple models with various strategies
- `StackedEnsemble`: Use meta-learner on base model predictions

**Strategies:**
- `simple`: Equal weight averaging
- `weighted`: Custom weighted averaging
- `confidence`: Weight by model agreement
- `performance`: Weight by recent performance
- `adaptive`: Adjust weights based on errors

**Usage Example:**
```python
from src.models import ModelEnsemble

# Create ensemble
ensemble = ModelEnsemble(models={
    'lightgbm': gbm_model,
    'xgboost': xgb_model,
    'lstm': lstm_model
})

# Make predictions
predictions, confidence = ensemble.predict_ensemble(
    X_test,
    feature_cols,
    strategy='confidence',
    return_all=False
)

# Calculate confidence
confidence_scores = ensemble.calculate_ensemble_confidence(all_predictions)
```

### 3. Dynamic Allocation (`src/models/allocation_strategy.py`)

**New Method:**
- `confidence_based_allocation()`: Adjusts position size based on model confidence

**Usage Example:**
```python
from src.models import AllocationStrategy

allocator = AllocationStrategy(max_volatility_multiplier=1.2)

# Get allocation with confidence
allocation = allocator.confidence_based_allocation(
    predicted_return=0.002,
    confidence_score=0.85,
    market_volatility=0.15
)

# Ensemble allocation with confidence
allocation = allocator.ensemble_allocation(
    predictions={'gbm': 0.002, 'lstm': 0.0018, 'xgb': 0.0022},
    market_volatility=0.15,
    use_confidence=True
)
```

### 4. Feature Selection (`src/models/feature_selection.py`)

**Class:** `FeatureSelector`

**Methods:**
- `select_by_importance()`: Select top K important features
- `remove_correlated_features()`: Remove highly correlated features
- `select_by_variance()`: Remove low-variance features
- `select_by_null_ratio()`: Remove features with too many nulls
- `comprehensive_selection()`: Apply all criteria

**Usage Example:**
```python
from src.models import FeatureSelector

selector = FeatureSelector(
    max_features=100,
    correlation_threshold=0.95
)

# Comprehensive selection
selected_features = selector.comprehensive_selection(
    df=train_df,
    feature_cols=all_features,
    target_col='market_forward_excess_returns',
    feature_importance=model.feature_importance,
    use_correlation=True,
    use_variance=True,
    use_null_filter=True
)

# Save/Load
selector.save('models/feature_selector.pkl')
```

### 5. Hyperparameter Tuning (`src/models/hyperparameter_tuning.py`)

**Classes:**
- `HyperparameterOptimizer`: Bayesian optimization
- `GridSearchOptimizer`: Grid search alternative

**Usage Example:**
```python
from src.models import HyperparameterOptimizer

# Bayesian optimization
optimizer = HyperparameterOptimizer(model_type='lightgbm')
best_params = optimizer.optimize_lightgbm(
    train_data=train_df,
    feature_cols=features,
    target_col='market_forward_excess_returns',
    n_iter=30,
    cv_folds=3
)

# Use best parameters
model = GradientBoostingPredictor(params=best_params)
```

### 6. Alternative Models (`src/models/alternative_models.py`)

**Classes:**
- `XGBoostPredictor`: XGBoost gradient boosting
- `CatBoostPredictor`: CatBoost gradient boosting

**Usage Example:**
```python
from src.models import XGBoostPredictor, CatBoostPredictor

# XGBoost
xgb_model = XGBoostPredictor()
xgb_model.train_full(train_df, feature_cols, target_col)
xgb_pred = xgb_model.predict(test_df, feature_cols)

# CatBoost
cat_model = CatBoostPredictor()
cat_model.train_full(train_df, feature_cols, target_col)
cat_pred = cat_model.predict(test_df, feature_cols)
```

### 7. Rolling Window Retraining (`src/models/rolling_window.py`)

**Classes:**
- `RollingWindowTrainer`: Retrain on expanding/rolling window
- `AdaptiveWindowTrainer`: Retrain based on performance

**Usage Example:**
```python
from src.models import RollingWindowTrainer

# Create trainer
trainer = RollingWindowTrainer(
    model_factory=lambda: GradientBoostingPredictor(),
    feature_engineer=feature_engineer,
    window_type='expanding',
    min_train_size=1000,
    retrain_frequency=50
)

# Predict with retraining
results = trainer.predict_with_rolling_window(
    test_data=test_df,
    feature_cols=features,
    target_col='market_forward_excess_returns',
    initial_train_data=train_df
)
```

## Complete Training Example

See `notebooks/03_train_with_improvements.py` for a complete example that:

1. Performs feature selection (reducing from 200+ to top 100 features)
2. Optionally runs hyperparameter optimization
3. Trains multiple models (LightGBM, XGBoost, CatBoost, LSTM, GRU)
4. Creates an ensemble with confidence scoring
5. Tests dynamic allocation with confidence

Run the script:

```bash
python notebooks/03_train_with_improvements.py
```

## Performance Comparison

### Before Improvements
- Single LightGBM model
- Fixed hyperparameters
- All features used
- Simple volatility scaling

### After Improvements
- Ensemble of 5 models
- Optimized hyperparameters
- Top 100 selected features
- Confidence-based allocation

**Expected Benefits:**
- Better prediction accuracy (ensemble)
- Reduced overfitting (feature selection)
- Improved parameters (optimization)
- More robust allocation (confidence)
- Adaptive learning (rolling window)

## Configuration Options

In `03_train_with_improvements.py`, you can configure:

```python
USE_FEATURE_SELECTION = True  # Automated feature selection
USE_HYPERPARAMETER_TUNING = False  # Bayesian optimization (slow)
USE_NEURAL_NETWORKS = True  # LSTM/GRU models
USE_ALTERNATIVE_MODELS = True  # XGBoost/CatBoost
USE_ENSEMBLE = True  # Combine all models
```

## Model Files

After training, the following files are saved to `models/`:

- `gbm_model.txt` - LightGBM model
- `xgb_model.json` - XGBoost model
- `catboost_model.cbm` - CatBoost model
- `lstm_model.h5` - LSTM neural network
- `gru_model.h5` - GRU neural network
- `ensemble.pkl` - Ensemble configuration
- `feature_selector.pkl` - Feature selector
- `feature_engineer.pkl` - Feature engineer
- `allocation_strategy.pkl` - Allocation strategy
- `hyperparameter_optimizer.pkl` - Best hyperparameters
- `model_list.pkl` - List of trained models

## Integration with Kaggle Submission

To use the ensemble in your Kaggle submission:

1. Upload all model files to Kaggle dataset
2. Modify `kaggle_submission.py` to load ensemble
3. Use `ensemble.predict_ensemble()` for predictions
4. Use `allocator.confidence_based_allocation()` for allocation

Example modification:

```python
class HullTacticalPredictor:
    def __init__(self):
        self.ensemble = ModelEnsemble()
        # Load models
        self.ensemble.add_model('lightgbm', gbm_model)
        self.ensemble.add_model('xgboost', xgb_model)
        # ... load other models
        
    def predict_allocation(self, test_row):
        # Get ensemble prediction
        pred, confidence = self.ensemble.predict_ensemble(
            test_row,
            self.feature_names,
            strategy='confidence'
        )
        
        # Dynamic allocation
        allocation = self.allocator.confidence_based_allocation(
            pred,
            confidence,
            market_volatility
        )
        
        return allocation
```

## Tips and Best Practices

1. **Start Simple**: Begin with feature selection and one alternative model
2. **Gradual Improvement**: Add components one at a time
3. **Monitor Performance**: Track correlation and RMSE for each model
4. **Resource Management**: Neural networks and optimization are slow
5. **Ensemble Strategy**: Start with 'confidence' strategy
6. **Rolling Window**: Use for live trading, not initial training

## Troubleshooting

**TensorFlow not available:**
- Install: `pip install tensorflow>=2.13.0`
- Or set `USE_NEURAL_NETWORKS = False`

**XGBoost/CatBoost not available:**
- Install: `pip install xgboost catboost`
- Or set `USE_ALTERNATIVE_MODELS = False`

**Bayesian optimization slow:**
- Reduce `n_iter` parameter
- Use `GridSearchOptimizer` instead
- Or set `USE_HYPERPARAMETER_TUNING = False`

**Out of memory:**
- Reduce `max_features` in feature selection
- Use fewer models in ensemble
- Reduce neural network units
- Process data in batches

## Future Enhancements

Potential additional improvements:

1. **Attention Mechanisms**: Add transformer models
2. **Meta-Learning**: Learn ensemble weights dynamically
3. **Online Learning**: Update models with each new sample
4. **Multi-Objective Optimization**: Optimize for multiple metrics
5. **Explainability**: Add SHAP values for interpretability
6. **Risk Management**: Add drawdown constraints
7. **Market Regime Detection**: Switch models based on regime

## References

- LightGBM: https://lightgbm.readthedocs.io/
- XGBoost: https://xgboost.readthedocs.io/
- CatBoost: https://catboost.ai/
- TensorFlow/Keras: https://www.tensorflow.org/
- Bayesian Optimization: https://github.com/fmfn/BayesianOptimization

