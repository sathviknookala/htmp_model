# Quick Validation & Anti-Overfitting Guide

## Immediate Actions You Can Take

### 1. Run Comprehensive Validation (5 minutes)
```bash
cd /Users/eshaanganguly/Documents/projects/htmp_model
python notebooks/04_comprehensive_validation.py
```

This will:
- Test your model on held-out data
- Calculate overfitting score (0-10 scale)
- Provide specific warnings and recommendations
- Save detailed validation report

### 2. Optimize Hyperparameters (15 minutes)
```bash
python notebooks/05_optimize_for_generalization.py
```

This will:
- Test 4 different regularization configurations
- Find optimal balance between performance and generalization
- Save recommended parameters
- Show comparison table

### 3. Use Recommended Parameters

After running optimization, update your training script to use the recommended parameters saved in `models/recommended_params.pkl`.

## Quick Overfitting Checks

### Red Flags (Severe Overfitting)
- ❌ Train RMSE << Test RMSE (gap > 50%)
- ❌ Train correlation > 0.9, Test correlation < 0.6
- ❌ Using 100+ features with < 2000 samples
- ❌ Model has very deep trees (depth > 10)
- ❌ No regularization (lambda_l1=0, lambda_l2=0)

### Green Flags (Good Generalization)
- ✅ Train and test performance are similar (< 20% gap)
- ✅ Correlation on test set > 0.3
- ✅ Using regularization (L1/L2)
- ✅ Early stopping is working
- ✅ Cross-validation scores are stable across folds

## Quick Fixes for Overfitting

### If Overfitting Score > 6 (Severe)
1. **Reduce features immediately**
   ```python
   # Use top 30-50 most important features only
   from src.models import FeatureSelector
   selector = FeatureSelector(max_features=50)
   selected_features = selector.select_by_importance(feature_importance)
   ```

2. **Increase regularization**
   ```python
   params = {
       'lambda_l1': 1.0,  # Increase from 0.1
       'lambda_l2': 1.0,  # Increase from 0.1
       'min_data_in_leaf': 100,  # Increase from 20
   }
   ```

3. **Reduce model complexity**
   ```python
   params = {
       'num_leaves': 15,  # Reduce from 31
       'max_depth': 5,    # Set explicit limit
   }
   ```

### If Overfitting Score 3-5 (Moderate)
1. **Tune feature fraction**
   ```python
   params = {
       'feature_fraction': 0.7,  # Reduce from 0.8
       'bagging_fraction': 0.7,  # Reduce from 0.8
   }
   ```

2. **Use ensemble**
   ```python
   from src.models import ModelEnsemble
   ensemble = ModelEnsemble(models={'lgb': model1, 'xgb': model2})
   predictions = ensemble.predict_ensemble(X, feature_names, strategy='confidence')
   ```

### If Overfitting Score < 3 (Minimal)
- You're doing well! Focus on optimization:
  1. Try hyperparameter tuning for better performance
  2. Experiment with feature engineering
  3. Consider ensemble methods

## Performance Optimization Checklist

### Data Quality
- [ ] Remove outliers (values > 3 standard deviations)
- [ ] Handle missing values properly (avoid look-ahead bias)
- [ ] Scale/normalize features if needed
- [ ] Use clean data (date_id >= 7000)

### Feature Engineering
- [ ] Select top 30-100 features by importance
- [ ] Remove highly correlated features (> 0.95)
- [ ] Remove low-variance features
- [ ] Check feature stability across CV folds

### Model Configuration
- [ ] Use Time Series Cross-Validation (not random split)
- [ ] Enable early stopping (50-100 rounds)
- [ ] Set appropriate learning rate (0.03-0.05)
- [ ] Add L1/L2 regularization
- [ ] Limit tree depth (5-8)
- [ ] Set min_data_in_leaf (30-100)

### Validation Strategy
- [ ] Keep held-out test set (10-15% of data)
- [ ] Use walk-forward validation
- [ ] Monitor train vs validation gap
- [ ] Check prediction distribution
- [ ] Analyze feature importance stability

### Ensemble (Optional but Recommended)
- [ ] Train multiple diverse models
- [ ] Use different algorithms (LightGBM, XGBoost)
- [ ] Apply confidence-based weighting
- [ ] Monitor individual model performance

## Monitoring Metrics

### Key Metrics to Track
1. **RMSE**: Lower is better, but watch train/test gap
2. **Correlation**: Higher is better (aim for > 0.3 on test)
3. **MAE**: Mean absolute error for interpretability
4. **Prediction Variance**: Should match actual variance

### Expected Performance
- **Good**: Test Correlation > 0.35, Overfitting Score < 3
- **Acceptable**: Test Correlation > 0.25, Overfitting Score < 5
- **Needs Work**: Test Correlation < 0.20 or Overfitting Score > 5

## Common Mistakes to Avoid

1. ❌ Training on full dataset without held-out test set
2. ❌ Using too many features relative to samples
3. ❌ No regularization
4. ❌ Training too many iterations without early stopping
5. ❌ Not checking train vs validation performance gap
6. ❌ Using random split instead of time series split
7. ❌ Ignoring feature importance stability
8. ❌ Setting max_depth=-1 (unlimited depth)

## Recommended Default Parameters

```python
# Conservative parameters (good starting point)
params = {
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

# Train with early stopping
gbm_model.train_with_cv(
    train_data,
    feature_names,
    target_col='market_forward_excess_returns',
    n_splits=5,
    num_boost_round=1000,
    early_stopping_rounds=50
)
```

## Next Steps After Validation

1. **If model is overfitting**: Apply fixes above, re-train, validate again
2. **If model is performing well**: Submit to Kaggle, monitor leaderboard
3. **For continuous improvement**: 
   - Try different feature combinations
   - Experiment with ensemble methods
   - Use walk-forward validation
   - Monitor performance over time

## Getting Help

If you see unexpected results:
1. Check the validation report (`validation_report.pkl`)
2. Review feature importance for unstable features
3. Compare train/test distributions
4. Check for data leakage or look-ahead bias
5. Ensure you're using time series split, not random split

## Quick Python Snippets

### Load and Check Previous Validation
```python
import pickle
with open('validation_report.pkl', 'rb') as f:
    report = pickle.load(f)

print(f"Overfitting Score: {report['overfitting_score']}/10")
print(f"Warnings: {report['warnings']}")
print(f"Recommendations: {report['recommendations']}")
```

### Use Recommended Parameters
```python
import pickle
with open('models/recommended_params.pkl', 'rb') as f:
    params = pickle.load(f)

model = GradientBoostingPredictor(params=params)
```

### Quick Feature Selection
```python
# Keep only top N features
top_n = 50
top_features = feature_importance.head(top_n)['feature'].tolist()
```

