# Complete Model Architecture Breakdown

## 🏗️ Overall System Architecture

```
Raw Market Data → Feature Engineering → ML Model → Allocation Strategy → Portfolio Weights
     (94 features)      (160 features)     (LightGBM)     (Vol Scaling)        (0-200%)
```

## 📊 Phase 1: Feature Engineering

### Input Data (94 base features)
- **M Features**: Market microstructure (spreads, depth, imbalances)
- **P Features**: Price-based signals (momentum, mean reversion)
- **V Features**: Volume patterns (participation, flow)
- **D Features**: Derived market data
- **E Features**: Economic/external factors
- **I Features**: Intraday patterns
- **S Features**: Statistical measures

### Feature Engineering Process (94 → 160 features)

#### 1. **Lag Features** (8 features)
```python
target_lag_1, target_lag_2, target_lag_5, target_lag_10, 
target_lag_20, target_lag_40, target_lag_60
```
- **Purpose**: Capture autoregressive patterns
- **Logic**: "If market was up/down N days ago, what happens next?"

#### 2. **Rolling Statistics** (25 features)
```python
# For windows [5, 10, 20, 40, 60]:
target_rolling_mean_X    # Moving averages
target_rolling_std_X     # Volatility measures  
target_rolling_min_X     # Support levels
target_rolling_max_X     # Resistance levels
target_rolling_skew_X    # Distribution shape
```
- **Purpose**: Market regime detection
- **Logic**: "Is market trending, volatile, or stable?"

#### 3. **Momentum Features** (11 features)
```python
momentum_5 = today - 5_days_ago
momentum_20 = today - 20_days_ago
roc_10 = (today / 10_days_ago) - 1    # Rate of change
ma_cross_5_20 = MA5 - MA20            # Moving average crossover
```
- **Purpose**: Trend detection
- **Logic**: "Is market accelerating up/down?"

#### 4. **Volatility Features** (11 features)
```python
realized_vol_20 = std(returns_20) * sqrt(252)  # Annualized
vol_of_vol_20 = std(realized_vol_20)           # Volatility clustering
range_5 = (max_5 - min_5) / mean_5             # Relative range
```
- **Purpose**: Risk assessment
- **Logic**: "How risky is the market right now?"

#### 5. **Technical Indicators** (15 features)
```python
zscore_20 = (current - mean_20) / std_20       # How extreme vs normal
rank_40 = percentile_rank(current, window=40)  # Relative position
bb_position = (price - bb_lower) / (bb_upper - bb_lower)  # Bollinger Bands
williams_r = (high_14 - current) / (high_14 - low_14)     # Williams %R
```
- **Purpose**: Mean reversion signals
- **Logic**: "Is market overbought/oversold?"

## 🤖 Phase 2: Machine Learning Model (LightGBM)

### Model Architecture
```
Input: 160 features → LightGBM → Output: Predicted excess return
```

### LightGBM Configuration
```python
{
    'objective': 'regression',           # Predict continuous values
    'metric': 'rmse',                   # Minimize prediction error
    'boosting_type': 'gbdt',            # Gradient boosting decision trees
    
    # Tree Structure
    'num_leaves': 31,                   # Complexity per tree
    'max_depth': -1,                    # No depth limit (dangerous!)
    'min_data_in_leaf': 20,             # Min samples per leaf
    
    # Learning Process  
    'learning_rate': 0.05,              # How fast to learn
    'num_boost_round': 1000,            # Max trees to build
    'early_stopping_rounds': 50,        # Stop if no improvement
    
    # Regularization (Anti-overfitting)
    'feature_fraction': 0.8,            # Use 80% of features per tree
    'bagging_fraction': 0.8,            # Use 80% of data per tree  
    'lambda_l1': 0.1,                   # L1 regularization
    'lambda_l2': 0.1,                   # L2 regularization
}
```

### How LightGBM Works
1. **Builds trees sequentially** (each corrects previous mistakes)
2. **Each tree predicts residuals** (what previous trees got wrong)
3. **Final prediction** = sum of all tree outputs
4. **Feature importance** = how often/effectively features are used

### Training Process
```python
# Time Series Cross-Validation (5 folds)
Fold 1: Train[7000-7338] → Validate[7339-7677]
Fold 2: Train[7000-7677] → Validate[7678-8016] 
Fold 3: Train[7000-8016] → Validate[8017-8355]
Fold 4: Train[7000-8355] → Validate[8356-8694]
Fold 5: Train[7000-8694] → Validate[8695-8989]

# Final Model: Train on ALL data [7000-8989]
```

## 📈 Phase 3: Allocation Strategy

### Input → Output
```
Predicted Excess Return → Allocation Weight (0% to 200%)
```

### Allocation Methods

#### 1. **Simple Allocation**
```python
if predicted_return > 0:
    allocation = 1.0 + confidence    # 100-200% (bullish)
else:
    allocation = 1.0 - confidence    # 0-100% (bearish)
```

#### 2. **Volatility-Scaled Allocation**
```python
# Kelly Criterion inspired
base_allocation = predicted_return / predicted_volatility²
scaled_allocation = base_allocation * volatility_target
final_allocation = clip(scaled_allocation, 0, 2.0)
```

#### 3. **Risk-Adjusted Allocation**
```python
market_vol = std(market_returns) * sqrt(252)  # Annualized volatility
max_vol = market_vol * 1.2                    # 120% constraint
allocation = min(allocation, max_vol / market_vol)
```

## 🔄 Complete Flow Example

### Example Prediction Day
```
Day T-1: Market data comes in
   ↓
Feature Engineering: 
   - market_return_lag_1 = 0.002
   - rolling_mean_20 = 0.001  
   - momentum_5 = 0.003
   - zscore_20 = 1.5 (overbought)
   - volatility_20 = 0.15
   ↓
LightGBM Model:
   Input: [0.002, 0.001, 0.003, 1.5, 0.15, ...]
   Output: predicted_excess_return = -0.001 (bearish)
   ↓
Allocation Strategy:
   predicted_return = -0.001 (negative)
   → allocation = 1.0 - confidence = 0.3 (30% equity)
   ↓
Final Position: 30% stocks, 70% risk-free assets
```

## 🎯 Model Performance Breakdown

### What Each Component Contributes

#### **Feature Engineering (Biggest Impact)**
- Transforms raw data into predictive signals
- **Most important features** (from validation):
  1. `zscore_60` - Long-term mean reversion
  2. `zscore_20` - Short-term mean reversion  
  3. `target_rolling_min_5` - Recent lows
  4. `target_ewm_5` - Recent trend

#### **LightGBM Model**
- **Strength**: Captures complex non-linear patterns
- **Current performance**: 0.924 correlation on test set
- **Issue**: Some overfitting (train=0.992 vs test=0.924)

#### **Allocation Strategy** 
- **Purpose**: Convert predictions to portfolio weights
- **Constraint**: Maximum 120% of market volatility
- **Output range**: 0% to 200% equity allocation

## ⚙️ Model Configurations Tested

| Configuration | Test Correlation | Overfitting Score | Notes |
|--------------|------------------|-------------------|--------|
| **Baseline** | 0.924 | 5/10 | Your current setup |
| **Conservative** | 0.787 | 0/10 | Less overfit, lower performance |
| **Moderate** | 0.906 | 5/10 | Good balance |
| **High Capacity** | 0.924 | 5/10 | Same as baseline |

## 🚨 Current Issues & Solutions

### Issues Detected:
1. **Moderate overfitting** (score 5/10)
2. **Feature instability** (101 features have high variance)
3. **High train correlation** (0.992 - suspiciously perfect)

### Recommended Fixes:
1. **Use conservative parameters** (already saved to `models/recommended_params.pkl`)
2. **Reduce features** to top 50-60 most stable ones
3. **Increase regularization** (lambda_l1=0.5, lambda_l2=0.5)
4. **Use ensemble** of multiple models

## 🎮 How to Use the Models

### Training a New Model
```python
from src.feature_engineering import FeatureEngineer
from src.models import GradientBoostingPredictor, AllocationStrategy

# 1. Feature Engineering
engineer = FeatureEngineer(lookback_window=20)
data_dict = engineer.prepare_data(train_df)
features = data_dict['feature_names']

# 2. Train Model  
model = GradientBoostingPredictor()
model.train_full(train_tabular, features, 'market_forward_excess_returns')

# 3. Make Predictions
predictions = model.predict(test_data, features)

# 4. Convert to Allocations
allocator = AllocationStrategy(max_volatility_multiplier=1.2)
allocations = allocator.get_allocation_batch(
    predictions, market_returns, risk_free_rates, strategy='volatility_scaled'
)
```

This is a sophisticated system that combines traditional financial features with modern machine learning to predict market movements and size positions appropriately!
