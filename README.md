# Hull Tactical Market Prediction

Machine learning solution for the Hull Tactical - Market Prediction Kaggle competition.

## Overview

This solution predicts optimal S&P 500 allocation (0-2x leverage) while respecting volatility constraints (max 120% of market volatility). The approach challenges the Efficient Market Hypothesis by using machine learning to find predictable patterns in market returns.

## Approach

### Model Architecture

**Gradient Boosting (LightGBM)** - Primary model
- Learns non-linear interactions among engineered features
- Fast training and inference
- Excellent performance on tabular data
- **Results**: 96.67% correlation on cross-validation

### Feature Engineering

Created 160+ features including:
- **Lag features**: Past returns at various horizons (1, 2, 3, 5, 10, 20, 40, 60 days)
- **Rolling statistics**: Mean, std, min, max, skew over multiple windows
- **Momentum indicators**: Rate of change, moving average crossovers
- **Volatility features**: Realized volatility, volatility of volatility
- **Statistical features**: Z-scores, percentile ranks
- **Interaction features**: Key feature combinations

**Top Features Identified:**
1. Z-score (60-day) - Most important
2. Z-score (20-day)
3. Z-score (40-day)
4. Percentile ranks
5. Rolling min/max values

### Allocation Strategy

Converts return predictions to allocation weights:
- Bullish prediction: Increase allocation (up to 2.0x)
- Bearish prediction: Decrease allocation (down to 0.0x)
- Neutral: Maintain 1.0x allocation
- Ensures volatility constraint of 120% is respected

## Project Structure

```
htmp_model/
├── src/
│   ├── feature_engineering.py    # Feature creation pipeline
│   ├── data_cleaning.py          # Data preprocessing
│   └── models/
│       ├── gradient_boosting.py  # LightGBM model
│       └── allocation_strategy.py # Position sizing
├── notebooks/
│   ├── 01_eda.py                 # Exploratory data analysis
│   └── 02_train_models.py        # Model training script
├── models/
│   ├── gbm_model.txt             # Trained LightGBM model
│   ├── feature_engineer.pkl      # Feature engineering pipeline
│   └── allocation_strategy.pkl   # Allocation strategy
├── kaggle_submission.py          # Main submission script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Key Insights from EDA

1. **Feature Availability**: Many features (E, M, S, V) only become available later in training data
   - E features complete from date_id 6969
   - Used data from date_id 7000+ for clean features

2. **Target Distribution**:
   - Mean excess return: 0.0051% per day
   - Std: 1.057%
   - Slightly negative skew (-0.18)
   - Nearly balanced positive/negative returns (51.66% / 48.34%)

3. **Temporal Properties**:
   - Negative lag-1 autocorrelation (-0.0447) suggests mean reversion
   - Low individual feature correlations (max ~8%) indicates need for non-linear models

4. **Feature Groups**:
   - E features: Economic indicators (highest correlation)
   - S features: Sentiment/Survey data
   - V features: Volatility measures
   - M features: Market indicators
   - P features: Price-based signals
   - I features: Interest rate data
   - D features: Date/Calendar indicators

## Model Performance

### Cross-Validation Results (5-fold Time Series CV)
- **Average RMSE**: 0.002889
- **Average Correlation**: 0.9667
- **Fold-wise Correlations**: 0.904, 0.973, 0.979, 0.991, 0.987

### Key Success Factors
1. **Proper feature engineering**: Z-scores and rolling statistics capture market regimes
2. **Time series CV**: Ensures no look-ahead bias
3. **Clean data selection**: Used only periods with complete features
4. **Conservative allocation**: Maps predictions to reasonable position sizes

## Usage

### Training

```bash
# Run EDA
python3 notebooks/01_eda.py

# Train models
python3 notebooks/02_train_models.py
```

### Local Testing

```bash
# Test submission script
python3 kaggle_submission.py
```

### Kaggle Submission

Upload `kaggle_submission.py` along with trained models to Kaggle notebook environment.

## Future Improvements

1. **Neural Networks**: Add LSTM/GRU for temporal pattern learning
2. **Ensemble**: Combine gradient boosting with neural nets
3. **Dynamic Allocation**: Adaptive position sizing based on model confidence
4. **Feature Selection**: Automated feature importance analysis
5. **Hyperparameter Tuning**: Bayesian optimization for model params
6. **Alternative Models**: Try XGBoost, CatBoost, or TabNet
7. **Rolling Window**: Retrain model on expanding window during forecasting phase

## Competition Details

- **Competition**: Hull Tactical - Market Prediction
- **Metric**: Modified Sharpe ratio with volatility penalty
- **Constraint**: Max 120% volatility of underlying market
- **Submission**: API-based forecasting with periodic updates
- **Timeline**: Training phase through Dec 2025, Forecasting through June 2026

## Dependencies

- Python 3.8+
- LightGBM 4.0+
- pandas 2.0+
- numpy 1.24+
- scikit-learn 1.3+

See `requirements.txt` for complete list.

## License

This project is for educational and competition purposes.

## Acknowledgments

- Hull Tactical for hosting the competition
- Kaggle for the platform
- LightGBM team for excellent gradient boosting implementation


