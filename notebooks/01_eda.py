"""
Exploratory Data Analysis for Hull Tactical Market Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 8)

# Load data
data_dir = Path(__file__).parent.parent
train_df = pd.read_csv(data_dir / 'train.csv')
test_df = pd.read_csv(data_dir / 'test.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain date range: {train_df['date_id'].min()} to {train_df['date_id'].max()}")
print(f"Test date range: {test_df['date_id'].min()} to {test_df['date_id'].max()}")

print("\n" + "=" * 80)
print("TARGET VARIABLES")
print("=" * 80)
target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
print("\nTarget statistics:")
print(train_df[target_cols].describe())

print("\nTarget correlation:")
print(train_df[target_cols].corr())

print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)

# Group features by prefix
feature_groups = {
    'D': [col for col in train_df.columns if col.startswith('D')],
    'E': [col for col in train_df.columns if col.startswith('E')],
    'I': [col for col in train_df.columns if col.startswith('I')],
    'M': [col for col in train_df.columns if col.startswith('M')],
    'P': [col for col in train_df.columns if col.startswith('P')],
    'S': [col for col in train_df.columns if col.startswith('S')],
    'V': [col for col in train_df.columns if col.startswith('V')],
}

print("\nMissing values by feature group (TRAIN):")
for group_name, cols in feature_groups.items():
    if cols:
        missing_pct = train_df[cols].isnull().sum().sum() / (len(train_df) * len(cols)) * 100
        print(f"  {group_name}: {missing_pct:.2f}% missing")

print("\nMissing values by feature group (TEST):")
for group_name, cols in feature_groups.items():
    if cols:
        missing_pct = test_df[cols].isnull().sum().sum() / (len(test_df) * len(cols)) * 100
        print(f"  {group_name}: {missing_pct:.2f}% missing")

# Check when features become available
print("\n" + "=" * 80)
print("FEATURE AVAILABILITY TIMELINE")
print("=" * 80)

for group_name, cols in feature_groups.items():
    if cols and train_df[cols].isnull().any().any():
        # Find first row where all features in group are non-null
        non_null_mask = train_df[cols].notnull().all(axis=1)
        if non_null_mask.any():
            first_complete = train_df.loc[non_null_mask, 'date_id'].min()
            print(f"  {group_name} features complete from date_id: {first_complete}")
        else:
            print(f"  {group_name} features: NEVER COMPLETE in train")

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

# Target distribution
target = train_df['market_forward_excess_returns'].dropna()
print(f"\nTarget statistics (market_forward_excess_returns):")
print(f"  Mean: {target.mean():.6f}")
print(f"  Std: {target.std():.6f}")
print(f"  Min: {target.min():.6f}")
print(f"  Max: {target.max():.6f}")
print(f"  Skewness: {target.skew():.4f}")
print(f"  Kurtosis: {target.kurtosis():.4f}")

# Positive vs negative returns
pos_returns = (target > 0).sum()
neg_returns = (target < 0).sum()
print(f"\n  Positive returns: {pos_returns} ({pos_returns/len(target)*100:.2f}%)")
print(f"  Negative returns: {neg_returns} ({neg_returns/len(target)*100:.2f}%)")

print("\n" + "=" * 80)
print("FEATURE CORRELATIONS WITH TARGET")
print("=" * 80)

# Calculate correlations for available features
correlations = {}
for group_name, cols in feature_groups.items():
    if cols:
        # Only use rows where features are available
        valid_data = train_df[cols + ['market_forward_excess_returns']].dropna()
        if len(valid_data) > 0:
            corr = valid_data[cols].corrwith(valid_data['market_forward_excess_returns'])
            if len(corr) > 0:
                correlations[group_name] = corr.abs().mean()

print("\nMean absolute correlation with target by group:")
for group_name, corr_value in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
    print(f"  {group_name}: {corr_value:.4f}")

print("\n" + "=" * 80)
print("TOP INDIVIDUAL FEATURES")
print("=" * 80)

# Find top correlated features
all_features = []
for cols in feature_groups.values():
    all_features.extend(cols)

valid_data = train_df[all_features + ['market_forward_excess_returns']].dropna()
if len(valid_data) > 0:
    feature_corrs = valid_data[all_features].corrwith(valid_data['market_forward_excess_returns']).abs()
    top_features = feature_corrs.nlargest(15)
    print("\nTop 15 features by absolute correlation:")
    for feat, corr in top_features.items():
        print(f"  {feat}: {corr:.4f}")

print("\n" + "=" * 80)
print("TIME SERIES PROPERTIES")
print("=" * 80)

# Check for autocorrelation in target
from scipy import stats
target_series = train_df.set_index('date_id')['market_forward_excess_returns'].dropna()

# Simple lag correlations
print("\nTarget autocorrelation:")
for lag in [1, 2, 3, 5, 10, 20]:
    autocorr = target_series.autocorr(lag=lag)
    print(f"  Lag {lag}: {autocorr:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)


