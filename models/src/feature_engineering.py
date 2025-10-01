"""
Feature Engineering for Hull Tactical Market Prediction
Creates both tabular features (for gradient boosting) and sequences (for neural nets)
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class FeatureEngineer:
    """
    Handles feature engineering for both gradient boosting and neural networks
    """
    
    def __init__(self, lookback_window: int = 20):
        """
        Args:
            lookback_window: Number of time steps to look back for sequence models
        """
        self.lookback_window = lookback_window
        self.feature_cols = None
        self.rolling_windows = [5, 10, 20, 40, 60]
        
    def get_base_features(self, df: pd.DataFrame) -> list:
        """Get list of base feature columns"""
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored',
                       'lagged_forward_returns', 'lagged_risk_free_rate',
                       'lagged_market_forward_excess_returns']
        return [col for col in df.columns if col not in exclude_cols]
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'market_forward_excess_returns',
                           lags: list = [1, 2, 3, 5, 10, 20, 40, 60]) -> pd.DataFrame:
        """Create lagged features of the target"""
        df = df.copy()
        
        for lag in lags:
            df[f'target_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                               target_col: str = 'market_forward_excess_returns') -> pd.DataFrame:
        """Create rolling statistics features"""
        df = df.copy()
        
        for window in self.rolling_windows:
            # Rolling mean
            df[f'target_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            
            # Rolling std (volatility)
            df[f'target_rolling_std_{window}'] = df[target_col].rolling(window).std()
            
            # Rolling min/max
            df[f'target_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'target_rolling_max_{window}'] = df[target_col].rolling(window).max()
            
            # Rolling skew
            df[f'target_rolling_skew_{window}'] = df[target_col].rolling(window).skew()
            
            # Exponential weighted moving average
            df[f'target_ewm_{window}'] = df[target_col].ewm(span=window).mean()
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame,
                                 target_col: str = 'market_forward_excess_returns') -> pd.DataFrame:
        """Create momentum and trend features"""
        df = df.copy()
        
        # Simple momentum (difference from past)
        for window in [5, 10, 20, 40, 60]:
            df[f'momentum_{window}'] = df[target_col] - df[target_col].shift(window)
        
        # Rate of change
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = df[target_col].pct_change(periods=window)
        
        # Moving average crossovers
        df['ma_cross_5_20'] = df[target_col].rolling(5).mean() - df[target_col].rolling(20).mean()
        df['ma_cross_10_40'] = df[target_col].rolling(10).mean() - df[target_col].rolling(40).mean()
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame,
                                   target_col: str = 'market_forward_excess_returns') -> pd.DataFrame:
        """Create volatility-based features"""
        df = df.copy()
        
        # Realized volatility at different windows
        for window in [5, 10, 20, 40, 60]:
            df[f'realized_vol_{window}'] = df[target_col].rolling(window).std() * np.sqrt(252)
        
        # Volatility of volatility
        df['vol_of_vol_20'] = df['realized_vol_20'].rolling(20).std()
        
        # High-low range as proxy for intraday volatility
        for window in [5, 10, 20]:
            rolling_max = df[target_col].rolling(window).max()
            rolling_min = df[target_col].rolling(window).min()
            df[f'range_{window}'] = rolling_max - rolling_min
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame,
                                    target_col: str = 'market_forward_excess_returns') -> pd.DataFrame:
        """Create statistical features"""
        df = df.copy()
        
        # Z-scores relative to rolling windows
        for window in [20, 40, 60]:
            rolling_mean = df[target_col].rolling(window).mean()
            rolling_std = df[target_col].rolling(window).std()
            df[f'zscore_{window}'] = (df[target_col] - rolling_mean) / (rolling_std + 1e-8)
        
        # Percentile ranks
        for window in [20, 40, 60]:
            df[f'rank_{window}'] = df[target_col].rolling(window).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
            )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between base features"""
        df = df.copy()
        
        # Get base features that are available
        base_features = self.get_base_features(df)
        
        # Only create interactions for top features to avoid explosion
        top_features = ['P10', 'P8', 'M3', 'P2', 'V13', 'E3', 'E14']
        available_top_features = [f for f in top_features if f in df.columns]
        
        # Create some key interactions
        if 'P10' in df.columns and 'P8' in df.columns:
            df['P10_x_P8'] = df['P10'] * df['P8']
        
        if 'M3' in df.columns and 'V13' in df.columns:
            df['M3_x_V13'] = df['M3'] * df['V13']
        
        if 'E3' in df.columns and 'E14' in df.columns:
            df['E3_x_E14'] = df['E3'] * df['E14']
        
        return df
    
    def create_all_tabular_features(self, df: pd.DataFrame, 
                                   target_col: str = 'market_forward_excess_returns') -> pd.DataFrame:
        """
        Create all tabular features for gradient boosting models
        """
        df = df.copy()
        
        # Create temporal features
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_momentum_features(df, target_col)
        df = self.create_volatility_features(df, target_col)
        df = self.create_statistical_features(df, target_col)
        df = self.create_interaction_features(df)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame,
                        feature_cols: list,
                        target_col: str = 'market_forward_excess_returns') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequences for neural network models
        
        Returns:
            X: Array of shape (samples, timesteps, features)
            y: Array of shape (samples,)
        """
        df = df.copy()
        
        # Fill missing values
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(0)
        
        X_sequences = []
        y_targets = []
        
        for i in range(self.lookback_window, len(df)):
            # Get sequence of features
            sequence = df[feature_cols].iloc[i - self.lookback_window:i].values
            X_sequences.append(sequence)
            
            # Get target
            if target_col in df.columns and pd.notna(df[target_col].iloc[i]):
                y_targets.append(df[target_col].iloc[i])
            else:
                y_targets.append(np.nan)
        
        X = np.array(X_sequences)
        y = np.array(y_targets)
        
        return X, y
    
    def prepare_data(self, train_df: pd.DataFrame, 
                    test_df: Optional[pd.DataFrame] = None,
                    target_col: str = 'market_forward_excess_returns') -> dict:
        """
        Prepare both tabular and sequence data
        
        Returns:
            Dictionary with keys:
                - 'train_tabular': DataFrame with engineered features
                - 'test_tabular': DataFrame with engineered features (if test_df provided)
                - 'train_sequences': Tuple of (X, y) for neural nets
                - 'test_sequences': Tuple of (X, y) for neural nets (if test_df provided)
                - 'feature_names': List of tabular feature names
                - 'sequence_feature_names': List of features used in sequences
        """
        # Create tabular features
        train_tabular = self.create_all_tabular_features(train_df, target_col)
        
        # Get base features for sequences
        base_features = self.get_base_features(train_df)
        
        # Store feature names
        tabular_feature_names = [col for col in train_tabular.columns 
                                if col not in ['date_id', target_col, 'forward_returns', 
                                             'risk_free_rate', 'is_scored',
                                             'lagged_forward_returns', 'lagged_risk_free_rate',
                                             'lagged_market_forward_excess_returns']]
        
        # Create sequences for neural nets
        train_sequences = self.create_sequences(train_df, base_features, target_col)
        
        result = {
            'train_tabular': train_tabular,
            'train_sequences': train_sequences,
            'feature_names': tabular_feature_names,
            'sequence_feature_names': base_features
        }
        
        # Process test data if provided
        if test_df is not None:
            # Concatenate for proper rolling calculations
            combined = pd.concat([train_df, test_df], ignore_index=False)
            combined_tabular = self.create_all_tabular_features(combined, target_col)
            
            # Split back
            test_tabular = combined_tabular.loc[test_df.index]
            
            # Create test sequences
            test_sequences = self.create_sequences(test_df, base_features, target_col)
            
            result['test_tabular'] = test_tabular
            result['test_sequences'] = test_sequences
        
        return result


