"""
Automated Feature Selection
Uses feature importance, correlation analysis, and recursive elimination
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import pickle


class FeatureSelector:
    """
    Automated feature selection based on importance and other criteria
    """
    
    def __init__(self, max_features: Optional[int] = None, 
                 correlation_threshold: float = 0.95):
        """
        Args:
            max_features: Maximum number of features to select (None = no limit)
            correlation_threshold: Remove features with correlation above this
        """
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.feature_scores = None
        
    def select_by_importance(self, feature_importance: pd.DataFrame, 
                            top_k: Optional[int] = None) -> List[str]:
        """
        Select top K features by importance
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_k: Number of features to select (uses self.max_features if None)
            
        Returns:
            List of selected feature names
        """
        if top_k is None:
            top_k = self.max_features
        
        if top_k is not None:
            selected = feature_importance.head(top_k)['feature'].tolist()
        else:
            selected = feature_importance['feature'].tolist()
        
        return selected
    
    def remove_correlated_features(self, df: pd.DataFrame, 
                                   feature_cols: List[str]) -> List[str]:
        """
        Remove highly correlated features
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature columns to consider
            
        Returns:
            List of features with correlations removed
        """
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.where(upper_triangle)
        
        # Find features with correlation above threshold
        to_drop = set()
        for column in corr_matrix.columns:
            correlated_features = corr_matrix[column][corr_matrix[column] > self.correlation_threshold].index
            if len(correlated_features) > 0:
                # Keep the first feature, drop the rest
                to_drop.update(correlated_features)
        
        # Remove dropped features
        selected = [f for f in feature_cols if f not in to_drop]
        
        print(f"Removed {len(to_drop)} highly correlated features")
        if len(to_drop) > 0:
            print(f"Dropped features: {list(to_drop)[:10]}...")  # Show first 10
        
        return selected
    
    def select_by_variance(self, df: pd.DataFrame, feature_cols: List[str],
                          threshold: float = 0.01) -> List[str]:
        """
        Remove low-variance features
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature columns
            threshold: Minimum variance threshold
            
        Returns:
            List of features with sufficient variance
        """
        variances = df[feature_cols].var()
        selected = variances[variances > threshold].index.tolist()
        
        print(f"Removed {len(feature_cols) - len(selected)} low-variance features")
        
        return selected
    
    def select_by_null_ratio(self, df: pd.DataFrame, feature_cols: List[str],
                            max_null_ratio: float = 0.5) -> List[str]:
        """
        Remove features with too many null values
        
        Args:
            df: DataFrame containing features
            feature_cols: List of feature columns
            max_null_ratio: Maximum ratio of null values allowed
            
        Returns:
            List of features with acceptable null ratios
        """
        null_ratios = df[feature_cols].isnull().mean()
        selected = null_ratios[null_ratios <= max_null_ratio].index.tolist()
        
        print(f"Removed {len(feature_cols) - len(selected)} features with high null ratios")
        
        return selected
    
    def mutual_information_selection(self, X: pd.DataFrame, y: pd.Series,
                                    top_k: Optional[int] = None) -> List[str]:
        """
        Select features using mutual information
        
        Args:
            X: Feature DataFrame
            y: Target series
            top_k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X.fillna(0), y, random_state=42)
            
            # Create DataFrame
            mi_df = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)
            
            # Select top K
            if top_k is None:
                top_k = self.max_features or len(mi_df)
            
            selected = mi_df.head(top_k)['feature'].tolist()
            
            print(f"Selected {len(selected)} features by mutual information")
            
            return selected
            
        except ImportError:
            print("Warning: sklearn not available for mutual information. Skipping.")
            return list(X.columns)
    
    def comprehensive_selection(self, df: pd.DataFrame, feature_cols: List[str],
                               target_col: str,
                               feature_importance: Optional[pd.DataFrame] = None,
                               use_correlation: bool = True,
                               use_variance: bool = True,
                               use_null_filter: bool = True) -> List[str]:
        """
        Comprehensive feature selection using multiple criteria
        
        Args:
            df: DataFrame containing features and target
            feature_cols: List of feature columns to consider
            target_col: Target column name
            feature_importance: Optional feature importance DataFrame
            use_correlation: Whether to remove correlated features
            use_variance: Whether to remove low-variance features
            use_null_filter: Whether to remove high-null features
            
        Returns:
            List of selected features
        """
        print("=" * 60)
        print("COMPREHENSIVE FEATURE SELECTION")
        print("=" * 60)
        print(f"Starting with {len(feature_cols)} features")
        
        selected = feature_cols.copy()
        
        # Step 1: Remove high-null features
        if use_null_filter:
            print("\n1. Filtering by null ratio...")
            selected = self.select_by_null_ratio(df, selected, max_null_ratio=0.5)
            print(f"   Remaining: {len(selected)} features")
        
        # Step 2: Remove low-variance features
        if use_variance:
            print("\n2. Filtering by variance...")
            selected = self.select_by_variance(df, selected, threshold=0.01)
            print(f"   Remaining: {len(selected)} features")
        
        # Step 3: Remove correlated features
        if use_correlation:
            print("\n3. Removing highly correlated features...")
            selected = self.remove_correlated_features(df, selected)
            print(f"   Remaining: {len(selected)} features")
        
        # Step 4: Select by importance
        if feature_importance is not None:
            print("\n4. Selecting by feature importance...")
            important_features = self.select_by_importance(
                feature_importance,
                top_k=self.max_features
            )
            # Keep features that are both selected and important
            selected = [f for f in selected if f in important_features]
            print(f"   Remaining: {len(selected)} features")
        
        # Step 5: Final limit
        if self.max_features is not None and len(selected) > self.max_features:
            print(f"\n5. Limiting to top {self.max_features} features...")
            if feature_importance is not None:
                # Sort by importance
                importance_dict = dict(zip(
                    feature_importance['feature'],
                    feature_importance['importance']
                ))
                selected = sorted(
                    selected,
                    key=lambda x: importance_dict.get(x, 0),
                    reverse=True
                )[:self.max_features]
            else:
                selected = selected[:self.max_features]
            print(f"   Final: {len(selected)} features")
        
        self.selected_features = selected
        print("\n" + "=" * 60)
        print(f"FEATURE SELECTION COMPLETE: {len(selected)} features selected")
        print("=" * 60)
        
        return selected
    
    def get_feature_scores(self, df: pd.DataFrame, feature_cols: List[str],
                          target_col: str) -> pd.DataFrame:
        """
        Calculate various scores for features
        
        Returns:
            DataFrame with feature scores
        """
        scores = []
        
        for feature in feature_cols:
            feature_data = df[feature].fillna(0)
            target_data = df[target_col].dropna()
            
            # Align lengths
            min_len = min(len(feature_data), len(target_data))
            feature_data = feature_data[:min_len]
            target_data = target_data[:min_len]
            
            # Calculate metrics
            variance = feature_data.var()
            null_ratio = df[feature].isnull().mean()
            
            # Correlation with target
            if len(feature_data) > 1 and len(target_data) > 1:
                corr = np.corrcoef(feature_data, target_data)[0, 1]
                corr = 0.0 if np.isnan(corr) else corr
            else:
                corr = 0.0
            
            scores.append({
                'feature': feature,
                'variance': variance,
                'null_ratio': null_ratio,
                'target_correlation': abs(corr)
            })
        
        self.feature_scores = pd.DataFrame(scores).sort_values(
            'target_correlation',
            ascending=False
        )
        
        return self.feature_scores
    
    def save(self, filepath: str):
        """Save feature selector"""
        metadata = {
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'max_features': self.max_features,
            'correlation_threshold': self.correlation_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load feature selector"""
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.selected_features = metadata['selected_features']
        self.feature_scores = metadata['feature_scores']
        self.max_features = metadata['max_features']
        self.correlation_threshold = metadata['correlation_threshold']

