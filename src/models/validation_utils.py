"""
Validation Utilities for Model Evaluation and Overfitting Detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
import warnings


class ValidationMetrics:
    """Calculate various validation metrics"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson Correlation"""
        return np.corrcoef(y_true, y_pred)[0, 1]
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared Score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics"""
        return {
            'rmse': ValidationMetrics.rmse(y_true, y_pred),
            'mae': ValidationMetrics.mae(y_true, y_pred),
            'correlation': ValidationMetrics.correlation(y_true, y_pred),
            'r2': ValidationMetrics.r2_score(y_true, y_pred)
        }


class OverfittingDetector:
    """Detect overfitting in models"""
    
    def __init__(self):
        self.thresholds = {
            'rmse_gap_severe': 0.5,
            'rmse_gap_moderate': 0.3,
            'rmse_gap_mild': 0.15,
            'corr_gap_severe': 0.3,
            'corr_gap_moderate': 0.15,
            'corr_gap_mild': 0.08,
            'train_corr_high': 0.9,
            'feature_sample_ratio_high': 0.1
        }
    
    def diagnose(self, train_metrics: Dict[str, float], 
                 val_metrics: Dict[str, float],
                 n_features: int,
                 n_samples: int) -> Dict:
        """
        Diagnose overfitting based on train/validation metrics
        
        Returns:
            Dictionary with overfitting score, warnings, and recommendations
        """
        score = 0
        warnings_list = []
        recommendations = []
        
        # Calculate gaps
        rmse_gap = (val_metrics['rmse'] - train_metrics['rmse']) / train_metrics['rmse']
        corr_gap = train_metrics['correlation'] - val_metrics['correlation']
        
        # Check RMSE gap
        if rmse_gap > self.thresholds['rmse_gap_severe']:
            score += 3
            warnings_list.append("SEVERE: Test RMSE is 50%+ higher than train RMSE")
            recommendations.append("Increase regularization (lambda_l1, lambda_l2)")
            recommendations.append("Reduce model complexity (num_leaves, max_depth)")
        elif rmse_gap > self.thresholds['rmse_gap_moderate']:
            score += 2
            warnings_list.append("MODERATE: Test RMSE is 30%+ higher than train RMSE")
            recommendations.append("Consider adding more regularization")
        elif rmse_gap > self.thresholds['rmse_gap_mild']:
            score += 1
            warnings_list.append("MILD: Test RMSE is 15%+ higher than train RMSE")
        
        # Check correlation gap
        if corr_gap > self.thresholds['corr_gap_severe']:
            score += 3
            warnings_list.append("SEVERE: Correlation drops by 0.3+ from train to test")
            recommendations.append("Use fewer features or feature selection")
        elif corr_gap > self.thresholds['corr_gap_moderate']:
            score += 2
            warnings_list.append("MODERATE: Correlation drops by 0.15+ from train to test")
            recommendations.append("Consider ensemble methods to improve generalization")
        elif corr_gap > self.thresholds['corr_gap_mild']:
            score += 1
            warnings_list.append("MILD: Correlation drops by 0.08+ from train to test")
        
        # Check training performance
        if train_metrics['correlation'] > self.thresholds['train_corr_high']:
            score += 2
            warnings_list.append("CONCERN: Very high training correlation (>0.9)")
            recommendations.append("Model may be memorizing training data")
        
        # Check feature to sample ratio
        feature_sample_ratio = n_features / n_samples
        if feature_sample_ratio > self.thresholds['feature_sample_ratio_high']:
            score += 2
            warnings_list.append(f"CONCERN: High feature-to-sample ratio ({feature_sample_ratio:.2%})")
            recommendations.append("Reduce features using feature selection")
        
        return {
            'overfitting_score': score,
            'severity': self._get_severity(score),
            'warnings': warnings_list,
            'recommendations': recommendations,
            'metrics': {
                'rmse_gap': rmse_gap,
                'corr_gap': corr_gap,
                'feature_sample_ratio': feature_sample_ratio
            }
        }
    
    def _get_severity(self, score: int) -> str:
        """Get severity level from score"""
        if score >= 6:
            return "SEVERE"
        elif score >= 3:
            return "MODERATE"
        elif score > 0:
            return "MILD"
        else:
            return "MINIMAL"


class WalkForwardValidator:
    """
    Walk-forward validation for time series
    More realistic than standard cross-validation
    """
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        """
        Args:
            n_splits: Number of train/test splits
            test_size: Size of test set (if None, uses equal splits)
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        
        if self.test_size is None:
            # Equal splits
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            # Test set starts after all previous splits
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            # Train on all data before test set
            train_indices = np.arange(0, test_start)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def validate(self, model, X: pd.DataFrame, y: pd.Series, 
                 feature_cols: List[str]) -> Dict:
        """
        Perform walk-forward validation
        
        Args:
            model: Model object with train and predict methods
            X: Feature DataFrame
            y: Target Series
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with validation results
        """
        splits = self.split(X)
        results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"Walk-forward fold {i+1}/{len(splits)}")
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Combine for model training
            train_df = X_train.copy()
            train_df['target'] = y_train
            
            # Train model
            model.train_full(train_df, feature_cols, 'target', num_boost_round=500)
            
            # Predict
            y_pred = model.predict(X_test, feature_cols)
            
            # Calculate metrics
            metrics = ValidationMetrics.all_metrics(y_test.values, y_pred)
            metrics['fold'] = i + 1
            results.append(metrics)
        
        # Aggregate results
        avg_metrics = {
            'rmse': np.mean([r['rmse'] for r in results]),
            'mae': np.mean([r['mae'] for r in results]),
            'correlation': np.mean([r['correlation'] for r in results]),
            'r2': np.mean([r['r2'] for r in results])
        }
        
        std_metrics = {
            'rmse_std': np.std([r['rmse'] for r in results]),
            'correlation_std': np.std([r['correlation'] for r in results])
        }
        
        return {
            'fold_results': results,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics
        }


class LearningCurveAnalyzer:
    """Analyze learning curves to detect overfitting/underfitting"""
    
    def __init__(self):
        pass
    
    def analyze_curve(self, train_scores: List[float], 
                     val_scores: List[float]) -> Dict:
        """
        Analyze learning curve
        
        Args:
            train_scores: Training scores over iterations
            val_scores: Validation scores over iterations
            
        Returns:
            Dictionary with analysis results
        """
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)
        
        # Calculate final gap
        final_gap = val_scores[-1] - train_scores[-1]
        
        # Check if validation is improving
        recent_window = min(50, len(val_scores) // 5)
        recent_trend = val_scores[-recent_window:].mean() - val_scores[-2*recent_window:-recent_window].mean()
        
        # Detect early signs of overfitting
        overfitting_start = None
        for i in range(len(val_scores) - 1):
            if val_scores[i+1] > val_scores[i]:  # Validation score getting worse
                overfitting_start = i
                break
        
        diagnosis = {
            'final_gap': final_gap,
            'recent_trend': recent_trend,
            'overfitting_detected': final_gap > 0.1,
            'overfitting_start': overfitting_start,
            'still_improving': recent_trend < 0
        }
        
        if diagnosis['overfitting_detected']:
            diagnosis['recommendation'] = "Model is overfitting. Use early stopping or increase regularization."
        elif diagnosis['still_improving']:
            diagnosis['recommendation'] = "Model can benefit from more training iterations."
        else:
            diagnosis['recommendation'] = "Model has converged appropriately."
        
        return diagnosis


class PredictionAnalyzer:
    """Analyze prediction quality and distribution"""
    
    @staticmethod
    def analyze_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Comprehensive analysis of predictions
        
        Returns:
            Dictionary with analysis results
        """
        # Basic statistics
        residuals = y_true - y_pred
        
        analysis = {
            'residuals': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max()
            },
            'predictions': {
                'mean': y_pred.mean(),
                'std': y_pred.std(),
                'min': y_pred.min(),
                'max': y_pred.max()
            },
            'actuals': {
                'mean': y_true.mean(),
                'std': y_true.std(),
                'min': y_true.min(),
                'max': y_true.max()
            }
        }
        
        # Check for bias
        if abs(residuals.mean()) > 0.01:
            analysis['bias_warning'] = f"Model has bias: {residuals.mean():.4f}"
        
        # Check for variance issues
        pred_std_ratio = y_pred.std() / y_true.std()
        analysis['std_ratio'] = pred_std_ratio
        
        if pred_std_ratio < 0.7:
            analysis['variance_warning'] = "Predictions are too conservative (low variance)"
        elif pred_std_ratio > 1.3:
            analysis['variance_warning'] = "Predictions are too aggressive (high variance)"
        
        # Check for outliers
        outlier_threshold = 3 * residuals.std()
        outliers = np.abs(residuals) > outlier_threshold
        analysis['outlier_count'] = outliers.sum()
        analysis['outlier_ratio'] = outliers.mean()
        
        return analysis


def create_validation_report(train_metrics: Dict, val_metrics: Dict, 
                            test_metrics: Dict, n_features: int, 
                            n_samples: int) -> str:
    """
    Create a formatted validation report
    
    Returns:
        Formatted string report
    """
    detector = OverfittingDetector()
    diagnosis = detector.diagnose(train_metrics, test_metrics, n_features, n_samples)
    
    report = []
    report.append("=" * 80)
    report.append("MODEL VALIDATION REPORT")
    report.append("=" * 80)
    
    report.append("\nPerformance Metrics:")
    report.append(f"{'Metric':<20} {'Train':<15} {'Validation':<15} {'Test':<15}")
    report.append("-" * 65)
    
    for metric in ['rmse', 'mae', 'correlation', 'r2']:
        train_val = train_metrics.get(metric, 0)
        val_val = val_metrics.get(metric, 0)
        test_val = test_metrics.get(metric, 0)
        report.append(f"{metric.upper():<20} {train_val:<15.6f} {val_val:<15.6f} {test_val:<15.6f}")
    
    report.append(f"\nOverfitting Score: {diagnosis['overfitting_score']}/10")
    report.append(f"Severity: {diagnosis['severity']}")
    
    if diagnosis['warnings']:
        report.append("\nWarnings:")
        for warning in diagnosis['warnings']:
            report.append(f"  - {warning}")
    
    if diagnosis['recommendations']:
        report.append("\nRecommendations:")
        for rec in diagnosis['recommendations']:
            report.append(f"  - {rec}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

