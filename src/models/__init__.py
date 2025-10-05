"""Models package for Hull Tactical Market Prediction"""

from .gradient_boosting import GradientBoostingPredictor
from .allocation_strategy import AllocationStrategy

# Optional imports with graceful fallback
try:
    from .neural_networks import LSTMPredictor, GRUPredictor
    __all_nn = ['LSTMPredictor', 'GRUPredictor']
except ImportError:
    __all_nn = []

try:
    from .alternative_models import XGBoostPredictor, CatBoostPredictor
    __all_alt = ['XGBoostPredictor', 'CatBoostPredictor']
except ImportError:
    __all_alt = []

from .ensemble import ModelEnsemble, StackedEnsemble
from .feature_selection import FeatureSelector
from .hyperparameter_tuning import HyperparameterOptimizer, GridSearchOptimizer
from .rolling_window import RollingWindowTrainer, AdaptiveWindowTrainer
from .validation_utils import (
    ValidationMetrics,
    OverfittingDetector,
    WalkForwardValidator,
    LearningCurveAnalyzer,
    PredictionAnalyzer,
    create_validation_report
)

__all__ = [
    'GradientBoostingPredictor',
    'AllocationStrategy',
    'ModelEnsemble',
    'StackedEnsemble',
    'FeatureSelector',
    'HyperparameterOptimizer',
    'GridSearchOptimizer',
    'RollingWindowTrainer',
    'AdaptiveWindowTrainer',
    'ValidationMetrics',
    'OverfittingDetector',
    'WalkForwardValidator',
    'LearningCurveAnalyzer',
    'PredictionAnalyzer',
    'create_validation_report'
] + __all_nn + __all_alt
