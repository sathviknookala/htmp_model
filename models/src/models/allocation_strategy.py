"""
Allocation Strategy with Volatility Constraints
Converts return predictions to allocation weights respecting 120% volatility constraint
"""
import numpy as np
import pandas as pd
from typing import Optional


class AllocationStrategy:
    """
    Manages position sizing with volatility constraints
    """
    
    def __init__(self, max_volatility_multiplier: float = 1.2, 
                 lookback_window: int = 252,
                 target_vol: Optional[float] = None):
        """
        Args:
            max_volatility_multiplier: Maximum volatility as multiple of market (1.2 = 120%)
            lookback_window: Window for calculating realized volatility
            target_vol: Target volatility (if None, uses market volatility)
        """
        self.max_volatility_multiplier = max_volatility_multiplier
        self.lookback_window = lookback_window
        self.target_vol = target_vol
        self.market_vol_history = []
        
    def calculate_realized_volatility(self, returns: pd.Series, 
                                     window: int = 252) -> float:
        """
        Calculate realized volatility from recent returns
        Annualized (assuming 252 trading days)
        """
        if len(returns) < window:
            window = len(returns)
        
        recent_returns = returns.tail(window)
        realized_vol = recent_returns.std() * np.sqrt(252)
        
        return realized_vol
    
    def simple_allocation(self, predicted_return: float, 
                         predicted_sign_confidence: float = 1.0) -> float:
        """
        Simple allocation based on predicted return sign
        
        Args:
            predicted_return: Predicted excess return
            predicted_sign_confidence: Confidence in prediction (0-1)
            
        Returns:
            Allocation weight (0 to 2)
        """
        if predicted_return > 0:
            # Bullish: scale allocation by confidence
            allocation = 1.0 + predicted_sign_confidence  # 1.0 to 2.0
        else:
            # Bearish: reduce allocation
            allocation = 1.0 - predicted_sign_confidence  # 0.0 to 1.0
        
        # Ensure within bounds
        return np.clip(allocation, 0.0, 2.0)
    
    def kelly_allocation(self, predicted_return: float, 
                        predicted_volatility: float,
                        risk_free_rate: float = 0.0) -> float:
        """
        Kelly criterion-inspired allocation
        
        Args:
            predicted_return: Expected excess return
            predicted_volatility: Expected volatility
            risk_free_rate: Risk-free rate
            
        Returns:
            Allocation weight (0 to 2)
        """
        if predicted_volatility <= 0:
            return 1.0  # Neutral allocation
        
        # Kelly fraction: f = (expected_return - risk_free_rate) / variance
        variance = predicted_volatility ** 2
        kelly_fraction = (predicted_return - risk_free_rate) / variance
        
        # Scale to reasonable range (Kelly can be too aggressive)
        # Use fractional Kelly (e.g., 0.25 Kelly)
        allocation = 1.0 + kelly_fraction * 0.25
        
        # Ensure within bounds
        return np.clip(allocation, 0.0, 2.0)
    
    def volatility_scaled_allocation(self, predicted_return: float,
                                     market_volatility: float,
                                     position_base: float = 1.0) -> float:
        """
        Scale allocation based on volatility targeting
        
        Args:
            predicted_return: Expected excess return
            market_volatility: Current market volatility (realized)
            position_base: Base position size
            
        Returns:
            Allocation weight (0 to 2)
        """
        # Determine target volatility
        if self.target_vol is None:
            target_vol = market_volatility
        else:
            target_vol = self.target_vol
        
        # Scale position to target volatility
        if market_volatility > 0:
            vol_scalar = target_vol / market_volatility
            vol_scalar = np.clip(vol_scalar, 0.0, self.max_volatility_multiplier)
        else:
            vol_scalar = 1.0
        
        # Direction from predicted return
        if predicted_return > 0:
            direction = 1.0
        elif predicted_return < 0:
            direction = -1.0
        else:
            direction = 0.0
        
        # Base allocation scaled by volatility
        allocation = position_base + (direction * (vol_scalar - 1.0))
        
        # Ensure within bounds and volatility constraint
        allocation = np.clip(allocation, 0.0, 2.0)
        
        # Check volatility constraint
        allocation_volatility = allocation * market_volatility
        max_allowed_volatility = self.max_volatility_multiplier * market_volatility
        
        if allocation_volatility > max_allowed_volatility:
            allocation = max_allowed_volatility / market_volatility
            allocation = np.clip(allocation, 0.0, 2.0)
        
        return allocation
    
    def ensemble_allocation(self, predictions: dict,
                           market_volatility: float,
                           risk_free_rate: float = 0.0,
                           strategy: str = 'volatility_scaled') -> float:
        """
        Combine multiple model predictions into allocation
        
        Args:
            predictions: Dictionary with keys like 'gbm', 'lstm', 'gru', etc.
            market_volatility: Current market volatility
            risk_free_rate: Risk-free rate
            strategy: Allocation strategy to use
            
        Returns:
            Final allocation weight (0 to 2)
        """
        # Average predictions
        avg_prediction = np.mean(list(predictions.values()))
        
        # Calculate confidence as agreement between models
        prediction_std = np.std(list(predictions.values()))
        max_std = market_volatility  # Use market vol as reference
        confidence = 1.0 - np.clip(prediction_std / max_std, 0.0, 1.0)
        
        # Choose strategy
        if strategy == 'simple':
            allocation = self.simple_allocation(avg_prediction, confidence)
        elif strategy == 'kelly':
            allocation = self.kelly_allocation(avg_prediction, market_volatility, risk_free_rate)
        elif strategy == 'volatility_scaled':
            allocation = self.volatility_scaled_allocation(avg_prediction, market_volatility)
        else:
            # Default: weighted by confidence
            if avg_prediction > 0:
                allocation = 1.0 + confidence
            else:
                allocation = 1.0 - confidence
            allocation = np.clip(allocation, 0.0, 2.0)
        
        return allocation
    
    def adaptive_allocation(self, predicted_return: float,
                           market_returns: pd.Series,
                           recent_performance: Optional[pd.Series] = None) -> float:
        """
        Adaptive allocation that adjusts based on recent performance
        
        Args:
            predicted_return: Model prediction
            market_returns: Historical market returns
            recent_performance: Recent model performance (optional)
            
        Returns:
            Allocation weight (0 to 2)
        """
        # Calculate market volatility
        market_vol = self.calculate_realized_volatility(market_returns, self.lookback_window)
        
        # Base allocation from volatility scaling
        base_allocation = self.volatility_scaled_allocation(
            predicted_return, market_vol
        )
        
        # Adjust based on recent performance if available
        if recent_performance is not None and len(recent_performance) > 0:
            # Calculate model accuracy over recent period
            recent_accuracy = (recent_performance > 0).mean()
            
            # Scale allocation by recent accuracy
            if recent_accuracy > 0.5:
                # Model has been accurate, increase confidence
                confidence_boost = (recent_accuracy - 0.5) * 2  # 0 to 1
                if predicted_return > 0:
                    base_allocation += confidence_boost * 0.5
                else:
                    base_allocation -= confidence_boost * 0.5
            else:
                # Model has been inaccurate, reduce position
                confidence_penalty = (0.5 - recent_accuracy) * 2
                base_allocation = base_allocation * (1 - confidence_penalty * 0.5)
        
        # Ensure within bounds
        allocation = np.clip(base_allocation, 0.0, 2.0)
        
        # Final volatility check
        allocation_volatility = allocation * market_vol
        max_allowed_volatility = self.max_volatility_multiplier * market_vol
        
        if allocation_volatility > max_allowed_volatility:
            allocation = max_allowed_volatility / market_vol
            allocation = np.clip(allocation, 0.0, 2.0)
        
        return allocation
    
    def get_allocation_batch(self, predictions: np.ndarray,
                            market_returns: pd.Series,
                            risk_free_rates: pd.Series,
                            strategy: str = 'volatility_scaled') -> np.ndarray:
        """
        Get allocations for a batch of predictions
        
        Args:
            predictions: Array of predicted returns
            market_returns: Historical market returns
            risk_free_rates: Historical risk-free rates
            strategy: Allocation strategy
            
        Returns:
            Array of allocations (0 to 2)
        """
        allocations = []
        
        for i, pred in enumerate(predictions):
            # Use returns up to current point
            current_returns = market_returns.iloc[:len(market_returns) - len(predictions) + i + 1]
            
            if len(current_returns) > 0:
                market_vol = self.calculate_realized_volatility(current_returns, self.lookback_window)
                rf_rate = risk_free_rates.iloc[len(risk_free_rates) - len(predictions) + i]
                
                if strategy == 'simple':
                    alloc = self.simple_allocation(pred)
                elif strategy == 'kelly':
                    alloc = self.kelly_allocation(pred, market_vol, rf_rate)
                elif strategy == 'volatility_scaled':
                    alloc = self.volatility_scaled_allocation(pred, market_vol)
                else:
                    alloc = 1.0  # Neutral
            else:
                alloc = 1.0  # Neutral
            
            allocations.append(alloc)
        
        return np.array(allocations)


