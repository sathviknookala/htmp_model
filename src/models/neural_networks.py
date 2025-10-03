"""
Neural Network Models (LSTM/GRU) for Temporal Pattern Learning
Uses TensorFlow/Keras for sequence modeling
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural network models will not work.")


class LSTMPredictor:
    """
    LSTM model for time series prediction
    """
    
    def __init__(self, input_shape: Tuple[int, int] = None, 
                 lstm_units: List[int] = None,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Args:
            input_shape: (timesteps, features)
            lstm_units: List of LSTM layer units, e.g., [64, 32]
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        self.input_shape = input_shape
        self.lstm_units = lstm_units or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None
        
    def build_model(self):
        """Build LSTM architecture"""
        if self.input_shape is None:
            raise ValueError("input_shape must be set before building model")
        
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(
            self.lstm_units[0],
            input_shape=self.input_shape,
            return_sequences=len(self.lstm_units) > 1
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (samples, timesteps, features)
            y_train: Training targets (samples,)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Set input shape if not set
        if self.input_shape is None:
            self.input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Build model if not built
        if self.model is None:
            self.build_model()
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        corr = np.corrcoef(y, predictions)[0, 1] if len(y) > 1 else 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr
        }
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath + '.h5')
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'history': self.history
        }
        
        with open(filepath + '.metadata', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath + '.h5')
        
        with open(filepath + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        
        self.input_shape = metadata['input_shape']
        self.lstm_units = metadata['lstm_units']
        self.dropout = metadata['dropout']
        self.learning_rate = metadata['learning_rate']
        self.history = metadata.get('history')


class GRUPredictor:
    """
    GRU model for time series prediction (lighter than LSTM)
    """
    
    def __init__(self, input_shape: Tuple[int, int] = None,
                 gru_units: List[int] = None,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Args:
            input_shape: (timesteps, features)
            gru_units: List of GRU layer units, e.g., [64, 32]
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU models")
        
        self.input_shape = input_shape
        self.gru_units = gru_units or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build GRU architecture"""
        if self.input_shape is None:
            raise ValueError("input_shape must be set before building model")
        
        model = keras.Sequential()
        
        # First GRU layer
        model.add(layers.GRU(
            self.gru_units[0],
            input_shape=self.input_shape,
            return_sequences=len(self.gru_units) > 1
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional GRU layers
        for i, units in enumerate(self.gru_units[1:], 1):
            return_sequences = i < len(self.gru_units) - 1
            model.add(layers.GRU(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.dropout))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> dict:
        """Train the GRU model"""
        if self.input_shape is None:
            self.input_shape = (X_train.shape[1], X_train.shape[2])
        
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - predictions))
        corr = np.corrcoef(y, predictions)[0, 1] if len(y) > 1 else 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'correlation': corr
        }
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath + '.h5')
        
        metadata = {
            'input_shape': self.input_shape,
            'gru_units': self.gru_units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'history': self.history
        }
        
        with open(filepath + '.metadata', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, filepath: str):
        """Load model from file"""
        self.model = keras.models.load_model(filepath + '.h5')
        
        with open(filepath + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        
        self.input_shape = metadata['input_shape']
        self.gru_units = metadata['gru_units']
        self.dropout = metadata['dropout']
        self.learning_rate = metadata['learning_rate']
        self.history = metadata.get('history')

