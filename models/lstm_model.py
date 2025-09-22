"""LSTM Model Module for AI Trading System
This module defines a reusable, well-documented LSTM model class in TensorFlow/Keras
for univariate or multivariate stock time series forecasting. It includes utilities for
scaling, sequence generation, model building, compilation, training, prediction, and
saving/loading artifacts.

Author: AI Trading System Team
Created: September 2025
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model


# Set TensorFlow log level for cleaner output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class LSTMConfig:
    """Configuration dataclass for LSTMTimeSeriesModel hyperparameters and behavior."""
    # Data/sequence
    lookback: int = 60
    horizon: int = 1  # steps ahead to predict
    feature_columns: Optional[List[str]] = None  # None -> use all columns
    target_column: Optional[str] = None  # None -> last column
    shuffle_sequences: bool = False
    batch_size: int = 32

    # Scaling
    scaling: str = "minmax"  # one of {"minmax", "standard", "robust", "none"}
    scaling_range: Tuple[float, float] = (0.0, 1.0)  # for MinMaxScaler

    # Architecture
    lstm_units: List[int] = (64, 32)
    dropout_rates: List[float] = (0.2, 0.2)
    dense_units: int = 1  # output size; for multistep set to horizon
    return_sequences_last: bool = False  # if True, last LSTM also returns sequences

    # Compile/fit
    loss: str = "mse"
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "adam"
    metrics: Tuple[str, ...] = ("mae",)
    epochs: int = 50
    validation_split: float = 0.1
    patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_delta: float = 1e-4
    model_dir: str = "models/checkpoints"
    checkpoint_best_only: bool = True
    seed: int = 42


class LSTMTimeSeriesModel:
    """Stacked LSTM model for time series forecasting with scaling and utilities.

    Workflow:
    1) fit_scalers on train data; 2) make_sequences; 3) build/compile; 4) train; 5) predict.

    Example:
        cfg = LSTMConfig(lookback=60, horizon=1, lstm_units=[64, 32])
        model = LSTMTimeSeriesModel(cfg)
        model.fit_scalers(train_df)
        X_train, y_train = model.make_sequences(train_df)
        model.build(input_shape=(X_train.shape[1], X_train.shape[2]))
        history = model.train(X_train, y_train)
        preds = model.predict(X_test)
        model.save("models/lstm_baseline")
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        self.config = config or LSTMConfig()
        self.model: Optional[tf.keras.Model] = None
        self.feature_scaler = self._create_scaler(self.config.scaling)
        self.target_scaler = self._create_scaler(self.config.scaling)
        # ensure reproducibility
        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)
        # will be set after fit_scalers
        self._feature_cols_: Optional[List[str]] = None
        self._target_col_: Optional[str] = None

    # ---------------------------- Scaling ---------------------------------- #
    def _create_scaler(self, scaling: str):
        if scaling == "minmax":
            return MinMaxScaler(feature_range=self.config.scaling_range)
        if scaling == "standard":
            return StandardScaler()
        if scaling == "robust":
            return RobustScaler()
        if scaling == "none":
            return None
        raise ValueError(f"Unsupported scaling type: {scaling}")

    def fit_scalers(self, df: pd.DataFrame) -> None:
        """Fit feature and target scalers on provided DataFrame.

        Args:
            df: DataFrame containing features and target.
        """
        feat_cols, tgt_col = self._resolve_columns(df)
        features = df[feat_cols].astype(float).values
        target = df[[tgt_col]].astype(float).values
        if self.feature_scaler is not None:
            self.feature_scaler.fit(features)
        if self.target_scaler is not None:
            self.target_scaler.fit(target)
        self._feature_cols_ = feat_cols
        self._target_col_ = tgt_col

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform features and target using fitted scalers.

        Returns:
            X: scaled features (numpy array)
            y: scaled target (numpy array)
        """
        feat_cols, tgt_col = self._resolve_columns(df)
        features = df[feat_cols].astype(float).values
        target = df[[tgt_col]].astype(float).values
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)
        if self.target_scaler is not None:
            target = self.target_scaler.transform(target)
        return features, target

    # ------------------------- Sequences ----------------------------------- #
    def make_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences from a DataFrame.

        Returns:
            X_seq: shape (num_samples, lookback, num_features)
            y_seq: shape (num_samples, horizon)
        """
        X, y = self.transform(df)
        lb, hz = self.config.lookback, self.config.horizon
        X_seq, y_seq = [], []
        for i in range(lb, len(X) - hz + 1):
            X_seq.append(X[i - lb:i, :])
            y_seq.append(y[i:i + hz, 0])  # next hz targets
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        if self.config.shuffle_sequences:
            idx = np.arange(len(X_seq))
            np.random.shuffle(idx)
            X_seq, y_seq = X_seq[idx], y_seq[idx]
        return X_seq, y_seq

    # ------------------------- Model --------------------------------------- #
    def build(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build a stacked LSTM architecture with Dropout and Dense head.

        Args:
            input_shape: (timesteps, num_features)
        Returns:
            Compiled tf.keras.Model
        """
        cfg = self.config
        model = Sequential(name="lstm_time_series")
        # Add LSTM layers
        for i, units in enumerate(cfg.lstm_units):
            return_sequences = True if i < len(cfg.lstm_units) - 1 else cfg.return_sequences_last
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            # Dropout
            dr = cfg.dropout_rates[i] if i < len(cfg.dropout_rates) else 0.0
            if dr and dr > 0:
                model.add(Dropout(dr))
        # Dense output for horizon steps
        out_units = cfg.dense_units if cfg.horizon == 1 else cfg.horizon
        model.add(Dense(out_units, name="output"))
        model.compile(optimizer=cfg.optimizer, loss=cfg.loss, metrics=list(cfg.metrics))
        self.model = model
        return model

    def compile(self, optimizer: Optional[Union[str, tf.keras.optimizers.Optimizer]] = None,
                loss: Optional[str] = None, metrics: Optional[List[str]] = None) -> None:
        """Compile the existing model with optional overrides."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build(input_shape) first.")
        self.model.compile(
            optimizer=optimizer or self.config.optimizer,
            loss=loss or self.config.loss,
            metrics=metrics or list(self.config.metrics),
        )

    # ------------------------- Training ------------------------------------ #
    def _default_callbacks(self, save_prefix: str) -> List[tf.keras.callbacks.Callback]:
        os.makedirs(self.config.model_dir, exist_ok=True)
        ckpt_path = os.path.join(self.config.model_dir, f"{save_prefix}_best.keras")
        callbacks: List[tf.keras.callbacks.Callback] = [
            EarlyStopping(monitor="val_loss", patience=self.config.patience, restore_best_weights=True,
                          min_delta=self.config.min_delta, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=self.config.reduce_lr_factor,
                              patience=self.config.reduce_lr_patience, verbose=1),
            ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=self.config.checkpoint_best_only,
                            save_weights_only=False, verbose=1),
        ]
        return callbacks

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              validation_split: Optional[float] = None,
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
              save_prefix: str = "lstm_model",
              verbose: int = 1) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            X: Input sequences, shape (samples, timesteps, features)
            y: Targets, shape (samples, horizon)
            validation_split: Fraction for validation. Defaults to config.validation_split
            epochs: Training epochs. Defaults to config.epochs
            batch_size: Batch size. Defaults to config.batch_size
            callbacks: Optional list of callbacks; defaults include EarlyStopping, ReduceLROnPlateau, and Checkpoint
            save_prefix: Prefix for checkpoint filenames
            verbose: Verbosity level for fit
        Returns:
            Keras History object
        """
        if self.model is None:
            # Build automatically if not built
            self.build(input_shape=(X.shape[1], X.shape[2]))
        history = self.model.fit(
            X, y,
            validation_split=self.config.validation_split if validation_split is None else validation_split,
            epochs=self.config.epochs if epochs is None else epochs,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            callbacks=self._default_callbacks(save_prefix) if callbacks is None else callbacks,
            verbose=verbose,
            shuffle=True,
        )
        return history

    # ------------------------- Inference ----------------------------------- #
    def predict(self, X: np.ndarray, inverse_transform: bool = True) -> np.ndarray:
        """Generate predictions for provided sequences.

        Args:
            X: Input sequences, shape (samples, timesteps, features)
            inverse_transform: If True, inverse scale the predictions using target_scaler
        Returns:
            Array of predictions, shape (samples, horizon)
        """
        if self.model is None:
            raise ValueError("Model not built/loaded.")
        preds = self.model.predict(X)
        preds = np.array(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if inverse_transform and self.target_scaler is not None:
            preds = self.target_scaler.inverse_transform(preds)
        return preds

    # ------------------------- Persistence --------------------------------- #
    def save(self, export_dir: str) -> None:
        """Save model and scalers to a directory.

        Files produced:
            - {export_dir}/model.keras
            - {export_dir}/feature_scaler.json
            - {export_dir}/target_scaler.json
            - {export_dir}/config.json
            - {export_dir}/meta.json
        """
        os.makedirs(export_dir, exist_ok=True)
        if self.model is None:
            raise ValueError("No model to save.")
        # Save model
        self.model.save(os.path.join(export_dir, "model.keras"))
        # Save config
        with open(os.path.join(export_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        # Save scalers
        self._save_scaler(self.feature_scaler, os.path.join(export_dir, "feature_scaler.json"))
        self._save_scaler(self.target_scaler, os.path.join(export_dir, "target_scaler.json"))
        # Save column info
        meta = {"feature_cols": self._feature_cols_, "target_col": self._target_col_}
        with open(os.path.join(export_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, export_dir: str) -> None:
        """Load model and scalers from a directory."""
        model_path = os.path.join(export_dir, "model.keras")
        cfg_path = os.path.join(export_dir, "config.json")
        meta_path = os.path.join(export_dir, "meta.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = load_model(model_path)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg_dict = json.load(f)
            self.config = LSTMConfig(**cfg_dict)
        # Load scalers
        self.feature_scaler = self._load_scaler(os.path.join(export_dir, "feature_scaler.json"))
        self.target_scaler = self._load_scaler(os.path.join(export_dir, "target_scaler.json"))
        # Load column info
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._feature_cols_ = meta.get("feature_cols")
            self._target_col_ = meta.get("target_col")

    # ------------------------- Utilities ----------------------------------- #
    def _resolve_columns(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        feat_cols = self.config.feature_columns or list(df.columns)
        tgt_col = self.config.target_column or feat_cols[-1]
        if tgt_col not in df.columns:
            raise ValueError(f"Target column '{tgt_col}' not found in DataFrame.")
        feat_cols = [c for c in feat_cols if c != tgt_col]
        return feat_cols, tgt_col

    def _save_scaler(self, scaler, path: str) -> None:
        if scaler is None:
            return
        payload: Dict[str, Any] = {"class": scaler.__class__.__name__}
        # dump essential attributes if present
        for attr in ("min_", "scale_", "data_min_", "data_max_", "data_range_", "mean_", "var_"):
            val = getattr(scaler, attr, None)
            if val is not None:
                if hasattr(val, "tolist"):
                    val = val.tolist()
                payload[attr] = val
        with open(path, "w") as f:
            json.dump(payload, f)

    def _load_scaler(self, path: str):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            payload = json.load(f)
        cls = payload.get("class")
        if cls == "MinMaxScaler":
            scaler = MinMaxScaler(feature_range=self.config.scaling_range)
        elif cls == "StandardScaler":
            scaler = StandardScaler()
        elif cls == "RobustScaler":
            scaler = RobustScaler()
        else:
            return None
        # restore attributes if available
        for attr, val in payload.items():
            if attr == "class":
                continue
            setattr(scaler, attr, np.array(val) if isinstance(val, list) else val)
        return scaler


__all__ = ["LSTMConfig", "LSTMTimeSeriesModel"]
