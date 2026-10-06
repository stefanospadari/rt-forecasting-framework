# lstm_utils.py
import numpy as np
import pandas as pd
from math import ceil
from typing import Callable, Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler

import time
import random
import tensorflow as tf
from copy import deepcopy
from pathlib import Path
import pickle


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras import Model

from libs.benchmark_utils import RealTimeForecaster

# =========================
# Build model
# =========================

def baseline_lstm(window, features, output_steps, units):
    """
    Builds a simple LSTM model for forecasting time series data.

    Parameters
    ----------
    window : int
        Input sequence length.
    features : int
        Number of features per timestep.
    output_steps : int
        Output size (number of steps predicted per forward pass).
    units : int
        Number of LSTM units.

    Returns
    -------
    Model
        A Keras model instance.
    """
    return Sequential([
        Input(shape=(window, features)),
        LSTM(units),
        Dense(output_steps)
    ])


def stacked_lstm(window, features, output_steps, units):    
    """
    Builds a stacked LSTM model for forecasting time series data.

    The model is composed of two LSTM layers, the first one with return_sequences=True
    and the second one without it. The output of the second LSTM layer is then passed
    to a Dense layer with output_steps neurons.

    Parameters
    ----------
    window : int
        Input sequence length.
    features : int
        Number of features per timestep.
    output_steps : int
        Output size (number of steps predicted per forward pass).
    units : int
        Number of LSTM units.

    Returns
    -------
    Model
        A Keras model instance.
    """
    return Sequential([
        Input(shape=(window, features)),
        LSTM(units, return_sequences=True),
        LSTM(units),
        Dense(output_steps)
    ])


def lstm_dense(window, features, output_steps, units, dense_units):
    """
    Builds an LSTM model with an additional Dense layer for forecasting time series data.

    The model is composed of two LSTM layers, the first one with return_sequences=True
    and the second one without it. The output of the second LSTM layer is then passed
    to a Dense layer with dense_units neurons, and finally to another Dense layer
    with output_steps neurons.

    Parameters
    ----------
    window : int
        Input sequence length.
    features : int
        Number of features per timestep.
    output_steps : int
        Output size (number of steps predicted per forward pass).
    units : int
        Number of LSTM units.
    dense_units : int
        Number of neurons in the Dense layer.

    Returns
    -------
    Model
        A Keras model instance.
    """
    return Sequential([
        Input(shape=(window, features)),
        LSTM(units, return_sequences=True),
        LSTM(units),
        Dense(dense_units),
        Dense(output_steps)
    ])

def lstm_dropout(window, features, output_steps, units, dense_units, dropout_rate):
    """
    Builds an LSTM model with a Dropout layer for forecasting time series data.

    The model is composed of two LSTM layers, the first one with return_sequences=True
    and the second one without it. The output of the second LSTM layer is then passed
    to a Dense layer with dense_units neurons, followed by a Dropout layer with
    dropout rate dropout_rate, and finally to another Dense layer with output_steps
    neurons.

    Parameters
    ----------
    window : int
        Input sequence length.
    features : int
        Number of features per timestep.
    output_steps : int
        Output size (number of steps predicted per forward pass).
    units : int
        Number of LSTM units.
    dense_units : int
        Number of neurons in the Dense layer.
    dropout_rate : float
        Dropout rate.

    Returns
    -------
    Model
        A Keras model instance.
    """
    return Sequential([
        Input(shape=(window, features)),
        LSTM(units, return_sequences=True),
        LSTM(units),
        Dense(dense_units),
        Dropout(dropout_rate),
        Dense(output_steps)
    ])
# lstm_utils.py (versione stateful minimalista)

# =========================
# Seeding
# =========================
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



# =========================
# LSTM class
# =========================
class LSTMForecaster(RealTimeForecaster):
    """
    Real-time LSTM forecaster with internal state management.

    This class maintains internal historical data, supports optional
    retraining when new data arrives, and performs recursive multi-step
    forecasting.

    Parameters
    ----------
    model_builder : Callable[[int, int, Dict[str, Any]], Model]
        Function that builds and compiles a Keras model.
        Signature: model_builder(window, features, model_params)

    model_params : dict
        Parameters forwarded to model_builder.

    train_params : dict
        Parameters forwarded to model.fit().
        Can include:
            - epochs
            - batch_size
            - verbose
            - callbacks
            - validation_split (handled internally)

    window : int
        Input sequence length.

    n_steps : int
        Output size (number of steps predicted per forward pass).

    features : int, default=1
        Number of features per timestep.

    scaler : object, optional
        Any sklearn-like scaler implementing fit/transform/inverse_transform.
        Defaults to StandardScaler().
    """

    # ------------------------------------------------------------------ #
    # INIT
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        model_func: Callable,
        model_params: Dict[str, Any],
        train_params: Dict[str, Any],
        window: int,
        output_steps: int,
        features: int = 1,
        scaler: Optional[Any] = None,
    ) -> None:

        self.model_func = model_func
        self.model_params = model_params
        self.train_params = train_params

        self.window = window
        self.output_steps = output_steps
        self.features = features

        self.scaler = scaler if scaler is not None else StandardScaler()

        # Build model (pass core args + **model_params)
        self.model: Model = self.model_func(
            window=self.window,
            features=self.features,
            output_steps=self.output_steps,
            **self.model_params
        )

        # Compile centrally
        self.model.compile(
            optimizer=self.train_params["optimizer"],
            loss=self.train_params["loss"],
            metrics=self.train_params.get("metrics")
        )

        self.history: Optional[pd.Series] = None
        self.trained: bool = False

    # ------------------------------------------------------------------ #
    # INTERNAL UTILITIES
    # ------------------------------------------------------------------ #

    def _create_sequences(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create supervised sequences from scaled data.

        Returns
        -------
        X : shape (n_samples, window, features)
        y : shape (n_samples, n_steps)
        """
        X, y = [], []

        for i in range(len(data) - self.window - self.output_steps + 1):
            X.append(data[i : i + self.window])
            y.append(data[i + self.window : i + self.window + self.output_steps])

        X = np.array(X).reshape(-1, self.window, self.features)
        y = np.array(y).reshape(-1, self.output_steps)

        return X, y

    def _split_train_val(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Temporal split based on train_params['validation_split'].
        """
        val_ratio = self.train_params.get("validation_split", 0.0)

        if val_ratio <= 0.0:
            return X, y, None, None

        val_size = int(len(X) * val_ratio)

        if val_size == 0:
            return X, y, None, None

        X_train, y_train = X[:-val_size], y[:-val_size]
        X_val, y_val = X[-val_size:], y[-val_size:]

        return X_train, y_train, X_val, y_val

    @staticmethod
    def _save_meta(meta: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @staticmethod
    def _load_meta(path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)


    # ------------------------------------------------------------------ #
    # FIT
    # ------------------------------------------------------------------ #

    def fit(self, series: pd.Series) -> None:
        """
        Train or retrain the model on the provided full time series.

        Parameters
        ----------
        series : pd.Series
            Complete historical time series.
        """

        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series.")

        if len(series) < self.window + self.output_steps:
            raise ValueError("Series too short for given window and output_steps.")

        self.history = series.copy()

        values = self.history.values.astype(float).reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        X, y = self._create_sequences(scaled)

        X_train, y_train, X_val, y_val = self._split_train_val(X, y)

        fit_kwargs = self.train_params.copy()

        # Remove non-fit keys
        fit_kwargs.pop("validation_split", None)
        fit_kwargs.pop("optimizer", None)
        fit_kwargs.pop("loss", None)
        fit_kwargs.pop("metrics", None)

        if X_val is not None:
            self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                **fit_kwargs
            )
        else:
            self.model.fit(
                X_train,
                y_train,
                **fit_kwargs
            )

        self.trained = True
        tf.keras.backend.clear_session()

    # ------------------------------------------------------------------ #
    # UPDATE
    # ------------------------------------------------------------------ #

    def update(self, new_data: pd.Series, refit: bool = False) -> None:
        """
        Append new observations to internal history.

        Parameters
        ----------
        new_data : pd.Series
            New data strictly following the last timestamp.

        refit : bool, default=False
            If True, retrain the model using updated history.
        """

        if not self.trained:
            raise RuntimeError("Model must be fitted first.")
        if not isinstance(new_data, pd.Series):
            raise TypeError("new_data must be a pandas Series.")
        if self.history is None:
            raise RuntimeError("Internal history is not initialized. Call fit() first.")
        if len(new_data) == 0:
            return

        # Keep only truly new points (by index)
        last_idx = self.history.index[-1]
        try:
            new_data = new_data[new_data.index > last_idx]
        except Exception:
            # If index is not comparable, fall back to strict behavior
            raise ValueError("Cannot compare indices to filter new points.")

        # If nothing new, just return
        if len(new_data) == 0:
            return

        # Optional: ensure monotonic increasing inside the chunk
        new_data = new_data[~new_data.index.duplicated(keep="last")].sort_index()

        self.history = pd.concat([self.history, new_data])

        if refit:
            self.fit(self.history)

    # ------------------------------------------------------------------ #
    # PREDICT
    # ------------------------------------------------------------------ #

    def predict(self, H: int) -> np.ndarray:
        """
        Forecast the next H time steps using recursive prediction.

        Parameters
        ----------
        H : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Forecasted values in original scale.
        """

        # print("DEBUG: LSTM PREDICT")
        # print("DEBUG: H:", H)
        # print("DEBUG: self.history:", self.history)
        # print("DEBUG: self.history.shape:", self.history.shape)
        # print("DEBUG: self.window:", self.window)
        # print("DEBUG: self.output_steps:", self.output_steps)
        # print("-"*80)

        if not self.trained:
            raise RuntimeError("Model not fitted.")

        if H <= 0:
            raise ValueError("H must be positive.")

        if len(self.history) < self.window:
            raise RuntimeError("Not enough history.")

        last_window = self.history.iloc[-self.window:].values.reshape(-1, 1)
        scaled_window = self.scaler.transform(last_window)

        current_window = scaled_window.reshape(
            1, self.window, self.features
        )

        predictions = []
        n_iter = ceil(H / self.output_steps)

        for _ in range(n_iter):

            y_hat = self.model.predict(current_window, verbose=0).flatten()
            predictions.extend(y_hat)

            updated_window = np.append(
                current_window.flatten(),
                y_hat
            )[-self.window:]

            current_window = updated_window.reshape(
                1, self.window, self.features
            )

        predictions = np.array(predictions[:H]).reshape(-1, 1)

        return self.scaler.inverse_transform(predictions).flatten()
    
    def copy(self):
        """
        Return an independent copy of the forecaster.

        Returns
        -------
        copy : RealTimeForecaster
            A deep copy of the model.
        """
        return deepcopy(self)

    def save(self, dirpath: str | Path) -> None:
        if not self.trained or self.model is None:
            raise RuntimeError("Cannot save: model not trained.")

        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # 1) keras model
        model_path = dirpath / "model.keras"
        self.model.save(model_path)

        # 2) meta (wrapper state)
        meta = {
            "class_name": self.__class__.__name__,
            "saved_at": time.time(),
            "model_func_name": getattr(self.model_func, "__name__", None),
            "model_params": self.model_params,
            "train_params": self.train_params,
            "window": self.window,
            "output_steps": self.output_steps,
            "features": self.features,
            "history": self.history,
            "scaler": self.scaler,
            "trained": self.trained,
        }
        self.__class__._save_meta(meta, dirpath / "meta.pkl")

    @classmethod
    def load(cls, dirpath: str | Path):
        dirpath = Path(dirpath)
        meta = cls._load_meta(dirpath / "meta.pkl")

        obj = cls.__new__(cls)

        # restore simple attrs
        obj.model_func = None              # opzionale: vedi nota sotto
        obj.model_params = meta.get("model_params", {})
        obj.train_params = meta.get("train_params", {})
        obj.window = meta.get("window")
        obj.output_steps = meta.get("output_steps")
        obj.features = meta.get("features", 1)
        obj.history = meta.get("history")
        obj.scaler = meta.get("scaler")
        obj.trained = bool(meta.get("trained", True))

        # load keras model
        obj.model = tf.keras.models.load_model(dirpath / "model.keras")

        return obj