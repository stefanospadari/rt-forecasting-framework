# arima_utils.py (minimal, RealTimeForecaster-compatible)

from typing import Optional, Dict, Any, Tuple
from copy import deepcopy
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from libs.benchmark_utils import RealTimeForecaster


# =========================
# Fourier features
# =========================

def fourier_features(t: np.ndarray, period: int, K: int, index=None) -> pd.DataFrame:
    """Generate Fourier sin/cos features for seasonal patterns."""
    X = pd.DataFrame(index=index if index is not None else range(len(t)))
    for k in range(1, K + 1):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return X


def infer_offset(index: pd.Index):
    """
    Infer a pandas DateOffset from a DatetimeIndex if possible.
    Returns None if frequency cannot be inferred.
    """
    freq = getattr(index, "freq", None) or pd.infer_freq(index)
    if freq is None:
        return None
    return pd.tseries.frequencies.to_offset(freq)


# =========================
# Forecaster
# =========================

class ArimaForecaster(RealTimeForecaster):
    """
    Minimal SARIMAX real-time forecaster.

    Design:
    - order/seasonal_order are provided by config (no auto stationarity logic).
    - optional log transform is applied consistently in fit/update/predict.
    - history is kept (raw scale) to support refit and robust time indexing.
    - update(new_data, refit=False) appends observations to the internal state
      without re-fitting (unless refit=True).
    """

    def __init__(
        self,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        fourier_params: Optional[Dict[str, Any]] = None,
        use_log: bool = False,
    ):
        self.order = order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self.fourier_params = fourier_params
        self.use_log = use_log

        self.history: Optional[pd.Series] = None  # raw series (original scale)
        self.model = None  # SARIMAXResults
        self.trained: bool = False

    # -------------------------
    # Internal helpers
    # -------------------------

    def _transform_endog(self, s: pd.Series) -> pd.Series:
        """Apply endog transform used by the model (currently only optional log)."""
        s = s.astype(float)
        if self.use_log:
            if (s <= 0).any():
                raise ValueError("use_log=True but series contains non-positive values.")
            return np.log(s)
        return s

    def _inverse_transform(self, s: pd.Series) -> pd.Series:
        """Invert endog transform (log -> exp)."""
        if self.use_log:
            return np.exp(s)
        return s

    def _build_exog(self, index: pd.Index, t_start: int) -> Optional[pd.DataFrame]:
        """
        Build Fourier exog for a specific index segment.
        t_start must be aligned with the absolute position of the model endog.
        """
        if not self.fourier_params:
            return None
        period = self.fourier_params["period"]
        K = self.fourier_params["K"]
        t = np.arange(t_start, t_start + len(index))
        return fourier_features(t, period=period, K=K, index=index)

    @staticmethod
    def _save_meta(meta: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @staticmethod
    def _load_meta(path: Path) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    # -------------------------
    # API methods
    # -------------------------

    def fit(self, series: pd.Series) -> None:
        """Fit SARIMAX on the provided series (up to its end)."""
        if not isinstance(series, pd.Series):
            raise TypeError("series must be a pandas Series.")
        if len(series) < 10:
            raise ValueError("Series too short for SARIMAX fit.")

        # Store raw history
        self.history = series.copy()

        # Transform endog (raw or log)
        endog = self._transform_endog(self.history)

        # Build exog aligned with endog (t starts at 0)
        exog = self._build_exog(endog.index, t_start=0)

        # Fit model
        self.model = SARIMAX(
            endog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        self.trained = True

    def predict(self, H: int) -> pd.Series:
        """Forecast the next H steps from the current internal state."""
        if not self.trained or self.model is None or self.history is None:
            raise RuntimeError("Model must be fitted first.")
        if H <= 0:
            raise ValueError("H must be positive.")

        # Build future exog continuing the same t scale
        exog_future = None
        if self.fourier_params:
            t_last = len(self.model.data.endog)
            # For exog we only need a correctly-sized index; values depend on t
            exog_future = self._build_exog(pd.RangeIndex(H), t_start=t_last)

        # Forecast in model space (raw/log)
        y_hat = pd.Series(self.model.forecast(steps=H, exog=exog_future))

        # Attach a meaningful datetime index if possible
        offset = infer_offset(self.history.index)
        if offset is not None and isinstance(self.history.index[-1], pd.Timestamp):
            y_hat.index = pd.date_range(start=self.history.index[-1] + offset, periods=H, freq=offset)

        # Back-transform to original scale
        y_hat = self._inverse_transform(y_hat)
        return y_hat

    def update(self, new_data: pd.Series, refit: bool = False) -> None:
        """
        Update internal state with new observations.

        Policy:
        - Overlapping / already-seen timestamps are ignored (append-only semantics).
        - If refit=True: re-fit from scratch using expanded history.
        - Else: append step-by-step to the state space model (fast).
        """
        if not self.trained or self.model is None or self.history is None:
            raise RuntimeError("Model must be fitted first.")
        if not isinstance(new_data, pd.Series):
            raise TypeError("new_data must be a pandas Series.")
        if len(new_data) == 0:
            return

        # Clean the incoming chunk: drop duplicates, sort by index
        new_data = new_data[~new_data.index.duplicated(keep="last")].sort_index()

        # Keep only truly new points (by index)
        last_idx = self.history.index[-1]
        try:
            new_only = new_data[new_data.index > last_idx]
        except Exception:
            # If indices are not comparable, we can't safely filter overlap
            raise ValueError("Cannot compare indices to filter new points.")

        # If nothing new, do nothing
        if len(new_only) == 0:
            return

        # Update raw history first (append-only)
        self.history = pd.concat([self.history, new_only])

        # Append one point at a time to keep exog aligned
        for ts, y_raw in new_only.items():
            y_series = pd.Series([y_raw], index=pd.Index([ts]))
            y_endog = self._transform_endog(y_series).iloc[0]

            exog_new = None
            if self.fourier_params:
                t_next = len(self.model.data.endog)
                exog_new = self._build_exog(pd.Index([ts]), t_start=t_next)

            self.model = self.model.append([y_endog], exog=exog_new, refit=refit)


    def copy(self):
        """Return an independent deep copy of the forecaster."""
        return deepcopy(self)

    # -------------------------
    # Save / Load (bundle)
    # -------------------------
    def save(self, dirpath: str | Path) -> None:
        if not self.trained or self.model is None:
            raise RuntimeError("Cannot save: model not trained.")

        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # 1) model (statsmodels)
        model_path = dirpath / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # 2) meta (everything else)
        meta = {
            "class_name": self.__class__.__name__,
            "saved_at": time.time(),
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "fourier_params": self.fourier_params,
            "use_log": self.use_log,
            "history": self.history,
            "trained": self.trained,
        }
        self.__class__._save_meta(meta, dirpath / "meta.pkl")

    @classmethod
    def load(cls, dirpath: str | Path):
        dirpath = Path(dirpath)

        meta = cls._load_meta(dirpath / "meta.pkl")

        obj = cls.__new__(cls)

        obj.order = meta.get("order")
        obj.seasonal_order = meta.get("seasonal_order", (0, 0, 0, 0))
        obj.fourier_params = meta.get("fourier_params")
        obj.use_log = meta.get("use_log", False)
        obj.history = meta.get("history")
        obj.trained = bool(meta.get("trained", True))

        with open(dirpath / "model.pkl", "rb") as f:
            obj.model = pickle.load(f)

        return obj