# benchmark.py

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from arima_utils import *
# from lstm_utils import *
# from benchmark_utils import ModelRolling



# =========================
# Metrics
# =========================


def rmse(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat)**2)))

def mae(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))

def smape(y, yhat, eps=1e-8):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = np.maximum(np.abs(y) + np.abs(yhat), eps)
    return float(np.mean(2.0 * np.abs(yhat - y) / denom))

def latency_stats_from_preds(preds):
    """
    Compute latency statistics for update, prediction, and total cycle.

    update/predict/total are computed on the same set of forecast origins
    (any origin without an update is excluded; with origins starting at
    position 1 there is none), so the three statistics stay consistent:
    mean(total) == mean(update) + mean(predict).

    Returns
    -------
    dict
        Dictionary containing mean, P95, max, and number of observations
        for each latency component.
    """

    update_vals = np.asarray(preds.attrs.get("update_latency_ms", []), dtype=float)
    predict_vals = np.asarray(preds.attrs.get("predict_latency_ms", []), dtype=float)
    total_vals = np.asarray(preds.attrs.get("total_cycle_latency_ms", []), dtype=float)

    valid = ~np.isnan(update_vals)

    results = {}

    for name, values in (
        ("update", update_vals[valid]),
        ("predict", predict_vals[valid]),
        ("total", total_vals[valid]),
    ):
        if values.size == 0:
            results[name] = {
                "mean_ms": np.nan,
                "p95_ms": np.nan,
                "max_ms": np.nan,
                "n": 0,
            }
        else:
            results[name] = {
                "mean_ms": float(values.mean()),
                "p95_ms": float(np.percentile(values, 95)),
                "max_ms": float(values.max()),
                "n": int(values.size),
            }

    return results

def forecast_origin_positions(n_test: int, H: int, n_predictions: int) -> np.ndarray:
    """Positions (in the test series) of the evenly spaced forecast origins.

    The first origin is at position 1 (not 0), so that every origin is a
    complete real-time cycle: one new sample arrives -> timed update -> timed
    forecast. With origin 0 the model would forecast without any new sample,
    and that cycle would have no update. Shared by rolling_predict and
    oracle_rolling_predict so that both use the same origins.
    """
    n_rolls_full = n_test - H + 1
    if n_rolls_full <= 0:
        raise ValueError("Test series too short for selected horizon H.")
    first = 1 if n_rolls_full > 1 else 0
    n_predictions = max(1, min(n_predictions, n_rolls_full - first))
    return np.unique(np.linspace(first, n_rolls_full - 1, n_predictions, dtype=int))


def rolling_predict(
    model,
    test_series,
    H=1,
    n_predictions=10,
    verbose=False,
    measure_latency=True
    ) -> pd.DataFrame:
    """
    Perform rolling multi-step forecasting on a subset of forecast origins.

    Measures separately:
        - update latency
        - prediction latency
        - total cycle latency

    The model must implement:
        - predict(H): returns H-step forecast
        - update(new_data: pd.Series, refit=False): updates internal state
        - copy(): returns an independent copy
    """

    if H <= 0:
        raise ValueError("H must be a positive integer.")
    if not isinstance(test_series, pd.Series):
        raise ValueError("test_series must be a pandas Series.")
    if n_predictions <= 0:
        raise ValueError("n_predictions must be a positive integer.")

    # The test set must contain enough observations for the
    # requested forecast horizon.
    n_rolls_full = len(test_series) - H + 1

    if n_rolls_full <= 0:
        raise ValueError("Test series too short for selected horizon H.")

    n_predictions = min(n_predictions, n_rolls_full)

    target_positions = forecast_origin_positions(len(test_series), H, n_predictions)

    model_copy = model.copy()

    preds = []
    start_indices = []

    update_latencies_ms = []
    predict_latencies_ms = []
    total_cycle_latencies_ms = []
    chunk_sizes = []

    current_pos = 0

    for target_pos in target_positions:

        # Number of new observations incorporated since
        # the previous forecast origin.
        chunk_size = target_pos - current_pos

        # --------------------------------------------------
        # Update model state
        # --------------------------------------------------
        if chunk_size > 0:

            update_data = test_series.iloc[current_pos:target_pos]

            # Untimed catch-up: absorb the whole gap except the last point.
            # Needed by the sparse walk-forward, but NOT representative of
            # the real-time per-cycle cost.
            if chunk_size > 1:
                model_copy.update(update_data.iloc[:-1])

            # Timed update: a single new sample, as in the real deployment
            # (one value per sampling interval).
            last_point = update_data.iloc[[-1]]

            if measure_latency:
                t0 = time.perf_counter()
                model_copy.update(last_point)
                t1 = time.perf_counter()

                update_latency_ms = (t1 - t0) * 1000.0
            else:
                model_copy.update(last_point)
                update_latency_ms = np.nan

            current_pos = target_pos

        else:
            # No update is performed for the first forecast origin.
            update_latency_ms = np.nan

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------
        if measure_latency:
            t0 = time.perf_counter()
            forecast = model_copy.predict(H)
            t1 = time.perf_counter()

            predict_latency_ms = (t1 - t0) * 1000.0
        else:
            forecast = model_copy.predict(H)
            predict_latency_ms = np.nan

        # --------------------------------------------------
        # Total cycle latency
        # --------------------------------------------------
        if measure_latency:
            if np.isnan(update_latency_ms):
                total_cycle_latency_ms = predict_latency_ms
            else:
                total_cycle_latency_ms = (
                    update_latency_ms + predict_latency_ms
                )
        else:
            total_cycle_latency_ms = np.nan

        forecast = np.asarray(forecast).reshape(-1)

        preds.append(forecast)
        start_indices.append(test_series.index[target_pos])

        chunk_sizes.append(chunk_size)
        update_latencies_ms.append(update_latency_ms)
        predict_latencies_ms.append(predict_latency_ms)
        total_cycle_latencies_ms.append(total_cycle_latency_ms)

    pred_df = pd.DataFrame(
        data=np.vstack(preds),
        index=pd.Index(
            start_indices,
            name="forecast_origin"
        ),
        columns=[f"h{h}" for h in range(1, H + 1)]
    )

    if measure_latency:
        pred_df.attrs["chunk_sizes"] = chunk_sizes
        pred_df.attrs["update_latency_ms"] = update_latencies_ms
        pred_df.attrs["predict_latency_ms"] = predict_latencies_ms
        pred_df.attrs["total_cycle_latency_ms"] = total_cycle_latencies_ms

    return pred_df

def plot_rolling_forecast(
    test_series,
    forecast_df,
    max_plots=5,
    title=None
):
    """
    Plot rolling multi-step forecasts stored in a structured DataFrame.

    Parameters
    ----------
    test_series : pd.Series
        Ground truth test time series.

    forecast_df : pd.DataFrame
        Output of rolling_predict_n.

        - Index: forecast origin
        - Columns: h1, h2, ..., hH

    max_plots : int
        Maximum number of forecast trajectories to display.

    title : str, optional
        Plot title.
    """

    if not isinstance(test_series, pd.Series):
        raise ValueError("test_series must be a pandas Series.")
    if not isinstance(forecast_df, pd.DataFrame):
        raise ValueError("forecast_df must be a pandas DataFrame.")
    if len(forecast_df) == 0:
        raise ValueError("forecast_df is empty.")

    H = forecast_df.shape[1]
    n_preds = len(forecast_df)

    plot_step = max(n_preds // max_plots, 1)

    plt.figure(figsize=(12, 6))
    plt.plot(test_series, color="black", linewidth=2, label="True")

    freq = getattr(test_series.index, "freq", None) or pd.infer_freq(test_series.index)
    offset = pd.tseries.frequencies.to_offset(freq) if freq is not None else None

    for i in range(0, n_preds, plot_step):

        origin = forecast_df.index[i]
        forecast_values = forecast_df.iloc[i].to_numpy()

        # Build forecast time index: origin is h1 (first predicted timestamp)
        if offset is not None and isinstance(origin, pd.Timestamp):
            forecast_index = pd.date_range(start=origin, periods=H, freq=offset)
        else:
            # Fallback: locate origin in test_series and take the next H indices from there
            pos = test_series.index.get_loc(origin)
            forecast_index = test_series.index[pos:pos + H]

        origin_value = test_series.loc[origin]

        # 1. X axis: the origin timestamp followed by the forecast timestamps
        x_axis = [pd.to_datetime(origin)] + pd.to_datetime(forecast_index).tolist()

        # 2. Y axis: the observed value at the origin followed by the forecast values
        y_axis = [origin_value] + list(forecast_values.flatten())

        # 3. Both lists must have the same length (H + 1, e.g. 11 for H=10)
        if len(x_axis) == len(y_axis):
            plt.plot(
                x_axis, 
                y_axis, 
                linestyle="--", 
                marker=".", 
                alpha=0.8
            )
        else:
            print(f"Length mismatch: X has {len(x_axis)}, Y has {len(y_axis)}")

    plt.title(title or "Rolling Forecast")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
import numpy as np


def evaluate_rolling_forecast(
    test_series: pd.Series,
    forecast_df: pd.DataFrame,
    metric_fn,
    reduce_fn=np.mean
) -> float:
    """
    Evaluate rolling multi-step forecasts.

    Parameters
    ----------
    test_series : pd.Series
        Ground truth time series.

    forecast_df : pd.DataFrame
        Structured forecast output from rolling_predict_n.

    metric_fn : callable
        Function metric(y_true, y_pred) -> float.

    reduce_fn : callable
        Aggregation function (default: mean).

    Returns
    -------
    float
        Aggregated forecast error.
    """

    if not isinstance(test_series, pd.Series):
        raise ValueError("test_series must be a pandas Series.")
    if not isinstance(forecast_df, pd.DataFrame):
        raise ValueError("forecast_df must be a pandas DataFrame.")
    if len(forecast_df) == 0:
        raise ValueError("forecast_df is empty.")

    H = forecast_df.shape[1]

    freq = getattr(test_series.index, "freq", None) or pd.infer_freq(test_series.index)
    offset = pd.tseries.frequencies.to_offset(freq) if freq is not None else None

    scores = []

    # Iterate row-by-row to be robust to duplicate origins
    for origin, row in forecast_df.iterrows():
        if offset is not None and isinstance(origin, pd.Timestamp):
            end = origin + (H - 1) * offset
            true_slice = test_series.loc[origin:end].to_numpy()
        else:
            pos = test_series.index.get_loc(origin)
            true_slice = test_series.iloc[pos:pos + H].to_numpy()

        pred_slice = row.to_numpy()

        # Safety: if slicing returns fewer than H points, skip (shouldn't happen if rolling_predict is correct)
        if len(true_slice) != H:
            continue

        scores.append(metric_fn(true_slice, pred_slice))

    if len(scores) == 0:
        raise ValueError("No valid forecast rows could be evaluated (check indices and horizon).")

    return float(reduce_fn(scores))


def oracle_rolling_predict(model, test_series: pd.Series, origins_by_H: dict) -> dict:
    """Oracle ("teacher forcing") multi-step forecasts.

    For every forecast origin p and every step h = 1..H the model predicts ONE
    step ahead, and before step h+1 its state is updated with the TRUE value
    observed at step h (not with its own prediction). Step h of the oracle
    trajectory is therefore the one-step-ahead forecast of y[p+h-1] given all
    true observations before it.

    Because of this, the oracle forecast of a position does not depend on the
    origin it belongs to: the function walks the test series once, forecasting
    one step ahead at every position that is needed and updating the model with
    the true values in between (untimed: latencies of this test are not
    representative of a deployment and are not measured).

    Parameters
    ----------
    model : forecaster with predict(1), update(series) and copy()
    test_series : pd.Series
    origins_by_H : dict {H: iterable of origin positions in test_series}

    Returns
    -------
    dict {H: pd.DataFrame}, same layout as rolling_predict:
        index = forecast_origin (timestamp of step h1), columns h1..hH
    """
    n = len(test_series)
    needed = set()
    for H, positions in origins_by_H.items():
        for p in positions:
            if p + H > n:
                raise ValueError(f"Origin {p} with H={H} goes beyond the test series.")
            needed.update(range(int(p), int(p) + int(H)))

    m = model.copy()
    one_step = {}
    cur = 0  # first test position not yet given to the model
    for q in sorted(needed):
        if q > cur:
            m.update(test_series.iloc[cur:q])   # true values up to q-1
            cur = q
        one_step[q] = float(np.asarray(m.predict(1)).reshape(-1)[0])

    out = {}
    for H, positions in origins_by_H.items():
        positions = [int(p) for p in positions]
        out[H] = pd.DataFrame(
            data=[[one_step[p + h] for h in range(H)] for p in positions],
            index=pd.Index(test_series.index[positions], name="forecast_origin"),
            columns=[f"h{h}" for h in range(1, H + 1)],
        )
    return out


def per_origin_errors(test_series: pd.Series, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, MAPE (%) and sMAPE (%) of every forecast row (origin).

    Mean and variance over origins can then be taken directly on the result.
    """
    H = forecast_df.shape[1]
    rows = []
    for origin, row in forecast_df.iterrows():
        pos = test_series.index.get_loc(origin)
        y = test_series.iloc[pos:pos + H].to_numpy(dtype=float)
        yhat = row.to_numpy(dtype=float)
        if len(y) != H:
            continue
        err = yhat - y
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.nanmean(np.abs(err) / np.abs(y)) * 100.0
            smape_v = np.nanmean(2.0 * np.abs(err) / (np.abs(y) + np.abs(yhat))) * 100.0
        rows.append({
            "forecast_origin": origin,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mape": float(mape),
            "smape": float(smape_v),
        })
    return pd.DataFrame(rows).set_index("forecast_origin")


class RealTimeForecaster:
    """
    RealTimeForecaster is an abstract class designed for real-time forecasting scenarios. It defines a standard workflow: the model is trained on historical data up to a known point, and from there it forecasts future steps in a real-time manner. New incoming data can be incorporated, and optionally, the model can be re-fitted with these new data.

    Workflow:
    - fit(series): trains the model using the input time series up to its end.
    - update(new_data, refit=False): integrates new observations. If refit is True, the model is re-trained on the expanded dataset; otherwise, it simply updates the internal state.
    - predict(h=1): generates a forecast of h steps ahead from the current internal state.

    Assumptions:
    - The model is trained up to a known timestamp; all predictions rely on that base data.
    - update allows incorporation of new data, and when refit is True, the model is retrained to adapt to the larger dataset.
    - predict returns a sequence of h steps based on the internal state.

    Methods:
    """

    def fit(self, series):
        """
        Train the model on the provided time series.
        
        Parameters:
            series (pd.Series): A time-ordered series of values used for training.
        
        Returns:
            None. After fitting, the model is trained up to the end of the series and ready for updates or predictions.
        """
        raise NotImplementedError("fit method must be implemented by the subclass.")

    def predict(self, h=1):
        """
        Produce a forecast of h steps ahead from the current known internal state.

        Parameters:
            h (int): The number of future steps to forecast. Defaults to 1.

        Returns:
            np.array: An array containing the h forecasted values.

        Notes:
            The model uses its internal state, so h-step forecasts respect the model's horizon capability.
        """
        raise NotImplementedError("predict method must be implemented by the subclass.")

    def update(self, new_data, refit=False):
        """
        Update the model with new incoming data.

        Parameters:
            new_data (pd.Series): A series of new data points not available during the original training.
            refit (bool): If True, the model is retrained with both the old and new data; if False, only the internal state is updated without re-fitting.

        Returns:
            None. After update, the model is either ready for prediction or has been re-trained.

        Notes:
            This method is the only way to modify the internal state. Use refit sparingly, as retraining is computationally expensive.
        """
        raise NotImplementedError("update method must be implemented by the subclass.")
    
    
    def copy(self):
        """
        Return an independent copy of the forecaster.

        Notes:
            This method should create a deep copy of the model, such that the returned copy is completely independent of the original model.
            The copy should be able to predict, update and copy itself without affecting the original model.
        """
        raise NotImplementedError

    def save(self, path):

        raise NotImplementedError