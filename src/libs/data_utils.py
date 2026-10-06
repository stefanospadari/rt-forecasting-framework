"""
data_utils.py

Utility functions for loading, cleaning, and resampling time series datasets.

This module provides a standardized pipeline for preparing time series data
before model-specific preprocessing (scaling, windowing, etc.).

Author: Stefano Spadari
"""

from typing import Optional, Union, List

import pandas as pd
import numpy as np


# ======================================================
# Loading
# ======================================================

def load_csv(
    path: str,
    time_col: Optional[str] = None,
    value_cols: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    If column names are provided, they are used.
    Otherwise, the first column is assumed to be time
    and the remaining columns are considered values.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    time_col : str, optional
        Name of the timestamp column.

    value_cols : str or list of str, optional
        Name(s) of the value column(s).

    Returns
    -------
    pd.DataFrame
        DataFrame with time column and value columns.
    """

    # Case 1: CSV with header
    if time_col is not None and value_cols is not None:

        df = pd.read_csv(path)

        if isinstance(value_cols, str):
            value_cols = [value_cols]

        cols = [time_col] + value_cols
        df = df[cols]

    # Case 2: CSV without header
    else:
        df = pd.read_csv(path, header=None)

        # Rename columns
        n_cols = df.shape[1]

        # Case: time + single value
        if n_cols == 2:
            df.columns = ["time", "value"]

        # Case: time + multiple values
        else:
            cols = ["time"] + [
                f"value_{i}" for i in range(1, n_cols)
            ]
            df.columns = cols

    return df


# ======================================================
# Timestamp handling
# ======================================================

def parse_timestamp(
    df: pd.DataFrame,
    time_col: str,
    time_format: str = "epoch"
) -> pd.DataFrame:
    """
    Convert a timestamp column to pandas datetime.

    Supported formats:
    - 'epoch'    : seconds since epoch
    - 'ms_epoch' : milliseconds since epoch
    - 'iso'      : ISO datetime string

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    time_col : str
        Name of timestamp column.

    time_format : str
        Timestamp format.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index.
    """

    df = df.copy()

    if time_format == "epoch":
        df[time_col] = pd.to_datetime(df[time_col], unit="s")

    elif time_format == "ms_epoch":
        df[time_col] = pd.to_datetime(df[time_col], unit="ms")

    elif time_format == "iso":
        df[time_col] = pd.to_datetime(df[time_col])

    else:
        raise ValueError(
            "Unsupported time_format. Use: 'epoch', 'ms_epoch', 'iso'."
        )

    df = df.set_index(time_col)
    df = df.sort_index()

    return df


# ======================================================
# Cleaning
# ======================================================

def clean_value_ranges(
    df: pd.DataFrame,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> pd.DataFrame:
    """
    Remove values outside a given range.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    min_val : float, optional
        Minimum allowed value.

    max_val : float, optional
        Maximum allowed value.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    df = df.copy()

    for col in df.columns:

        if min_val is not None:
            df.loc[df[col] < min_val, col] = np.nan

        if max_val is not None:
            df.loc[df[col] > max_val, col] = np.nan

    return df


def remove_spikes(
    df: pd.DataFrame,
    max_diff: Optional[float] = None
) -> pd.DataFrame:
    """
    Remove sudden spikes based on first difference.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    max_diff : float, optional
        Maximum allowed absolute difference.

    Returns
    -------
    pd.DataFrame
        Dataframe with spikes removed.
    """

    if max_diff is None:
        return df

    df = df.copy()

    for col in df.columns:

        diff = df[col].diff().abs()
        df.loc[diff > max_diff, col] = np.nan

    return df


# ======================================================
# Resampling & interpolation
# ======================================================

def resample_timeseries(
    df: pd.DataFrame,
    freq: str = "1min",
    agg: str = "mean"
) -> pd.DataFrame:
    """
    Resample time series to a fixed frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with datetime index.

    freq : str
        Resampling frequency (e.g., '1min', '10min').

    agg : str
        Aggregation method ('mean', 'median', etc.).

    Returns
    -------
    pd.DataFrame
        Resampled dataframe.
    """

    if agg == "mean":
        df_res = df.resample(freq).mean()

    elif agg == "median":
        df_res = df.resample(freq).median()

    else:
        raise ValueError("Unsupported aggregation method.")

    return df_res


def interpolate_missing(
    df: pd.DataFrame,
    method: str = "time",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Interpolate missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    method : str
        Interpolation method.

    limit : int, optional
        Maximum number of consecutive missing values to interpolate.

    Returns
    -------
    pd.DataFrame
        Interpolated dataframe.
    """

    df = df.copy()

    df = df.interpolate(method=method, limit=limit)

    return df


# ======================================================
# High-level pipeline
# ======================================================

def prepare_dataset(
    path: str,

    time_col: Optional[str] = None,
    value_cols: Optional[Union[str, List[str]]] = None,

    time_format: str = "epoch",

    freq: str = "1min",

    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    max_diff: Optional[float] = None,

    interpolate: bool = True,
    interpolation_method: str = "time",
    interpolation_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prepare a time series dataset for forecasting.

    This function performs:
    - CSV loading
    - timestamp parsing
    - value cleaning
    - spike removal
    - resampling
    - optional interpolation

    Parameters
    ----------
    path : str
        Path to CSV file.

    time_col : str, optional
        Timestamp column name.

    value_cols : str or list of str, optional
        Value column name(s).

    time_format : str
        Timestamp format ('epoch', 'ms_epoch', 'iso').

    freq : str
        Resampling frequency.

    min_val : float, optional
        Minimum valid value.

    max_val : float, optional
        Maximum valid value.

    max_diff : float, optional
        Maximum allowed step difference.

    interpolate : bool
        Whether to interpolate missing values.

    Returns
    -------
    pd.DataFrame
        Prepared time series dataframe.
    """

    # Load
    df = load_csv(path, time_col, value_cols)

    # Determine time column
    if time_col is None:
        time_col = df.columns[0]

    # Parse timestamp
    df = parse_timestamp(df, time_col, time_format)

    # Clean ranges
    df = clean_value_ranges(df, min_val, max_val)

    # Remove spikes
    if max_diff is not None:    
        df = remove_spikes(df, max_diff)

    # Resample
    df = resample_timeseries(df, freq)

    print(interpolate)
    # Interpolate
    if interpolate:
        print("Interpolating missing values...")
        df = interpolate_missing(
            df,
            method=interpolation_method,
            limit=interpolation_limit,
        )

    return df
