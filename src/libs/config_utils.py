"""Helpers that read experiment settings from the YAML config.

Nothing about rooms, metrics, horizons or sampling interval is hard-coded in
training.py / testing.py: everything comes from the config file.
"""

# Used only if the config has no `data.metrics` section (old configs).
_LEGACY_METRICS = {
    "temperature": {"min_val": 0, "max_val": 50},
    "humidity": {"min_val": 0, "max_val": 100},
    "co2": {"min_val": 0, "max_val": 2000},
}

_DEFAULT_EVALUATION = {
    "horizons": [1, 10, 30, 60],
    "delta_t_sec": 60.0,
    "n_origins": 100,
}


def resolve_metric(data_cfg: dict, metric: str | None) -> str:
    """Select the metric to use and apply its valid range to data_cfg.

    The metric comes from --metric (or `data.metric` in the config) and must be
    one of the keys of `data.metrics`. Its optional `min_val` / `max_val` are
    copied into data_cfg, where prepare_dataset uses them to drop outliers.
    """
    metrics = data_cfg.get("metrics") or _LEGACY_METRICS
    metric = metric or data_cfg.get("metric")
    if metric is None:
        raise ValueError(
            f"No metric given: pass --metric <name>. Available: {sorted(metrics)}"
        )
    if metric not in metrics:
        raise ValueError(
            f"Metric '{metric}' is not declared under data.metrics in the config. "
            f"Available: {sorted(metrics)}"
        )
    bounds = metrics[metric] or {}
    data_cfg["metric"] = metric
    data_cfg["min_val"] = bounds.get("min_val")
    data_cfg["max_val"] = bounds.get("max_val")
    if "max_diff" in bounds:
        data_cfg["max_diff"] = bounds["max_diff"]
    return metric


def resolve_rooms(data_cfg: dict, room) -> list:
    """Rooms to process: the one given with --room, otherwise data.rooms."""
    if room is not None:
        return [int(room)] if str(room).isdigit() else [room]
    rooms = data_cfg.get("rooms")
    if not rooms:
        raise ValueError("No rooms: set data.rooms in the config or pass --room.")
    return list(rooms)


def evaluation_settings(config: dict, n_origins: int | None = None) -> dict:
    """Evaluation settings (horizons, delta_t_sec, n_origins) from `evaluation:`.

    --n-origins on the command line overrides evaluation.n_origins.
    `is_default_n` tells whether the run uses the configured number of origins
    (used to name the output files of non-default, quick runs differently).
    """
    ev = {**_DEFAULT_EVALUATION, **(config.get("evaluation") or {})}
    default_n = int(ev["n_origins"])
    n = int(n_origins) if n_origins is not None else default_n
    return {
        "horizons": [int(h) for h in ev["horizons"]],
        "delta_t_sec": float(ev["delta_t_sec"]),
        "n_origins": n,
        "is_default_n": n == default_n,
    }
