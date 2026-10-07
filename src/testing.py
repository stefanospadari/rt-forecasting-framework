from pathlib import Path
import argparse

import pandas as pd
import yaml

from libs.data_utils import prepare_dataset
from libs.paths import PROJECT_ROOT, project_path, config_path
from libs.config_utils import resolve_metric, resolve_rooms, evaluation_settings
from libs.arima_utils import ArimaForecaster
from libs.lstm_utils import LSTMForecaster
from libs.benchmark_utils import (
    rolling_predict,
    evaluate_rolling_forecast,
    rmse,
    mae,
    smape,
    latency_stats_from_preds,
)


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/base.yaml")
parser.add_argument("--metric", type=str, default=None)
parser.add_argument("--n-origins", type=int, default=None,
                    help="forecast origins per room x model x horizon (default: evaluation.n_origins)")
parser.add_argument("--room", type=str, default=None,
                    help="evaluate only this room (default: all data.rooms)")
args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = config_path(args.config)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print(f"Configuration loaded: {CONFIG_PATH}")
print(f"Project root:         {PROJECT_ROOT}")


# ============================================================
# Configuration sections
# ============================================================

data_cfg = config["data"]
output_cfg = config["output"]

metric = resolve_metric(data_cfg, args.metric)
rooms = resolve_rooms(data_cfg, args.room)

eval_cfg = evaluation_settings(config, args.n_origins)
H_test = eval_cfg["horizons"]
N_SPARSE = eval_cfg["n_origins"]
DELTA_T_SEC = eval_cfg["delta_t_sec"]

results_rows = []


# ============================================================
# Models
# ============================================================

models = config["models"]

print("\nModels to benchmark:")
for model_cfg in models:
    print(f"  {model_cfg['name']} -> {model_cfg['builder']}")

print(f"\nRooms: {rooms}")
print(f"Metric: {metric}")
print(f"Horizons: {H_test}")
print(f"Sparse predictions per horizon: {N_SPARSE}")
print(f"Sampling interval (delta_t): {DELTA_T_SEC} s")


# ============================================================
# Rooms
# ============================================================

for room in rooms:

    print("\n" + "=" * 60)
    print(f"ROOM {room} - {metric.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_path = (
        project_path(data_cfg.get("path", "data/archive/KETI"))
        / str(room)
        / f"{metric}.csv"
    )

    print("\nDataset:")
    print(f"  {dataset_path}")

    df = prepare_dataset(
        path=str(dataset_path),
        time_col=data_cfg.get("time_col"),
        value_cols=metric,
        time_format=data_cfg.get("time_format", "epoch"),
        freq=data_cfg["freq"],
        min_val=data_cfg.get("min_val"),
        max_val=data_cfg.get("max_val"),
        max_diff=data_cfg.get("max_diff"),
        interpolate=data_cfg["interpolate"],
    )

    series = df["value"]

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    train_ratio = data_cfg["train_ratio"]
    train_size = int(len(series) * train_ratio)

    train_series = series.iloc[:train_size]
    test_series = series.iloc[train_size:]

    print(f"  Train: {len(train_series)}")
    print(f"  Test:  {len(test_series)}")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    for model_cfg in models:

        model_name = model_cfg["name"]
        builder_name = model_cfg["builder"]

        print("\n" + "-" * 60)
        print(f"MODEL: {model_name}")
        print("-" * 60)

        # ----------------------------------------------------
        # Resolve builder
        # ----------------------------------------------------

        builder = globals()[builder_name]

        # ----------------------------------------------------
        # Model path
        # ----------------------------------------------------

        model_path = (
            project_path(output_cfg["path"])
            / str(room)
            / metric
            / model_name
        )

        if not model_path.exists():
            print(f"Skipping {model_name}: model not found")
            continue

        print("Loading model from:")
        print(f"  {model_path}")

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model = builder.load(model_path)

        print("Model loaded.")

        # ----------------------------------------------------
        # Predictions directory
        # ----------------------------------------------------

        predictions_path = model_path / "predictions"
        predictions_path.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # Test all horizons
        # ----------------------------------------------------

        for H in H_test:

            print(f"\n--- Benchmark: {model_name} | H={H} ---")

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            preds = rolling_predict(
                model=model,
                test_series=test_series,
                H=H,
                n_predictions=N_SPARSE,
                measure_latency=True,
            )

            print(f"Predictions shape: {preds.shape}")

            # ------------------------------------------------
            # Save predictions
            # ------------------------------------------------

            prediction_file = (
                predictions_path
                / f"predictions_H{H}.csv"
            )

            preds.to_csv(prediction_file)

            print(f"Predictions saved to:")
            print(f"  {prediction_file}")

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            rmse_val = evaluate_rolling_forecast(
                test_series,
                preds,
                rmse,
            )

            mae_val = evaluate_rolling_forecast(
                test_series,
                preds,
                mae,
            )

            smape_val = evaluate_rolling_forecast(
                test_series,
                preds,
                smape,
            )

            # ------------------------------------------------
            # Latency
            # ------------------------------------------------

            latency_stats = latency_stats_from_preds(preds)

            update_stats = latency_stats["update"]
            predict_stats = latency_stats["predict"]
            total_stats = latency_stats["total"]

            # Save detailed timing information for every forecast origin.
            timing_df = pd.DataFrame(
                {
                    "forecast_origin": preds.index,
                    "chunk_size": preds.attrs["chunk_sizes"],
                    "update_latency_ms": preds.attrs["update_latency_ms"],
                    "predict_latency_ms": preds.attrs["predict_latency_ms"],
                    "total_cycle_latency_ms": preds.attrs["total_cycle_latency_ms"],
                }
            )

            timing_file = (
                model_path
                / f"timings_sparse_H{H}.csv"
            )

            timing_df.to_csv(timing_file, index=False)

            print(f"Timing data saved to:")
            print(f"  {timing_file}")

            # Total-cycle latency is the latency relevant to the
            # end-to-end real-time constraint.
            lat_mean_ms = total_stats["mean_ms"]
            lat_p95_ms = total_stats["p95_ms"]
            lat_max_ms = total_stats["max_ms"]
            n_origins = total_stats["n"]

            lat_per_step_ms = (
                lat_mean_ms / H
                if pd.notna(lat_mean_ms) and H > 0
                else float("nan")
            )

            feas_p95_ratio = (
                (lat_p95_ms / 1000.0) / DELTA_T_SEC
                if pd.notna(lat_p95_ms)
                else float("nan")
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            results_rows.append(
                {
                    "room": room,
                    "metric": metric,
                    "model": model_name,
                    "H": H,

                    "rmse": rmse_val,
                    "mae": mae_val,
                    "smape": smape_val,

                    # Update latency
                    "lat_update_mean_ms": update_stats["mean_ms"],
                    "lat_update_p95_ms": update_stats["p95_ms"],
                    "lat_update_max_ms": update_stats["max_ms"],

                    # Prediction latency
                    "lat_pred_mean_ms": predict_stats["mean_ms"],
                    "lat_pred_p95_ms": predict_stats["p95_ms"],
                    "lat_pred_max_ms": predict_stats["max_ms"],

                    # Total cycle latency
                    "lat_total_mean_ms": total_stats["mean_ms"],
                    "lat_total_p95_ms": total_stats["p95_ms"],
                    "lat_total_max_ms": total_stats["max_ms"],

                    "lat_per_step_ms": lat_per_step_ms,
                    "feas_p95_ratio": feas_p95_ratio,
                    "n_origins": n_origins,
                }
            )

            print("\nResults:")
            print(f"  RMSE:             {rmse_val:.6f}")
            print(f"  MAE:              {mae_val:.6f}")
            print(f"  sMAPE:            {smape_val:.6f}")

            print(
                f"  Update latency:   "
                f"{update_stats['mean_ms']:.3f} ms "
                f"(P95: {update_stats['p95_ms']:.3f} ms)"
            )

            print(
                f"  Predict latency:  "
                f"{predict_stats['mean_ms']:.3f} ms "
                f"(P95: {predict_stats['p95_ms']:.3f} ms)"
            )

            print(
                f"  Total latency:    "
                f"{total_stats['mean_ms']:.3f} ms "
                f"(P95: {total_stats['p95_ms']:.3f} ms)"
            )

            print(f"  Latency/step:     {lat_per_step_ms:.3f} ms")
            print(f"  P95/Δt:           {feas_p95_ratio:.6f}")
            print(f"  N origins:        {n_origins}")


# ============================================================
# Results table
# ============================================================

df_results = pd.DataFrame(results_rows)

print("\n\n")
print("=" * 60)
print(f"BENCHMARK RESULTS - {metric.upper()}")
print("=" * 60)

if not df_results.empty:
    print(df_results.to_string(index=False))
else:
    print("No benchmark results produced.")


# ============================================================
# Save results
# ============================================================

results_dir = project_path(output_cfg.get("results_path", "results"))
results_dir.mkdir(parents=True, exist_ok=True)

suffix = ("" if eval_cfg["is_default_n"] else f"_n{N_SPARSE}") + (f"_room{args.room}" if args.room else "")
output_file = results_dir / f"benchmark_results_sparse_{metric}{suffix}.csv"

df_results.to_csv(output_file, index=False)

print("\nSaved results to:")
print(f"  {output_file}")
