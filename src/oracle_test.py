"""Oracle ("teacher forcing") benchmark.

How much of the error that a recursive model accumulates at long horizons comes
from feeding back its own (wrong) predictions, and how much is intrinsic to the
model? For each selected model and horizon H, every forecast origin is replayed
step by step: predict 1 step -> record -> update the model state with the TRUE
value of that step -> next step.

Forecast origins are taken from the reference predictions of the same model
(<model>/predictions/predictions_H<H>.csv, written by testing.py), so the oracle
can be compared origin by origin with the recursive / direct runs that produced
them. If a reference file is missing, origins are computed with the same rule
as testing.py (evaluation.n_origins or --n-origins).

Outputs (per room / metric / model):
  <model>/predictions_oracle/predictions_oracle_H<H>.csv   same layout as predictions_H<H>.csv
  <results_path>/oracle_<metric>_room<room>.csv            per room: recursive vs oracle,
                                                           mean and variance over origins
  <results_path>/oracle_<metric>_all_rooms.csv             all rooms pooled (mean, variance)
  <results_path>/oracle_<metric>_per_origin.csv            every origin of every room (for custom stats/plots)
Latency is NOT measured: replaying with true values is not a deployment scenario.
"""
from pathlib import Path
import argparse

import pandas as pd
import yaml

from libs.paths import PROJECT_ROOT, project_path, config_path
from libs.config_utils import resolve_metric, resolve_rooms, evaluation_settings
from libs.data_utils import prepare_dataset
from libs.arima_utils import ArimaForecaster
from libs.lstm_utils import LSTMForecaster
from libs.benchmark_utils import (
    forecast_origin_positions,
    oracle_rolling_predict,
    per_origin_errors,
)

BUILDERS = {"ArimaForecaster": ArimaForecaster, "LSTMForecaster": LSTMForecaster}
DEFAULT_MODELS = "LSTM_DENSE_MED_W60_H1,LSTM_DENSE_MED_W240_H1"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default="configs/base.yaml")
parser.add_argument("--metric", required=True)
parser.add_argument("--room", default=None,
                    help="one room or a comma-separated list, e.g. 413,419 (default: all data.rooms)")
parser.add_argument("--models", default=DEFAULT_MODELS,
                    help=f"comma-separated model names from the config (default: {DEFAULT_MODELS})")
parser.add_argument("--horizons", default=None,
                    help="comma-separated, e.g. 10,30,60 (default: evaluation.horizons without H=1)")
parser.add_argument("--reference-root", default=None,
                    help="folder with the same layout as output.path (<room>/<metric>/<model>/predictions/"
                         "predictions_H<H>.csv) to take the origins from, e.g. an older run extracted "
                         "elsewhere (default: output.path, i.e. the latest testing.py run)")
parser.add_argument("--n-origins", type=int, default=None,
                    help="only used when a reference predictions file is missing")
parser.add_argument("--tag", default="",
                    help="suffix for every output (predictions_oracle_<tag>/, oracle_<metric>_<tag>_*.csv), "
                         "so runs against different references do not overwrite each other, e.g. n100")
parser.add_argument("--require-reference", action="store_true",
                    help="stop with an error if a reference predictions file is missing "
                         "(instead of falling back to the config rule)")
args = parser.parse_args()
TAG = f"_{args.tag}" if args.tag else ""

CONFIG_PATH = config_path(args.config)
config = yaml.safe_load(open(CONFIG_PATH))
print(f"Configuration loaded: {CONFIG_PATH}")
print(f"Project root:         {PROJECT_ROOT}")

data_cfg, output_cfg = config["data"], config["output"]
metric = resolve_metric(data_cfg, args.metric)
rooms = [r for part in (args.room.split(",") if args.room else [None])
         for r in resolve_rooms(data_cfg, part.strip() if part else None)]
eval_cfg = evaluation_settings(config, args.n_origins)
horizons = ([int(h) for h in args.horizons.split(",")] if args.horizons
            else [h for h in eval_cfg["horizons"] if h > 1])
model_names = [m.strip() for m in args.models.split(",") if m.strip()]
models_cfg = {m["name"]: m for m in config["models"]}
missing = [m for m in model_names if m not in models_cfg]
if missing:
    raise ValueError(f"Models not in the config: {missing}")

print(f"Metric: {metric} | Rooms: {rooms} | Models: {model_names} | Horizons: {horizons}")

results_dir = project_path(output_cfg.get("results_path", "results"))
results_dir.mkdir(parents=True, exist_ok=True)
all_errors = []   # per-origin errors of every room, for the summary over all rooms

for room in rooms:
    print("\n" + "=" * 60 + f"\nROOM {room} - {metric.upper()}\n" + "=" * 60)
    df = prepare_dataset(
        path=str(project_path(data_cfg.get("path", "data/archive/KETI")) / str(room) / f"{metric}.csv"),
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
    train_size = int(len(series) * data_cfg["train_ratio"])
    test_series = series.iloc[train_size:]
    print(f"Test points: {len(test_series)}")

    summary = []
    for name in model_names:
        model_path = project_path(output_cfg["path"]) / str(room) / metric / name
        if not model_path.exists():
            print(f"Skipping {name}: model not found ({model_path})")
            continue
        model = BUILDERS[models_cfg[name]["builder"]].load(model_path)
        print(f"\n--- {name} ---")

        # Origins: from the reference predictions of the same model, if available
        origins_by_H, reference = {}, {}
        for H in horizons:
            ref_root = project_path(args.reference_root) if args.reference_root else project_path(output_cfg["path"])
            ref_file = ref_root / str(room) / metric / name / "predictions" / f"predictions_H{H}.csv"
            if ref_file.exists():
                ref = pd.read_csv(ref_file, index_col="forecast_origin", parse_dates=True)
                reference[H] = ref
                origins_by_H[H] = [test_series.index.get_loc(o) for o in ref.index]
                print(f"  H={H}: {len(ref)} origins from {ref_file}")
            elif args.require_reference:
                raise FileNotFoundError(f"Reference predictions missing: {ref_file}")
            else:
                origins_by_H[H] = list(forecast_origin_positions(len(test_series), H, eval_cfg["n_origins"]))
                print(f"  H={H}: no reference file, {len(origins_by_H[H])} origins from the config rule")

        oracle = oracle_rolling_predict(model, test_series, origins_by_H)

        out_dir = model_path / f"predictions_oracle{TAG}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for H, pred in oracle.items():
            out_file = out_dir / f"predictions_oracle_H{H}.csv"
            pred.to_csv(out_file)
            print(f"  saved {out_file.relative_to(PROJECT_ROOT)}")

            variants = {"oracle": pred}
            if H in reference:
                # *_H1 models (output_steps=1) reach H by feeding back their own predictions
                if models_cfg[name]["builder"] == "ArimaForecaster":
                    ref_label = "arima_multistep"      # statsmodels H-step forecast
                else:
                    steps = models_cfg[name]["params"].get("output_steps", 1)
                    ref_label = "recursive" if steps == 1 else f"block{steps}"
                variants[ref_label] = reference[H]
                # sanity check: step h1 is the same one-step forecast in both runs
                d = (pred["h1"] - reference[H]["h1"]).abs().max()
                print(f"  H={H}: max |oracle h1 - reference h1| = {d:.3g} (expected ~0)")

            for variant, p in variants.items():
                e = per_origin_errors(test_series, p)
                all_errors.append(e.reset_index().assign(room=room, model=name, H=H, variant=variant))
                row = {"room": room, "metric": metric, "model": name, "H": H,
                       "variant": variant, "n_origins": len(e)}
                for col in ["mae", "rmse", "mape", "smape"]:
                    row[f"{col}_mean"] = e[col].mean()
                    row[f"{col}_var"] = e[col].var(ddof=1)
                summary.append(row)

    if summary:
        out = results_dir / f"oracle_{metric}{TAG}_room{room}.csv"
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"\nSummary saved to {out}")
        print(pd.DataFrame(summary).to_string(index=False))

if all_errors:
    errs = pd.concat(all_errors, ignore_index=True)
    errs.insert(0, "metric", metric)
    errs.to_csv(results_dir / f"oracle_{metric}{TAG}_per_origin.csv", index=False)
    pooled = (errs.groupby(["model", "H", "variant"])
                  .agg(n_rooms=("room", "nunique"), n_origins=("mae", "size"),
                       **{f"{c}_{a}": (c, a) for c in ["mae", "rmse", "mape", "smape"] for a in ["mean", "var"]})
                  .reset_index())
    pooled.insert(0, "metric", metric)
    out = results_dir / f"oracle_{metric}{TAG}_all_rooms.csv"
    pooled.to_csv(out, index=False)
    print(f"Rooms in the summary: {sorted(errs['room'].unique().tolist())}")
    print(f"\nSummary over all rooms (every origin of every room pooled) saved to {out}")
    print(pooled.to_string(index=False))
    print(f"Per-origin errors saved to {results_dir / f'oracle_{metric}{TAG}_per_origin.csv'}")

print("\nOracle benchmark completed.")
