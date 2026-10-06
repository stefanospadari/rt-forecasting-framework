from pathlib import Path
import argparse

import tensorflow as tf
import yaml

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

from libs.arima_utils import ArimaForecaster
from libs.lstm_utils import (
    LSTMForecaster,
    lstm_dense,
    lstm_dropout,
    baseline_lstm,
    stacked_lstm,
    set_seeds,
)
from libs.data_utils import prepare_dataset
from libs.paths import PROJECT_ROOT, project_path, config_path


# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/base.yaml",
)
parser.add_argument(
    "--model-index",
    type=int,
    default=None,
)
parser.add_argument(
    "--metric",
    type=str,
    default=None,
)
parser.add_argument(
    "--room",
    type=str,
    default=None,
)
args = parser.parse_args()

CONFIG_PATH = config_path(args.config)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print(f"Configuration loaded: {CONFIG_PATH}")
print(f"Project root:         {PROJECT_ROOT}")

# ============================================================
# Device
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"GPU detected: {len(gpus)}")

    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("No GPU detected. Using CPU.")


# ============================================================
# Reproducibility
# ============================================================

set_seeds(config["training"]["seed"])


# ============================================================
# Dataset
# ============================================================

data_cfg = config["data"]
output_cfg = config["output"]

if args.metric is not None:
    data_cfg["metric"] = args.metric

    metric_limits = {
        "temperature": (0, 50),
        "humidity": (0, 100),
        "co2": (0, 2000),
    }

    if args.metric not in metric_limits:
        raise ValueError(f"Unsupported metric: {args.metric}")

    data_cfg["min_val"], data_cfg["max_val"] = metric_limits[args.metric]

rooms = data_cfg["rooms"]

if args.room is not None:
    rooms = [int(args.room)]

metric = data_cfg["metric"]

for room in rooms:

    print("\n" + "=" * 60)
    print(f"ROOM {room} - {metric.upper()}")
    print("=" * 60)

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

    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # ========================================================
    # Train / test split
    # ========================================================

    series = df["value"]

    train_ratio = data_cfg["train_ratio"]
    train_size = int(len(series) * train_ratio)

    train_series = series.iloc[:train_size]
    test_series = series.iloc[train_size:]

    print("\nDataset split:")
    print(f"  Total: {len(series)}")
    print(f"  Train: {len(train_series)}")
    print(f"  Test:  {len(test_series)}")

    # ========================================================
    # Output path
    # ========================================================

    output_path = (
        project_path(output_cfg["path"])
        / str(room)
        / metric
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nOutput path:")
    print(f"  {output_path}")


    # Build model configurations
    # ============================================================

    models = []


    for cfg in config["models"]:

        params = cfg["params"].copy()

        # --------------------------------------------------------
        # Resolve builder
        # --------------------------------------------------------

        builder = globals()[cfg["builder"]]


        # --------------------------------------------------------
        # Resolve LSTM model function
        # --------------------------------------------------------

        if "model_func" in params:

            params["model_func"] = globals()[
                params["model_func"]
            ]


        # --------------------------------------------------------
        # Build LSTM training parameters
        # --------------------------------------------------------

        if "train_params" in params:

            train_cfg = params["train_params"]

            callbacks = []

            if "early_stopping" in train_cfg:
                callbacks.append(
                    EarlyStopping(**train_cfg["early_stopping"])
                )

            if "reduce_lr" in train_cfg:
                callbacks.append(
                    ReduceLROnPlateau(**train_cfg["reduce_lr"])
                )

            params["train_params"] = {
                "batch_size": train_cfg["batch_size"],
                "epochs": train_cfg["epochs"],
                "validation_split": train_cfg["validation_split"],
                "optimizer": Adam(
                    learning_rate=train_cfg["learning_rate"],
                    clipnorm=train_cfg["clipnorm"],
                ),
                "loss": Huber(),
                "callbacks": callbacks,
            }

        # --------------------------------------------------------
        # Store resolved configuration
        # --------------------------------------------------------

        models.append(
            {
                "name": cfg["name"],
                "builder": builder,
                "params": params,
            }
        )

    if args.model_index is not None:
        if not 0 <= args.model_index < len(models):
            raise ValueError(
                f"Invalid model index: {args.model_index}. "
                f"Valid range: 0-{len(models) - 1}."
            )

        models = [models[args.model_index]]

    # ============================================================
    # Print models
    # ============================================================

    print("\nModels to train:")

    for cfg in models:

        print(
            f"  {cfg['name']}"
            f" -> {cfg['builder'].__name__}"
        )


    # ============================================================
    # Training
    # ============================================================

    print("\nStarting training...\n")


    for cfg in models:

        model = cfg["builder"](**cfg["params"])

        model.fit(train_series)

        model_path = output_path / cfg["name"]

        model.save(model_path)

        print(
            f"Saved: {model_path}"
        )


    print("\nTraining completed.")
