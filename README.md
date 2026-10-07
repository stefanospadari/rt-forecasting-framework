# rt-forecasting-framework

A framework to **train, evaluate and analyse** forecasting models (ARIMA, LSTM)
used as a predictive fallback in real-time IoT pipelines: when a sensor stops
sending data, a model fills the gap with forecasts, and it is useful only if it
produces them within the sampling interval.

For every model the framework measures accuracy (RMSE, MAE, sMAPE) and, separately,
the latency of the two operations a deployed model performs at each sampling interval,
so that the real-time constraint can be checked:

```
T_update(one new sample) + T_forecast(H)  ≤  Δt
```

Everything that defines an experiment — which series (rooms), which metrics, which
models and hyper-parameters, horizons, sampling interval, number of evaluation
points — lives in a YAML config. The reference setup (`configs/base.yaml`) uses the
public [KETI smart-building dataset](https://www.kaggle.com/datasets/ranakrc/smart-building-system), but any dataset with the same file layout works
(see [Using your own data](#using-your-own-data)).

## Layout

```
rt-forecasting-framework/
  README.md
  setup.sh                 # creates the folders, checks data + container against a config
  scripts/
    get_data.sh            # downloads and prepares the KETI dataset (Kaggle)
    config_info.py         # helper used by setup.sh to read a config
  configs/
    base.yaml              # reference experiment: paths, rooms, metrics, models, evaluation
  src/
    training.py            # trains one or more models on one or more rooms
    testing.py             # evaluates trained models (accuracy + latency)
    libs/
      paths.py             # PROJECT_ROOT: all paths are relative to the project root
      config_utils.py      # reads metrics / rooms / evaluation settings from the config
      arima_utils.py       # ArimaForecaster (SARIMAX)
      lstm_utils.py        # LSTMForecaster (Keras, block-wise predict + truncation)
      data_utils.py        # prepare_dataset (loading, cleaning, resampling, interpolation)
      benchmark_utils.py   # rolling_predict, metrics, latency statistics
  slurm/
    train.slurm            # training via Slurm + Apptainer
    test.slurm             # testing via Slurm + Apptainer
    smoke.slurm            # quick end-to-end check (~10 min), writes to smoke/<JOBID>/
    validate.sh            # smoke tests on several GPU partitions + report
    _common.sh             # shared logic (paths, bind mounts, banner)
  containers/
    tf-gpu.def             # Apptainer image definition (TF 2.21, CUDA/cuDNN from pip)
    requirements.lock.txt  # exact versions of the Python packages
    build.sbatch           # image build (Slurm job)
    test_gpu.sbatch        # GPU smoke test of the image
    check_gpu.py
    submit_build.sh        # build + GPU test, chained
  results/                 # aggregated CSVs produced by testing.py        (in Git)
  data/                    # datasets                                      (not in Git)
  trained_models/          # output of training.py, one folder per model   (not in Git)
  logs/                    # Slurm logs                                    (not in Git)
```

All paths in a config (`data.path`, `output.path`, `output.results_path`) are
**relative to the project root**, not to the directory you launch from. The root is
derived in `src/libs/paths.py` and can be forced with the `PROJECT_ROOT` environment
variable.

## Quick start (from scratch, after `git clone`)

Requirements: Apptainer ≥ 1.1, NVIDIA driver ≥ 525 (CUDA 12), `curl`, `unzip`.
Slurm is optional (see [Without Slurm](#without-slurm-gpu-workstation)).

```bash
git clone https://github.com/stefanospadari/rt-forecasting-framework.git
cd rt-forecasting-framework

# 1) KETI dataset (public, Kaggle) -> data/archive/KETI/<room>/<metric>.csv
./scripts/get_data.sh
#    if the automatic download fails, download archive.zip from
#    https://www.kaggle.com/datasets/ranakrc/smart-building-system  and run
#    ./scripts/get_data.sh /path/to/archive.zip

# 2) container (first time only): build + GPU test, 1-3 h depending on the disk
./containers/submit_build.sh
#    without Slurm:  apptainer build --fakeroot containers/tf-gpu.sif containers/tf-gpu.def
#                    apptainer exec --nv containers/tf-gpu.sif python /opt/check_gpu.py

# 3) check folders, dataset (rooms x metrics of the config) and container
./setup.sh                      # or: ./setup.sh configs/<your_config>.yaml

# 4) small end-to-end run (~10 min): must end with "SMOKE END - OK"
sbatch slurm/smoke.slurm
```

Then train and evaluate as described in [Running with Slurm](#running-with-slurm).
Trained models are not in the repository: they are produced by the training step.

The container holds **only the environment** (Python 3.10, TensorFlow 2.21, Keras 3.12,
CUDA 12.x and cuDNN 9 from pip, statsmodels, …), built from `containers/tf-gpu.def`
with the exact versions in `containers/requirements.lock.txt`. Code, configs, data and
models stay outside and are mounted at run time: changing the code or a config never
requires rebuilding the image.

> Technical note: the `tensorflow[and-cuda]==2.21.0` wheels do not include
> `nvidia/cusolver/lib` (nor `curand`, `nvrtc`, `nvjitlink`) in their RUNPATH, so
> TensorFlow cannot find `libcusolver.so.11` and silently drops the GPU
> ("Cannot dlopen some GPU libraries"). The image therefore exports every
> `site-packages/nvidia/*/lib` directory in `LD_LIBRARY_PATH` (see `tf-gpu.def`).

### On another cluster

The default partitions in the scripts (`l40`, `l40s`, `sbuild`) are those of the
DISI (University of Bologna) cluster. Elsewhere, pass another one to `sbatch`
(`sbatch -p <partition> ...`) or edit the `#SBATCH --partition` line.
Options given on the `sbatch` command line always override the `#SBATCH` lines.
`slurm/validate.sh` lists the DISI GPU partitions and must be adapted as well.

### Without Slurm (GPU workstation)

The scripts in `slurm/` also run as plain bash scripts, from the project root:
```bash
METRIC=co2 bash slurm/train.slurm
METRIC=co2 N_ORIGINS=10 bash slurm/test.slurm
bash slurm/smoke.slurm
```

## Configuring an experiment

A config has five sections. Annotated excerpt of `configs/base.yaml`:

```yaml
data:
  path: data/archive/KETI        # dataset root (relative to the project root)
  rooms: [413, 419, 442, 510, 621]   # series to use: one sub-folder each
  metrics:                       # metrics that can be passed with --metric
    temperature: {min_val: 0, max_val: 50}   # optional valid range: values outside
    humidity:    {min_val: 0, max_val: 100}  # are treated as sensor errors and
    co2:         {min_val: 0, max_val: 2000} # dropped before interpolation
  freq: 1min                     # resampling frequency (= sampling interval of the series)
  interpolate: true              # fill gaps after resampling
  train_ratio: 0.8               # chronological train/test split
  # optional: time_col, time_format (epoch | ms_epoch | iso), max_diff (spike filter)

training:
  seed: 42

evaluation:
  horizons: [1, 10, 30, 60]      # forecast horizons H (in steps) evaluated by testing.py
  delta_t_sec: 60                # sampling interval Δt: the real-time budget (feasibility = P95 / Δt)
  n_origins: 100                 # forecast origins per room x model x horizon

output:
  path: trained_models           # where models are saved: <path>/<room>/<metric>/<model>/
  results_path: results          # aggregated CSVs from testing.py

models:                          # every entry is trained and evaluated
  - name: ARIMA_311
    builder: ArimaForecaster
    params: {order: [3, 1, 1]}
  - name: LSTM_DENSE_MED_W60_H30
    builder: LSTMForecaster
    params:
      model_func: lstm_dense
      model_params: {units: 128, dense_units: 128}
      train_params: {batch_size: 32, epochs: 200, early_stopping: {...}, reduce_lr: {...}, ...}
      window: 60                 # input window (steps)
      output_steps: 30           # steps predicted per call (longer horizons are recursive)
  # ... 14 models in base.yaml
```

- **Rooms, metrics, horizons, Δt, number of origins and models are all read from the
  config**; nothing about them is hard-coded in the scripts. A run on a single room
  or with fewer origins can also be requested from the command line
  (`--room`, `--n-origins`).
- **Adding a model** = adding an entry under `models:`; training and testing pick it up
  automatically. Available builders: `ArimaForecaster` (`order`, optional
  `fourier_params: {period, K}`) and `LSTMForecaster` with `model_func` one of
  `lstm_dense`, `lstm_dropout`, `baseline_lstm`, `stacked_lstm`
  (defined in `src/libs/lstm_utils.py`).
- **Adding a metric** = adding it under `data.metrics` (the range is optional:
  `light: null` or `light: {}`), provided `<data.path>/<room>/light.csv` exists.
- `./setup.sh <config>` checks that every room × metric of a config is on disk.

### Using your own data

Any dataset organised as

```
<data.path>/<series_id>/<metric>.csv
```

can be used by pointing `data.path` to it and listing the series under `data.rooms`
(the name "room" comes from KETI; it can be any sensor, device or location id).
Each CSV holds a timestamp column and a value column: without a header the first
column is the time and the second the value; with a header, set `data.time_col`
(the value column must be named like the metric). Timestamps are parsed according to
`data.time_format` (`epoch` seconds by default, `ms_epoch`, or `iso`). Set `data.freq`
and `evaluation.delta_t_sec` to the sampling interval of your series.

### Trying new configurations (experiments)

Do **not** edit `configs/base.yaml` (it reproduces the reference results). Copy it and
give the copy **its own output folders**, so that nothing gets overwritten:

```bash
cp configs/base.yaml configs/exp_lstm_wide.yaml
# in configs/exp_lstm_wide.yaml change at least:
#   output.path:         trained_models/exp_lstm_wide
#   output.results_path: results/exp_lstm_wide
# then rooms, metrics, models, horizons ... as needed

./setup.sh configs/exp_lstm_wide.yaml
sbatch --export=METRIC=co2,CONFIG=configs/exp_lstm_wide.yaml slurm/train.slurm
sbatch --export=METRIC=co2,CONFIG=configs/exp_lstm_wide.yaml slurm/test.slurm
```

- Before a long run, try a quick one: `MODEL_INDEX=<i>,ROOM=<room>` for training and
  `N_ORIGINS=10` for testing.
- The container needs rebuilding only if new Python packages are required: add them,
  with a pinned version, to `containers/requirements.lock.txt` and run
  `./containers/submit_build.sh`.
- Quick check that everything works (container + GPU + paths, ~10 min):
  `sbatch slurm/smoke.slurm`, or `./slurm/validate.sh` to try several GPU partitions.

## `training.py` — arguments

| Argument        | Default             | Meaning |
|-----------------|---------------------|---------|
| `--config`      | `configs/base.yaml` | YAML config |
| `--metric`      | — (required)        | one of the keys of `data.metrics`; its `min_val`/`max_val` are applied |
| `--room`        | all `data.rooms`    | train only this room |
| `--model-index` | all models          | train only the model at this (0-based) index of `models:` |

```bash
# inside the container, from the project root
python src/training.py --config configs/base.yaml --metric co2
python src/training.py --config configs/base.yaml --metric temperature --model-index 3 --room 413
```

Index of each model by name:
```bash
python3 -c "
import yaml
for i, m in enumerate(yaml.safe_load(open('configs/base.yaml'))['models']):
    print(i, m['name'])"
```

## `testing.py` — arguments

| Argument      | Default                | Meaning |
|---------------|------------------------|---------|
| `--config`    | `configs/base.yaml`    | YAML config |
| `--metric`    | — (required)           | as for training |
| `--room`      | all `data.rooms`       | evaluate only this room (results file gets `_room<id>`) |
| `--n-origins` | `evaluation.n_origins` | forecast origins per room × model × horizon. With a value different from the config, the results file gets `_n<N>` so it does not overwrite the reference run |

Models listed in the config but not yet trained are skipped
(`Skipping <model>: model not found`).

Output: `<results_path>/benchmark_results_sparse_<metric>[_n<N>][_room<id>].csv`, one
row per room × model × horizon with RMSE, MAE, sMAPE, update / predict / total latency
(mean, P95, max, in ms), latency per step and the feasibility ratio `P95_total / Δt`
(feasible if < 1). Per-model predictions and timings are also written inside each model
folder.

## Running with Slurm

Always **from the project root** (logs go to `logs/`, paths are resolved from there).
Parameters are passed with `--export`; any `sbatch` option (partition, time, memory…)
can be added on the command line.

```bash
# train ALL models / rooms of the config for one metric (hours)
sbatch --export=METRIC=co2 slurm/train.slurm

# targeted training: one model, one room (minutes)
sbatch --export=METRIC=temperature,MODEL_INDEX=3,ROOM=413 slurm/train.slurm

# another config, another partition
sbatch -p h100sxm5 --export=METRIC=co2,CONFIG=configs/exp_lstm_wide.yaml slurm/train.slurm

# quick evaluation (10 origins) before the real run
sbatch --export=METRIC=humidity,N_ORIGINS=10 slurm/test.slurm

# full evaluation (evaluation.n_origins from the config)
sbatch --export=METRIC=humidity slurm/test.slurm
```

**Latency measurements:** latencies depend on the hardware. Run all the evaluations you
want to compare on the same partition/GPU type (the reference results use `l40`).
Training can run on any GPU.

Monitoring:
```bash
squeue -u $USER
tail -f logs/<jobname>_<jobid>.out
```
Logs are unbuffered (`python -u`): if a job prints nothing for several minutes, check
whether it is really stuck (see below) before cancelling it.

## Known issues / open points

1. **LSTM predict latency is dominated by Keras overhead.** `LSTMForecaster.predict`
   (`src/libs/lstm_utils.py`) calls `self.model.predict(...)` at every step;
   `predict()` has a fixed cost of tens of ms per call regardless of the network size
   (smoke test on L40S: ~80 ms/step for the smallest LSTM). Calling
   `self.model(x, training=False)` should bring it down to a few ms (to be verified).
   Changing it invalidates latencies already measured (not the trained models).
2. **`LSTM_DENSE_SMALL_W60_H1` in `base.yaml` has `epochs: 5`** and no early stopping,
   unlike all other LSTMs (200 epochs + early stopping). Probably left over from a test.
3. **`testing.py` overwrites the per-model files** (`predictions/predictions_H*.csv`,
   `timings_sparse_H*.csv`) at every run, even with a non-default `--n-origins`: only
   the aggregated CSV in `results/` gets the `_n<N>` suffix. A quick check with
   `N_ORIGINS=10` on the reference models therefore overwrites the reference per-model files.

## Diagnosing silent or slow jobs

- `WARNING: Could not find any nv files on this host!` at the top of a log is
  **harmless**: it also appears when the GPU is used correctly (check the
  `GPU detected: 1` line, or the per-epoch times of the LSTMs).
- To check whether a running job is actually working (without killing it):
  ```bash
  srun --jobid=<jobid> --overlap --pty nvidia-smi
  srun --jobid=<jobid> --overlap --pty top -bn1
  ```
- ARIMA models (statsmodels/SARIMAX) run on CPU: 0% GPU during their turn is normal.
- A container build spends most of its time in `Installing collected packages`
  without printing anything: on a slow shared disk this can take more than an hour.

## Latency measurement methodology

For every forecast origin, `rolling_predict` (`src/libs/benchmark_utils.py`) separates:

1. an **untimed catch-up**, which feeds the model all observations of the gap
   except the last one (needed because evaluation origins are sparse, not consecutive);
2. a **timed update with a single point** (the last observation before the origin):
   this is the number that represents the real per-cycle cost in a real-time
   deployment (one new sample per sampling interval).

Without this separation, the update latency would include the cost of catching up a
gap of tens or hundreds of points, inflating it by orders of magnitude (observed during
development: from ~8.4 s to ~34 ms on ARIMA_311).

`latency_stats_from_preds` computes update / predict / total statistics on the same set
of origins (the first, cold-start origin, where no update happened, is excluded), so
that `mean(total) == mean(update) + mean(predict)` always holds exactly — a quick
sanity check that nothing is broken.
