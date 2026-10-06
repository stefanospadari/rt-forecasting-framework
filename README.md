# rt-forecasting-framework

Framework per **training, valutazione e analisi** di modelli di forecasting
(ARIMA, LSTM) usati come fallback predittivo in pipeline IoT real-time.

Allena e testa modelli di forecasting (ARIMA, LSTM) sul
dataset [KETI](https://www.kaggle.com/datasets/ranakrc/smart-building-system) (5 stanze, 3 metriche: co2, temperature, humidity), con
misura separata di latenza di update/predict per il vincolo real-time
`T_update(un nuovo campione) + T_forecast ≤ T_s`.

## Struttura

```
rt-forecasting-framework/
  README.md
  setup.sh                 # crea le cartelle e verifica dati + container
  scripts/
    get_data.sh            # scarica e prepara il dataset KETI da Kaggle
  configs/
    base.yaml              # UNICA fonte di verità: percorsi, stanze, modelli, iperparametri
  src/
    training.py            # allena uno o più modelli, su una o più stanze
    testing.py             # valuta i modelli già allenati (accuratezza + latenza)
    libs/
      paths.py             # PROJECT_ROOT: tutti i percorsi sono relativi alla radice del progetto
      arima_utils.py       # ArimaForecaster (SARIMAX)
      lstm_utils.py        # LSTMForecaster (Keras, predict a blocchi + troncamento)
      data_utils.py        # prepare_dataset (caricamento + interpolazione)
      benchmark_utils.py   # rolling_predict, evaluate_rolling_forecast, metriche, latenza
  slurm/
    train.slurm            # training via Slurm + Apptainer
    test.slurm             # testing via Slurm + Apptainer
    smoke.slurm            # quick end-to-end check (~10 min), writes to smoke/<JOBID>/
    validate.sh            # smoke tests on all GPU partitions + report
    _common.sh             # shared logic (paths, bind, banner)
  containers/
    tf-gpu.def             # definizione dell'immagine (TF 2.21, CUDA/cuDNN da pip)
    requirements.lock.txt  # versioni esatte dei pacchetti Python
    build.sbatch           # build dell'immagine (partizione sbuild)
    test_gpu.sbatch        # smoke test GPU dell'immagine
    check_gpu.py
    submit_build.sh        # build + test in catena
  results/                 # CSV aggregati prodotti da testing.py      (in Git)
  data/archive/KETI/       # dataset                                   (NON in Git)
  trained_models/          # output di training.py, un modello = una cartella (NON in Git)
  logs/                    # log Slurm                                 (NON in Git)
```

Tutti i percorsi in `configs/base.yaml` (`data.path`, `output.path`,
`output.results_path`) sono **relativi alla radice del progetto**, non alla
cartella da cui si lancia lo script. La radice è ricavata da `src/libs/paths.py`
e si può forzare con la variabile d'ambiente `PROJECT_ROOT`.

## Avvio rapido (da zero, dopo `git clone`)

Requisiti sulla macchina: Apptainer ≥ 1.1, driver NVIDIA ≥ 525 (CUDA 12), `curl`, `unzip`.
Slurm è opzionale (vedi "Senza Slurm" sotto).

```bash
git clone https://github.com/stefanospadari/rt-forecasting-framework.git
cd rt-forecasting-framework

# 1) dataset KETI (Kaggle, pubblico) -> data/archive/KETI/<stanza>/<metrica>.csv
./scripts/get_data.sh
#    se il download automatico fallisce: scaricare archive.zip da
#    https://www.kaggle.com/datasets/ranakrc/smart-building-system  e poi
#    ./scripts/get_data.sh /percorso/archive.zip

# 2) container (solo la prima volta): build + test GPU, ~1-3 h a seconda del disco
./containers/submit_build.sh          # con Slurm (partizione sbuild: cambiare con sbatch -p, vedi sotto)
#    senza Slurm:  apptainer build --fakeroot containers/tf-gpu.sif containers/tf-gpu.def
#                  apptainer exec --nv containers/tf-gpu.sif python /opt/check_gpu.py

# 3) verifica cartelle, dataset, container
./setup.sh

# 4) prova end-to-end in piccolo (~10 min): deve finire con "SMOKE END - OK"
sbatch slurm/smoke.slurm
```

Poi training e testing come descritto in "Lanciare con Slurm". I modelli allenati
non sono nel repository: si ottengono lanciando il training (`trained_models/`).

Il container contiene **solo l'ambiente** (Python 3.10, TensorFlow 2.21,
Keras 3.12, CUDA 12.x e cuDNN 9 da pip, statsmodels, …), costruito da
`containers/tf-gpu.def` con le versioni esatte di `containers/requirements.lock.txt`.
Codice, config, dati e modelli restano fuori e vengono montati a runtime:
modificare il codice non richiede di ricostruire l'immagine.

> Nota tecnica: i wheel `tensorflow[and-cuda]==2.21.0` non includono
> `nvidia/cusolver/lib` (né `curand`, `nvrtc`, `nvjitlink`) nel loro RUNPATH,
> quindi TF non trova `libcusolver.so.11` e scarta la GPU
> ("Cannot dlopen some GPU libraries"). L'immagine esporta per questo tutte le
> cartelle `site-packages/nvidia/*/lib` in `LD_LIBRARY_PATH` (vedi `tf-gpu.def`).

### Su un altro cluster

Le partizioni di default negli script (`l40`, `l40s`, `sbuild`) sono quelle del
cluster DISI UniBo. Su un altro cluster basta passarne una diversa a `sbatch`
(`sbatch -p <partizione> ...`), oppure cambiare la riga `#SBATCH --partition`
nello script. `slurm/validate.sh` contiene l'elenco delle partizioni DISI da adattare.

### Senza Slurm (workstation con GPU)

Gli script in `slurm/` funzionano anche come normali script bash, lanciati dalla
radice del progetto:
```bash
METRIC=co2 bash slurm/train.slurm
METRIC=co2 N_ORIGINS=10 bash slurm/test.slurm
bash slurm/smoke.slurm
```

## `configs/base.yaml`

```yaml
data:
  rooms: [413, 419, 442, 510, 621]
  freq: 1min
  interpolate: true
  train_ratio: 0.8
training:
  seed: 42
output:
  path: trained_models
models:
  - name: ARIMA_311
    builder: ArimaForecaster
    params: {order: [3, 1, 1]}
  - name: LSTM_DENSE_MED_W60_H30
    builder: LSTMForecaster
    params:
      model_func: lstm_dense
      model_params: {units: 128, dense_units: 128}
      train_params: {...}
      window: 60
      output_steps: 30
  # ... 14 modelli in totale
```

Per aggiungere un modello nuovo: aggiungi una voce a `models:` qui, basta —
training e testing lo vedono automaticamente, nessun altro file da toccare.

## `training.py` — argomenti

| Argomento        | Default        | Significato                                                              |
|-------------------|----------------|----------------------------------------------------------------------------|
| `--config`         | `configs/base.yaml` | Path del config YAML (di norma `configs/base.yaml`)                         |
| `--metric`         | `None`         | `co2` \| `temperature` \| `humidity`. Sovrascrive `data.metric` e imposta i `min_val`/`max_val` corretti per quella metrica (0–2000 per co2, 0–50 temperature, 0–100 humidity) |
| `--room`           | tutte (da config) | Allena solo quella stanza, invece di tutte quelle in `data.rooms`        |
| `--model-index`    | tutti (da config) | Allena solo il modello a quell'indice (0-based) nella lista `models:` del config, invece di tutti |

Esempi diretti (senza Slurm, per debug locale dentro il container):
```bash
# allena TUTTI i 14 modelli, TUTTE le 5 stanze, metrica co2
python src/training.py --config configs/base.yaml --metric co2

# allena solo il modello indice 3 (controlla l'indice con lo snippet sotto), solo stanza 413
python src/training.py --config configs/base.yaml --metric temperature --model-index 3 --room 413
```

Per sapere l'indice di un modello per nome (dalla radice del progetto):
```bash
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/base.yaml'))
for i, m in enumerate(cfg['models']):
    print(i, m['name'])
"
```

## `testing.py` — argomenti

| Argomento        | Default        | Significato                                                              |
|-------------------|----------------|----------------------------------------------------------------------------|
| `--config`         | `configs/base.yaml` | Path del config YAML (di norma `configs/base.yaml`)                         |
| `--metric`         | `None`         | Come sopra per training                                                     |
| `--n-origins`      | `100`          | Numero di forecast origin per combinazione stanza×modello×orizzonte. Con valori diversi da 100 il file di output cambia nome (`results/benchmark_results_sparse_{metric}_n{N}.csv`) per non sovrascrivere il run "ufficiale" |

Esempio diretto:
```bash
# giro veloce, 10 origin, solo per sanity-check
python src/testing.py --config configs/base.yaml --metric co2 --n-origins 10

# run completo, 100 origin (default)
python src/testing.py --config configs/base.yaml --metric co2
```

`testing.py` NON supporta `--room`/`--model-index`: itera sempre su tutte
le stanze e tutti i modelli del config (filtrando in automatico quelli non
ancora allenati, con un messaggio `Skipping {model}: model not found`).

## Lanciare con Slurm

Sempre **dalla radice del progetto** (i log finiscono in `logs/`, i percorsi
sono risolti da lì). Parametri via `--export`. La partizione di default è `l40`;
si sovrascrive con `--partition` su `sbatch`.

```bash
# training di TUTTI i modelli/stanze per una metrica (ore)
sbatch --export=METRIC=co2 slurm/train.slurm

# training mirato: solo un modello, solo una stanza (minuti)
sbatch --export=METRIC=temperature,MODEL_INDEX=3,ROOM=413 slurm/train.slurm

# training su GPU h100 invece di l40
sbatch --partition=h100sxm5 --export=METRIC=co2 slurm/train.slurm

# testing veloce, 10 origin, per sanity-check prima del run vero
sbatch --export=METRIC=humidity,N_ORIGINS=10 slurm/test.slurm

# testing completo, 100 origin (default)
sbatch --export=METRIC=humidity slurm/test.slurm
```

**Latenze del paper:** sono misurate su L40 (partizione `l40`). Per i run
ufficiali di `test.slurm` non cambiare partizione, altrimenti le latenze non
sono confrontabili. Il training invece può girare su qualunque GPU.

Monitoraggio:
```bash
squeue -u $USER
tail -f logs/<nomejob>_<jobid>.out
```
I log sono in tempo reale (`python -u`, output non bufferizzato) — se un
job sembra "fermo" per diversi minuti senza nuove righe, controllare che
sia davvero bloccato (vedi sotto) prima di ucciderlo.

## Provare nuove configurazioni (esperimenti)

Regola d'oro: **non modificare `configs/base.yaml`** (è la configurazione dei
risultati del paper). Per un esperimento si crea un config nuovo, con una
**cartella di output propria**, così nulla viene sovrascritto:

```bash
cp configs/base.yaml configs/exp_lstm_wide.yaml
# in configs/exp_lstm_wide.yaml cambiare almeno:
#   output.path:         trained_models/exp_lstm_wide
#   output.results_path: results/exp_lstm_wide
# poi stanze, modelli, iperparametri a piacere (vedi sezione configs/base.yaml)

sbatch --export=METRIC=co2,CONFIG=configs/exp_lstm_wide.yaml slurm/train.slurm
sbatch --export=METRIC=co2,CONFIG=configs/exp_lstm_wide.yaml slurm/test.slurm
```

- Modelli disponibili: `ArimaForecaster` (parametro `order`, opzionale
  `fourier_params`) e `LSTMForecaster` con `model_func` tra `lstm_dense`,
  `lstm_dropout`, `baseline_lstm`, `stacked_lstm` (definite in `src/libs/lstm_utils.py`).
- Prima di un run lungo, una prova veloce: `--export=...,MODEL_INDEX=<i>,ROOM=413`
  sul training e `N_ORIGINS=10` sul testing.
- Modificare il codice in `src/` **non** richiede di ricostruire il container.
  Va ricostruito solo se servono pacchetti Python nuovi: aggiungerli a
  `containers/requirements.lock.txt` con versione fissata e lanciare
  `./containers/submit_build.sh`.
- Controllo rapido che tutto funzioni (container + GPU + percorsi, ~10 min):
  `sbatch slurm/smoke.slurm` (oppure `./slurm/validate.sh` per provarlo su tutte le partizioni GPU).

## Problemi noti / punti aperti

1. **Latenza di predict delle LSTM dominata dall'overhead di Keras.**
   `LSTMForecaster.predict` (`src/libs/lstm_utils.py`) chiama `self.model.predict(...)`
   a ogni passo; `predict()` ha un costo fisso di decine di ms per chiamata,
   indipendente dalla dimensione della rete (smoke test su L40S: ~80 ms/passo per
   la LSTM più piccola). Con `self.model(x, training=False)` il costo dovrebbe
   scendere a pochi ms (da verificare). Cambiarlo invalida le latenze già
   misurate (non i modelli allenati).
2. **`LSTM_DENSE_SMALL_W60_H1` in `base.yaml` ha `epochs: 5`** e niente early
   stopping, a differenza di tutte le altre LSTM (200 epoche + early stopping).
   Probabile residuo di una prova.
3. **`testing.py` sovrascrive i file per-modello** (`predictions/predictions_H*.csv`,
   `timings_sparse_H*.csv`) a ogni run, anche con `N_ORIGINS` diverso da 100:
   solo il CSV aggregato in `results/` ha il suffisso `_n{N}`. Un sanity-check con
   `N_ORIGINS=10` sui modelli ufficiali sovrascrive quindi i file del run ufficiale.
4. **GPU delle latenze:** il paper riporta tempi su L40 (partizione `l40`);
   i run ufficiali di `test.slurm` vanno fatti tutti sulla stessa partizione.

## Diagnostica job "silenziosi" o lenti

- `WARNING: Could not find any nv files on this host!` all'inizio del log
  è **innocuo**, non significa che la GPU non è disponibile — compare
  comunque anche quando la GPU viene usata correttamente (verificabile dal
  `Loaded cuDNN version ...` poco dopo, o dai tempi per-epoch delle LSTM).
- Per controllare se un job in corso sta davvero lavorando (senza ucciderlo):
```bash
  srun --jobid=<jobid> --overlap --pty nvidia-smi
  srun --jobid=<jobid> --overlap --pty top -bn1
```
- I modelli ARIMA (statsmodels/SARIMAX) girano su CPU, non su GPU: `nvidia-smi`
  a 0% durante il loro turno è normale.

## Metodologia di misura della latenza (importante per il paper)

`rolling_predict` (in `libs/benchmark_utils.py`) separa due cose per ogni
forecast origin:
1. un **catch-up non cronometrato**, che incorpora tutte le osservazioni
   del gap tranne l'ultima (necessario perché gli origin di valutazione
   sono sparsi, non consecutivi);
2. un **update cronometrato di un solo punto** (l'ultima osservazione prima
   dell'origin) — questo è il numero che rappresenta il vero costo per-ciclo
   nel deployment real-time (un nuovo campione per intervallo di campionamento).

Senza questa separazione, `Update latency` includerebbe anche il costo di
"recuperare" un gap di decine/centinaia di punti, gonfiando il numero di
ordini di grandezza rispetto al caso reale (visto e corretto durante questa
sessione: da ~8.4s a ~34ms su ARIMA_311).

`latency_stats_from_preds` calcola update/predict/total sempre sullo stesso
insieme di origin (si esclude il primo, cold-start, dove nessun update è
avvenuto), così `mean(total) == mean(update) + mean(predict)` torna sempre
esatto — utile per controllare a colpo d'occhio se qualcosa si è rotto.
