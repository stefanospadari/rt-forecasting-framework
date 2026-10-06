"""Container smoke test: run this BEFORE launching the training runs."""
import ctypes, os, sys, site, time
print("python      :", sys.version.split()[0], "|", sys.executable)
print("user site on:", site.ENABLE_USER_SITE, "(must be False)")
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

# cuDNN shipped with the image (pip wheel nvidia-cudnn-cu12): must be >= 9.3 for TF 2.21
try:
    import nvidia.cudnn as _c
    _lib = os.path.join(list(_c.__path__)[0], "lib", "libcudnn.so.9")
except ImportError:
    _lib = "libcudnn.so.9"
v = ctypes.CDLL(_lib).cudnnGetVersion()
print(f"cuDNN runtime: {v//10000}.{(v%10000)//100}.{v%100}  ({_lib})")
if v < 90300:
    sys.exit("ERROR: cuDNN < 9.3, TF 2.21 LSTMs will fail")

import numpy as np, tensorflow as tf
print("numpy       :", np.__version__)
print("tensorflow  :", tf.__version__, "| keras", tf.keras.__version__)
bi = tf.sysconfig.get_build_info()
print("built for   : CUDA", bi.get("cuda_version"), "| cuDNN", bi.get("cudnn_version"))
gpus = tf.config.list_physical_devices("GPU")
print("GPU visible :", gpus)
if not gpus:
    sys.exit("ERROR: no GPU. Did you forget --nv? Does the Slurm job have --gres=gpu:1?")
print("  ", tf.config.experimental.get_device_details(gpus[0]))

with tf.device("/GPU:0"):
    a = tf.random.normal((2000, 2000)); t = time.perf_counter()
    tf.matmul(a, a).numpy(); print(f"matmul OK   : {1e3*(time.perf_counter()-t):.1f} ms")

# LSTM with default arguments -> cuDNN kernel: this is the test that used to fail
x = np.random.rand(256, 60, 1).astype("float32"); y = np.random.rand(256, 1).astype("float32")
m = tf.keras.Sequential([tf.keras.Input((60, 1)), tf.keras.layers.LSTM(32), tf.keras.layers.Dense(1)])
m.compile("adam", tf.keras.losses.Huber())
m.fit(x, y, epochs=2, batch_size=32, verbose=0)
print("LSTM fit OK : loss", float(m.evaluate(x, y, verbose=0)))

p = "/tmp/_check.keras"; m.save(p); tf.keras.models.load_model(p); os.remove(p)
print("save/load OK")

import statsmodels
from statsmodels.tsa.arima.model import ARIMA
ARIMA(np.cumsum(np.random.randn(300)), order=(3, 1, 1)).fit()
print("ARIMA OK    : statsmodels", statsmodels.__version__)
print("\nALL OK")
