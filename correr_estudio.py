"""Ejecuta el estudio de los capitulos 02 y 03 fuera de Jupyter, por grupos.

Uso (doble clic en los .bat, o desde la terminal con el entorno activado):

    python correr_estudio.py --grupo gpu     # Random Forest, XGBoost y KNN (GPU)
    python correr_estudio.py --grupo gpu2    # los mismos, en orden inverso
    python correr_estudio.py --grupo cpu     # Logistica, NB, arbol, SVM, Ridge, Lasso, SVR
    python correr_estudio.py --grupo cpu2    # los mismos, en orden inverso
    python correr_estudio.py --grupo knn_cpu # KNN con FAISS en la CPU (sin GPU)
    python correr_estudio.py --grupo cpu3    # tercer trabajador de CPU (SVM primero)

Los grupos estan pensados para correr AL MISMO TIEMPO, en varias ventanas. ``gpu``
y ``gpu2`` recorren la misma lista de modelos desde extremos opuestos (la GPU
quedaba ociosa ~30 % del tiempo con un solo proceso); lo mismo ``cpu`` y ``cpu2``.
Cada unidad se reserva con un archivo de bloqueo (``resultados/bloqueos/``), de
modo que dos ventanas nunca corren la misma unidad: cuando se encuentran, se
reparten lo que falta. Cada ventana usa solo su parte de los nucleos
(``--nucleos``), escribe sus temporales en una carpeta propia y guarda cada
unidad terminada en ``resultados/corridas/``, el mismo registro que usan los
notebooks. No cambia nada del diseno: es el mismo estudio, en otro orden.

Se puede cerrar la ventana o apagar el equipo en cualquier momento: al volver a
ejecutarlo continua donde quedo. El progreso queda en resultados/log_<grupo>.txt.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Cada grupo es una lista ordenada de (tarea, modelo, balanceos); None = todos.
# El orden prioriza lo mas informativo (26-sep): terminar modelos completos antes
# que las combinaciones con remuestreo de los modelos mas lentos. No cambia el
# diseno ni los resultados de ninguna unidad, solo el orden en que se ejecutan.
C, R = "clasificacion", "regresion"
GRUPOS = {
    "gpu": [(C, "random_forest", None),
            (C, "xgboost", ["ninguno", "class_weight"]),
            (R, "xgboost", None),
            (R, "random_forest", None),
            (C, "xgboost", ["smote", "adasyn"]),
            (C, "knn", None), (R, "knn", None)],
    "cpu": [(C, "logistica", None), (C, "arbol", None), (C, "naive_bayes", None),
            (C, "svm", None),
            (R, "ridge", None), (R, "lasso", None), (R, "arbol", None), (R, "svr", None)],
    "cpu2": [(C, "arbol", None), (C, "naive_bayes", None), (C, "logistica", None),
             (C, "svm", None)],
    "cpu3": [(C, "naive_bayes", None), (C, "arbol", None), (C, "logistica", None),
             (C, "svm", None)],
    # KNN con busqueda exacta de FAISS en la CPU (mismos vecinos que PyTorch)
    "knn_cpu": [(C, "knn", None), (R, "knn", None)],
}
GRUPOS["gpu2"] = list(reversed(GRUPOS["gpu"]))
NUCLEOS_DEFECTO = {"gpu": 2, "gpu2": 2, "cpu": 4, "cpu2": 3, "knn_cpu": 3, "cpu3": 3}

ap = argparse.ArgumentParser()
ap.add_argument("--grupo", choices=list(GRUPOS), required=True)
ap.add_argument("--nucleos", type=int, default=None,
                help="hilos de CPU para este grupo (por defecto: gpu=2, gpu2=2, cpu=4, cpu2=3, knn_cpu=3, cpu3=3)")
args = ap.parse_args()

total = os.cpu_count() or 4
nucleos = args.nucleos or min(NUCLEOS_DEFECTO[args.grupo], max(1, total - 1))
# antes de importar numpy: limita los hilos de BLAS/OpenMP de este proceso
for var in ("E3_NUCLEOS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[var] = str(nucleos)
if args.grupo == "knn_cpu":
    os.environ["E3_KNN_MOTOR"] = "faiss"

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
os.environ["PYTHONPATH"] = str(RAIZ) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.chdir(RAIZ)

import warnings  # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)

from src import config  # noqa: E402

config.crear_directorios()
LOG = config.DIR_RESULTADOS / f"log_{args.grupo}.txt"


class Tee:
    def __init__(self, *flujos):
        self.flujos = flujos

    def write(self, s):
        for f in self.flujos:
            try:
                f.write(s)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.flujos:
            try:
                f.flush()
            except Exception:
                pass


archivo_log = open(LOG, "a", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, archivo_log)
sys.stderr = Tee(sys.__stderr__, archivo_log)

from threadpoolctl import threadpool_limits  # noqa: E402
from src import datos, experimento as ex  # noqa: E402

print(f"\n===== Grupo {args.grupo.upper()} | inicio {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
print(config.resumen())
if not datos.existe_traspaso():
    print("\nERROR: falta el traspaso de datos. Ejecute primero ESTIMAR_TIEMPOS.bat o el capitulo 01.")
    sys.exit(1)
if not (config.DIR_CACHE / "particiones" / "manifiesto.json").is_file():
    from src import preprocesamiento as prep
    print("\nConstruyendo particiones (una sola vez)...")
    prep.preparar_particiones(datos.cargar_traspaso())

t0 = time.time()
with threadpool_limits(limits=nucleos):
    for tarea, m, balanceos in GRUPOS[args.grupo]:
        etiqueta = f"{tarea} / {m}" + (f" / {', '.join(balanceos)}" if balanceos else "")
        print(f"\n----- {etiqueta} | {time.strftime('%Y-%m-%d %H:%M:%S')} -----", flush=True)
        try:
            ex.ejecutar(tarea, modelos=[m], balanceos=balanceos)
        except KeyboardInterrupt:
            print("\nDetenido por el usuario. Lo terminado esta guardado; "
                  "vuelva a ejecutar para continuar.")
            sys.exit(0)

print(f"\n===== Grupo {args.grupo.upper()} terminado en {(time.time() - t0) / 3600:.1f} h "
      f"| {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
av = ex.avance()
grupo = {(t, m) for t, m, _ in GRUPOS[args.grupo]}
print(av[[(t, m) in grupo for t, m in zip(av["tarea"], av["modelo"])]].to_string(index=False))
