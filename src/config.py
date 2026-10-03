"""Configuracion global del Entregable 3.

Todo lo que fija el comportamiento del proyecto vive aqui: la semilla unica, las
rutas (siempre relativas a la raiz del libro, nunca absolutas) y el presupuesto
computacional del estudio combinatorio. Los notebooks importan este modulo y no
redefinen ninguno de estos valores.

Modo prueba
-----------
Si la variable de entorno ``E3_MODO_PRUEBA=1`` esta definida, el proyecto usa
datos sinteticos pequenos y presupuestos minimos. Sirve exclusivamente para
verificar que el codigo corre de principio a fin antes de lanzar la corrida real;
ninguna cifra producida en ese modo tiene valor analitico.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Semilla unica del proyecto
# ---------------------------------------------------------------------------
SEMILLA: int = 42
"""Semilla global. Se propaga a numpy, scikit-learn, imbalanced-learn, XGBoost,
FAISS (PCA previo), Optuna y DEAP. Es la misma del notebook de la linea base, de
modo que la particion de prueba coincide con la del modelo logistico."""


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """Fija las semillas de ``random`` y ``numpy`` y la del hash de Python.

    Parameters
    ----------
    semilla : int
        Valor de la semilla.

    Notes
    -----
    Los estimadores reciben ademas ``random_state=semilla`` de forma explicita al
    construirse (ver :mod:`src.modelos`); fijar la semilla global no basta para
    scikit-learn, que usa su propio generador por estimador.
    """
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)


# ---------------------------------------------------------------------------
# Rutas: todas relativas a la raiz del libro
# ---------------------------------------------------------------------------
def _raiz_proyecto() -> Path:
    """Localiza la raiz del libro (la carpeta que contiene ``_config.yml`` y ``src``).

    Se busca hacia arriba desde el directorio de trabajo, de modo que los notebooks
    funcionan igual en cualquier computador sin editar ninguna ruta. La variable de
    entorno ``E3_RAIZ`` permite forzarla.
    """
    forzada = os.environ.get("E3_RAIZ")
    if forzada:
        return Path(forzada).resolve()
    actual = Path.cwd().resolve()
    for carpeta in (actual, *actual.parents):
        if (carpeta / "_config.yml").is_file() and (carpeta / "src").is_dir():
            return carpeta
    return actual


RAIZ = _raiz_proyecto()
DIR_DATOS = RAIZ / "datos"                 # traspaso que produce el capitulo 01
DIR_CACHE = RAIZ / "cache"                 # particiones ya preprocesadas (.npy)
DIR_RESULTADOS = RAIZ / "resultados"
DIR_CORRIDAS = DIR_RESULTADOS / "corridas"         # un .json por unidad de trabajo
DIR_PREDICCIONES = DIR_RESULTADOS / "predicciones"  # puntajes fuera de muestra
DIR_FINALES = DIR_RESULTADOS / "finales"            # modelos finales y prueba
DIR_ESTUDIOS = DIR_RESULTADOS / "estudios"          # estudios especificos (cap. 04-05)
DIR_FIGURAS = RAIZ / "figuras"

MODO_PRUEBA: bool = os.environ.get("E3_MODO_PRUEBA", "0") == "1"
if MODO_PRUEBA:
    # En modo prueba todo se escribe en una carpeta aparte, para que una
    # verificacion jamas mezcle sus archivos con los de la corrida real.
    _p = RAIZ / "_prueba"
    DIR_DATOS = _p / "datos"
    DIR_CACHE = _p / "cache"
    DIR_RESULTADOS = _p / "resultados"
    DIR_CORRIDAS = DIR_RESULTADOS / "corridas"
    DIR_PREDICCIONES = DIR_RESULTADOS / "predicciones"
    DIR_FINALES = DIR_RESULTADOS / "finales"
    DIR_ESTUDIOS = DIR_RESULTADOS / "estudios"
    DIR_FIGURAS = _p / "figuras"


def crear_directorios() -> None:
    """Crea (si faltan) todas las carpetas de trabajo del proyecto."""
    for d in (DIR_DATOS, DIR_CACHE, DIR_RESULTADOS, DIR_CORRIDAS,
              DIR_PREDICCIONES, DIR_FINALES, DIR_ESTUDIOS, DIR_FIGURAS):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Presupuesto del estudio combinatorio (Seccion 2 y 3.2 de la guia)
# ---------------------------------------------------------------------------
def _entero_env(nombre: str, defecto: int) -> int:
    valor = os.environ.get(nombre)
    return int(valor) if valor else defecto


N_EXTERNOS: int = _entero_env("E3_N_EXTERNOS", 3 if MODO_PRUEBA else 5)
"""Pliegues del bucle externo. Solo estiman el desempeno de generalizacion."""

N_INTERNOS: int = _entero_env("E3_N_INTERNOS", 2 if MODO_PRUEBA else 3)
"""Pliegues del bucle interno, donde corre cada optimizador."""

N_EVALUACIONES: int = _entero_env("E3_N_EVALUACIONES", 4 if MODO_PRUEBA else 20)
"""Presupuesto de evaluaciones por optimizador y por pliegue externo. Es el MISMO
para Grid, Random, Bayesiano y Genetico: comparar metodos con presupuestos
distintos no dice nada sobre su eficiencia."""

PROPORCION_PARADA_TEMPRANA: float = 0.10
"""Fraccion del entrenamiento (real, antes de balancear) que XGBoost reserva para
la parada temprana. Nunca se toma del pliegue de validacion."""

SMOTE_PROPORCION: float = 0.5
"""``sampling_strategy`` de SMOTE y ADASYN: la clase minoritaria sube hasta la
mitad de la mayoritaria (de 8 % a 33 % de positivos). Llevarla a 1:1 duplicaria
el tamano del entrenamiento y generaria mas de 150 mil filas sinteticas por
pliegue, casi todas interpoladas entre los mismos ~16 mil casos reales."""

COSTE_FN: int = 10
"""Un incumplimiento no detectado cuesta 10 veces lo que rechazar a un buen
cliente. Es el mismo criterio de coste de la linea base logistica."""

NYSTROEM_COMPONENTES: int = 60 if MODO_PRUEBA else 500
"""Dimension de la aproximacion de Nystroem al nucleo RBF en SVM/SVR."""

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
NUCLEOS: int = _entero_env("E3_NUCLEOS", max(1, (os.cpu_count() or 2) - 1))
"""Hilos disponibles: por defecto se deja uno libre para el sistema operativo.
``E3_NUCLEOS`` lo limita cuando se ejecutan dos grupos de modelos a la vez
(``correr_estudio.py``), para que no compitan por los mismos nucleos."""

PARALELIZAR_INTERNOS: bool = os.environ.get("E3_PARALELO", "1") == "1"
"""Si es True, los pliegues internos de los modelos de un solo hilo (Naive Bayes,
arbol, lineales, SVM) se ajustan en paralelo con joblib/loky. Las matrices se
comparten por memoria mapeada, de modo que no se copian por proceso."""

BLOQUE_PREDICCION: int = 50_000
"""Filas por bloque al predecir: acota el pico de memoria."""


def gpu_disponible() -> bool:
    """Indica si XGBoost puede usar una GPU NVIDIA con CUDA.

    Requiere dos cosas: que la rueda de XGBoost este compilada con CUDA y que el
    controlador de NVIDIA responda (``nvidia-smi``).
    """
    if os.environ.get("E3_SIN_GPU") == "1":
        return False
    try:
        import xgboost as xgb
        if not xgb.build_info().get("USE_CUDA", False):
            return False
    except Exception:
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=15)
        return True
    except Exception:
        return False


DISPOSITIVO_XGB: str = "cuda" if gpu_disponible() else "cpu"
"""Dispositivo de XGBoost en el estudio principal (Seccion 4.3). Tambien lo usa el
Random Forest, que se entrena con el modo bosque aleatorio de XGBoost."""

RF_MOTOR: str = os.environ.get("E3_RF_MOTOR", "xgbrf")
"""Implementacion del Random Forest: ``'xgbrf'`` (bosque aleatorio de XGBoost, en
GPU si existe) o ``'sklearn'`` (RandomForest de scikit-learn, CPU). La medicion
del capitulo 02 mostro que el de scikit-learn se llevaba el 72 % del tiempo total
del estudio; el capitulo 05 compara ambos."""


def _torch_cuda() -> bool:
    if os.environ.get("E3_SIN_GPU") == "1":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


KNN_MOTOR: str = os.environ.get("E3_KNN_MOTOR", "torch" if _torch_cuda() else "faiss")
"""Motor de busqueda de vecinos del KNN: ``'torch'`` (busqueda exacta por fuerza
bruta en la GPU con PyTorch) o ``'faiss'`` (busqueda exacta en CPU). Ambas
devuelven los mismos vecinos; solo cambia la velocidad."""


def resumen() -> str:
    """Texto con la configuracion vigente, para imprimir al inicio de cada notebook."""
    lineas = [
        f"Raiz del proyecto     : {RAIZ.name}",
        f"Modo prueba           : {MODO_PRUEBA}",
        f"Semilla               : {SEMILLA}",
        f"CV anidada            : {N_EXTERNOS} externos x {N_INTERNOS} internos",
        f"Evaluaciones/optimiz. : {N_EVALUACIONES}",
        f"Nucleos               : {NUCLEOS}   paralelo interno: {PARALELIZAR_INTERNOS}",
        f"XGBoost               : device='{DISPOSITIVO_XGB}'",
        f"Random Forest         : {RF_MOTOR}   KNN: {KNN_MOTOR}",
    ]
    return "\n".join(lineas)
