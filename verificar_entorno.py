"""Comprueba el entorno antes de ejecutar el libro.

Uso:  python verificar_entorno.py

Revisa la version de Python, las librerias fijadas en requirements.txt, la GPU
para XGBoost, FAISS, el espacio libre en disco y la ubicacion de los datos.
"""
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

print(f"Python {platform.python_version()} ({platform.system()})")
if sys.version_info[:2] != (3, 10):
    print("  AVISO: el entorno esta fijado para Python 3.10")

requeridas = {}
for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
    linea = linea.split("#")[0].strip()
    if "==" in linea:
        n, v = linea.split("==")
        requeridas[n.strip()] = v.strip()
modulo = {"deap": "deap", "scikit-learn": "sklearn", "imbalanced-learn": "imblearn", "faiss-cpu": "faiss",
          "memory-profiler": "memory_profiler", "scikit-posthocs": "scikit_posthocs",
          "jupyter-book": "jupyter_book"}
problemas = 0
for paquete, version in requeridas.items():
    try:
        m = importlib.import_module(modulo.get(paquete, paquete.replace("-", "_")))
        v = getattr(m, "__version__", "?")
        estado = "ok" if (v == version or v == "?" or version.startswith(str(v) + ".")) else f"instalada {v}"
        if estado != "ok":
            problemas += 1
        print(f"  {paquete:18s} {version:12s} {estado}")
    except Exception as e:
        problemas += 1
        print(f"  {paquete:18s} {version:12s} FALTA ({type(e).__name__})")

from src import config  # noqa: E402
print(f"\nNucleos disponibles para el estudio: {config.NUCLEOS}")
print(f"XGBoost usara device='{config.DISPOSITIVO_XGB}'"
      + ("" if config.DISPOSITIVO_XGB == "cuda" else
         "  (sin GPU detectada: revise el controlador de NVIDIA si esperaba usarla)"))
try:
    import torch
    print(f"PyTorch {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
except ImportError:
    print("PyTorch no instalado: KNN usara FAISS en CPU (ejecute INSTALAR_GPU.bat)")
print(f"Random Forest: {config.RF_MOTOR} | KNN: {config.KNN_MOTOR}")
libre = shutil.disk_usage(RAIZ).free / 2**30
print(f"Espacio libre en disco: {libre:,.0f} GB" + ("  AVISO: se recomiendan >= 20 GB" if libre < 20 else ""))
datos = os.environ.get("HOME_CREDIT_DIR")
print(f"HOME_CREDIT_DIR: {datos or '(no definida; el capitulo 01 buscara los CSV automaticamente)'}")
print(f"Traspaso del capitulo 01: {'existe' if (config.DIR_DATOS / 'meta.json').is_file() else 'aun no'}")
print("\nEntorno listo." if problemas == 0 else f"\n{problemas} paquete(s) con version distinta o ausentes.")
