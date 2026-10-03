"""Ejecuta el cierre del Entregable 3 de principio a fin, sin abrir Jupyter.

Uso (doble clic en CORRER_CAPITULOS.bat, o desde la terminal con el entorno activado):

    python correr_capitulos.py                 # todo
    python correr_capitulos.py --desde 06      # retoma desde un capitulo
    python correr_capitulos.py --solo 05       # un solo capitulo (o varios: --solo 01 05 08)
    python correr_capitulos.py --solo-libro    # solo compila el libro (tras editar texto)
    python correr_capitulos.py --solo 06 A1 --con-libro   # unos capitulos y el libro
    python correr_capitulos.py --sin-reparar   # omite la revision de fallos
    python correr_capitulos.py --esperar       # espera a que termine el estudio y arranca solo

Pasos, en orden:

0. Revision de las evaluaciones fallidas del estudio y su reparacion
   (``src/reparacion.py``): lista en ``resultados/evaluaciones_fallidas.csv``,
   acciones en ``resultados/reparaciones.csv``.
1. Capitulos 01 a 09 y el anexo A1, cada uno ejecutado completo de
   arriba abajo en un kernel nuevo (lo que pide la correccion del profesor). Las
   salidas se guardan en el propio .ipynb. Si un capitulo falla, se guarda hasta
   donde llego, se registra el error y se sigue con el siguiente.
2. ``jupyter-book build .`` para generar el libro en ``_build/html``.

El capitulo 01 reescribe el traspaso de datos; si su contenido cambiara (no
deberia: misma semilla), se restaura la copia original para no invalidar el
estudio.

Todo queda registrado en ``resultados/log_capitulos.txt``; la salida completa de la
compilacion del libro (con todas sus advertencias) en ``resultados/log_libro.txt``.
"""
import argparse
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import nbformat
from jupyter_client.kernelspec import find_kernel_specs
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
CAPITULOS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "A1"]

ap = argparse.ArgumentParser()
ap.add_argument("--desde", default=None, help="capitulo desde el que empezar (p. ej. 04)")
ap.add_argument("--solo", nargs="+", default=None, help="ejecutar solo estos capitulos")
ap.add_argument("--solo-libro", action="store_true", help="solo compilar el libro")
ap.add_argument("--con-libro", action="store_true", help="con --solo: compilar el libro al final")
ap.add_argument("--sin-reparar", action="store_true", help="omitir la revision de fallos")
ap.add_argument("--sin-libro", action="store_true", help="no compilar el Jupyter Book")
ap.add_argument("--esperar", action="store_true",
                help="esperar a que las 700 unidades del estudio esten terminadas")
args = ap.parse_args()

from src import config  # noqa: E402

config.crear_directorios()
LOG = config.DIR_RESULTADOS / "log_capitulos.txt"
_log = open(LOG, "a", encoding="utf-8")


def log(msg=""):
    linea = f"{msg}"
    print(linea, flush=True)
    _log.write(linea + "\n")
    _log.flush()


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ejecutar_notebook(ruta: Path, kernel: str) -> bool:
    nb = nbformat.read(ruta, as_version=4)
    codigo = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
    for i in codigo:
        nb.cells[i].outputs = []
        nb.cells[i].execution_count = None
    cliente = NotebookClient(nb, timeout=None, kernel_name=kernel,
                             resources={"metadata": {"path": str(RAIZ)}})
    ok = True
    t0 = time.time()
    with cliente.setup_kernel():
        for n, i in enumerate(codigo, 1):
            celda = nb.cells[i]
            primera = celda.source.strip().splitlines()[0][:70] if celda.source.strip() else ""
            log(f"   [{n}/{len(codigo)}] {ahora()}  {primera}")
            try:
                cliente.execute_cell(celda, i)
            except Exception as e:
                log(f"\n   ERROR en la celda {n}:\n{str(e)[-3000:]}")
                ok = False
                break
            if n % 5 == 0:                        # guarda el avance cada 5 celdas
                nbformat.write(nb, ruta)
    nbformat.write(nb, ruta)
    log(f"   {'OK' if ok else 'CON ERROR'} en {(time.time() - t0) / 60:.1f} min")
    return ok


def _estudio_terminado():
    from src import experimento as ex
    from src.espacios import MODELOS
    todas = [u for t in ("clasificacion", "regresion") for u in ex.unidades(t, MODELOS[t])]
    faltan = [u for u in todas if not u.hecha]
    vivos = []
    for b in (config.DIR_RESULTADOS / "bloqueos").glob("*.lock"):
        try:
            pid = int(b.read_text().strip() or 0)
        except Exception:
            pid = 0
        if pid and ex._proceso_vivo(pid):
            vivos.append(b.stem)
    return len(todas), len(faltan), vivos


if args.esperar:
    log(f"\n===== Esperando a que termine el estudio | {ahora()} =====")
    while True:
        total, faltan, vivos = _estudio_terminado()
        if faltan == 0 and not vivos:
            log(f"{ahora()}  Estudio completo: {total} de {total} unidades. Se arranca el cierre.")
            break
        log(f"{ahora()}  faltan {faltan} de {total} unidades | en curso: {len(vivos)}"
            + ("" if vivos else "  (ninguna ventana del estudio esta corriendo: abralas)"))
        time.sleep(300)

log(f"\n===== Cierre del entregable | inicio {ahora()} =====")
kernels = find_kernel_specs()
KERNEL = "entregable3" if "entregable3" in kernels else "python3"
log(f"Kernel: {KERNEL}")

# --- 0. revision y reparacion de evaluaciones fallidas ------------------------
if not args.sin_reparar and not args.solo and not args.desde and not args.solo_libro:
    log(f"\n--- Revision de evaluaciones fallidas | {ahora()} ---")
    try:
        from src import reparacion as rep
        df = rep.listar_fallidas()
        log(f"Evaluaciones fallidas: {len(df)}")
        if len(df):
            log(df.groupby(["causa", "modelo", "optimizador"]).size().to_string())
        tabla = rep.reparar_todo()
        if len(tabla):
            log(tabla.to_string(index=False))
        log("Revision terminada.")
    except Exception:
        log("ERROR en la revision (se continua con los capitulos):\n" + traceback.format_exc())

# --- 1. capitulos -----------------------------------------------------------------
lista = CAPITULOS
if args.solo_libro:
    lista = []
elif args.solo:
    lista = [c.zfill(2) for c in args.solo]
elif args.desde:
    lista = [c for c in CAPITULOS if c >= args.desde.zfill(2)]

estado = {}
for cap in lista:
    ruta = next(RAIZ.glob(f"{cap}_*.ipynb"))
    log(f"\n--- Capitulo {cap}: {ruta.name} | {ahora()} ---")
    respaldo = None
    if cap == "01":                              # protege el traspaso del estudio
        from src import datos, preprocesamiento as prep
        huella_antes = prep._huella(datos.cargar_traspaso())
        respaldo = config.DIR_CACHE / "respaldo_traspaso"
        shutil.rmtree(respaldo, ignore_errors=True)
        shutil.copytree(config.DIR_DATOS, respaldo)
    try:
        estado[cap] = ejecutar_notebook(ruta, KERNEL)
    except Exception:
        log("ERROR al ejecutar el capitulo:\n" + traceback.format_exc())
        estado[cap] = False
    if respaldo is not None:
        huella_despues = prep._huella(datos.cargar_traspaso())
        if huella_despues != huella_antes:
            log("   AVISO: el traspaso cambio; se restaura la copia original.")
            shutil.rmtree(config.DIR_DATOS, ignore_errors=True)
            shutil.copytree(respaldo, config.DIR_DATOS)
        shutil.rmtree(respaldo, ignore_errors=True)

# --- 2. libro -----------------------------------------------------------------------
if not args.sin_libro and (not args.solo or args.con_libro):
    log(f"\n--- Compilando el Jupyter Book | {ahora()} ---")
    r = subprocess.run([sys.executable, "-m", "jupyter_book", "build", str(RAIZ)],
                       capture_output=True, text=True)
    if r.returncode != 0:              # versiones antiguas usan el ejecutable directo
        r = subprocess.run(["jupyter-book", "build", str(RAIZ)], capture_output=True,
                           text=True, shell=(sys.platform == "win32"))
    (config.DIR_RESULTADOS / "log_libro.txt").write_text(
        r.stdout + "\n\n===== ADVERTENCIAS Y ERRORES (stderr) =====\n" + r.stderr, encoding="utf-8")
    avisos = [l for l in r.stderr.splitlines() if "WARNING" in l or "ERROR" in l]
    log(f"Advertencias de la compilacion: {len(avisos)} (detalle en resultados/log_libro.txt)")
    if r.returncode != 0:
        log("ERROR al compilar el libro:\n" + r.stderr[-3000:])
    else:
        log("Libro generado en _build/html/index.html")

log(f"\n===== Resumen | {ahora()} =====")
for cap, ok in estado.items():
    log(f"  Capitulo {cap}: {'OK' if ok else 'CON ERROR (ver arriba)'}")
