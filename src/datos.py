"""Carga del conjunto de modelado y traspaso desde el capitulo 01.

El capitulo 01 (EDA + linea base logistica) construye la matriz de diseno: une las
siete tablas, aplica el embudo de la Fase 3.5 y parte en entrenamiento, validacion
y prueba con la semilla del proyecto. Al final de ese capitulo se llama a
:func:`guardar_traspaso`, que deja en ``datos/`` exactamente las mismas filas y
columnas con las que se ajusto la logistica. Los capitulos siguientes leen de ahi
con :func:`cargar_traspaso` y no dependen de nada que quede vivo en memoria.

Particion usada en el Entregable 3
----------------------------------
* ``desarrollo`` = entrenamiento + validacion de la linea base (80 %). Sobre el
  corre la validacion cruzada anidada.
* ``prueba`` = la misma prueba de la linea base (20 %). No participa en ninguna
  decision; solo se usa al final para evaluar los modelos finales y compararlos
  con la logistica en igualdad de condiciones.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

ARCHIVO_DESARROLLO = "desarrollo.parquet"
ARCHIVO_PRUEBA = "prueba.parquet"
ARCHIVO_META = "meta.json"
OBJETIVO = "TARGET"
COL_PARTICION_BASE = "PARTICION_BASE"


@dataclass
class ConjuntoModelado:
    """Contenedor del conjunto de modelado.

    Attributes
    ----------
    X_dev, y_dev : DataFrame, Series
        Desarrollo (entrenamiento + validacion de la linea base).
    particion_base : Series
        ``'entrenamiento'`` o ``'validacion'`` para cada fila de desarrollo, segun
        la particion del notebook logistico. Se usa en la grafica de ajuste de la
        Seccion 5.2.1 (entrenamiento / validacion / prueba).
    X_te, y_te : DataFrame, Series
        Prueba final, intocada hasta los capitulos de evaluacion.
    cuantitativas, cualitativas : list of str
        Tipos de columna, necesarios para el preprocesamiento.
    meta : dict
        Metadatos del traspaso (fecha, semilla, dimensiones).
    """

    X_dev: pd.DataFrame
    y_dev: pd.Series
    particion_base: pd.Series
    X_te: pd.DataFrame
    y_te: pd.Series
    cuantitativas: list
    cualitativas: list
    meta: dict = field(default_factory=dict)

    def describir(self) -> pd.DataFrame:
        """Tabla con filas, positivos y tasa de mora por particion."""
        filas = []
        for nombre, y in (("desarrollo", self.y_dev), ("prueba", self.y_te)):
            filas.append({"Particion": nombre, "Filas": len(y),
                          "Positivos": int(y.sum()),
                          "Mora_%": round(100 * float(y.mean()), 2)})
        tabla = pd.DataFrame(filas)
        tabla["Columnas"] = self.X_dev.shape[1]
        tabla["Cuantitativas"] = len(self.cuantitativas)
        tabla["Cualitativas"] = len(self.cualitativas)
        return tabla


# ---------------------------------------------------------------------------
# Traspaso (lo llama el capitulo 01)
# ---------------------------------------------------------------------------
def guardar_traspaso(X_tr, y_tr, X_va, y_va, X_te, y_te,
                     cuantitativas, cualitativas, semilla=config.SEMILLA,
                     directorio=None) -> dict:
    """Guarda en disco el conjunto de modelado tal como lo uso la linea base.

    Parameters
    ----------
    X_tr, y_tr, X_va, y_va, X_te, y_te : DataFrame / Series
        Las tres particiones del notebook logistico. Se conserva su indice, que es
        la posicion de la fila en ``application_train.csv`` (archivo ordenado por
        ``SK_ID_CURR``); ese orden es el que usan las pruebas de independencia de
        residuos.
    cuantitativas, cualitativas : list of str
        Columnas de cada tipo.
    semilla : int
        Semilla con que se hizo la particion.
    directorio : Path, optional
        Carpeta destino. Por defecto ``config.DIR_DATOS``.

    Returns
    -------
    dict
        Metadatos escritos en ``meta.json``.
    """
    directorio = directorio or config.DIR_DATOS
    directorio.mkdir(parents=True, exist_ok=True)

    dev = pd.concat([X_tr, X_va], axis=0).copy()     # copia consolidada (no fragmentada)
    dev[OBJETIVO] = pd.concat([y_tr, y_va], axis=0).astype("int8").to_numpy()
    dev[COL_PARTICION_BASE] = (["entrenamiento"] * len(X_tr)
                               + ["validacion"] * len(X_va))
    dev.index.name = "FILA"
    te = X_te.copy()
    te[OBJETIVO] = y_te.astype("int8").to_numpy()
    te.index.name = "FILA"

    for c in cualitativas:            # parquet no admite 'object' mixto
        dev[c] = dev[c].astype("string")
        te[c] = te[c].astype("string")

    dev.to_parquet(directorio / ARCHIVO_DESARROLLO)
    te.to_parquet(directorio / ARCHIVO_PRUEBA)

    meta = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "semilla": int(semilla),
        "filas_desarrollo": int(len(dev)),
        "filas_prueba": int(len(te)),
        "columnas": int(X_tr.shape[1]),
        "cuantitativas": list(cuantitativas),
        "cualitativas": list(cualitativas),
        "tasa_mora_desarrollo": float(dev[OBJETIVO].mean()),
        "tasa_mora_prueba": float(te[OBJETIVO].mean()),
    }
    (directorio / ARCHIVO_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def existe_traspaso(directorio=None) -> bool:
    """True si el capitulo 01 ya dejo el conjunto de modelado en disco."""
    directorio = directorio or config.DIR_DATOS
    return all((directorio / a).is_file()
               for a in (ARCHIVO_DESARROLLO, ARCHIVO_PRUEBA, ARCHIVO_META))


def cargar_traspaso(directorio=None) -> ConjuntoModelado:
    """Lee el conjunto de modelado que dejo el capitulo 01.

    En modo prueba (``E3_MODO_PRUEBA=1``), si no hay traspaso se genera uno
    sintetico con :func:`generar_sinteticos`.

    Returns
    -------
    ConjuntoModelado

    Raises
    ------
    FileNotFoundError
        Si el traspaso no existe (hay que ejecutar primero el capitulo 01).
    """
    directorio = directorio or config.DIR_DATOS
    if not existe_traspaso(directorio):
        if config.MODO_PRUEBA:
            generar_sinteticos(directorio=directorio)
        else:
            raise FileNotFoundError(
                f"No existe el traspaso en {directorio}. Ejecute completo el "
                "capitulo 01 (01_datos_y_linea_base.ipynb): su ultima seccion "
                "guarda el conjunto de modelado.")

    meta = json.loads((directorio / ARCHIVO_META).read_text(encoding="utf-8"))
    dev = pd.read_parquet(directorio / ARCHIVO_DESARROLLO)
    te = pd.read_parquet(directorio / ARCHIVO_PRUEBA)
    cuanti, cuali = meta["cuantitativas"], meta["cualitativas"]

    for d in (dev, te):
        for c in cuali:
            # OneHotEncoder y SimpleImputer esperan 'object' con np.nan
            d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
        for c in cuanti:
            if d[c].dtype == "float64":
                d[c] = d[c].astype("float32")

    return ConjuntoModelado(
        X_dev=dev[cuanti + cuali], y_dev=dev[OBJETIVO].astype(int),
        particion_base=dev[COL_PARTICION_BASE].astype(str),
        X_te=te[cuanti + cuali], y_te=te[OBJETIVO].astype(int),
        cuantitativas=cuanti, cualitativas=cuali, meta=meta)


# ---------------------------------------------------------------------------
# Datos sinteticos (solo para verificar el codigo)
# ---------------------------------------------------------------------------
def generar_sinteticos(n: int = 3000, p_num: int = 24, semilla: int = config.SEMILLA,
                       directorio=None) -> dict:
    """Genera un traspaso sintetico con la misma estructura que el real.

    Reproduce los rasgos que importan para que el codigo se comporte igual: ~8 %
    de positivos, faltantes, indicadoras enteras, tres variables cualitativas con
    niveles raros y una relacion logistica debil con el objetivo. No tiene ningun
    valor analitico.
    """
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame(rng.normal(size=(n, p_num)).astype("float32"),
                     columns=[f"NUM_{i:02d}" for i in range(p_num)])
    X["FLAG_A"] = rng.integers(0, 2, n).astype("int8")
    X["FLAG_B"] = rng.integers(0, 2, n).astype("int8")
    X.loc[rng.random(n) < 0.15, "NUM_03"] = np.nan
    X.loc[rng.random(n) < 0.40, "NUM_07"] = np.nan
    X["CAT_TIPO"] = rng.choice(["A", "B", "C"], n, p=[.6, .3, .1]).astype(object)
    X["CAT_OFICIO"] = rng.choice([f"o{i}" for i in range(12)], n).astype(object)
    X["CAT_RARA"] = rng.choice(["x", "y", "z", "w"], n, p=[.9, .07, .02, .01]).astype(object)
    X.loc[rng.random(n) < 0.1, "CAT_OFICIO"] = np.nan

    lin = (0.8 * X["NUM_00"].fillna(0) - 0.6 * X["NUM_01"] + 0.4 * X["NUM_02"] * X["NUM_04"]
           + 0.5 * (X["CAT_TIPO"] == "C") + 0.3 * X["FLAG_A"])
    lin = lin - np.quantile(lin, 0.95) - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-lin.to_numpy()))).astype(int)

    cuanti = [c for c in X.columns if not c.startswith("CAT_")]
    cuali = [c for c in X.columns if c.startswith("CAT_")]
    idx = np.arange(n)
    rng.shuffle(idx)
    X.index = idx
    y = pd.Series(y, index=idx)

    from sklearn.model_selection import train_test_split
    X_aj, X_te, y_aj, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=semilla)
    X_tr, X_va, y_tr, y_va = train_test_split(X_aj, y_aj, test_size=0.2,
                                              stratify=y_aj, random_state=semilla)
    return guardar_traspaso(X_tr, y_tr, X_va, y_va, X_te, y_te, cuanti, cuali,
                            semilla, directorio)
