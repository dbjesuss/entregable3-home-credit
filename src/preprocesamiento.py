"""Preprocesamiento sin fuga de informacion y particiones de la CV anidada.

Principio
---------
Todo estadistico que se estima de los datos (mediana de imputacion, media y
desviacion del escalado, vocabulario de categorias) se ajusta **solo** con el
entrenamiento de cada particion y se aplica despues a su validacion. Es el mismo
pipeline de la linea base logistica (``construir`` en el capitulo 01).

Por que se guarda en cache
--------------------------
El preprocesamiento no depende de ningun hiperparametro del modelo: la mediana de
una columna es la misma para ``C=0.01`` que para ``C=10``. Ajustarlo una vez por
particion y reutilizarlo en todas las evaluaciones es matematicamente identico a
reajustarlo dentro de cada evaluacion (lo que haria ``GridSearchCV`` con un
``Pipeline``), pero evita repetir el mismo calculo miles de veces. El capitulo 04
lo verifica numericamente sobre una configuracion.

Lo que si depende de la evaluacion -- el balanceo (SMOTE/ADASYN), el PCA de KNN,
el Nystroem del SVM -- se ajusta dentro de cada evaluacion, solo con el
entrenamiento de esa particion (ver :mod:`src.experimento`).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def construir_preprocesador(cuantitativas, cualitativas) -> ColumnTransformer:
    """Mismo preprocesamiento de la linea base logistica, con salida float32.

    Rama cuantitativa: imputacion por mediana + estandarizacion.
    Rama cualitativa : imputacion por ``'Desconocido'`` + indicadoras (categorias
    con menos de 20 casos se agrupan en un nivel ``infrequent``) + estandarizacion.

    Parameters
    ----------
    cuantitativas, cualitativas : list of str

    Returns
    -------
    ColumnTransformer
    """
    rama_num = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalador", StandardScaler()),
    ])
    rama_cat = Pipeline([
        ("imputador", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("indicadora", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=20, sparse_output=False,
                                     dtype=np.float32)),
        ("escalador", StandardScaler()),
    ])
    return ColumnTransformer([("num", rama_num, list(cuantitativas)),
                              ("cat", rama_cat, list(cualitativas))],
                             sparse_threshold=0.0, verbose_feature_names_out=False)


def transformar_por_bloques(pre, X: pd.DataFrame,
                            bloque: int = config.BLOQUE_PREDICCION) -> np.ndarray:
    """Aplica un preprocesador ya ajustado por bloques de filas, en float32.

    Transformar las ~200 mil filas de golpe materializa copias intermedias en
    float64; por bloques el pico de memoria queda acotado.
    """
    partes = [np.asarray(pre.transform(X.iloc[i:i + bloque]), dtype=np.float32)
              for i in range(0, len(X), bloque)]
    return np.ascontiguousarray(np.vstack(partes))


@dataclass
class Particion:
    """Una particion entrenamiento/validacion ya preprocesada.

    Attributes
    ----------
    nombre : str
        Por ejemplo ``'e2'`` (externa 2) o ``'e2/i1'`` (interna 1 del externo 2).
    X_tr, y_tr, X_va, y_va : ndarray
        Matrices float32 y objetivos int8.
    fila_tr, fila_va : ndarray
        Indice original (fila de ``application_train``) de cada observacion.
    columnas : list of str
        Nombres de las columnas tras el preprocesamiento.
    """

    nombre: str
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    fila_tr: np.ndarray
    fila_va: np.ndarray
    columnas: list

    @property
    def tamano_mb(self) -> float:
        """Tamano en memoria de las matrices de la particion, en MB."""
        return (self.X_tr.nbytes + self.X_va.nbytes) / 2**20


# ---------------------------------------------------------------------------
# Escritura / lectura de una particion
# ---------------------------------------------------------------------------
_ARCHIVOS = ("X_tr", "y_tr", "X_va", "y_va", "fila_tr", "fila_va")


def _guardar(directorio: Path, p: Particion) -> None:
    directorio.mkdir(parents=True, exist_ok=True)
    for a in _ARCHIVOS:
        np.save(directorio / f"{a}.npy", getattr(p, a))
    (directorio / "columnas.json").write_text(json.dumps(p.columnas), encoding="utf-8")
    (directorio / "LISTO").write_text("ok", encoding="utf-8")   # marca de completitud


def _leer(directorio: Path, nombre: str) -> Particion:
    if not (directorio / "LISTO").is_file():
        raise FileNotFoundError(f"Particion incompleta o inexistente: {directorio}")
    datos = {a: np.load(directorio / f"{a}.npy") for a in _ARCHIVOS}
    columnas = json.loads((directorio / "columnas.json").read_text(encoding="utf-8"))
    return Particion(nombre=nombre, columnas=columnas, **datos)


def _ajustar_y_transformar(X_tr, y_tr, X_va, y_va, cuanti, cuali, nombre) -> Particion:
    pre = construir_preprocesador(cuanti, cuali)
    pre.fit(X_tr)
    return Particion(
        nombre=nombre,
        X_tr=transformar_por_bloques(pre, X_tr), y_tr=np.asarray(y_tr, dtype=np.int8),
        X_va=transformar_por_bloques(pre, X_va), y_va=np.asarray(y_va, dtype=np.int8),
        fila_tr=np.asarray(X_tr.index), fila_va=np.asarray(X_va.index),
        columnas=list(pre.get_feature_names_out()))


# ---------------------------------------------------------------------------
# CV anidada
# ---------------------------------------------------------------------------
def _huella(conj) -> str:
    """Huella del conjunto + esquema: si cambia algo, la cache se invalida."""
    h = hashlib.sha1()
    h.update(str(conj.X_dev.shape).encode())
    h.update(",".join(conj.X_dev.columns).encode())
    h.update(str(int(conj.y_dev.sum())).encode())
    h.update(str(int(np.asarray(conj.X_dev.index).sum())).encode())
    h.update(f"{config.N_EXTERNOS}-{config.N_INTERNOS}-{config.SEMILLA}".encode())
    return h.hexdigest()[:16]


def dir_particiones() -> Path:
    """Carpeta de la cache de particiones preprocesadas."""
    return config.DIR_CACHE / "particiones"


def pliegues_externos(y_dev) -> list:
    """Indices posicionales (train, val) de los pliegues externos."""
    cv = StratifiedKFold(n_splits=config.N_EXTERNOS, shuffle=True,
                         random_state=config.SEMILLA)
    return list(cv.split(np.zeros(len(y_dev)), y_dev))


def pliegues_internos(y_tr_externo, k: int) -> list:
    """Indices posicionales (train, val) de los pliegues internos del externo k,
    relativos al entrenamiento de ese externo."""
    cv = StratifiedKFold(n_splits=config.N_INTERNOS, shuffle=True,
                         random_state=config.SEMILLA + 1 + k)
    return list(cv.split(np.zeros(len(y_tr_externo)), y_tr_externo))


def preparar_particiones(conj, forzar: bool = False, verboso: bool = True) -> Path:
    """Construye y guarda en disco todas las particiones de la CV anidada.

    Para cada pliegue externo ``k`` se guardan la particion externa (``e{k}``) y
    las ``N_INTERNOS`` internas (``e{k}/i{j}``), cada una con su preprocesador
    ajustado solo sobre su entrenamiento. Es reanudable: una particion ya escrita
    (con su marca ``LISTO``) no se recalcula.

    Parameters
    ----------
    conj : ConjuntoModelado
    forzar : bool
        Borra la cache y la reconstruye.

    Returns
    -------
    Path
        Carpeta de las particiones.
    """
    base = dir_particiones()
    huella = _huella(conj)
    manifiesto = base / "manifiesto.json"
    if base.exists() and (forzar or not manifiesto.is_file()
                          or json.loads(manifiesto.read_text())["huella"] != huella):
        if verboso and not forzar and manifiesto.is_file():
            print("La cache de particiones no corresponde a los datos/esquema actuales: se reconstruye")
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    X, y = conj.X_dev, conj.y_dev.to_numpy()
    cq, cl = conj.cuantitativas, conj.cualitativas
    for k, (tr, va) in enumerate(pliegues_externos(y)):
        d = base / f"e{k}"
        if not (d / "LISTO").is_file():
            p = _ajustar_y_transformar(X.iloc[tr], y[tr], X.iloc[va], y[va], cq, cl, f"e{k}")
            _guardar(d, p)
            if verboso:
                print(f"  e{k}: {p.X_tr.shape[0]:,} x {p.X_tr.shape[1]} entrenamiento | "
                      f"{p.X_va.shape[0]:,} validacion")
            del p
        for j, (itr, iva) in enumerate(pliegues_internos(y[tr], k)):
            di = d / f"i{j}"
            if (di / "LISTO").is_file():
                continue
            Xk, yk = X.iloc[tr], y[tr]
            p = _ajustar_y_transformar(Xk.iloc[itr], yk[itr], Xk.iloc[iva], yk[iva],
                                       cq, cl, f"e{k}/i{j}")
            _guardar(di, p)
            if verboso:
                print(f"    e{k}/i{j}: {p.X_tr.shape[0]:,} entrenamiento | "
                      f"{p.X_va.shape[0]:,} validacion")
            del p
    manifiesto.write_text(json.dumps({"huella": huella,
                                      "externos": config.N_EXTERNOS,
                                      "internos": config.N_INTERNOS,
                                      "semilla": config.SEMILLA}, indent=2))
    return base


def cargar_externa(k: int) -> Particion:
    """Particion externa ``k`` (entrenamiento completo del externo vs su validacion)."""
    return _leer(dir_particiones() / f"e{k}", f"e{k}")


def cargar_interna(k: int, j: int) -> Particion:
    """Particion interna ``j`` del pliegue externo ``k``."""
    return _leer(dir_particiones() / f"e{k}" / f"i{j}", f"e{k}/i{j}")


# ---------------------------------------------------------------------------
# Particiones para los modelos finales
# ---------------------------------------------------------------------------
def preparar_final(conj, forzar: bool = False) -> Particion:
    """Desarrollo completo (80 %) frente a prueba (20 %), para los modelos finales."""
    d = config.DIR_CACHE / "final"
    if forzar and d.exists():
        shutil.rmtree(d)
    if not (d / "LISTO").is_file():
        p = _ajustar_y_transformar(conj.X_dev, conj.y_dev, conj.X_te, conj.y_te,
                                   conj.cuantitativas, conj.cualitativas, "final")
        _guardar(d, p)
    return _leer(d, "final")


def preparar_base(conj, forzar: bool = False) -> tuple:
    """Particion de la linea base: entrenamiento (64 %) frente a validacion (16 %)
    y prueba (20 %). Se usa solo en la grafica de ajuste de la Seccion 5.2.1.

    Returns
    -------
    (Particion, ndarray, ndarray, ndarray)
        Particion entrenamiento/validacion, X de prueba transformada, y de prueba
        y filas originales de prueba.
    """
    d = config.DIR_CACHE / "base"
    if forzar and d.exists():
        shutil.rmtree(d)
    if not (d / "LISTO").is_file():
        es_tr = (conj.particion_base == "entrenamiento").to_numpy()
        X_tr, X_va = conj.X_dev[es_tr], conj.X_dev[~es_tr]
        pre = construir_preprocesador(conj.cuantitativas, conj.cualitativas).fit(X_tr)
        p = Particion("base", transformar_por_bloques(pre, X_tr),
                      conj.y_dev[es_tr].to_numpy().astype(np.int8),
                      transformar_por_bloques(pre, X_va),
                      conj.y_dev[~es_tr].to_numpy().astype(np.int8),
                      np.asarray(X_tr.index), np.asarray(X_va.index),
                      list(pre.get_feature_names_out()))
        _guardar(d, p)
        np.save(d / "X_te.npy", transformar_por_bloques(pre, conj.X_te))
    p = _leer(d, "base")
    return (p, np.load(d / "X_te.npy"), conj.y_te.to_numpy().astype(np.int8),
            np.asarray(conj.X_te.index))
