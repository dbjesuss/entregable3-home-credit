# Entregable 3 · Machine Learning sobre Home Credit

Jupyter Book con **140 modelos** (112 de clasificación y 28 de regresión) entrenados para predecir si un
solicitante de Home Credit tendrá dificultades de pago (`TARGET` = 1), con validación cruzada anidada
5 × 3, cuatro métodos de optimización de hiperparámetros, optimización computacional, interpretabilidad
(SHAP y LIME) y comparación estadística jerárquica.

**Integrantes:** Jesús David Barrios Valdes, Samuel David Chamorro Solorzano y Jacobo Londoño Baquero.
**Curso:** Machine Learning, Pregrado en Ciencia de Datos.

El libro publicado está en GitHub Pages (enlace en la descripción del repositorio).

## Resultado principal

| Modelo (mejor combinación) | AUC-ROC en prueba | AUC-ROC CV anidada |
|---|---|---|
| **XGBoost** (sin balanceo, optimizador genético) | **0.7754** | 0.7721 ± 0.0038 |
| Regresión logística L1 (`class_weight`, bayesiana) | 0.7597 | 0.7576 ± 0.0028 |
| SVM (`class_weight`, genética) | 0.7595 | 0.7578 ± 0.0031 |
| Línea base del entregable anterior (logística) | 0.7603 | – |

XGBoost supera a la línea base por +0.0151 de AUC (IC 95 % [0.0119, 0.0179], DeLong con ajuste de Holm).
SMOTE y ADASYN no mejoran el AUC en ningún modelo; la optimización de hiperparámetros cambia la
eficiencia de la búsqueda, pero no el desempeño final. Los detalles están en las conclusiones del libro.

## Contenido

| Capítulo | Contenido |
|---|---|
| `01_datos_y_linea_base.ipynb` | EDA en tres fases, embudo de selección de variables y logística de referencia |
| `02_clasificacion.ipynb` | 7 modelos × 4 balanceos × 4 optimizadores (112 combinaciones) |
| `03_regresion.ipynb` | 7 regresores × 4 optimizadores sobre `TARGET` (28 combinaciones) |
| `04_optimizacion_hiperparametros.ipynb` | Rejilla, aleatoria, bayesiana (Optuna/TPE) y genética (DEAP); convergencia; Successive Halving |
| `05_optimizacion_computacional.ipynb` | Complejidad teórica frente a tiempos medidos, perfilamiento, GPU y paralelismo |
| `06_evaluacion_clasificacion.ipynb` | Métricas, matrices de confusión, curvas ROC, balanceo y calibración |
| `07_evaluacion_regresion.ipynb` | RMSE, MAE, ajuste en tres conjuntos y análisis de residuos |
| `08_interpretabilidad.ipynb` | SHAP global y local, SHAP frente a LIME |
| `09_comparacion_estadistica.ipynb` | Friedman, Nemenyi, DeLong, bootstrap BCa, MCS, Giacomini-White, Clark-West, Diebold-Mariano, SPA |
| `10_conclusiones.md` | Conclusiones |
| `A1_ejecucion_del_estudio.ipynb` | Anexo: cómo se ejecutó el estudio, fallos y reparaciones |

El código reutilizable está en `src/` (configuración, datos, preprocesamiento, espacios de búsqueda,
modelos, balanceo, optimizadores, motor del estudio, métricas, pruebas estadísticas). La tabla maestra
de las 140 combinaciones está en `resultados/tabla_maestra.csv`.

## Cómo reproducirlo

```bash
pip install -r requirements.txt        # Python 3.10 o 3.12
python verificar_entorno.py
# 1) 01_datos_y_linea_base.ipynb  -> Restart & Run All (guarda datos/)
# 2) 02_clasificacion.ipynb y 03_regresion.ipynb (reanudables)
# 3) 04 a 09 y el anexo A1 en orden
jupyter-book build .
```

Los datos de la competencia [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
no se incluyen (sus reglas no permiten redistribuirlos): se descargan de Kaggle y el código los busca
automáticamente o en la carpeta indicada por la variable de entorno `HOME_CREDIT_DIR`.

El estudio completo consumió unas 444 horas de cómputo en un portátil con GPU, así que se ejecutó fuera
de Jupyter con el mismo código: `python correr_estudio.py --grupo gpu` y `--grupo cpu` en varias
terminales a la vez (sin pisarse: cada unidad tiene su bloqueo) y luego `python correr_capitulos.py`,
que repara las evaluaciones fallidas, ejecuta todos los capítulos de arriba abajo y compila el libro
(anexo A).

Para verificar el código sin los datos reales (datos sintéticos y presupuestos mínimos, en una carpeta
`_prueba/` aparte), defina `E3_MODO_PRUEBA=1` antes de abrir Jupyter.

De `resultados/` se incluyen la tabla maestra (`tabla_maestra.csv`, una fila por combinación) y el
registro por pliegue (`registro_pliegues.csv`, una fila por cada una de las 700 unidades), con
hiperparámetros, métricas del bucle externo, tiempos y semilla. Lo demás se regenera al ejecutar el
proyecto y no se incluye: la caché de particiones (`cache/`, ~10 GB), los datos de modelado (`datos/`),
los registros JSON de cada unidad, las predicciones fuera de muestra y los modelos finales.
