"""Métricas de la clase elevada y predicciones legibles del clasificador."""

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from config import CLAVES


def predecir(modelo, datos):
    """Devuelve etiqueta, clase predicha y probabilidad de la clase 1."""
    # transform usa el escalado aprendido en entrenamiento; no vuelve a ajustarlo.
    return (modelo.transform(datos)
            .withColumn("score", vector_to_array("probability")[1])
            .select(*CLAVES, "anio_objetivo", "tasa_siguiente", "label", "prediction", "score"))


def evaluar(predicciones):
    """Evalúa especialmente la clase 1; accuracy sola ocultaría el desbalance."""
    predicciones = predicciones.select(*[F.col(c).cast("double").alias(c) for c in ["label", "prediction", "score"]])
    filas = predicciones.groupBy("label", "prediction").count().collect()
    matriz = {(int(r.label), int(r.prediction)): r["count"] for r in filas}
    tn, fp = matriz.get((0, 0), 0), matriz.get((0, 1), 0)
    fn, tp = matriz.get((1, 0), 0), matriz.get((1, 1), 0)
    total = tn + fp + fn + tp
    if not total:
        raise ValueError("No hay predicciones para evaluar.")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "filas": total, "accuracy": (tp + tn) / total,
        "precision_clase_1": precision, "recall_clase_1": recall,
        "f1_clase_1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auc_pr": BinaryClassificationEvaluator(rawPredictionCol="score", metricName="areaUnderPR").evaluate(predicciones),
        "auc_roc": BinaryClassificationEvaluator(rawPredictionCol="score", metricName="areaUnderROC").evaluate(predicciones),
        "matriz_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def referencia_persistencia(datos, umbral):
    """Referencia sencilla: el año siguiente se parecerá al actual."""
    return (datos.withColumn("prediction", (F.col("indice_abandono_secundaria") > umbral).cast("double"))
            .withColumn("score", F.col("indice_abandono_secundaria"))
            .select(*CLAVES, "anio_objetivo", "tasa_siguiente", "label", "prediction", "score"))


def referencia_mayoritaria(datos, clase):
    """Predice siempre la clase mayoritaria de entrenamiento, sin mirar el futuro."""
    return datos.withColumn("prediction", F.lit(float(clase))).withColumn("score", F.lit(float(clase)))
