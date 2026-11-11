"""Pipeline de variables, MLP pequeño, comparación y guardado de resultados."""

import json

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.util import JavaMLReader
from pyspark.sql import functions as F

from .datos import VARIABLES, preparar_datos
from .evaluacion import evaluar, predecir, referencia_persistencia, referencia_mayoritaria

SEMILLA = 42
CAPAS = [len(VARIABLES), 8, 2]


def cargar_modelo(ruta):
    """Recupera el Pipeline completo guardado por las etapas nativas de MLlib."""
    # Todas nuestras etapas son Java. El lector Java evita el paso por Python RDD
    # de PipelineModel.load, que falla en este entorno Windows/Python 3.12.
    return JavaMLReader(PipelineModel).load(str(ruta.resolve()))


def crear_pipeline(nombre):
    """La preparación y el clasificador se ajustan juntos, solo en entrenamiento."""
    vector = VectorAssembler(inputCols=VARIABLES, outputCol="vector", handleInvalid="error")
    escala = StandardScaler(inputCol="vector", outputCol="features", withMean=True, withStd=True)
    if nombre == "mlp":
        clasificador = MultilayerPerceptronClassifier(layers=CAPAS, maxIter=100, seed=SEMILLA, blockSize=128)
    elif nombre == "regresion_logistica":
        clasificador = LogisticRegression(maxIter=100, regParam=0.01)
    else:
        raise ValueError(f"Clasificador desconocido: {nombre}")
    return Pipeline(stages=[vector, escala, clasificador])


def ejecutar_ml(spark, origen, destino):
    """Entrena candidatos, elige por validación y evalúa ambos en años reservados."""
    if destino.exists():
        raise FileExistsError(f"La salida de ML ya existe: {destino}")
    base = spark.read.parquet(str(origen / "sobreedad_segmentos"))
    particiones, configuracion = preparar_datos(base)
    destino.mkdir(parents=True)
    modelos = {}
    predicciones = []
    guardadas = []
    metricas = {"configuracion": {**configuracion, "capas_mlp": CAPAS, "semilla": SEMILLA,
                                  "max_iteraciones": 100, "reg_param_logistica": 0.01},
                "validacion": {}, "prueba": {}}
    try:
        for nombre in ["regresion_logistica", "mlp"]:
            print(f"  Entrenando {nombre}...", flush=True)
            modelo = crear_pipeline(nombre).fit(particiones["entrenamiento"])
            modelos[nombre] = modelo
            pred = predecir(modelo, particiones["validacion"]).cache()
            guardadas.append(pred)
            metricas["validacion"][nombre] = evaluar(pred)
            predicciones.append(pred.withColumn("modelo", F.lit(nombre)).withColumn("particion", F.lit("validacion")))
            print(f"    Validación: {metricas['validacion'][nombre]}", flush=True)

        # La decisión se toma ANTES de evaluar prueba. No se elige mirando 2023–2024.
        elegido = max(modelos, key=lambda nombre: metricas["validacion"][nombre]["auc_pr"])
        metricas["seleccion"] = {"modelo": elegido, "criterio": "Mayor AUC-PR en validación; empate favorece regresión logística"}
        modelos[elegido].write().save(str(destino / "modelo"))

        for nombre, modelo in modelos.items():
            pred = predecir(modelo, particiones["prueba"]).cache()
            guardadas.append(pred)
            metricas["prueba"][nombre] = evaluar(pred)
            predicciones.append(pred.withColumn("modelo", F.lit(nombre)).withColumn("particion", F.lit("prueba")))

        distribucion = configuracion["distribucion"]["entrenamiento"]
        mayoritaria = 1 if distribucion["elevada"] > distribucion["normal"] else 0
        for particion in ["validacion", "prueba"]:
            datos = particiones[particion]
            metricas[particion]["persistencia"] = evaluar(referencia_persistencia(datos, configuracion["umbral_porcentaje"]))
            metricas[particion]["clase_mayoritaria"] = evaluar(referencia_mayoritaria(datos, mayoritaria))

        salida = predicciones[0]
        for frame in predicciones[1:]:
            salida = salida.unionByName(frame)
        salida.write.partitionBy("particion", "modelo").parquet(str(destino / "predicciones"))
        # Solo se guarda el pequeño resumen en JSON; los datos permanecen en Spark/Parquet.
        (destino / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"  Modelo elegido por validación: {elegido}", flush=True)
        print(f"  Prueba final: {metricas['prueba'][elegido]}", flush=True)
        return metricas
    finally:
        for frame in particiones.values():
            frame.unpersist()
        # Las predicciones de cada modelo se persistieron para reutilizarlas al escribir.
        for frame in guardadas:
            frame.unpersist()
