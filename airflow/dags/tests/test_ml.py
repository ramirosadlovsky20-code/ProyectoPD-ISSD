"""Controles de orden temporal, etiqueta y métricas del clasificador."""

import unittest

from pyspark.sql import functions as F

from radar_spark.session import create_session
from radar_spark.ml.datos import crear_pares, preparar_datos, VARIABLES
from radar_spark.ml.evaluacion import evaluar


class MLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = create_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def base(self):
        return (self.spark.range(2011, 2025)
                .crossJoin(self.spark.sql("SELECT * FROM VALUES ('A'), ('B') AS t(departamento)"))
                .select(F.col("id").cast("int").alias("anio"), F.lit("Provincia").alias("provincia"),
                        "departamento", F.lit("Estatal").alias("sector"), F.lit("Rural").alias("ambito"),
                        F.when(F.col("departamento") == "A", F.col("id") - 2010).otherwise(0.1)
                        .cast("double").alias("indice_abandono_secundaria"),
                        F.lit(20.0).alias("tasa_sobreedad_secundaria")))

    def test_pares_exigen_anios_consecutivos_y_excluyen_enmascarados(self):
        base = self.base().filter(F.col("anio").isin(2011, 2013, 2014))
        oculto = base.filter(F.col("departamento") == "A").withColumn("departamento", F.lit("Enmascarado"))
        pares = crear_pares(base.unionByName(oculto)).collect()
        self.assertEqual(len(pares), 2)
        self.assertTrue(all(r.anio == 2013 and r.anio_objetivo == 2014 for r in pares))
        a = next(r for r in pares if r.departamento == "A")
        self.assertEqual((a.indice_abandono_secundaria, a.tasa_siguiente), (3.0, 4.0))
        self.assertTrue(set(VARIABLES).isdisjoint({"tasa_siguiente", "anio_objetivo", "label"}))
        with self.assertRaisesRegex(ValueError, "una sola fila"):
            crear_pares(base.unionByName(base))

    def test_umbral_no_usa_prueba_y_particiones_no_mezclan_fechas(self):
        base = self.base()
        modificada = base.withColumn("indice_abandono_secundaria",
            F.when((F.col("anio") >= 2023) & (F.col("departamento") == "A"), 1000.0)
            .otherwise(F.col("indice_abandono_secundaria")))
        primera, info = preparar_datos(base)
        segunda = {}
        try:
            segunda, otro = preparar_datos(modificada)
            self.assertEqual(info["umbral_porcentaje"], otro["umbral_porcentaje"])
            self.assertEqual(info["anio_objetivo_validacion"], [2022])
            self.assertEqual(info["anio_objetivo_prueba"], [2023, 2024])
            self.assertEqual(primera["entrenamiento"].agg(F.max("anio_objetivo")).first()[0], 2021)
            self.assertEqual(primera["prueba"].agg(F.min("anio_objetivo")).first()[0], 2023)
        finally:
            for frame in [*primera.values(), *segunda.values()]:
                frame.unpersist()

    def test_metricas_clase_positiva_con_matriz_conocida(self):
        datos = self.spark.sql("""SELECT * FROM VALUES
            (0.0, 0.0, 0.1), (1.0, 1.0, 0.9), (0.0, 1.0, 0.8), (1.0, 0.0, 0.2)
            AS t(label, prediction, score)""")
        resultado = evaluar(datos)
        self.assertEqual(resultado["matriz_confusion"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(resultado["f1_clase_1"], 0.5)
        self.assertEqual(resultado["accuracy"], 0.5)
        self.assertAlmostEqual(resultado["auc_roc"], 0.75)


if __name__ == "__main__":
    unittest.main()
