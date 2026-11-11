"""Comprobaciones de equivalencia con las reglas de los notebooks de PP1."""

import unittest
import csv
from pathlib import Path
import tempfile

from pyspark.sql import functions as F

from radar_spark.session import create_session
from radar_spark.cleaning import limpiar_matricula, limpiar_trayectoria
from radar_spark.quality import asegurar_calidad
from radar_spark.formatting import formatear
from radar_spark.eda import (ETAPAS, tendencia, resumen_por_sexo, flujo_por_sexo,
                             promocion_por_sexo, base_secundaria, base_conectividad)


class FasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = create_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_formateo_suma_internet_en_lugar_de_concatenarlo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            annual = root / "BD Escuelas 2024"
            annual.mkdir()
            with (annual / "2024 Caracteristicas del establecimiento.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, delimiter=";")
                writer.writerow(["provincia", "departamento", "sector", "ambito",
                                 "InternetTipodeservicioGratuitoEstado", "InternetTipodeservicioGratuitoOtro"])
                writer.writerows([["Buenos Aires", "A", "Estatal", "Rural", "33", "11"],
                                  ["Buenos Aires", "B", "Estatal", "Rural", " ", "1"]])
            result = formatear(self.spark, root, "caracteristicas", [2024]).orderBy("departamento").collect()
            self.assertEqual([r.internettipodeserviciogratuito for r in result], ["44", "1"])
            self.assertTrue(all(r.anio == 2024 for r in result))

    def test_matricula_conserva_calculos_de_pp1(self):
        columnas = [F.lit(2024).alias("anio")]
        columnas += [F.lit("Ejemplo").alias(c) for c in ["provincia", "departamento", "sector", "ambito"]]
        for i in list(range(1, 13)) + [1314]:
            for prefijo, valor in [("_", "100"), ("v_", "40"), ("r_", "10"), ("s_", "20")]:
                columnas.append(F.lit(valor).alias(f"{prefijo}{i}"))
        df = self.spark.range(1).select(*columnas)
        df = df.withColumn("lactantes", F.lit("9")).withColumn("_sec1", F.lit("3"))
        limpio = limpiar_matricula(df)
        row = limpio.first()
        self.assertNotIn("lactantes", limpio.columns)
        self.assertNotIn("_sec1", limpio.columns)
        self.assertEqual(row.mujeres_grado_1, 60)
        self.assertEqual(row.tasa_repitencia_grado_1, 10)
        self.assertEqual(row.tasa_sobreedad_anio_1, 20)
        self.assertEqual(len(limpio.columns), 93)
        cero = limpiar_matricula(df.withColumn("_1", F.lit("0")).withColumn("r_1", F.lit("0"))).first()
        self.assertEqual(cero.tasa_repitencia_grado_1, 0)

    def test_trayectoria_selecciona_y_renombra_como_pp1(self):
        df = self.spark.sql("""SELECT 2024 anio, '100' inicial_1, '3' ssp_1,
                              '8' promovidos_ex_7, '9' m_otros_12, '2' entrados_1,
                              ' ' nopromo_1, '15' primaria_egresados""")
        limpio = limpiar_trayectoria(df)
        row = limpio.first()
        self.assertNotIn("m_otros_12", limpio.columns)
        self.assertNotIn("entrados_1", limpio.columns)
        self.assertEqual(row.inscritos_marzo_grado_1, 100)
        self.assertEqual(row.desercion_grado_1, 3)
        self.assertEqual(row.aprobados_mesas_anio_1, 8)
        self.assertEqual(row.reprobados_grado_1, 0)
        self.assertEqual(row.egresados_primaria, 15)

    def test_calidad_conserva_filas_e_imputa_como_pp1(self):
        df = self.spark.sql("""SELECT * FROM VALUES
                              (2024, 'buenos aires', 'ejemplo', 'estatal', 'urbano', 1.5625),
                              (2024, 'buenos aires', 'ejemplo', 'estatal', ' ', 2.0)
                              AS t(anio, provincia, departamento, sector, ambito, tasa_ejemplo)""")
        resultado = asegurar_calidad(df, "matricula_secciones").collect()
        self.assertEqual(len(resultado), 2)
        self.assertTrue(all(r.ambito == "Urbano" for r in resultado))
        self.assertTrue(all(r.provincia == "Buenos Aires" for r in resultado))
        self.assertAlmostEqual(float(resultado[0].tasa_ejemplo), 1.562)
        infra = asegurar_calidad(df, "caracteristicas").collect()
        self.assertEqual(infra[1].ambito, "0")

    def datos_eda(self):
        claves = [F.lit(2024).alias("anio"), F.lit("Ejemplo").alias("provincia"),
                  F.lit("Departamento").alias("departamento"), F.lit("Estatal").alias("sector"),
                  F.lit("Rural").alias("ambito")]
        t = [F.lit(valor).cast("long").alias(prefijo + sufijo) for sufijo, _ in ETAPAS
             for prefijo, valor in [("inscritos_marzo_", 100), ("inscritos_marzo_mujeres_", 60),
                                    ("aprobados_", 80), ("aprobados_mujeres_", 50), ("desercion_", 10)]]
        m = [F.lit(valor).cast("long").alias(prefijo + f"anio_{i}") for i in range(1, 7)
             for prefijo, valor in [("total_", 100), ("sobreedad_", 20)]]
        return self.spark.range(2).select(*claves, *m), self.spark.range(2).select(*claves, *t)

    def test_eda_conserva_totales_y_deriva_varones_por_diferencia(self):
        _, trayectoria = self.datos_eda()
        row = tendencia(trayectoria).first()
        self.assertEqual(row.Total_Alumnos, 2400)
        self.assertEqual(row.Total_Salidas, 240)
        self.assertEqual(row.Tasa_Abandono, 10)
        resumen = resumen_por_sexo(trayectoria)
        flujo = flujo_por_sexo(resumen)
        self.assertEqual(flujo.count(), 24)
        varones = flujo.filter((F.col("Orden") == 1) & (F.col("Sexo") == "Varones")).first()
        self.assertEqual((varones.Alumnos, varones.Aprobados), (80, 60))
        promocion = promocion_por_sexo(resumen).filter(F.col("Orden") == 1).first()
        self.assertEqual(promocion.Tasa_Varones, 75)
        self.assertAlmostEqual(promocion.Tasa_Mujeres, 100 * 100 / 120)

    def test_eda_no_multiplica_conteos_al_cruzar_claves_repetidas(self):
        matricula, trayectoria = self.datos_eda()
        base = base_secundaria(matricula, trayectoria)
        self.assertEqual(base.count(), 1)
        row = base.first()
        self.assertEqual(row.matricula_secundaria, 1200)
        self.assertEqual(row.inscritos_secundaria, 1200)
        self.assertEqual(row.tasa_sobreedad_secundaria, 20)
        self.assertEqual(row.indice_abandono_secundaria, 10)
        caracteristicas = matricula.select("anio", "provincia", "departamento", "sector", "ambito").select(
            "*", F.lit(1).alias("internettipodeserviciogratuito"),
            F.lit(0).alias("internettipodeserviciopago"), F.lit(0).alias("disponedesalaolaboratoriodeinformaticasi"))
        conectado = base_conectividad(base, caracteristicas)
        self.assertEqual(conectado.count(), 1)
        self.assertEqual(conectado.first().con_recursos, 1)


if __name__ == "__main__":
    unittest.main()
