"""Cálculos del dashboard de PP1 usando DataFrames de Spark."""

from functools import reduce

from pyspark.sql import functions as F

# F reúne funciones de Spark: col referencia columnas y lit agrega valores fijos.
# Estas expresiones construyen un plan; write y count hacen ejecutar ese plan.
# Los bucles Python de este módulo preparan expresiones, no recorren las filas del dataset.

from config import CLAVES

ETAPAS = [(f"grado_{i}", nombre) for i, nombre in enumerate(
    ["1ro Primaria", "2do Primaria", "3ro Primaria", "4to Primaria", "5to Primaria", "6to Primaria"], 1)]
ETAPAS += [(f"anio_{i}", nombre) for i, nombre in enumerate(
    ["1ro Secundaria", "2do Secundaria", "3ro Secundaria", "4to Secundaria", "5to Secundaria", "6to Secundaria"], 1)]


def sumar_columnas(prefijo, sufijos):
    """Construye una suma horizontal: varias columnas de una misma fila."""
    # reduce encadena los signos +; F.col indica qué columnas sumará Spark.
    # Es distinto de F.sum, que suma una columna entre las filas de un grupo.
    return reduce(lambda a, b: a + b, [F.col(prefijo + s) for s in sufijos])


def porcentaje(numerador, denominador, sin_datos=None):
    """Devuelve la expresión numerador/denominador × 100 o el valor sin_datos."""
    # when/otherwise equivale a CASE WHEN de SQL. cast convierte el tipo numérico.
    return F.when(denominador > 0, numerador.cast("double") / denominador * 100).otherwise(sin_datos)


def tendencia(trayectoria):
    """Datos del gráfico de evolución de salidas sin pase por provincia y año."""
    sufijos = [s for s, _ in ETAPAS]
    # groupBy define los grupos; agg calcula sus totales; alias nombra la salida.
    resumen = trayectoria.groupBy("anio", "provincia").agg(
        F.sum(sumar_columnas("inscritos_marzo_", sufijos)).alias("Total_Alumnos"),
        F.sum(sumar_columnas("desercion_", sufijos)).alias("Total_Salidas"),
    )
    # select elige columnas y cálculos. dropna quita las tasas que quedaron nulas.
    return resumen.select(
        F.col("anio").alias("Anio"), F.col("provincia").alias("Provincia"),
        "Total_Alumnos", "Total_Salidas",
        porcentaje(F.col("Total_Salidas"), F.col("Total_Alumnos")).alias("Tasa_Abandono"),
    ).dropna(subset=["Tasa_Abandono"])


def resumen_por_sexo(trayectoria):
    """Recibe trayectoria y devuelve una fila por año con totales por etapa."""
    columnas = [prefijo + sufijo for sufijo, _ in ETAPAS for prefijo in
                ["inscritos_marzo_", "inscritos_marzo_mujeres_", "aprobados_", "aprobados_mujeres_"]]
    # El * desempaqueta la lista: agg recibe una expresión de suma por columna.
    return trayectoria.groupBy("anio").agg(*[F.sum(c).alias(c) for c in columnas])


def flujo_por_sexo(resumen):
    """Mismas cantidades por etapa/sexo que el embudo original."""
    filas = []
    for orden, (sufijo, etapa) in enumerate(ETAPAS, 1):
        total, mujeres = F.col("inscritos_marzo_" + sufijo), F.col("inscritos_marzo_mujeres_" + sufijo)
        aprobados, aprobadas = F.col("aprobados_" + sufijo), F.col("aprobados_mujeres_" + sufijo)
        # struct reúne campos en un registro; lit convierte una constante en columna.
        for sexo, alumnos, apr in [("Mujeres", mujeres, aprobadas), ("Varones", total - mujeres, aprobados - aprobadas)]:
            filas.append(F.struct(F.lit(orden).alias("Orden"), F.lit(etapa).alias("Etapa"),
                                  F.lit(sexo).alias("Sexo"), alumnos.alias("Alumnos"), apr.alias("Aprobados")))
    # array agrupa esos registros; explode los convierte en filas (24 por año).
    # dato.* abre el registro para que Etapa, Sexo, Alumnos y Aprobados sean columnas.
    return resumen.select(F.col("anio").alias("Anio"), F.explode(F.array(*filas)).alias("dato")).select("Anio", "dato.*")


def promocion_por_sexo(resumen):
    """Las cuatro etapas del gráfico de mancuernas de PP1."""
    filas = []
    etapas = [("grado_1", "Inicio Primaria"), ("grado_6", "Fin Primaria"),
              ("anio_1", "Inicio Secundaria"), ("anio_6", "Fin Secundaria")]
    for orden, (sufijo, etapa) in enumerate(etapas, 1):
        im, am = F.col("inscritos_marzo_mujeres_" + sufijo), F.col("aprobados_mujeres_" + sufijo)
        # Las restas también son expresiones Spark: se evalúan para cada año.
        iv, av = F.col("inscritos_marzo_" + sufijo) - im, F.col("aprobados_" + sufijo) - am
        filas.append(F.struct(
            F.lit(orden).alias("Orden"), F.lit(etapa).alias("Etapa"),
            porcentaje(av, iv, 0).alias("Tasa_Varones"), porcentaje(am, im, 0).alias("Tasa_Mujeres"),
            av.alias("Apr_V"), iv.alias("Ins_V"), am.alias("Apr_M"), im.alias("Ins_M"),
        ))
    # El mismo mecanismo del embudo, pero aquí genera cuatro etapas por año.
    return resumen.select(F.col("anio").alias("Anio"), F.explode(F.array(*filas)).alias("dato")).select("Anio", "dato.*")


def base_secundaria(matricula, trayectoria):
    """Cruce de segmentos; agrupa claves repetidas antes del join."""
    sufijos = [f"anio_{i}" for i in range(1, 7)]
    # Cada tabla queda con una sola fila por CLAVES antes de cruzarlas.
    m = matricula.groupBy(*CLAVES).agg(
        F.sum(sumar_columnas("total_", sufijos)).alias("matricula_secundaria"),
        F.sum(sumar_columnas("sobreedad_", sufijos)).alias("sobreedad_secundaria"),
    )
    t = trayectoria.groupBy(*CLAVES).agg(
        F.sum(sumar_columnas("inscritos_marzo_", sufijos)).alias("inscritos_secundaria"),
        F.sum(sumar_columnas("desercion_", sufijos)).alias("salidas_secundaria"),
        *[F.sum(c).alias(c) for c in ["aprobados_anio_6", "aprobados_mujeres_anio_6",
                                    "inscritos_marzo_anio_6", "inscritos_marzo_mujeres_anio_6"]],
    )
    # join inner conserva las claves presentes en ambas; filter equivale a WHERE.
    base = m.join(t, CLAVES, "inner").filter(F.col("matricula_secundaria") > 0)
    # concat_ws arma un texto con separadores; sha2 lo transforma en el id_hash.
    return base.select(
        "*", F.sha2(F.concat_ws("-", *[F.col(c).cast("string") for c in CLAVES]), 256).alias("id_hash"),
        porcentaje(F.col("sobreedad_secundaria"), F.col("matricula_secundaria")).alias("tasa_sobreedad_secundaria"),
        porcentaje(F.col("salidas_secundaria"), F.col("inscritos_secundaria")).alias("indice_abandono_secundaria"),
    )


def sobreedad_por_provincia(base):
    """Devuelve una fila por provincia con la media de sobreedad para el mapa."""
    # PP1 usa la media simple de tasas de los segmentos, acumulando los años.
    # trim quita espacios, upper pasa a mayúsculas y avg calcula el promedio.
    return (base.dropna(subset=["tasa_sobreedad_secundaria", "indice_abandono_secundaria"])
            .groupBy(F.upper(F.trim("provincia")).alias("provincia"))
            .agg(F.avg("tasa_sobreedad_secundaria").alias("sobreedad_promedio")))


def base_conectividad(base, caracteristicas):
    """Agrega al cruce educativo si el segmento tiene algún recurso tecnológico."""
    # | es OR entre condiciones Spark. Se usan paréntesis en cada comparación.
    conectada = ((F.col("internettipodeserviciogratuito") > 0)
                 | (F.col("internettipodeserviciopago") > 0)
                 | (F.col("disponedesalaolaboratoriodeinformaticasi") > 0))
    # max de indicadores 0/1 da 1 si al menos un registro del segmento tiene recursos.
    recursos = caracteristicas.groupBy(*CLAVES).agg(
        F.max(F.when(conectada, 1).otherwise(0)).alias("con_recursos"))
    return base.join(recursos, CLAVES, "inner")


def infraestructura(base):
    """Prepara los segmentos rurales estatales y sus categorías para el boxplot."""
    # & es AND entre condiciones Spark; when asigna una etiqueta a cada categoría.
    return (base.filter((F.col("sector") == "Estatal") & (F.col("ambito") == "Rural"))
            .select(*CLAVES, "id_hash",
                    F.when(F.col("con_recursos") == 1, "Conectada / Equipada")
                    .otherwise("Sin Recursos Tecnológicos").alias("nivel_conectividad"),
                    "indice_abandono_secundaria")
            .dropna(subset=["indice_abandono_secundaria"]))


def perfil_riesgo(base):
    """Perfil descriptivo del radar existente; no es un modelo predictivo."""
    im, am = F.col("inscritos_marzo_mujeres_anio_6"), F.col("aprobados_mujeres_anio_6")
    iv, av = F.col("inscritos_marzo_anio_6") - im, F.col("aprobados_anio_6") - am
    # withColumn crea o reemplaza una columna y devuelve otro DataFrame.
    # Aquí agrupamos por perfil y calculamos las medias que usa el radar original.
    perfil = (base.withColumn("conectividad", F.when(F.col("con_recursos") == 1, "Conectada").otherwise("Sin Conectividad"))
              .groupBy("sector", "ambito", "conectividad").agg(
                  F.avg("indice_abandono_secundaria").alias("tasa_abandono"),
                  F.avg("tasa_sobreedad_secundaria").alias("tasa_sobreedad"),
                  F.avg(porcentaje(av, iv)).alias("aprobacion_varones_fin_sec"),
                  F.avg(porcentaje(am, im)).alias("aprobacion_mujeres_fin_sec"),
              ).dropna()
              .withColumn("brecha_genero", F.col("aprobacion_mujeres_fin_sec") - F.col("aprobacion_varones_fin_sec")))
    ejes = [("tasa_abandono", "n_abandono"), ("tasa_sobreedad", "n_sobreedad"), ("brecha_genero", "n_brecha")]
    # agg sin groupBy devuelve una sola fila con los mínimos y máximos globales.
    limites = perfil.agg(*[exp for c, _ in ejes for exp in [F.min(c).alias("min_" + c), F.max(c).alias("max_" + c)]])
    # crossJoin adjunta esa única fila de límites a cada perfil para normalizarlo.
    unido = perfil.crossJoin(limites)
    normalizados = []
    for c, nombre in ejes:
        minimo, maximo = F.col("min_" + c), F.col("max_" + c)
        valor = F.when(maximo > minimo, (F.col(c) - minimo) / (maximo - minimo) * 100).otherwise(50.0)
        normalizados.append(F.bround(valor, 1).alias(nombre))  # Redondeo a un decimal, con empates al par.
    return (unido.select(*perfil.columns, *normalizados)
            .withColumn("perfil", F.concat_ws(" / ", "sector", "ambito", "conectividad"))
            .withColumn("riesgo_total", F.col("n_abandono") + F.col("n_sobreedad") + F.col("n_brecha")))


def ejecutar_eda(spark, origen, destino):
    """Lee las tres tablas de calidad y escribe los siete resultados del EDA."""
    # read.parquet devuelve DataFrames distribuidos a partir de las carpetas guardadas.
    trayectoria = spark.read.parquet(str(origen / "trayectoria_por_sexo"))
    matricula = spark.read.parquet(str(origen / "matricula_secciones"))
    caracteristicas = spark.read.parquet(str(origen / "caracteristicas"))
    sexo = resumen_por_sexo(trayectoria)
    base = base_secundaria(matricula, trayectoria)
    conectividad = base_conectividad(base, caracteristicas)
    resultados = {
        "tendencia": tendencia(trayectoria),
        "flujo_por_sexo": flujo_por_sexo(sexo),
        "promocion_por_sexo": promocion_por_sexo(sexo),
        "sobreedad_segmentos": base.select(*CLAVES, "id_hash", "tasa_sobreedad_secundaria", "indice_abandono_secundaria")
                                  .dropna(subset=["tasa_sobreedad_secundaria", "indice_abandono_secundaria"]),
        "sobreedad_provincias": sobreedad_por_provincia(base),
        "infraestructura": infraestructura(conectividad),
        "perfil_riesgo": perfil_riesgo(conectividad),
    }
    for nombre, df in resultados.items():
        ruta = str(destino / nombre)
        df.write.parquet(ruta)  # Acción: ejecuta los cálculos pendientes y guarda sus resultados.
        # count es otra acción: lee la salida y cuenta sus filas para comprobarla.
        print(f"  {nombre}: {spark.read.parquet(ruta).count()} filas", flush=True)
