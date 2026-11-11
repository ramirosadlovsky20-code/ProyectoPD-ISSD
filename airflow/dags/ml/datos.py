"""Ejemplos temporales: indicadores del año t y clase observada en t+1."""

from pyspark.sql import functions as F

from config import CLAVES

VARIABLES = [
    "indice_abandono_secundaria",
    "tasa_sobreedad_secundaria",
    "es_estatal",
    "es_rural"
]

SEGMENTO = CLAVES[1:]
PERCENTIL = 0.75


def crear_pares(base):
    """Cruza únicamente años consecutivos del mismo segmento identificado."""

    if base.groupBy(*CLAVES).count().filter("count > 1").limit(1).count():
        raise ValueError(
            "La entrada del EDA debe tener una sola fila por segmento y año."
        )

    # Un nombre enmascarado no identifica un departamento estable entre años.
    base = base.filter(
        ~F.lower("departamento").contains("enmascarad")
    )

    base = base.filter(
        F.col("sector").isin("Estatal", "Privado")
        & F.col("ambito").isin("Urbano", "Rural")
    )

    # Se exigen razones finitas y no negativas.
    for c in VARIABLES[:2]:
        base = base.filter(
            F.col(c).isNotNull()
            & ~F.isnan(c)
            & (F.col(c) >= 0)
            & (F.col(c) < float("inf"))
        )

    actual = (
        base.withColumn(
            "es_estatal",
            (F.col("sector") == "Estatal").cast("double")
        )
        .withColumn(
            "es_rural",
            (F.col("ambito") == "Rural").cast("double")
        )
    )

    # Desplazar la clave de año permite un join exacto t → t+1.
    futuro = base.select(
        *SEGMENTO,
        (F.col("anio") - 1).alias("anio"),
        F.col("anio").alias("anio_objetivo"),
        F.col("indice_abandono_secundaria").alias("tasa_siguiente")
    )

    return (
        actual.join(futuro, CLAVES, "inner")
        .select(
            *CLAVES,
            "anio_objetivo",
            *VARIABLES,
            "tasa_siguiente"
        )
    )


def preparar_datos(base):
    """Aprende el umbral solo con entrenamiento y separa por año objetivo."""

    pares = crear_pares(base).cache()
    particiones = {}

    try:
        anios = [
            r.anio_objetivo
            for r in pares.select("anio_objetivo")
            .distinct()
            .orderBy("anio_objetivo")
            .collect()
        ]

        if len(anios) < 5:
            raise ValueError(
                "Se necesitan al menos cinco años objetivo para separar "
                "entrenamiento, validación y prueba."
            )

        validacion = anios[-3]
        prueba = anios[-2:]

        entrenamiento = pares.filter(
            F.col("anio_objetivo") < validacion
        )

        # approxQuantile devuelve un número sin mover el dataset completo a Python.
        umbral = entrenamiento.approxQuantile(
            "tasa_siguiente",
            [PERCENTIL],
            0.001
        )[0]

        etiquetados = pares.withColumn(
            "label",
            (F.col("tasa_siguiente") > umbral).cast("double")
        )

        particiones = {
            "entrenamiento": etiquetados.filter(
                F.col("anio_objetivo") < validacion
            ).cache(),

            "validacion": etiquetados.filter(
                F.col("anio_objetivo") == validacion
            ).cache(),

            "prueba": etiquetados.filter(
                F.col("anio_objetivo").isin(prueba)
            ).cache(),
        }

        distribucion = {}

        for nombre, frame in particiones.items():

            clases = {
                int(r.label): r["count"]
                for r in frame.groupBy("label").count().collect()
            }

            if set(clases) != {0, 1}:
                raise ValueError(
                    f"{nombre} necesita ejemplos de ambas clases; "
                    f"encontrados: {clases}"
                )

            distribucion[nombre] = {
                "normal": clases[0],
                "elevada": clases[1],
                "total": sum(clases.values())
            }

        configuracion = {
            "objetivo": (
                "Tasa de salidas sin pase del siguiente archivo anual "
                "superior al umbral de entrenamiento"
            ),
            "variables": VARIABLES,
            "umbral_porcentaje": umbral,
            "percentil_entrenamiento": PERCENTIL,
            "error_relativo_cuantil": 0.001,
            "anio_objetivo_entrenamiento": [
                a for a in anios if a < validacion
            ],
            "anio_objetivo_validacion": [validacion],
            "anio_objetivo_prueba": prueba,
            "distribucion": distribucion,
            "regla_clase": "1 si tasa_siguiente > umbral; 0 en otro caso",
            "departamentos_enmascarados": (
                "Excluidos antes del cruce temporal"
            ),
            "alcance_temporal": (
                "Años de las carpetas fuente; "
                "no se afirma disponibilidad en tiempo real"
            ),
        }

        return particiones, configuracion

    except Exception:
        for frame in particiones.values():
            frame.unpersist()
        raise

    finally:
        pares.unpersist()