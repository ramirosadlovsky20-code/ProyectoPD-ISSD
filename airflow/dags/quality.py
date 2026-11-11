"""Fase 04: nulos, texto, tasas y detección de atípicos heredados de PP1."""

from pyspark.sql import functions as F

from config import TEXTO


def asegurar_calidad(df, tabla):
    """Devuelve la tabla con las imputaciones, textos y redondeos de QA de PP1."""
    # Los espacios vacíos también se consideran nulos, como en el notebook.
    # trim quita espacios; length mide el texto; when sin otherwise deja null si está vacío.
    df = df.select(*[
        F.when(F.length(F.trim(F.col(c))) > 0, F.trim(F.col(c))).alias(c)
        if c in TEXTO else F.col(c) for c in df.columns
    ])
    for c in TEXTO:
        # isNull detecta faltantes. limit(1).count() ejecuta la consulta para saber si hay alguno.
        if df.filter(F.col(c).isNull()).limit(1).count():
            if tabla == "caracteristicas":
                valor = "0"  # Equivalente al fillna(0) original para esta tabla.
            else:
                # groupBy/count obtiene frecuencias; orderBy ordena y first trae solo la moda.
                moda = (df.filter(F.col(c).isNotNull()).groupBy(c).count()
                        .orderBy(F.desc("count"), F.asc(c)).first())
                if moda is None:
                    raise ValueError(f"No hay valores para imputar la columna {c}.")
                valor = moda[c]
            df = df.fillna({c: valor})  # Completa nulos únicamente en esta columna.
    df = df.fillna(0)  # Completa nulos numéricos; no convierte los textos a números.
    # initcap normaliza mayúsculas; bround redondea las tasas con empates al par.
    return df.select(*[
        F.initcap(F.col(c)).alias(c) if c in TEXTO else
        F.bround(F.col(c), 3).alias(c) if c.startswith("tasa_") else F.col(c)
        for c in df.columns
    ])


def contar_atipicos(df, tabla):
    """Cuenta |z| > 3 sin borrar filas, igual que el QA original."""
    columnas = [c for c in df.columns if c not in TEXTO]
    # stddev_pop usa la población completa; stddev_samp aplica la fórmula muestral.
    desviacion = F.stddev_pop if tabla == "caracteristicas" else F.stddev_samp
    # Spark calcula las estadísticas; first trae al proceso Python una sola fila resumen.
    estadisticas = df.agg(*[expresion for c in columnas for expresion in
                           (F.avg(c).alias(f"media_{c}"), desviacion(c).alias(f"desvio_{c}"))]).first()
    conteos = []
    for c in columnas:
        media, desvio = estadisticas[f"media_{c}"], estadisticas[f"desvio_{c}"]
        # abs obtiene la distancia al promedio; when la convierte en 1 (atípico) o 0.
        condicion = F.abs((F.col(c) - media) / desvio) > 3 if desvio and desvio > 0 else F.lit(False)
        conteos.append(F.sum(F.when(condicion, 1).otherwise(0)).alias(c))
    resultado = df.agg(*conteos).first().asDict()  # Una fila de conteos pasa a diccionario Python.
    return sum(valor or 0 for valor in resultado.values())
