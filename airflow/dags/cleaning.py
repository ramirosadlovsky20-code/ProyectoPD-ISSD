"""Fase 03: selección, renombre y cálculos del notebook original."""

from pyspark.sql import functions as F

from config import TEXTO


def numericos(df):
    """Equivalente a to_numeric(errors='coerce').fillna(0).astype(int)."""
    # select arma la salida; col conserva textos; try_cast convierte a entero largo.
    # coalesce reemplaza conversiones fallidas por cero; alias mantiene el nombre.
    return df.select(*[
        F.col(c) if c in TEXTO else F.coalesce(F.expr(f"try_cast(`{c}` as bigint)"), F.lit(0)).alias(c)
        for c in df.columns
    ])


def etiqueta(numero):
    """Devuelve el sufijo de PP1: por ejemplo, 1 → grado_1 y 7 → anio_1."""
    # Se conserva la convención de nombres usada por PP1 y el dashboard.
    return f"grado_{numero}" if numero <= 6 else f"anio_{numero - 6}"


def limpiar_matricula(df):
    """Descarta columnas, renombra medidas y calcula mujeres y tasas de PP1."""
    quitar = ["lactantes", "deambulantes", "s2", "s3", "s4", "s5", "_snu", "v_snu",
              "multi_ini", "multi_pri", "multi_sec", "multinivel", "_seclactantes", "_secdeambu"]
    quitar += [f"{prefijo}{sala}" for prefijo in ("v_", "s_")
               for sala in ("lactantes", "deambulantes", "s2", "s3", "s4", "s5")]
    quitar += [f"_secs{i}" for i in range(2, 6)]
    quitar += [f"_sec{i}" for i in list(range(1, 13)) + [1314]]
    df = df.drop(*quitar)
    nombres = {}
    for i in list(range(1, 13)) + [1314]:
        sufijo = etiqueta(i) if i != 1314 else "anio_sup"
        for prefijo, nombre in [("_", "total"), ("v_", "varones"),
                                ("r_", "repitentes"), ("s_", "sobreedad")]:
            nombres[f"{prefijo}{i}"] = f"{nombre}_{sufijo}"
    # alias aplica el diccionario de nombres; después se convierten las medidas.
    df = numericos(df.select(*[F.col(c).alias(nombres.get(c, c)) for c in df.columns]))
    calculos = []
    for i in range(1, 13):
        sufijo = etiqueta(i)
        total = F.col(f"total_{sufijo}")
        # Misma regla de PP1: sustituir cero por uno SOLO en el denominador.
        denominador = F.when(total == 0, F.lit(1)).otherwise(total)
        calculos.extend([
            (total - F.col(f"varones_{sufijo}")).alias(f"mujeres_{sufijo}"),
            (F.col(f"repitentes_{sufijo}") / denominador * 100).alias(f"tasa_repitencia_{sufijo}"),
            (F.col(f"sobreedad_{sufijo}") / denominador * 100).alias(f"tasa_sobreedad_{sufijo}"),
        ])
    return df.select("*", *calculos)  # Conserva las columnas y agrega las expresiones calculadas.


def limpiar_trayectoria(df):
    """Conserva y renombra inscritos, salidas, aprobados y egresados de PP1."""
    quitar = ("entrados_", "m_entrados_", "scp_", "m_scp_", "regulares_", "m_regulares_", "otros_", "m_otros_")
    df = df.drop(*[c for c in df.columns if c.startswith(quitar)])
    prefijos = {
        "inicial_": "inscritos_marzo_", "m_inicial_": "inscritos_marzo_mujeres_",
        "ssp_": "desercion_", "m_ssp_": "desercion_mujeres_",
        "ultimo_": "alumnos_diciembre_", "m_ultimo_": "alumnos_diciembre_mujeres_",
        "promovidos_": "aprobados_", "m_promovidos_": "aprobados_mujeres_",
        "promovidos_ex_": "aprobados_mesas_", "m_promovidos_ex_": "aprobados_mesas_mujeres_",
        "nopromo_": "reprobados_", "m_nopromo_": "reprobados_mujeres_",
    }
    nombres = {"primaria_egresados": "egresados_primaria", "primaria_m_egresados": "egresados_mujeres_primaria",
               "secundaria_egresados": "egresados_secundaria", "secundaria_m_egresados": "egresados_mujeres_secundaria"}
    for i in list(range(1, 13)) + [1314]:
        sufijo = etiqueta(i) if i != 1314 else "anio_superior"
        for original, nuevo in prefijos.items():
            nombres[f"{original}{i}"] = nuevo + sufijo
    return numericos(df.select(*[F.col(c).alias(nombres.get(c, c)) for c in df.columns]))


def limpiar_caracteristicas(df):
    """Conserva el esquema de características y convierte sus conteos a números."""
    # PP1 terminó conservando estas columnas: su drop con mayúsculas no coincidía
    # con los encabezados en minúsculas. Se conserva el esquema final existente.
    return numericos(df)


LIMPIEZA = {"matricula_secciones": limpiar_matricula,
            "trayectoria_por_sexo": limpiar_trayectoria,
            "caracteristicas": limpiar_caracteristicas}
