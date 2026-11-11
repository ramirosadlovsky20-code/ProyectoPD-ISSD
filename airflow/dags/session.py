"""Creación de la sesión de Spark."""

from pyspark.sql import SparkSession


def create_session(master: str = "local[2]") -> SparkSession:
    """Abre o reutiliza Spark y devuelve la sesión que usan lectura y consultas."""
    # SparkSession es el punto de entrada. builder configura la sesión antes de abrirla.
    # local[2] ejecuta tareas localmente con dos hilos; no levanta dos servidores.
    spark = SparkSession.builder \
    .appName("mi_app") \
    .master("local[*]") \
    .getOrCreate()

    print("¡Spark inició correctamente!")
    return spark