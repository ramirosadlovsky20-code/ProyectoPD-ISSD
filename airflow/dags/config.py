"""Tablas y columnas compartidas por las fases del proyecto."""

# Estas columnas identifican un segmento agregado; no una escuela individual.
CLAVES = ["anio", "provincia", "departamento", "sector", "ambito"]
TEXTO = CLAVES[1:]  # El año es numérico; las otras cuatro claves son texto.
# Nombre de la tabla de salida → fragmento que buscamos en el nombre del CSV.
TABLAS = {
    "matricula_secciones": "matricula y secciones",
    "trayectoria_por_sexo": "trayectoria",
    "caracteristicas": "caracteristicas",
}
