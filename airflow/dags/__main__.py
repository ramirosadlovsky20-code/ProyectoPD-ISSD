"""Ejecuta las fases de PP1 en Spark y la comparación de clasificadores con MLlib."""

import argparse
import shutil
from pathlib import Path

from config import TABLAS


def main():
    """Lee las opciones de consola y ejecuta las fases elegidas en orden."""

    parser = argparse.ArgumentParser(
        description="Traducción de las fases de PP1 a Spark."
    )

    parser.add_argument(
        "--fase",
        choices=["02", "03", "04", "eda", "ml", "todas"],
        default="02"
    )

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2011, 2025))
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("01_raw_data")
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/spark")
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina los resultados anteriores antes de ejecutar."
    )

    args = parser.parse_args()

    raiz = args.output_dir.resolve()

    # Limpieza automática de los resultados anteriores.
    if args.clean:
        carpeta_outputs = (Path.cwd() / "outputs").resolve()

        if carpeta_outputs not in raiz.parents:
            parser.error(
                "La carpeta de salida debe estar dentro de outputs."
            )

        if raiz.exists():
            print(
                f"Limpiando resultados anteriores: {raiz}",
                flush=True
            )
            shutil.rmtree(raiz)

        raiz.mkdir(parents=True, exist_ok=True)

    fases = (
        ["02", "03", "04", "eda", "ml"]
        if args.fase == "todas"
        else [args.fase]
    )

    carpetas = {
        "02": "02_formateo",
        "03": "03_limpieza",
        "04": "04_calidad",
        "eda": "06_eda",
        "ml": "07_ml"
    }

    # Validación de resultados existentes y dependencias entre fases.
    for fase in fases:
        destino = raiz / carpetas[fase]

        if destino.exists():
            parser.error(
                f"Ya existe {destino}. "
                "Elegí otra salida o ejecutá con --clean."
            )

        if fase != "02" and args.fase != "todas":
            previa = {"eda": "04", "ml": "eda"}.get(fase)

            if previa is None:
                previa = f"{int(fase)-1:02}"

            anterior = raiz / carpetas[previa]

            necesarias = (
                ["sobreedad_segmentos"]
                if fase == "ml"
                else TABLAS
            )

            if not all(
                (anterior / tabla / "_SUCCESS").exists()
                for tabla in necesarias
            ):
                parser.error(
                    f"Primero completá la fase {previa} en {raiz}."
                )

    from session import create_session
    from formatting import formatear
    from cleaning import LIMPIEZA
    from quality import asegurar_calidad, contar_atipicos

    spark = create_session()

    try:
        for fase in fases:

            print(
                f"FASE {fase}: {carpetas[fase]}",
                flush=True
            )

            if fase == "ml":
                from ml.entrenamiento import ejecutar_ml

                ejecutar_ml(
                    spark,
                    raiz / carpetas["eda"],
                    raiz / carpetas["ml"]
                )
                continue

            if fase == "eda":
                from eda import ejecutar_eda

                ejecutar_eda(
                    spark,
                    raiz / carpetas["04"],
                    raiz / carpetas["eda"]
                )
                continue

            for tabla in TABLAS:

                if fase == "02":
                    df = formatear(
                        spark,
                        args.data_dir,
                        tabla,
                        args.years
                    )

                else:
                    anterior = (
                        raiz
                        / carpetas[f"{int(fase)-1:02}"]
                        / tabla
                    )

                    df = spark.read.parquet(str(anterior))

                    df = (
                        LIMPIEZA[tabla](df)
                        if fase == "03"
                        else asegurar_calidad(df, tabla)
                    )

                destino = raiz / carpetas[fase] / tabla

                df.write.parquet(str(destino))

                guardado = spark.read.parquet(str(destino))

                print(
                    f"  {tabla}: {guardado.count()} filas, "
                    f"{len(guardado.columns)} columnas",
                    flush=True
                )

                if fase == "04":
                    print(
                        "    Valores atípicos (no eliminados): "
                        f"{contar_atipicos(guardado, tabla)}",
                        flush=True
                    )

        print(f"Resultados: {raiz}", flush=True)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()