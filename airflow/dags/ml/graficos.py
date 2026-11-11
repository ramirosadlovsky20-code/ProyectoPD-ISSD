"""Gráficos interactivos a partir de metricas.json, sin ejecutar Spark ni entrenar."""

import argparse
from html import escape
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODELOS = {
    "regresion_logistica": ("Regresión logística", "#2563eb"),
    "mlp": ("MLP · 4–8–2", "#7c3aed"),
    "persistencia": ("Persistencia", "#0f766e"),
    "clase_mayoritaria": ("Clase mayoritaria", "#64748b"),
}
METRICAS = {
    "accuracy": "Aciertos<br>generales", "precision_clase_1": "Precisión<br>clase elevada",
    "recall_clase_1": "Detección<br>(recall)", "f1_clase_1": "F1<br>clase elevada",
    "auc_pr": "AUC-PR", "auc_roc": "AUC-ROC",
}


def estilo(figura, altura):
    """Aplica el mismo aspecto a los gráficos del reporte."""
    figura.update_layout(template="plotly_white", height=altura, font=dict(family="Arial", size=13, color="#172b4d"),
                         margin=dict(l=70, r=35, t=60, b=80),
                         legend=dict(orientation="h", y=-0.2, x=0),
                         paper_bgcolor="white", plot_bgcolor="white")
    return figura


def comparar_metricas(resultados):
    """Una barra por método y métrica; todos los valores están entre 0 y 1."""
    figura = go.Figure()
    for clave, (nombre, color) in MODELOS.items():
        valores = [resultados[clave][m] for m in METRICAS]
        figura.add_bar(name=nombre, x=list(METRICAS.values()), y=valores, marker_color=color,
                       text=[f"{v:.3f}" for v in valores], textposition="outside",
                       hovertemplate="%{x}: %{y:.4f}<extra>%{fullData.name}</extra>")
    figura.update_layout(barmode="group")
    figura.update_yaxes(title="Puntaje (0–1)", range=[0, 1.13], tickvals=[0, .2, .4, .6, .8, 1])
    return estilo(figura, 460)


def comparar_casos(resultados):
    """Muestra cuántos casos detecta, omite y señala incorrectamente cada método."""
    figura = go.Figure()
    nombres = [nombre for nombre, _ in MODELOS.values()]
    for campo, etiqueta, color in [("tp", "Elevados detectados", "#0f766e"),
                                   ("fn", "Elevados omitidos", "#dc2626"),
                                   ("fp", "Falsas alarmas", "#d97706")]:
        valores = [resultados[m]["matriz_confusion"][campo] for m in MODELOS]
        figura.add_bar(name=etiqueta, x=nombres, y=valores, marker_color=color,
                       text=valores, textposition="outside",
                       hovertemplate="%{x}: %{y} casos<extra>%{fullData.name}</extra>")
    figura.update_layout(barmode="group")
    figura.update_yaxes(title="Cantidad de segmentos", rangemode="tozero")
    return estilo(figura, 400)


def matrices_confusion(resultados):
    """Los números son conteos; el color representa el porcentaje de cada fila real."""
    figura = make_subplots(rows=2, cols=2, subplot_titles=[n for n, _ in MODELOS.values()],
                          horizontal_spacing=.18, vertical_spacing=.25)
    for i, modelo in enumerate(MODELOS):
        m = resultados[modelo]["matriz_confusion"]
        conteos = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fracciones = [[v / sum(fila) if sum(fila) else 0 for v in fila] for fila in conteos]
        textos = [[f"Correcto<br>{m['tn']}", f"Falsa alarma<br>{m['fp']}"],
                  [f"Omitido<br>{m['fn']}", f"Detectado<br>{m['tp']}"]]
        figura.add_trace(go.Heatmap(
            z=fracciones, customdata=conteos, text=textos, texttemplate="%{text}",
            textfont=dict(size=13, color="#172b4d"), coloraxis="coloraxis",
            x=["Predicho: no elevado", "Predicho: elevado"],
            y=["Real: no elevado", "Real: elevado"],
            hovertemplate="%{y}<br>%{x}<br>%{customdata} casos · %{z:.1%} de esa fila<extra></extra>",
        ), row=i // 2 + 1, col=i % 2 + 1)
    figura.update_yaxes(autorange="reversed")
    figura.update_layout(coloraxis=dict(cmin=0, cmax=1, colorscale=[[0, "#f1f5f9"], [1, "#83c9cf"]],
                                        colorbar=dict(title="% de fila", tickvals=[0, .5, 1], ticktext=["0%", "50%", "100%"])))
    return estilo(figura, 740)


def distribucion_clases(configuracion):
    """Compara los porcentajes de cada clase en los tres períodos temporales."""
    datos = configuracion["distribucion"]
    particiones = ["entrenamiento", "validacion", "prueba"]
    nombres = ["Entrenamiento", "Validación", "Prueba"]
    figura = go.Figure()
    for clase, etiqueta, color in [("normal", "No elevada", "#cbd5e1"), ("elevada", "Elevada", "#7c3aed")]:
        conteos = [datos[p][clase] for p in particiones]
        proporciones = [datos[p][clase] / datos[p]["total"] for p in particiones]
        figura.add_bar(name=etiqueta, x=nombres, y=proporciones, marker_color=color,
                       customdata=conteos, text=[f"{p:.1%}" for p in proporciones], textposition="inside",
                       hovertemplate="%{x}<br>%{customdata} casos · %{y:.1%}<extra>%{fullData.name}</extra>")
    figura.update_layout(barmode="stack")
    figura.update_yaxes(title="Proporción de ejemplos", range=[0, 1], tickformat=".0%")
    return estilo(figura, 350)


def generar_reporte(archivo, salida):
    """Lee las métricas y genera un HTML autónomo con Plotly incluido."""
    datos = json.loads(archivo.read_text(encoding="utf-8"))
    cfg = datos["configuracion"]
    # Comprueba que los gráficos de conteos y clases se apoyen en la misma población.
    for particion in ["prueba", "validacion"]:
        distribucion = cfg["distribucion"][particion]
        for modelo in MODELOS:
            resultado = datos[particion][modelo]
            matriz = resultado["matriz_confusion"]
            if (sum(matriz.values()) != resultado["filas"] or resultado["filas"] != distribucion["total"]
                    or matriz["tp"] + matriz["fn"] != distribucion["elevada"]):
                raise ValueError(f"Conteos inconsistentes: {particion}/{modelo}")

    elegido = datos["seleccion"]["modelo"]
    nombre_elegido = MODELOS[elegido][0]
    prueba = cfg["distribucion"]["prueba"]
    tarjetas = f"""
    <div class="cards">
      <article><span>Ejemplos de prueba</span><strong>{prueba['total']:,}</strong></article>
      <article><span>Casos elevados reales</span><strong>{prueba['elevada']} · {prueba['elevada']/prueba['total']:.1%}</strong></article>
      <article><span>Umbral de la clase elevada</span><strong>&gt; {cfg['umbral_porcentaje']:.2f}%</strong></article>
      <article><span>Elegido por validación</span><strong>{escape(nombre_elegido)}</strong></article>
    </div>"""
    bloques = []
    incluir_js = True

    def grafico(figura, identificador):
        nonlocal incluir_js
        # Se incluye Plotly una sola vez para poder abrir el archivo sin internet.
        html = figura.to_html(full_html=False, include_plotlyjs=incluir_js, div_id=identificador,
                              config=dict(responsive=True, displaylogo=False,
                                          toImageButtonOptions=dict(format="png", scale=2)))
        incluir_js = False
        return html

    for particion, titulo in [("prueba", "Prueba final"), ("validacion", "Validación")]:
        anios = ", ".join(str(a) for a in cfg[f"anio_objetivo_{particion}"])
        total = cfg["distribucion"][particion]["total"]
        bloques.append(f'<section id="{particion}"><h2>{titulo} · {anios}</h2><p>{total:,} segmentos/año evaluados.</p>')
        for nombre, funcion, explicacion in [
            ("Comparación de métricas", comparar_metricas,
             "Cada métrica mide algo distinto. Más alto es mejor, pero un alto porcentaje de aciertos puede ocultar casos elevados omitidos."),
            ("Casos detectados, omitidos y falsas alarmas", comparar_casos,
             "Detectados + omitidos = casos elevados reales. Las falsas alarmas son casos no elevados que se señalaron como elevados."),
            ("Matrices de confusión", matrices_confusion,
             "Las filas son las clases reales y las columnas las predicciones. Cada celda muestra casos; su color indica el porcentaje dentro de su fila real."),
        ]:
            bloques.append(f'<article class="panel"><h3>{nombre}</h3><p>{explicacion}</p>')
            bloques.append(grafico(funcion(datos[particion]), f"{particion}-{funcion.__name__}"))
            bloques.append("</article>")
        bloques.append("</section>")

    bloques.append('<section id="clases"><h2>Distribución de clases</h2><article class="panel"><p>La proporción de casos elevados cambia entre períodos. El umbral se aprende en entrenamiento y permanece fijo.</p>')
    bloques.append(grafico(distribucion_clases(cfg), "distribucion-clases"))
    bloques.append("</article></section>")
    resultado = datos["prueba"][elegido]
    matriz = resultado["matriz_confusion"]
    lectura = (f"El modelo seleccionado detectó {matriz['tp']} de {prueba['elevada']} casos elevados "
               f"({resultado['recall_clase_1']:.1%}) y omitió {matriz['fn']}. "
               "La selección se hizo por AUC-PR de validación. El criterio no implica ganar en todas las métricas de prueba.")
    html = """<!doctype html><html lang="es"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>Resultados de clasificación · Radar educativo</title>
    <style>
    *{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172b4d;font-family:Arial,sans-serif;line-height:1.5}
    main{max-width:1280px;margin:auto;padding:36px 24px}h1{font-size:32px;margin:0 0 8px}h2{margin:36px 0 4px}h3{margin:0 0 6px}
    p{color:#52647d;margin:8px 0 16px}nav{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}
    nav a{background:white;border:1px solid #dce3ec;border-radius:8px;padding:9px 16px;text-decoration:none;color:#1d4ed8}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards article,.panel{background:white;border:1px solid #e0e6ee;border-radius:12px;padding:20px}
    .cards span{display:block;color:#52647d;font-size:13px}.cards strong{display:block;font-size:22px;margin-top:8px}
    .panel{margin:18px 0;overflow:hidden}.note{padding:16px 20px;border-left:4px solid #d97706;background:#fff8e8;border-radius:8px}
    footer{font-size:13px;margin-top:24px;color:#52647d}section{scroll-margin-top:20px}
    @media(max-width:800px){main{padding:24px 12px}.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:26px}.panel{padding:12px}}
    </style></head><body><main><h1>Resultados de clasificación</h1>
    <p>Comparación del MLP, la regresión logística y dos referencias sencillas. Resultados guardados de Spark MLlib.</p>
    <nav><a href="#prueba">Prueba final</a><a href="#validacion">Validación</a><a href="#clases">Distribución de clases</a></nav>
    """ + tarjetas + f'<p class="note">{escape(lectura)}</p>' + "".join(bloques) + """
    <footer>Precisión: cuántas alertas fueron correctas. Recall: cuántos casos elevados se detectaron.
    F1 combina ambas. AUC-PR y AUC-ROC evalúan el orden de los puntajes, sin fijar una única decisión de clase.
    “Elevada” es una etiqueta experimental de salidas sin pase; no es una valoración oficial de riesgo.
    Este archivo muestra las métricas guardadas: no vuelve a entrenar ni calcula curvas a partir de valores agregados.</footer>
    </main></body></html>"""
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")
    return salida


def main():
    """Genera el reporte desde consola sin iniciar una sesión de Spark."""
    parser = argparse.ArgumentParser(description="Gráficos HTML de las métricas de clasificación.")
    parser.add_argument("--metricas", type=Path, default=Path("outputs/spark/07_ml/metricas.json"))
    parser.add_argument("--salida", type=Path)
    args = parser.parse_args()
    salida = args.salida or args.metricas.with_name("graficos.html")
    if salida.resolve() == args.metricas.resolve():
        parser.error("El HTML debe guardarse en un archivo diferente al JSON de métricas.")
    print(f"Reporte generado: {generar_reporte(args.metricas, salida).resolve()}")


if __name__ == "__main__":
    main()
