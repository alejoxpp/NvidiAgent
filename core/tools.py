"""Herramientas generales y de hardware para RetroTron, agente de InvidiAgent."""

import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import requests
from langchain_core.tools import tool

from core.rag import obtener_motor

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTOS_DIR = Path(os.getenv("DOCUMENTOS_DIR", "data/documentos"))
if not DOCUMENTOS_DIR.is_absolute():
    DOCUMENTOS_DIR = PROJECT_DIR / DOCUMENTOS_DIR


@tool
def calculadora(expresion: str) -> str:
    """Evalúa expresiones matemáticas sencillas."""
    try:
        contexto = {
            "sqrt": math.sqrt,
            "log": math.log,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
        }
        return f"{expresion} = {eval(expresion, {'__builtins__': {}}, contexto)}"
    except Exception as error:
        return f"Error al calcular: {error}"


@tool
def fecha_hora_actual(zona: str = "America/Bogota") -> str:
    """Devuelve la fecha y hora actual en la zona indicada."""
    try:
        now = datetime.now(ZoneInfo(zona))
        return now.strftime(f"%A %d de %B de %Y, %H:%M en {zona}")
    except Exception:
        return datetime.now().strftime("%A %d de %B de %Y, %H:%M")


@tool
def consultar_clima(ciudad: str) -> str:
    """Consulta el clima actual de una ciudad."""
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key or "TU_CLAVE" in api_key:
        return "La herramienta de clima no está configurada (falta OPENWEATHER_API_KEY)."
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": ciudad, "appid": api_key, "units": "metric", "lang": "es"},
            timeout=10,
        )
        if response.status_code == 404:
            return f"Ciudad '{ciudad}' no encontrada."
        response.raise_for_status()
        data = response.json()
        return (
            f"{data['name']}, {data['sys']['country']}: {data['main']['temp']:.1f}°C, "
            f"{data['weather'][0]['description']}, humedad {data['main']['humidity']}%"
        )
    except Exception as error:
        return f"Error al consultar el clima: {error}"


@tool
def buscar_documentos(pregunta: str) -> str:
    """Busca en manuales NVIDIA de drivers, compatibilidad y catálogo."""
    motor = obtener_motor()
    texto, fuentes = motor.buscar(pregunta)
    if not texto:
        return "No se encontró información relevante en los documentos NVIDIA disponibles."
    return texto + "\n\nFuentes consultadas: " + ", ".join(fuentes)


@tool
def analizar_ventas(
    agrupar_por: str = "categoria",
    metrica: str = "cantidad_vendida",
    producto: str = "",
    mes: str = "",
) -> str:
    """Analiza inventario NVIDIA por producto, categoría, marca o arquitectura.

    Métricas: cantidad_vendida, stock o precio_usd. El archivo no registra
    períodos de venta; mes se acepta para señalar esa limitación.
    """
    ruta = DOCUMENTOS_DIR / "04_ventas_e_inventario.csv"
    if not ruta.exists():
        return f"No se encontró el archivo de inventario: {ruta}"
    try:
        df = pd.read_csv(ruta)
    except Exception as error:
        return f"No se pudo leer el archivo de inventario: {error}"

    if mes:
        detalle = ""
        if producto:
            coincidencias = df[df["producto"].str.contains(producto, case=False, na=False)]
            if not coincidencias.empty:
                fila = coincidencias.iloc[0]
                detalle = (
                    f" Registro agregado: {int(fila['cantidad_vendida']):,} unidades; "
                    f"fecha de lanzamiento {fila['fecha_lanzamiento']}."
                )
        return (
            f"El CSV no registra ventas por mes, así que no puedo confirmar ventas de {mes}. "
            f"La fecha_lanzamiento no es un período de venta.{detalle}"
        )

    if producto:
        df = df[df["producto"].str.contains(producto, case=False, na=False)]
        if df.empty:
            return f"No se encontró el producto '{producto}' en el inventario."

    agrupaciones = {
        "producto": "producto",
        "categoria": "categoria",
        "categoría": "categoria",
        "marca": "marca",
        "arquitectura": "arquitectura",
    }
    metricas = {
        "cantidad_vendida": "cantidad_vendida",
        "unidades": "cantidad_vendida",
        "stock": "stock_disponible",
        "stock_disponible": "stock_disponible",
        "precio": "precio_usd",
        "precio_usd": "precio_usd",
    }
    group_column = agrupaciones.get(agrupar_por.strip().lower())
    metric_column = metricas.get(metrica.strip().lower())
    if group_column is None or metric_column is None:
        return "Agrupaciones: producto, categoria, marca, arquitectura. Métricas: cantidad_vendida, stock, precio_usd."

    df[metric_column] = pd.to_numeric(df[metric_column], errors="coerce").fillna(0)
    if metric_column == "precio_usd":
        resumen = df.groupby(group_column)[metric_column].mean().sort_values(ascending=False)
        operacion = "precio promedio"
        unidad = "USD"
        titulo = "Precio promedio"
    else:
        resumen = df.groupby(group_column)[metric_column].sum().sort_values(ascending=False)
        titulo = "Stock disponible" if metric_column == "stock_disponible" else "Unidades vendidas"
        unidad = "unidades"

    lineas = [f"- {nombre}: {valor:,.2f} {unidad}" for nombre, valor in resumen.items()]
    return f"{titulo} por {agrupar_por}:\n" + "\n".join(lineas)


_GPU_WATTS = {
    "rtx 4090": 450,
    "rtx 4080 super": 320,
    "rtx 4080": 320,
    "rtx 4070 ti super": 285,
    "rtx 4070 ti": 285,
    "rtx 4070 super": 220,
    "rtx 4070": 200,
    "rtx 4060 ti 16gb": 165,
    "rtx 4060 ti": 160,
    "rtx 4060": 115,
}
_GPU_MIN_PSU = {
    "rtx 4090": 1000,
    "rtx 4080 super": 850,
    "rtx 4080": 850,
    "rtx 4070 ti super": 850,
    "rtx 4070 ti": 850,
    "rtx 4070 super": 750,
    "rtx 4070": 750,
    "rtx 4060 ti 16gb": 650,
    "rtx 4060 ti": 650,
    "rtx 4060": 600,
}
_CPU_WATTS = {
    "i9-14900k": 253,
    "i7-14700k": 253,
    "i5-14600k": 181,
    "ryzen 9 7950x": 230,
    "ryzen 7 7800x3d": 162,
    "ryzen 7 7800x": 120,
    "ryzen 5 7600": 88,
}
_GPU_PERFORMANCE_RANK = {
    "rtx 4090": 100,
    "rtx 4080 super": 95,
    "rtx 4080": 90,
    "rtx 4070 ti super": 85,
    "rtx 4070 super": 80,
    "rtx 4070": 75,
    "rtx 3090": 70,
    "rtx 3080": 65,
    "rtx 3070": 60,
    "rtx 4060 ti 16gb": 55,
    "rtx 4060 ti": 54,
    "rtx 4060": 50,
    "rtx 3060 12gb": 45,
    "rtx 2060 super": 40,
}


@tool
def calcular_watts_fuente(gpu: str, cpu: str) -> str:
    """Estima consumo total y capacidad de fuente para modelos conocidos."""
    gpu_key = gpu.strip().casefold()
    cpu_key = cpu.strip().casefold()
    if gpu_key not in _GPU_WATTS or cpu_key not in _CPU_WATTS:
        return "No tengo un perfil de consumo estimado para esa GPU o CPU. Consulta la ficha del fabricante."

    gpu_watts = _GPU_WATTS[gpu_key]
    cpu_watts = _CPU_WATTS[cpu_key]
    base_watts = gpu_watts + cpu_watts + 100
    calculated_recommendation = math.ceil(base_watts * 1.25 / 50) * 50
    recommended = max(calculated_recommendation, _GPU_MIN_PSU[gpu_key])
    return (
        f"Estimación orientativa: GPU {gpu_watts} W + CPU {cpu_watts} W + "
        f"100 W para el resto del sistema; con 25% de margen, fuente de al menos {recommended} W. "
        "Verifica las recomendaciones del fabricante para los modelos exactos."
    )


@tool
def verificar_compatibilidad(
    socket_cpu: str,
    socket_motherboard: str,
    tipo_ram_motherboard: str,
    tipo_ram_memoria: str,
) -> str:
    """Compara el socket y el tipo de RAM indicados por el usuario."""
    socket_match = socket_cpu.strip().casefold() == socket_motherboard.strip().casefold()
    ram_match = tipo_ram_motherboard.strip().casefold() == tipo_ram_memoria.strip().casefold()
    if socket_match and ram_match:
        return "Compatible según socket y tipo de RAM indicados; verifica además generación de CPU y lista QVL de la motherboard."

    problemas = []
    if not socket_match:
        problemas.append(f"socket CPU {socket_cpu} != socket motherboard {socket_motherboard}")
    if not ram_match:
        problemas.append(f"RAM {tipo_ram_memoria} != tipo admitido {tipo_ram_motherboard}")
    return "No compatible: " + "; ".join(problemas)


@tool
def recomendar_build_presupuesto(presupuesto: float) -> str:
    """Sugiere una build orientativa en USD usando precios GPU del CSV.

    El precio de los demás componentes es un margen conjunto estimado, no
    una cotización ni una garantía de disponibilidad.
    """
    if presupuesto <= 0:
        return "El presupuesto debe ser mayor que cero USD."

    ruta = DOCUMENTOS_DIR / "04_ventas_e_inventario.csv"
    if not ruta.exists():
        return "No se encontró el inventario; no puedo respaldar el precio de una GPU."
    try:
        df = pd.read_csv(ruta)
        gpu_rows = df[df["categoria"].str.contains("GPU Consumidor", case=False, na=False)].copy()
        gpu_rows["precio_usd"] = pd.to_numeric(gpu_rows["precio_usd"], errors="coerce")
        gpu_rows = gpu_rows[gpu_rows["precio_usd"] > 0]
    except Exception as error:
        return f"No se pudo consultar el catálogo de GPU: {error}"

    if gpu_rows.empty:
        return "El CSV no contiene GPUs de consumo con un precio utilizable."

    def build_profile(product_name: str) -> dict:
        name = product_name.casefold()
        if any(model in name for model in ("4090", "4080", "3090")):
            return {
                "cpu": "Ryzen 7 7800X3D (estimado: $350)",
                "motherboard": "B650/X670, socket AM5 (estimado: $220)",
                "ram": "32 GB DDR5 (estimado: $120)",
                "storage": "SSD NVMe 2 TB (estimado: $150)",
                "psu": "850 W para RTX 4080/3090; 1000 W para RTX 4090 (estimado: $180)",
                "case": "Gabinete con flujo de aire amplio (estimado: $150)",
                "extras": 1170,
                "performance": "4K",
            }
        if any(model in name for model in ("4070", "3080", "3070")):
            return {
                "cpu": "Ryzen 5 7600 (estimado: $180)",
                "motherboard": "B650, socket AM5 (estimado: $160)",
                "ram": "32 GB DDR5 (estimado: $95)",
                "storage": "SSD NVMe 1 TB (estimado: $85)",
                "psu": "750 W 80+ Gold (estimado: $110)",
                "case": "Gabinete con flujo de aire (estimado: $100)",
                "extras": 730,
                "performance": "1440p",
            }
        return {
            "cpu": "Ryzen 5 7600 (estimado: $180)",
            "motherboard": "B650, socket AM5 (estimado: $130)",
            "ram": "32 GB DDR5 (estimado: $75)",
            "storage": "SSD NVMe 1 TB (estimado: $70)",
            "psu": "650 W 80+ Gold (estimado: $90)",
            "case": "Gabinete con flujo de aire (estimado: $75)",
            "extras": 620,
            "performance": "1080p",
        }

    gpu_rows["extras_estimados"] = gpu_rows["producto"].map(
        lambda nombre: build_profile(str(nombre))["extras"]
    )
    gpu_rows["ranking"] = gpu_rows["producto"].str.casefold().map(_GPU_PERFORMANCE_RANK).fillna(0)
    gpu_rows = gpu_rows[gpu_rows["precio_usd"] + gpu_rows["extras_estimados"] <= presupuesto]
    if gpu_rows.empty:
        return (
            f"No hay una GPU de consumo en el CSV que quepa en ${presupuesto:,.2f} USD, "
            "incluyendo CPU, motherboard, RAM, almacenamiento, fuente y gabinete estimados."
        )

    gpu = gpu_rows.sort_values(["ranking", "precio_usd"]).iloc[-1]
    gpu_name = str(gpu["producto"])
    gpu_price = float(gpu["precio_usd"])
    profile = build_profile(gpu_name)
    total = gpu_price + profile["extras"]
    if "4090" in gpu_name:
        profile["psu"] = "1000 W 80+ Gold (estimado: $180)"
    return (
        f"Build orientativa para {profile['performance']}:\n"
        f"- GPU: {gpu_name}, ${gpu_price:,.2f} USD según el CSV.\n"
        f"- CPU: {profile['cpu']}.\n"
        f"- Motherboard: {profile['motherboard']}.\n"
        f"- RAM: {profile['ram']}.\n"
        f"- Almacenamiento: {profile['storage']}.\n"
        f"- Fuente: {profile['psu']}.\n"
        f"- Gabinete: {profile['case']}.\n"
        f"- Total estimado: ${total:,.2f} USD, dentro del presupuesto de ${presupuesto:,.2f} USD.\n"
        "Los precios de GPU vienen del CSV; los demás son estimaciones locales, no cotizaciones. "
        "Verifica precios, stock y compatibilidad antes de comprar."
    )


TOOLS_BASE = [
    calculadora,
    fecha_hora_actual,
    consultar_clima,
    buscar_documentos,
    analizar_ventas,
    calcular_watts_fuente,
    verificar_compatibilidad,
    recomendar_build_presupuesto,
]