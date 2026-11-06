"""core/graph.py — Grafo coordinador (LangGraph)
RetroTron · InvidiAgent · Soluciones de Computación e IA de NVIDIA

       START → coordinador → [atencion_cliente | documentos | analisis | reporte] → END

   • atencion_cliente → sub-grafo multiagente para CLIENTES externos
   • documentos       → RAG para consultas TÉCNICAS / HARDWARE y SOFTWARE
   • analisis         → análisis de mercado, comparativas de productos, benchmarks
   • reporte          → Salida Estructurada con Pydantic (hardware/software)
"""

import os
import sys
from typing import TypedDict, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from s6_llm_factory import crear_llm
from core.tools import TOOLS_BASE, buscar_documentos, analizar_ventas
from core.rag import obtener_motor
from core.structured import ReporteInventarioHardware
from core.atencion_cliente import invoke_customer_agent as invoke_atencion_cliente

NOMBRE_AGENTE = os.getenv("AGENT_NAME", "RetroTron")
ROL_AGENTE = os.getenv("AGENT_ROLE", "Experto en soluciones de computación y IA")


def limpiar_pensamiento(texto: str) -> str:
    if texto and "\n\n" in texto:
        return texto.split("\n\n")[-1].strip()
    return (texto or "").strip()


# ════════════════════════════════════════════════════════════
#  STATE
# ════════════════════════════════════════════════════════════
class EstadoAgente(TypedDict):
    entrada_usuario: str
    historial: list
    categoria: str
    respuesta_final: str
    fuentes: list


# ════════════════════════════════════════════════════════════
#  COORDINADOR
# ════════════════════════════════════════════════════════════
def nodo_coordinador(estado: EstadoAgente) -> dict:
    llm = crear_llm(temperature=0.0)
    categorias = ["atencion_cliente", "documentos", "analisis", "reporte"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Eres {NOMBRE_AGENTE}, un experto en soluciones de computación y IA de NVIDIA.
Tu personalidad: técnico pero accesible, riguroso con especificaciones y rendimiento.

Clasifica en una categoría y responde solo su nombre en minúsculas:
- atencion_cliente: saludos, preguntas generales y catálogo. Ejemplo: "¿Cuánto cuesta la RTX 4070?"
- documentos: manuales, drivers, instalación y compatibilidad documentada. Ejemplo: "Manual de instalación RTX 4070"
- analisis: comparaciones, ventas o interpretación de datos. Ejemplo: "Compara RTX 4090 vs RTX 3080"
- reporte: solicitud explícita de reporte. Ejemplo: "Genera un reporte de inventario NVIDIA"""),
        ("human", "Pregunta: {entrada_usuario}\nResponde solo con la categoría.")
    ])

    respuesta = (prompt | llm).invoke({"entrada_usuario": estado["entrada_usuario"]}).content
    tokens = [
        token.strip(".,:;!?\"'`()[]")
        for token in limpiar_pensamiento(str(respuesta)).lower().split()
    ]
    categoria = next((opcion for opcion in categorias if opcion in tokens), "atencion_cliente")

    return {"categoria": categoria}


# ════════════════════════════════════════════════════════════
#  NODO: ATENCIÓN AL CLIENTE
# ════════════════════════════════════════════════════════════
def nodo_atencion_cliente(estado: EstadoAgente) -> dict:
    resultado = invoke_atencion_cliente(estado["entrada_usuario"])
    return {"respuesta_final": resultado["answer"]}


# ════════════════════════════════════════════════════════════
#  NODO: DOCUMENTOS (RAG)
# ════════════════════════════════════════════════════════════
def nodo_documentos(estado: EstadoAgente) -> dict:
    pregunta = estado["entrada_usuario"]
    motor = obtener_motor()
    contexto, fuentes = motor.buscar(pregunta)
    if not contexto:
        return {"respuesta_final": "No encontré información relevante en los documentos NVIDIA disponibles.", "fuentes": []}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente técnico NVIDIA. Responde en español usando solo el contexto; si falta un dato, dilo."),
        ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}"),
    ])
    respuesta = (prompt | crear_llm(temperature=0.2)).invoke({"contexto": contexto, "pregunta": pregunta}).content
    return {
        "respuesta_final": limpiar_pensamiento(str(respuesta)) + "\n\nFuentes: " + ", ".join(fuentes),
        "fuentes": fuentes,
    }


# ════════════════════════════════════════════════════════════
#  NODO: ANÁLISIS
# ════════════════════════════════════════════════════════════
def nodo_analisis(estado: EstadoAgente) -> dict:
    pregunta = estado["entrada_usuario"]
    texto = pregunta.casefold()
    if any(palabra in texto for palabra in ("venta", "inventario", "stock", "vendid", "csv")):
        agrupacion = next((campo for campo in ("arquitectura", "categoria", "marca", "producto") if campo in texto), "producto")
        metrica = "stock" if "stock" in texto else "precio_usd" if "precio" in texto else "cantidad_vendida"
        meses = (
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        )
        mes = next((nombre for nombre in meses if nombre in texto), "")
        anio = next((token for token in texto.split() if token.isdigit() and len(token) == 4), "")
        consulta_mes = " ".join(parte for parte in (mes, anio) if parte)
        producto = "RTX 4090" if "4090" in texto else ""
        datos = analizar_ventas.invoke({
            "agrupar_por": agrupacion,
            "metrica": metrica,
            "producto": producto,
            "mes": consulta_mes,
        })
        return {"respuesta_final": datos}

    contexto, fuentes = obtener_motor().buscar(pregunta)
    if not contexto:
        return {"respuesta_final": "No encontré datos suficientes en la base de conocimiento para realizar esa comparación.", "fuentes": []}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un analista técnico de NVIDIA. Analiza con rigor, no inventes datos y responde en español."),
        ("human", "Pregunta: {pregunta}\n\nInformación recopilada:\n{contexto}\n\nCompara y concluye."),
    ])
    respuesta = (prompt | crear_llm(temperature=0.3)).invoke({"pregunta": pregunta, "contexto": contexto}).content
    return {"respuesta_final": limpiar_pensamiento(str(respuesta)), "fuentes": fuentes}


# ════════════════════════════════════════════════════════════
#  NODO: REPORTE — Salida Estructurada con Pydantic
# ════════════════════════════════════════════════════════════
def nodo_reporte(estado: EstadoAgente) -> dict:
    modelo_estructurado = os.getenv("GROQ_MODEL_ESTRUCTURADO", "openai/gpt-oss-120b")
    llm_estructurado = crear_llm(temperature=0.4, modelo=modelo_estructurado).with_structured_output(ReporteInventarioHardware)

    datos_inventario = analizar_ventas.invoke({"agrupar_por": "categoria", "metrica": "stock"})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Genera un reporte estructurado del inventario NVIDIA basado solo en los datos disponibles.
No inventes cifras ni presentes estimaciones como datos reales.
Responde en español."""),
        ("human", "Solicitud: {solicitud}\n\nDatos disponibles:\n{datos}")
    ])
    chain = prompt | llm_estructurado
    reporte: ReporteInventarioHardware = chain.invoke({
        "solicitud": estado["entrada_usuario"],
        "datos": datos_inventario,
    })

    salida = f"REPORTE DE INVENTARIO NVIDIA\n\n{reporte.resumen_general}\n"
    if reporte.alertas:
        salida += "\nALERTAS:\n"
        for a in reporte.alertas:
            salida += f"- [{a.area}] {a.descripcion}\n  Acción: {a.accion_recomendada}\n"
    if reporte.oportunidades:
        salida += "\nOPORTUNIDADES:\n" + "\n".join(f"- {o}" for o in reporte.oportunidades) + "\n"
    if reporte.recomendaciones:
        salida += "\nRECOMENDACIONES:\n" + "\n".join(f"- {recomendacion}" for recomendacion in reporte.recomendaciones)

    return {"respuesta_final": salida}


# ════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL GRAFO
# ════════════════════════════════════════════════════════════
def construir_grafo():
    grafo = StateGraph(EstadoAgente)

    grafo.add_node("coordinador", nodo_coordinador)
    grafo.add_node("atencion_cliente", nodo_atencion_cliente)
    grafo.add_node("documentos", nodo_documentos)
    grafo.add_node("analisis", nodo_analisis)
    grafo.add_node("reporte", nodo_reporte)

    grafo.add_edge(START, "coordinador")
    grafo.add_conditional_edges("coordinador", router, {
        "atencion_cliente": "atencion_cliente",
        "documentos": "documentos",
        "analisis": "analisis",
        "reporte": "reporte",
    })
    grafo.add_edge("atencion_cliente", END)
    grafo.add_edge("documentos", END)
    grafo.add_edge("analisis", END)
    grafo.add_edge("reporte", END)

    return grafo.compile()


# ════════════════════════════════════════════════════════════
#  ROUTER — Función de clasificación
# ════════════════════════════════════════════════════════════
def router(estado: EstadoAgente) -> str:
    categoria = estado.get("categoria", "")
    return categoria
