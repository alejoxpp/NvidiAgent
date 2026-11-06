"""core/atencion_cliente.py — Sub-grafo multiagente de atención al cliente
RetroTron · InvidiAgent · Soluciones de Computación e IA de NVIDIA

Las bases de conocimiento se cargan desde TXT y usan PDFs homónimos como
respaldo en data/documentos/:

   KB_EMPRESA   ← manuales_controladores.pdf + guia_compatibilidad.pdf
   KB_PRODUCTOS ← catalogo_productos_nvidia.pdf

Flujo:
   START → supervisor → greeting_worker → company_worker
         → products_worker → fallback_worker → aggregator → END
"""

import os
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END

from s6_llm_factory import crear_llm


# ════════════════════════════════════════════════════════════
#  CARGA DE BASES DE CONOCIMIENTO DESDE TXT/PDF
# ════════════════════════════════════════════════════════════
# Rutas a los archivos de conocimiento NVIDIA
PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTOS_DIR = Path(os.getenv("DOCUMENTOS_DIR", "data/documentos"))
if not DOCUMENTOS_DIR.is_absolute():
    DOCUMENTOS_DIR = PROJECT_DIR / DOCUMENTOS_DIR

PDFS_KB_EMPRESA = [
    os.getenv("PDF_KB_EMPRESA_1", "01_manuales_controladores.txt"),
    os.getenv("PDF_KB_EMPRESA_2", "02_guia_compatibilidad.txt"),
]
PDF_KB_PRODUCTOS = os.getenv("PDF_KB_PRODUCTOS", "03_catalogo_productos.txt")



def _cargar_documento_texto(nombre_archivo: str) -> str:
    """Lee TXT directamente o extrae texto de un PDF."""
    ruta = DOCUMENTOS_DIR / nombre_archivo
    if not ruta.exists():
        respaldo = ruta.with_suffix(".pdf") if ruta.suffix.lower() == ".txt" else None
        if respaldo and respaldo.exists():
            ruta = respaldo
        else:
            return ""
    try:
        if ruta.suffix.lower() == ".txt":
            return ruta.read_text(encoding="utf-8")
        paginas = PyPDFLoader(str(ruta)).load()
        return "\n\n".join(p.page_content for p in paginas if p.page_content.strip())
    except Exception:
        return ""


def _cargar_multiples(nombres: list[str]) -> str:
    """Carga varios documentos y los concatena con un separador claro."""
    bloques = []
    for n in nombres:
        texto = _cargar_documento_texto(n)
        if texto:
            bloques.append(f"### Documento: {n}\n\n{texto}")
    return "\n\n".join(bloques)


# Se ejecutan UNA sola vez al importar el módulo
KB_EMPRESA = _cargar_multiples(PDFS_KB_EMPRESA)
KB_PRODUCTOS = _cargar_documento_texto(PDF_KB_PRODUCTOS)

if not KB_EMPRESA:
    KB_EMPRESA = "(No se pudo cargar la información de soporte.)"
if not KB_PRODUCTOS:
    KB_PRODUCTOS = "(No se pudo cargar el catálogo.)"


# ════════════════════════════════════════════════════════════
#  DOMINIOS SOPORTADOS (para el fallback)
# ════════════════════════════════════════════════════════════
DOMINIOS_SOPORTADOS = """
- 👋 Saludos y conversación sobre tecnología y computación.
- 🏢 Información sobre productos NVIDIA: GeForce, RTX, Quadro, Drive, IA.
- 🛒 Consultas de catálogo: especificaciones, precios, disponibilidad.
- ⚙️ Soporte técnico: drivers, instalación, resolución de problemas.
- 📦 Políticas: garantías, devoluciones, RMA para productos NVIDIA.
"""


# ════════════════════════════════════════════════════════════
#  CONTRATO: Clasificación
# ═════════════════════════════════════════════════════════════
class QuestionClassification(BaseModel):
    greeting: str = Field(default="", description="Saludo o conversación casual.")
    company: str = Field(default="", description="Preguntas sobre productos NVIDIA.")
    products: str = Field(default="", description="Preguntas sobre catálogo y especificaciones.")
    other: str = Field(default="", description="Fuera de alcance o consultas no técnicas.")


class CustomerState(TypedDict, total=False):
    messages: list[AnyMessage]
    greeting_question: str
    company_question: str
    products_question: str
    other_question: str
    greeting_response: str
    company_response: str
    products_response: str
    other_response: str
    final_response: str


def supervisor(state: CustomerState) -> dict:
    """Clasifica el mensaje actual para los workers de atención."""
    pregunta = state["messages"][-1].content
    prompt = [
        SystemMessage(content=(
            "Clasifica el mensaje en greeting, company, products u other. "
            "Devuelve el mensaje original solo en el campo que corresponda y deja los demás vacíos. "
            "greeting: saludos; company: información general NVIDIA; products: catálogo y especificaciones; "
            "other: fuera del dominio."
        )),
        HumanMessage(content=pregunta),
    ]
    clasificacion: QuestionClassification = crear_llm(temperature=0).with_structured_output(
        QuestionClassification
    ).invoke(prompt)
    return {
        "greeting_question": clasificacion.greeting,
        "company_question": clasificacion.company,
        "products_question": clasificacion.products,
        "other_question": clasificacion.other,
    }


# ════════════════════════════════════════════════════════════
#  TRABAJADORES ESPECIALIZADOS
# ════════════════════════════════════════════════════════════
def greeting_worker(state: dict) -> dict:
    greeting = state.get("greeting_question", "")
    if greeting:
        return {"greeting_response": f"¡Hola! Soy {NOMBRE_AGENTE}, especialista en tecnología NVIDIA. ¿En qué puedo ayudarte?"}
    return {"greeting_response": ""}


def company_worker(state: dict) -> dict:
    company = state.get("company_question", "")
    if company:
        respuesta = f"""**🏢 Información de NVIDIA**

Somos líderes en tecnología de computación visual e IA.

• 🎮 **GeForce RTX**: La mejor experiencia para gaming y creadores.
• 🤖 **IA y Aprendizaje Automático**: Plataformas NVIDIA AI para desarrollo.
• 🚗 **Automotive**: Soluciones para vehículos autónomos.
• 🏥 **Salud**: IA para descubrimiento de fármacos y imágenes médicas.
• 🌍 **Sostenimiento**: Centros de datos eficientes con tecnología NVIDIA.

{KB_EMPRESA[:600] if KB_EMPRESA else ""}"""
        return {"company_response": respuesta}
    return {"company_response": ""}


def products_worker(state: dict) -> dict:
    products = state.get("products_question", "")
    if products:
        respuesta = f"""**🛒 Catálogo de Productos NVIDIA**

Trabajamos con las tecnologías más avanzadas:

• 🃏 **GPUs GeForce**: Serie RTX 4090, 4080, 4070 para gaming definitivo.
• 💻 **Processadores Grace**: Para computación de alto rendimiento.
• 🧠 **IA y Software**: NVIDIA AI Enterprise, cuDIA, RAPIDS.
• 🚗 **Drive**: Plataformas para automoción y transporte autónomo.
• 📊 **Visualización**: Soluciones para diseño, renderizado y simulación.

Principales categorías:
• **Drivers y Software**: Controladores Game Ready, Studio, NVIDIA Omniverse.
• **Hardware**: Tarjetas gráficas, servidores, sistemas integrados.
• **Desarrollo**: Toolkits, bibliotecas optimizadas, SDKs.
• **Soluciones Empresariales**: Plataformas para data center y nube.

{KB_PRODUCTOS[:800] if KB_PRODUCTOS else ""}"""
        return {"products_response": respuesta}
    return {"products_response": ""}


def fallback_worker(state: dict) -> dict:
    other = state.get("other_question", "")
    if other:
        return {"other_response": "⚠️ Consulta fuera de alcance. Puedo ayudarte con: soporte de drivers, especificaciones de productos, información sobre suites de IA, o consulta nuestro catálogo de productos NVIDIA."}
    return {"other_response": ""}


def aggregator(state: dict) -> dict:
    parts = []
    if state.get("greeting_question"):
        parts.append(f"[Saludo]\n{state['greeting_response']}")
    if state.get("company_question"):
        parts.append(f"[Empresa]\n{state['company_response']}")
    if state.get("products_question"):
        parts.append(f"[Productos]\n{state['products_response']}")
    if state.get("other_question"):
        parts.append(f"[Fuera de alcance]\n{state['other_response']}")

    if not parts:
        return {"final_response": "Lo siento, no pude procesar tu consulta."}

    combined = "\n\n".join(parts)
    llm = crear_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=f"""Eres {NOMBRE_AGENTE}. Toma las respuestas de los especialistas y conviértelas en UNA SOLA respuesta natural.

Reglas:
- Empieza con el saludo (si lo hay)
- Responde cada tema en orden natural
- Si hay 'fuera de alcance', intégrala al final con amabilidad
- NO uses etiquetas como [Saludo] o [Productos]
- Cierra ofreciendo ayuda adicional
- Tono técnico pero accesible, experto en soluciones NVIDIA
- Cierra preguntando si necesitan ayuda específica con nuestros productos o tecnologías"""),
        HumanMessage(content=f"""Mensaje original del cliente: {state['messages'][-1].content}

Respuestas de los especialistas:
{combined}""")
    ])
    return {"final_response": response.content}


# ════════════════════════════════════════════════════════════
#  GRAFO
# ════════════════════════════════════════════════════════════
def build_customer_agent():
    wf = StateGraph(CustomerState)
    wf.add_node("supervisor", supervisor)
    wf.add_node("greeting_worker", greeting_worker)
    wf.add_node("company_worker", company_worker)
    wf.add_node("products_worker", products_worker)
    wf.add_node("fallback_worker", fallback_worker)
    wf.add_node("aggregator", aggregator)

    wf.add_edge(START, "supervisor")
    wf.add_edge("supervisor", "greeting_worker")
    wf.add_edge("greeting_worker", "company_worker")
    wf.add_edge("company_worker", "products_worker")
    wf.add_edge("products_worker", "fallback_worker")
    wf.add_edge("fallback_worker", "aggregator")
    wf.add_edge("aggregator", END)
    return wf.compile()


customer_agent = build_customer_agent()


# ════════════════════════════════════════════════════════════
#  API PÚBLICA
# ════════════════════════════════════════════════════════════
def invoke_customer_agent(query: str) -> dict:
    """Invoca el sub-grafo multiagente y devuelve un dict listo para la API."""
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "greeting_question": "",
        "company_question": "",
        "products_question": "",
        "other_question": "",
        "greeting_response": "",
        "company_response": "",
        "products_response": "",
        "other_response": "",
        "final_response": "",
    }
    result = customer_agent.invoke(initial_state)
    return {
        "answer": result["final_response"],
        "classification": {
            "saludo": result["greeting_question"],
            "empresa": result["company_question"],
            "productos": result["products_question"],
            "fuera_de_alcance": result["other_question"],
        },
        "worker_responses": {
            "saludo": result["greeting_response"],
            "empresa": result["company_response"],
            "productos": result["products_response"],
            "fuera_de_alcance": result["other_response"],
        },
    }


# ════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ════════════════════════════════════════════════════════════
NOMBRE_AGENTE = os.getenv("AGENT_NAME", "RetroTron")
ROL_AGENTE = os.getenv("AGENT_ROLE", "Experto en soluciones de computación y IA")
