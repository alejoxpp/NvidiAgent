"""
core/rag.py — Motor de RAG (Retrieval Augmented Generation)
Seminario: Agentes de IA · Sesión 6 (Despliegue)

Lee los documentos reales de data/documentos/ (TXT, PDF de respaldo y CSV) y responde
preguntas basándose en su contenido — el mismo concepto de la Sesión 4,
adaptado para desplegarse sin dependencias pesadas de embeddings
(nada de sentence-transformers/torch — más liviano y rápido de
construir en Railway).

Búsqueda HÍBRIDA en dos capas (la misma técnica validada en el curso):
  1) Coincidencia de palabras clave sobre fragmentos pequeños
  2) Texto exacto sobre la página COMPLETA (respaldo para datos puntuales)

✏️ VENTANA DE CONTEXTO — configurable en .env:
  TAMANO_FRAGMENTO, SUPERPOSICION_FRAGMENTO, K_FRAGMENTOS
"""

import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENTOS_DIR = Path(os.getenv("DOCUMENTOS_DIR", "data/documentos"))
if not DOCUMENTOS_DIR.is_absolute():
    DOCUMENTOS_DIR = PROJECT_DIR / DOCUMENTOS_DIR
TAMANO_FRAGMENTO = int(os.getenv("TAMANO_FRAGMENTO", "900"))
SUPERPOSICION_FRAGMENTO = int(os.getenv("SUPERPOSICION_FRAGMENTO", "120"))
K_FRAGMENTOS = int(os.getenv("K_FRAGMENTOS", "4"))

STOPWORDS_ES = {
    "de", "la", "el", "los", "las", "en", "y", "a", "que", "es", "un",
    "una", "unos", "unas", "para", "con", "por", "su", "sus", "se", "del",
    "al", "lo", "como", "mas", "pero", "le", "ya", "o", "fue", "fueron",
    "ha", "han", "si", "no", "sin", "sobre", "entre", "cuando", "muy",
    "este", "esta", "estos", "estas", "eso", "esa", "ese", "quien",
    "quienes", "cual", "cuales", "donde", "tu", "te", "me", "mi", "yo",
    "nos", "les", "que", "cómo", "dónde", "cuánto", "cuánta",
}


def normalizar(texto: str) -> str:
    """Quita tildes, puntuación y pasa a minúsculas."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = "".join(c if c.isalnum() or c.isspace() else " " for c in texto)
    return texto


def palabras_clave(texto: str) -> set:
    return {p for p in normalizar(texto).split() if p not in STOPWORDS_ES and len(p) > 1}


def _cargar_txt() -> list:
    documentos = []
    for ruta in sorted(DOCUMENTOS_DIR.glob("*.txt")):
        texto = ruta.read_text(encoding="utf-8")
        documentos.append(Document(
            page_content=texto,
            metadata={"source": str(ruta), "page": 0},
        ))
    return documentos


def _cargar_pdfs() -> list:
    documentos = []
    for ruta in sorted(DOCUMENTOS_DIR.glob("*.pdf")):
        if ruta.with_suffix(".txt").exists():
            continue
        documentos.extend(PyPDFLoader(str(ruta)).load())
    return documentos


def _cargar_csv() -> list:
    """Cada CSV entra como un único documento de texto (encabezado + filas)."""
    documentos = []
    for ruta in sorted(DOCUMENTOS_DIR.glob("*.csv")):
        texto = ruta.read_text(encoding="utf-8")
        documentos.append(Document(page_content=texto, metadata={"source": str(ruta), "page": 0}))
    return documentos


class MotorRAG:
    """Índice de búsqueda sobre la base de conocimiento NVIDIA."""

    def __init__(self):
        self.paginas: list = []
        self.fragmentos: list = []
        self._construir()

    def _construir(self):
        if not DOCUMENTOS_DIR.exists():
            print(f"  ⚠️  Carpeta de documentos no encontrada: {DOCUMENTOS_DIR}")
            return
        documentos = _cargar_txt() + _cargar_pdfs() + _cargar_csv()
        self.paginas = documentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=TAMANO_FRAGMENTO, chunk_overlap=SUPERPOSICION_FRAGMENTO
        )
        self.fragmentos = splitter.split_documents(documentos)
        print(f"  📚 RAG listo: {len(documentos)} documento(s), {len(self.fragmentos)} fragmentos")

    def buscar(self, pregunta: str, k: int = K_FRAGMENTOS):
        """Devuelve (texto_contexto, lista_de_fuentes)."""
        palabras_pregunta = palabras_clave(pregunta)
        if not palabras_pregunta or not self.fragmentos:
            return "", []

        puntuados = []
        for frag in self.fragmentos:
            score = len(palabras_pregunta & palabras_clave(frag.page_content))
            if score > 0:
                puntuados.append((score, frag))
        puntuados.sort(key=lambda x: x[0], reverse=True)

        elegidos = {}
        for _, frag in puntuados[:k]:
            clave = (frag.metadata.get("source"), frag.metadata.get("page"))
            elegidos[clave] = frag

        # Respaldo: texto exacto sobre la página completa (datos puntuales)
        umbral = max(2, len(palabras_pregunta) // 2)
        for pagina in self.paginas:
            if len(palabras_pregunta & palabras_clave(pagina.page_content)) >= umbral:
                clave = (pagina.metadata.get("source"), pagina.metadata.get("page"))
                elegidos.setdefault(clave, pagina)

        texto = ""
        fuentes = []
        for frag in list(elegidos.values())[: k + 2]:
            archivo = os.path.basename(str(frag.metadata.get("source", "?")))
            pagina_num = frag.metadata.get("page", 0)
            texto += f"\n[Fuente: {archivo}, página {pagina_num}]\n{frag.page_content}\n"
            fuentes.append(f"{archivo} (pág. {pagina_num})")

        return texto, sorted(set(fuentes))


_motor: MotorRAG | None = None


def obtener_motor() -> MotorRAG:
    global _motor
    if _motor is None:
        _motor = MotorRAG()
    return _motor
