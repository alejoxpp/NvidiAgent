import sys
import types


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


try:
    from langchain_core.tools import tool  # noqa: F401
except ModuleNotFoundError:
    core = sys.modules.get("langchain_core") or _module("langchain_core")
    core_documents = _module("langchain_core.documents")

    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    core_documents.Document = Document
    core_tools = _module("langchain_core.tools")
    core_tools.tool = lambda function: function

try:
    from langchain_community.document_loaders import PyPDFLoader  # noqa: F401
except ModuleNotFoundError:
    community = _module("langchain_community")
    loaders = _module("langchain_community.document_loaders")

    class PyPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            from langchain_core.documents import Document

            return [Document(page_content=f"PDF fallback: {self.path}", metadata={"source": self.path, "page": 0})]

    loaders.PyPDFLoader = PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
except ModuleNotFoundError:
    splitters = _module("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents):
            return documents

    splitters.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter