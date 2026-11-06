import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tests._langchain_stubs  # noqa: F401
from core.rag import MotorRAG


class NvidiaRagTests(unittest.TestCase):
    def test_txt_is_preferred_and_pdf_is_only_used_as_fallback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            documents_dir = Path(temporary_directory)
            (documents_dir / "manual.txt").write_text("TXT knowledge", encoding="utf-8")
            (documents_dir / "manual.pdf").write_bytes(b"not parsed because TXT exists")
            (documents_dir / "fallback.pdf").write_bytes(b"loaded by the PDF stub")

            with patch("core.rag.DOCUMENTOS_DIR", documents_dir):
                motor = MotorRAG()

        sources = {Path(document.metadata["source"]).name for document in motor.paginas}
        contents = " ".join(document.page_content for document in motor.paginas)
        self.assertEqual(sources, {"manual.txt", "fallback.pdf"})
        self.assertIn("TXT knowledge", contents)
        self.assertNotIn("manual.pdf", contents)


if __name__ == "__main__":
    unittest.main()