from __future__ import annotations

import shutil
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter
from .docling_common import build_converter


class TesseractAdapter(OCRAdapter):
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 7):
        self.lang = lang
        self.psm = psm
        self._converter = None

    def is_available(self) -> bool:
        return shutil.which("tesseract") is not None

    def setup(self) -> None:
        try:
            from docling.datamodel.pipeline_options import TesseractOcrOptions
        except ImportError as exc:
            raise AdapterUnavailableError(
                "docling is not installed. `pip install docling`."
            ) from exc

        if not self.is_available():
            raise AdapterUnavailableError(
                "tesseract binary not found on PATH. "
                "`sudo apt install tesseract-ocr` (or `brew install tesseract`)."
            )

        ocr_options = TesseractOcrOptions(lang=[self.lang], psm=self.psm)
        self._converter = build_converter(ocr_options)

    def recognize(self, image_path: Path) -> str:
        assert self._converter is not None, "call setup() before recognize()"
        result = self._converter.convert(str(image_path))
        return result.document.export_to_text().strip()
