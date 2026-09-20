from __future__ import annotations

import sys
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter
from .docling_common import build_converter


class OcrMacAdapter(OCRAdapter):
    name = "ocrmac"

    def __init__(self):
        self._converter = None

    def is_available(self) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            import ocrmac  # noqa: F401
        except ImportError:
            return False
        return True

    def setup(self) -> None:
        try:
            from docling.datamodel.pipeline_options import OcrMacOptions
        except ImportError as exc:
            raise AdapterUnavailableError(
                "docling is not installed. `pip install docling`."
            ) from exc

        if not self.is_available():
            raise AdapterUnavailableError(
                "ocrmac requires macOS with the ocrmac package installed "
                "(`pip install ocrmac`)."
            )

        ocr_options = OcrMacOptions()
        self._converter = build_converter(ocr_options)

    def recognize(self, image_path: Path) -> str:
        assert self._converter is not None, "call setup() before recognize()"
        result = self._converter.convert(str(image_path))
        return result.document.export_to_text().strip()
