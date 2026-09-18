"""Reference implementation #2: RapidOCR, routed through docling.

Same shape as TesseractAdapter — setup() builds one reusable converter,
recognize() just calls .convert(). This is the pattern to copy for every
stub adapter in this package.
"""

from __future__ import annotations

from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter
from .docling_common import build_converter


class RapidOCRAdapter(OCRAdapter):
    name = "rapidocr"

    def __init__(self, lang: str = "english"):
        self.lang = lang
        self._converter = None

    def is_available(self) -> bool:
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError:
            try:
                import rapidocr  # noqa: F401
            except ImportError:
                return False
        return True

    def setup(self) -> None:
        try:
            from docling.datamodel.pipeline_options import RapidOcrOptions
        except ImportError as exc:
            raise AdapterUnavailableError(
                "docling is not installed. `pip install docling`."
            ) from exc

        if not self.is_available():
            raise AdapterUnavailableError(
                "rapidocr is not installed. `pip install rapidocr-onnxruntime` "
                "(docling's RapidOcrOptions needs this package on the path)."
            )

        # Note (see RapidOcrOptions docstring in docling): `lang` isn't
        # actually wired through to rapidocr yet upstream — kept here for
        # forward-compat and so the CLI's --lang flag stays meaningful.
        ocr_options = RapidOcrOptions(lang=[self.lang])
        self._converter = build_converter(ocr_options)

    def recognize(self, image_path: Path) -> str:
        assert self._converter is not None, "call setup() before recognize()"
        result = self._converter.convert(str(image_path))
        return result.document.export_to_text().strip()
