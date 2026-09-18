"""All adapters, in one registry.

Import is intentionally lazy-friendly: importing this package does not
import any OCR library itself — each adapter module only imports its
underlying engine inside setup()/is_available(), so `import ocr_benchmark`
stays cheap even if you only have three of these nine engines installed.
"""

from __future__ import annotations

from .base import AdapterUnavailableError, OCRAdapter
from .easyocr_adapter import EasyOCRAdapter
from .granite_docling_adapter import GraniteDoclingAdapter
from .ocrmac_adapter import OcrMacAdapter
from .paddleocr_vl_adapter import PaddleOCRVLAdapter
from .rapidocr_adapter import RapidOCRAdapter
from .surya_adapter import SuryaAdapter
from .tesseract_adapter import TesseractAdapter
from .unlimited_ocr_adapter import UnlimitedOCRAdapter

#: name -> adapter class, used by cli.py's --adapters flag and by
#: tests/test_adapters.py to iterate "every registered adapter".
ADAPTER_REGISTRY: dict[str, type[OCRAdapter]] = {
    cls.name: cls
    for cls in (
        EasyOCRAdapter,
        GraniteDoclingAdapter,
        OcrMacAdapter,
        PaddleOCRVLAdapter,
        RapidOCRAdapter,
        SuryaAdapter,
        TesseractAdapter,
        UnlimitedOCRAdapter,
    )
}

__all__ = [
    "OCRAdapter",
    "AdapterUnavailableError",
    "ADAPTER_REGISTRY",
    "EasyOCRAdapter",
    "GraniteDoclingAdapter",
    "OcrMacAdapter",
    "PaddleOCRVLAdapter",
    "RapidOCRAdapter",
    "SuryaAdapter",
    "TesseractAdapter",
    "UnlimitedOCRAdapter",
]
