from __future__ import annotations

from .base import AdapterUnavailableError, OCRAdapter
from .chandra_adapter import ChandraOCRAdapter
from .easyocr_adapter import EasyOCRAdapter
from .granite_docling_adapter import GraniteDoclingAdapter
from .ocrmac_adapter import OcrMacAdapter
from .paddleocr_vl_adapter import PaddleOCRVLAdapter
from .rapidocr_adapter import RapidOCRAdapter
from .tesseract_adapter import TesseractAdapter
from .unlimited_ocr_adapter import UnlimitedOCRAdapter

#: name -> adapter class, used by cli.py's --adapters flag and by
#: tests/test_adapters.py to iterate "every registered adapter".
ADAPTER_REGISTRY: dict[str, type[OCRAdapter]] = {
    cls.name: cls
    for cls in (
        ChandraOCRAdapter,
        EasyOCRAdapter,
        GraniteDoclingAdapter,
        OcrMacAdapter,
        PaddleOCRVLAdapter,
        RapidOCRAdapter,
        TesseractAdapter,
        UnlimitedOCRAdapter,
    )
}

__all__ = [
    "OCRAdapter",
    "AdapterUnavailableError",
    "ADAPTER_REGISTRY",
    "ChandraOCRAdapter",
    "EasyOCRAdapter",
    "GraniteDoclingAdapter",
    "OcrMacAdapter",
    "PaddleOCRVLAdapter",
    "RapidOCRAdapter",
    "TesseractAdapter",
    "UnlimitedOCRAdapter",
]
