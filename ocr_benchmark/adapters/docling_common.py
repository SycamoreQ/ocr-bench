"""Shared plumbing for adapters that go through docling's DocumentConverter
rather than calling an OCR library directly.

Docling normally targets whole documents (PDF/image -> layout + text), which
is overkill per IAM line-crop, but it gives every one of these engines the
exact same call shape, which is what we want for a fair, low-effort-to-add
benchmark. Each adapter just supplies an `OcrOptions` instance and this
module does the conversion + text extraction.
"""

from __future__ import annotations

from pathlib import Path

from .base import AdapterUnavailableError


def convert_image_to_text(image_path: Path, ocr_options) -> str:
    """Run docling's PDF/image pipeline (OCR-only, no table/layout work
    needed for a single word/line crop) over one image and return the
    extracted plain text.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise AdapterUnavailableError(
            "docling is not installed. `pip install docling`."
        ) from exc

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        ocr_options=ocr_options,
    )
    pipeline_options.ocr_options.force_full_page_ocr = True

    converter = DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(image_path))
    return result.document.export_to_text().strip()


def build_converter(ocr_options):
    """Like convert_image_to_text but returns a reusable DocumentConverter,
    for adapters that want to build it once in setup() and reuse it across
    every recognize() call instead of rebuilding it (and re-warming the
    OCR engine) per image. Prefer this over convert_image_to_text() in real
    adapter implementations; convert_image_to_text() exists mainly as a
    minimal reference for how the pieces fit together.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise AdapterUnavailableError(
            "docling is not installed. `pip install docling`."
        ) from exc

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        ocr_options=ocr_options,
    )
    pipeline_options.ocr_options.force_full_page_ocr = True

    return DocumentConverter(
        format_options={
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
