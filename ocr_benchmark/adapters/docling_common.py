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
