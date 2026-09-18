"""IBM Granite-Docling (ibm-granite/granite-docling-258M), via docling's
VLM pipeline. This is structurally different from the other docling-backed
adapters: it's not an `OcrOptions` engine plugged into `PdfPipelineOptions`,
it's a full VLM document-conversion pipeline (`VlmPipeline` +
`VlmPipelineOptions`), so it gets its own converter-building code instead
of using docling_common.build_converter (which is wired for the classic
OCR-engine pipeline).

Much slower per-image than the classical OCR engines — it's running a
~258M-parameter vision-language model, not a detector+recognizer.
"""

from __future__ import annotations

from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter


class GraniteDoclingAdapter(OCRAdapter):
    name = "granite-docling"

    def __init__(self, device: str | None = None):
        # device: None lets docling/transformers auto-select (CUDA > MPS >
        # CPU). Override with "cpu" / "cuda" / "mps" if auto-detection
        # picks the wrong accelerator for your machine.
        self.device = device
        self._converter = None

    def is_available(self) -> bool:
        try:
            import docling.datamodel.vlm_model_specs
            import docling.pipeline.vlm_pipeline
            import transformers
        except ImportError:
            return False
        return True

    def setup(self) -> None:
        try:
            from docling.datamodel import vlm_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorOptions,
                VlmPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.vlm_pipeline import VlmPipeline
        except ImportError as exc:
            raise AdapterUnavailableError(
                "docling's VLM extras are not installed. "
                '`pip install "docling[vlm]"`.'
            ) from exc

        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
        )
        if self.device is not None:
            # AcceleratorDevice takes plain strings too ("cpu"/"cuda"/"mps").
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=self.device
            )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

    def recognize(self, image_path: Path) -> str:
        assert self._converter is not None, "call setup() before recognize()"
        result = self._converter.convert(str(image_path))
        return result.document.export_to_text().strip()
