from __future__ import annotations

import re
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
_MD_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_MD_CODE_RE = re.compile(r"`([^`]*)`")
_MD_EMPHASIS_RE = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)(.+?)\1")


def _markdown_to_text(markdown: str) -> str:
    text = _MD_IMAGE_RE.sub("", markdown)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_HR_RE.sub("", text)
    text = _MD_LIST_RE.sub("", text)
    text = _MD_CODE_RE.sub(r"\1", text)
    text = _MD_EMPHASIS_RE.sub(r"\2", text)
    return re.sub(r"\s+", " ", text).strip()


class ChandraOCRAdapter(OCRAdapter):
    name = "chandra"

    def __init__(self, prompt_type: str = "ocr", max_output_tokens: int = 256):
        # "ocr" (plain HTML, no layout blocks) rather than "ocr_layout"
        # (bbox-annotated layout blocks) -- the latter is built for full
        # pages, and its bbox/label wrapping is pure overhead on a single
        # line crop with nothing to lay out.
        self.prompt_type = prompt_type
        # The library's own default (settings.MAX_OUTPUT_TOKENS) is 12384,
        # sized for whole-page documents. A single handwritten line needs
        # nowhere near that -- generation still stops early via the
        # eos/im_end tokens regardless, but capping this bounds worst-case
        # latency if the model ever runs on without stopping cleanly.
        self.max_output_tokens = max_output_tokens
        self._manager = None

    def is_available(self) -> bool:
        try:
            import torch
            import transformers
            import chandra
        except ImportError:
            return False
        return True

    def setup(self) -> None:
        try:
            from chandra.model import InferenceManager
        except ImportError as exc:
            raise AdapterUnavailableError(
                "chandra-ocr is not installed. "
                '`pip install "chandra-ocr[hf]"` (the [hf] extra pulls in '
                "torch/transformers for local inference; without it, only "
                "the vLLM-server code path is installed)."
            ) from exc

        self._manager = InferenceManager(method="hf")

    def recognize(self, image_path: Path) -> str:
        assert self._manager is not None, "call setup() before recognize()"
        from PIL import Image
        from chandra.model.schema import BatchInputItem

        image = Image.open(image_path).convert("RGB")
        batch = [BatchInputItem(image=image, prompt_type=self.prompt_type)]

        result = self._manager.generate(
            batch, max_output_tokens=self.max_output_tokens
        )[0]

        if result.error:
            return ""

        return _markdown_to_text(result.markdown)
