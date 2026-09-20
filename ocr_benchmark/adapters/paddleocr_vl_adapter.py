from __future__ import annotations

from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter


class PaddleOCRVLAdapter(OCRAdapter):
    name = "paddleocr-vl"

    def __init__(self):
        self._pipeline = None

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return hasattr(paddleocr, "PaddleOCRVL")

    def setup(self) -> None:
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise AdapterUnavailableError(
                "paddleocr is not installed. "
                '`pip install -U "paddleocr[doc-parser]"` plus a matching '
                "paddlepaddle / paddlepaddle-gpu build."
            ) from exc

        self._pipeline = PaddleOCRVL()

    def recognize(self, image_path: Path) -> str:
        assert self._pipeline is not None, "call setup() before recognize()"
        outputs = self._pipeline.predict(str(image_path))
        texts = []
        for res in outputs:
            markdown = getattr(res, "markdown", None)
            if isinstance(markdown, dict) and markdown.get("markdown_texts"):
                texts.append(markdown["markdown_texts"])
                continue
            # Fallback: reconstruct from per-block parsing results if the
            # markdown accessor isn't populated for this result.
            blocks = res["parsing_res_list"] if "parsing_res_list" in res else []
            texts.append(
                "\n".join(
                    getattr(block, "content", "") for block in blocks if block
                )
            )
        return "\n".join(t for t in texts if t).strip()
