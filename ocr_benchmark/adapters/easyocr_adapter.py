from __future__ import annotations

from pathlib import Path

import cv2

from .base import AdapterUnavailableError, OCRAdapter


class EasyOCRAdapter(OCRAdapter):
    name = "easyocr"

    def __init__(self):
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def setup(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise AdapterUnavailableError(
                "easyocr is not installed. "
                "Install with: pip install easyocr"
            ) from exc

        # EasyOCR automatically selects CUDA -> MPS -> CPU.
        self._reader = easyocr.Reader(
            ["en"],
            gpu=True,
            verbose=False,
        )

    def recognize(self, image_path: Path) -> str:
        assert self._reader is not None, "call setup() before recognize()"

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]

        # IAM-Line is already a cropped text line.
        # Tell EasyOCR that the complete image is the text region.
        horizontal_list = [[0, width, 0, height]]
        free_list = []

        result = self._reader.recognize(
            image,
            horizontal_list=horizontal_list,
            free_list=free_list,
            decoder="greedy",
            batch_size=1,
            detail=0,
            paragraph=False,
        )

        if not result:
            return ""

        return " ".join(result).strip()
