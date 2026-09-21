from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .base import AdapterUnavailableError, OCRAdapter


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$", re.MULTILINE)
_MD_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+", re.MULTILINE)
_MD_CODE_RE = re.compile(r"`([^`]*)`")
_MD_EMPHASIS_RE = re.compile(
    r"(\*\*\*|\*\*|\*|___|__|_)(.+?)\1"
)


def _markdown_to_text(markdown: str) -> str:
    """
    Convert Chandra's markdown output into plain text so that
    WER/CER are computed against the same kind of text produced
    by the other OCR adapters.
    """
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

    def __init__(
        self,
        prompt_type: str = "ocr",
        max_output_tokens: int = 8192,
        vllm_api_base: str = "http://localhost:8000/v1",
        max_workers: int | None = None,
    ):
        self.prompt_type = prompt_type
        self.max_output_tokens = max_output_tokens
        self.vllm_api_base = vllm_api_base
        self.max_workers = max_workers

        self._manager = None

    def is_available(self) -> bool:
        try:
            import chandra  # noqa: F401
        except ImportError:
            return False

        return True

    def setup(self) -> None:
        try:
            from chandra.model import InferenceManager
        except ImportError as exc:
            raise AdapterUnavailableError(
                "chandra-ocr is not installed. "
                "Run `pip install chandra-ocr`."
            ) from exc

        # IMPORTANT:
        # We use Chandra's vLLM client rather than local HuggingFace
        # inference. The actual model runs in the vLLM server.
        self._manager = InferenceManager(method="vllm")

        print(
            f"[chandra] vLLM endpoint: {self.vllm_api_base}"
        )

        print(
            f"[chandra] max_output_tokens: {self.max_output_tokens}"
        )

        if self.max_workers is not None:
            print(
                f"[chandra] max_workers: {self.max_workers}"
            )

    def recognize(self, image_path: Path) -> str:
        """
        Compatibility path for the normal single-image OCRAdapter API.

        For the benchmark, BenchmarkRunner will call recognize_batch()
        whenever --batch-size > 1.
        """
        results = self.recognize_batch([image_path])

        return results[0]

    def recognize_batch(
        self,
        image_paths: Sequence[Path],
    ) -> list[str]:
        """
        Run multiple images through Chandra's vLLM backend.

        Chandra's vLLM backend sends the items concurrently, allowing
        the vLLM server to dynamically batch requests.
        """
        assert self._manager is not None, "call setup() before recognize_batch()"

        from PIL import Image
        from chandra.model.schema import BatchInputItem

        images = []
        batch = []

        try:
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                images.append(image)

                batch.append(
                    BatchInputItem(
                        image=image,
                        prompt_type=self.prompt_type,
                    )
                )

            generate_kwargs = {
                "max_output_tokens": self.max_output_tokens,
                "vllm_api_base": self.vllm_api_base,
            }

            if self.max_workers is not None:
                generate_kwargs["max_workers"] = self.max_workers

            results = self._manager.generate(
                batch,
                **generate_kwargs,
            )

            if len(results) != len(image_paths):
                raise RuntimeError(
                    f"Chandra returned {len(results)} results "
                    f"for {len(image_paths)} inputs"
                )

            outputs: list[str] = []

            for result in results:
                if result.error:
                    outputs.append("")
                else:
                    outputs.append(
                        _markdown_to_text(
                            result.markdown or ""
                        )
                    )

            return outputs

        finally:
            for image in images:
                image.close()
