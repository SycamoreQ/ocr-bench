from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter

_MODEL_NAME = "baidu/Unlimited-OCR"

_DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)


def _strip_det_markers(raw: str) -> str:
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = _DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def _select_dtype(explicit: "torch.dtype | None"):
    import torch

    if explicit is not None:
        return explicit
    return torch.bfloat16


@contextmanager
def _neutralize_cuda_calls():
    import torch

    if torch.cuda.is_available():
        yield
        return

    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **kw: self
    try:
        yield
    finally:
        torch.Tensor.cuda = original_cuda


class UnlimitedOCRAdapter(OCRAdapter):
    name = "unlimited-ocr"

    def __init__(
        self,
        crop_mode: bool = True,
        dtype: "torch.dtype | None" = None,
        attn_implementation: str | None = "eager",
    ):
        self.crop_mode = crop_mode
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self._tokenizer = None
        self._model = None

    def is_available(self) -> bool:
        try:
            import torch
            import transformers
        except ImportError:
            return False
        return True

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise AdapterUnavailableError(
                "torch/transformers are not installed. "
                "`pip install torch transformers`."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_NAME, trust_remote_code=True
        )

        dtype = _select_dtype(self.dtype)
        load_kwargs = dict(
            trust_remote_code=True,
            use_safetensors=True,
            dtype=dtype,  # `torch_dtype=` is deprecated as of transformers 4.5x+
        )

        model = None
        if self.attn_implementation is not None:
            try:
                model = AutoModel.from_pretrained(
                    _MODEL_NAME,
                    attn_implementation=self.attn_implementation,
                    **load_kwargs,
                )
            except (TypeError, ValueError):
                model = None

        if model is None:
            model = AutoModel.from_pretrained(_MODEL_NAME, **load_kwargs)

        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model

    def recognize(self, image_path: Path) -> str:
        assert self._model is not None, "call setup() before recognize()"

        image_size = 640 if self.crop_mode else 1024
        with tempfile.TemporaryDirectory() as out_dir:
            with _neutralize_cuda_calls():
                result = self._model.infer(
                    self._tokenizer,
                    prompt="<image>document parsing.",
                    image_file=str(image_path),
                    output_path=out_dir,
                    base_size=1024,
                    image_size=image_size,
                    crop_mode=self.crop_mode,
                    max_length=32768,
                    no_repeat_ngram_size=35,
                    ngram_window=128,
                    save_results=True,
                )

            raw_text = self._as_text(result)
            if raw_text is None:
                raw_text = self._read_from_output_dir(Path(out_dir))

        if not raw_text:
            return ""
        return _strip_det_markers(raw_text)

    @staticmethod
    def _as_text(result) -> str | None:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("text", "result", "output", "content"):
                if isinstance(result.get(key), str):
                    return result[key]
        return None

    @staticmethod
    def _read_from_output_dir(out_dir: Path) -> str:
        candidates = sorted(
            (*out_dir.rglob("*.md"), *out_dir.rglob("*.mmd"), *out_dir.rglob("*.txt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return ""
        return candidates[0].read_text(encoding="utf-8")
