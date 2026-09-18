"""Baidu's Unlimited-OCR (baidu/Unlimited-OCR on HuggingFace) — a
DeepSeek-OCR-derived vision-language document parser, "single-pass
long-horizon parsing". Loaded via `transformers` with `trust_remote_code`,
matching the model's own reference usage.

The model's own `.infer(...)` call is primarily used for its side effect
(writing results to `output_path` when `save_results=True`), but the
published evaluation snippet for this model treats its return value as
the raw generated string and post-processes it with a small regex to
strip `<|det|>...<|/det|>` region-type/bbox markers before scoring — so
we do the same: use the return value if we get a string back, and only
fall back to reading a file out of the output directory if the installed
model build doesn't return one.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from .base import AdapterUnavailableError, OCRAdapter

_MODEL_NAME = "baidu/Unlimited-OCR"

# From the model card's own OmniDocBench post-processing snippet: strips
# <|det|>type [bbox]<|/det|> region markers, keeps block text, drops
# "image" blocks entirely, separates blocks with a blank line.
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


class UnlimitedOCRAdapter(OCRAdapter):
    name = "unlimited-ocr"

    def __init__(self, crop_mode: bool = True):
        # crop_mode=True ("gundam" config): base_size=1024, image_size=640
        # — the model card's recommended single-image setting. Set False
        # for the "base" config (image_size=1024) on very dense pages.
        self.crop_mode = crop_mode
        self._tokenizer = None
        self._model = None

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
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
        model = AutoModel.from_pretrained(
            _MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self._model = model

    def recognize(self, image_path: Path) -> str:
        assert self._model is not None, "call setup() before recognize()"

        image_size = 640 if self.crop_mode else 1024
        with tempfile.TemporaryDirectory() as out_dir:
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
        # save_results=True writes generated output into output_path;
        # exact filenames aren't part of the model's documented contract,
        # so read whatever text/markdown file landed there most recently.
        candidates = sorted(
            (*out_dir.rglob("*.md"), *out_dir.rglob("*.mmd"), *out_dir.rglob("*.txt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return ""
        return candidates[0].read_text(encoding="utf-8")
