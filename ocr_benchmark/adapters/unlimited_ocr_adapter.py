"""Baidu's Unlimited-OCR (baidu/Unlimited-OCR on HuggingFace) — a
DeepSeek-OCR-derived vision-language document parser, "single-pass
long-horizon parsing". Loaded via `transformers` with `trust_remote_code`,
matching the model's own reference usage.

Runs on CPU (e.g. a Mac, for smoke-testing) or a real CUDA GPU (e.g. a
Modal T4) without changes -- see _select_dtype and _neutralize_cuda_calls
below for how each is handled.

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
from contextlib import contextmanager
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


def _select_dtype(explicit: "torch.dtype | None"):
    """bf16 needs Ampere+ (compute capability >= 8.0) for tensor-core
    support; on older CUDA GPUs like a T4 (7.5), fp16 is the faster
    choice for identical precision-class numerics. On CPU, fall back to
    fp32 -- bf16 is not consistently accelerated there and can end up
    slower than fp32 depending on the torch build.
    """
    import torch

    if explicit is not None:
        return explicit
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


@contextmanager
def _neutralize_cuda_calls():
    """baidu/Unlimited-OCR's own `.infer()` hardcodes `.cuda()` on its
    input tensors (e.g. `input_ids.unsqueeze(0).cuda()`) regardless of
    what device the model was actually loaded onto. That's a bug in the
    model's published remote code, not a device-selection problem on our
    side -- there's no equivalent of `.to("mps")` that fixes it, because
    `infer()` never consults the model's actual device before calling
    `.cuda()`.

    On a machine with a real CUDA device (e.g. a Modal T4), that call is
    correct as written and must be left alone. On any machine without one
    (CPU, Apple Silicon/MPS included), it always raises
    `AssertionError: Torch not compiled with CUDA enabled` -- so there we
    make `Tensor.cuda()` a no-op for the duration of the call instead,
    leaving a CPU tensor exactly where it is. Scoped and restored
    immediately after, so it can't affect anything else in the process.
    """
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
        attn_implementation: str | None = "sdpa",
    ):
        # crop_mode=True ("gundam" config): base_size=1024, image_size=640
        # — the model card's recommended single-image setting. Set False
        # for the "base" config (image_size=1024) on very dense pages.
        self.crop_mode = crop_mode
        # None = auto-select via _select_dtype (bf16 on Ampere+, fp16 on
        # older CUDA GPUs like a T4, fp32 on CPU). Pass an explicit
        # torch.dtype to override.
        self.dtype = dtype
        # "sdpa" avoids the model trying to auto-select flash-attention 2,
        # which isn't available on pre-Ampere GPUs (a T4 included) and
        # isn't relevant on CPU anyway. If this model's remote code
        # doesn't accept the kwarg at all, setup() falls back to loading
        # without it rather than failing -- pass None yourself to skip
        # the attempt entirely.
        self.attn_implementation = attn_implementation
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
            except TypeError:
                # This model's remote code doesn't accept
                # attn_implementation as a load kwarg -- fall back to its
                # own default rather than hard-failing setup over an
                # optimization flag.
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
