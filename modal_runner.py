"""
Run ocr_benchmark on a Modal GPU instead of locally.

Usage (from the repo root, alongside requirements.txt):

    pip install modal
    modal setup                     # one-time auth, opens a browser
    modal run modal_runner.py --max-samples 100 --adapters unlimited-ocr

Everything after --adapters is passed straight through as a comma-separated
list (or "all"), same as bench_line.py's own --adapters flag.

The HF model cache lives on a persistent Modal Volume, so the ~6.67GB
Unlimited-OCR shard is only downloaded once across all future runs, not
once per container cold start.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("or-bench-gpu")

# Persists across runs so the model weights are only fetched from HF once.
hf_cache_volume = modal.Volume.from_name("or-bench-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # libgl1/libglib2.0-0: cheap insurance against an indirect cv2 import
    # inside custom trust_remote_code modeling files, which is common for
    # DeepSeek-OCR-derived vision-language models. Drop if unused.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "transformers>=4.57,<5.0",  # see tesseract/unlimited-ocr adapter notes: 5.0 dropped
                                     # is_torch_fx_available, which older trust_remote_code
                                     # models (this one included) still import
        "accelerate",
        "safetensors",
        "einops",
        "addict",
        "matplotlib",
        "datasets",
        "Pillow",
        "numpy",
        "jiwer",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Ships your local ocr_benchmark/ package (adapters, dataset.py,
    # runner.py, bench_line.py, ...) into the image so the function below
    # can import it directly. Run this script from the repo root so the
    # relative package path resolves.
    .add_local_python_source("ocr_benchmark")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 2,  # 2h ceiling; a 100-sample VLM run shouldn't need this
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def run_benchmark(
    adapters_spec: str,
    split: str,
    max_samples: int,
    seed: int,
) -> dict[str, str]:
    from ocr_benchmark.adapters import ADAPTER_REGISTRY
    from ocr_benchmark.bench_line import build_samples, get_adapter_names
    from ocr_benchmark.runner import BenchmarkRunner

    adapter_names = get_adapter_names(adapters_spec)
    adapters = [ADAPTER_REGISTRY[name]() for name in adapter_names]

    # Ephemeral container-local paths -- only the HF model cache above
    # needs to survive between runs. The line-crop PNGs are cheap and
    # deterministic to regenerate from HF given the same split/seed.
    output_dir = Path("/root/output")
    image_cache_dir = output_dir / ".iam_line_images"

    samples = build_samples(
        split=split,
        max_samples=max_samples,
        seed=seed,
        image_cache_dir=image_cache_dir,
        no_cache=False,
    )

    runner = BenchmarkRunner(adapters=adapters, dataset=samples, output_dir=output_dir)
    runner.run()

    return {
        "summary_csv": (output_dir / "summary.csv").read_text(),
        "per_sample_csv": (output_dir / "per_sample.csv").read_text(),
    }


@app.local_entrypoint()
def main(
    adapters: str = "unlimited-ocr",
    split: str = "test",
    max_samples: int = 100,
    seed: int = 42,
    output_dir: str = "results/modal_run",
):
    result = run_benchmark.remote(
        adapters_spec=adapters,
        split=split,
        max_samples=max_samples,
        seed=seed,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.csv").write_text(result["summary_csv"])
    (out_dir / "per_sample.csv").write_text(result["per_sample_csv"])

    print(f"Wrote {out_dir / 'summary.csv'}")
    print(f"Wrote {out_dir / 'per_sample.csv'}")
    print()
    print(result["summary_csv"])
