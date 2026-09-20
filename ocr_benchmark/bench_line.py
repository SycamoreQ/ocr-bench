from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

from .adapters import ADAPTER_REGISTRY
from .dataset import Sample
from .runner import BenchmarkRunner


DATASET_NAME = "Teklia/IAM-line"
DIAGNOSE_FIRST_N = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark existing OCR adapters on Teklia/IAM-line."
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Hugging Face dataset split.",
    )

    parser.add_argument(
        "--adapters",
        default="all",
        help=(
            "Comma-separated adapter names, or 'all'. "
            f"Registered: {', '.join(sorted(ADAPTER_REGISTRY))}"
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples. Default: 100.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when selecting a subset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/iam_line"),
        help="Directory for benchmark output.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "Always re-save images from HF even if a PNG already exists "
            "at the target path. The cache key is (split, seed, index) "
            "only, not image content -- if you have any reason to think "
            "the cached PNGs under .iam_line_images/ are stale or bad, "
            "pass this instead of trusting the cache, or just delete "
            "that directory before rerunning."
        ),
    )

    return parser


def get_adapter_names(spec: str) -> list[str]:
    if spec == "all":
        return list(ADAPTER_REGISTRY)

    names = [x.strip() for x in spec.split(",") if x.strip()]

    unknown = set(names) - set(ADAPTER_REGISTRY)
    if unknown:
        raise SystemExit(
            f"Unknown adapter(s): {', '.join(sorted(unknown))}\n"
            f"Registered: {', '.join(sorted(ADAPTER_REGISTRY))}"
        )

    return names


def _image_stats(img: Image.Image) -> str:
    arr = np.array(img.convert("L"))
    return (
        f"mode={img.mode} size={img.size} "
        f"min={arr.min()} max={arr.max()} mean={arr.mean():.1f}"
    )


def build_samples(
    split: str,
    max_samples: int | None,
    seed: int,
    image_cache_dir: Path,
    no_cache: bool,
) -> list[Sample]:
    print(f"Loading Hugging Face dataset: {DATASET_NAME}")
    print(f"Split: {split}")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
    )

    total = len(dataset)

    print(f"Available samples: {total}")
    if max_samples is not None and max_samples < total:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))

    image_cache_dir.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    blank_count = 0

    for i, row in enumerate(dataset):
        image = row["image"]
        reference = str(row["text"]).strip()

        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL image for sample {i}, got {type(image)}"
            )

        if i < DIAGNOSE_FIRST_N:
            print(f"[sample {i}] raw from HF: {_image_stats(image)} | text={reference!r}")

        sample_id = f"iam-line-{split}-{i:06d}"
        image_path = image_cache_dir / f"{sample_id}.png"

        if no_cache or not image_path.exists():
            image.convert("RGB").save(image_path)

        # Verify what actually landed on disk rather than trusting the
        # in-memory object -- this catches save/codec issues (and stale
        # cache hits from a previous bad run) that the HF-side object
        # wouldn't show at all.
        with Image.open(image_path) as saved:
            saved.load()
            arr = np.array(saved.convert("L"))
            is_blank = arr.max() == arr.min()

        if i < DIAGNOSE_FIRST_N:
            with Image.open(image_path) as saved:
                print(f"[sample {i}] on disk:   {_image_stats(saved)}")

        if is_blank:
            blank_count += 1
            if blank_count <= DIAGNOSE_FIRST_N:
                print(
                    f"[sample {i}] WARNING: saved PNG is a flat/uniform "
                    f"image -- {image_path}"
                )

        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=image_path,
                reference_text=reference,
            )
        )

        if (i + 1) % 50 == 0:
            print(f"Prepared {i + 1}/{len(dataset)} images")

    if blank_count:
        print(
            f"\n*** {blank_count}/{len(samples)} saved images are flat/blank. "
            "That's a data problem upstream of any OCR engine -- every "
            "adapter will return nothing on these regardless of settings. "
            "Re-run with --no-cache after confirming this, and check the "
            "[sample N] diagnostics above to see whether the image is "
            "already blank straight out of load_dataset() or only goes "
            "blank after the .convert('RGB').save() step. ***\n"
        )

    print(f"Prepared {len(samples)} samples")
    return samples


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    adapter_names = get_adapter_names(args.adapters)

    adapters = [
        ADAPTER_REGISTRY[name]()
        for name in adapter_names
    ]

    image_cache_dir = args.output_dir / ".iam_line_images"

    samples = build_samples(
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
        image_cache_dir=image_cache_dir,
        no_cache=args.no_cache,
    )

    print()
    print("IAM-LINE OCR BENCHMARK")
    print(f"Dataset : {DATASET_NAME}")
    print(f"Split : {args.split}")
    print(f"Samples : {len(samples)}")
    print(
        "Adapters: "
        + ", ".join(adapter.name for adapter in adapters)
    )
    print(f"Output  : {args.output_dir}")
    print()

    runner = BenchmarkRunner(
        adapters=adapters,
        dataset=samples,
        output_dir=args.output_dir,
    )

    summary_path = runner.run()

    print()
    print("Done.")
    print(f"Summary: {summary_path}")
    print(
        f"Per-sample: "
        f"{args.output_dir / 'per_sample.csv'}"
    )


if __name__ == "__main__":
    main()
