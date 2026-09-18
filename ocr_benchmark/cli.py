"""CLI entry point.

Examples
--------
Run every registered adapter over 200 IAM lines:

    python -m ocr_benchmark.cli \\
        --iam-root /path/to/IAM \\
        --split lines \\
        --max-samples 200 \\
        --output-dir results/

Run just the two reference adapters (the ones guaranteed to work out of
the box, before you've filled in the stubs):

    python -m ocr_benchmark.cli \\
        --iam-root /path/to/IAM \\
        --adapters tesseract,rapidocr \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import ADAPTER_REGISTRY
from .dataset import IAMAsciiDataset
from .runner import BenchmarkRunner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iam-root",
        required=True,
        type=Path,
        help="Root of the extracted IAM Handwriting Database "
        "(must contain ascii/lines.txt or ascii/words.txt).",
    )
    parser.add_argument(
        "--split",
        choices=["lines", "words"],
        default="lines",
        help="Which IAM ground-truth granularity to benchmark against.",
    )
    parser.add_argument(
        "--adapters",
        default="all",
        help=f"Comma-separated adapter names, or 'all'. Registered: "
        f"{', '.join(sorted(ADAPTER_REGISTRY))}",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of dataset samples (useful while adapters "
        "are still being filled in / for a quick smoke test).",
    )
    parser.add_argument(
        "--include-err-segmented",
        action="store_true",
        help="By default, IAM rows flagged 'err' (known bad segmentation) "
        "are skipped. Pass this to include them anyway.",
    )
    parser.add_argument(
        "--words-txt",
        "--lines-txt",
        dest="gt_file",
        type=Path,
        default=None,
        help="Explicit path to words.txt/lines.txt, if auto-discovery "
        "under --iam-root picks the wrong file or your mirror uses a "
        "non-standard layout.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help="Explicit directory to index images under (default: "
        "--iam-root itself). Only needed if images live outside "
        "--iam-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Where to write per_sample.csv and summary.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.adapters == "all":
        adapter_names = list(ADAPTER_REGISTRY)
    else:
        adapter_names = [a.strip() for a in args.adapters.split(",") if a.strip()]
        unknown = set(adapter_names) - set(ADAPTER_REGISTRY)
        if unknown:
            raise SystemExit(
                f"Unknown adapter(s): {', '.join(sorted(unknown))}. "
                f"Registered: {', '.join(sorted(ADAPTER_REGISTRY))}"
            )

    adapters = [ADAPTER_REGISTRY[name]() for name in adapter_names]

    dataset = IAMAsciiDataset(
        root=args.iam_root,
        split=args.split,
        only_ok=not args.include_err_segmented,
        max_samples=args.max_samples,
        gt_file=args.gt_file,
        images_root=args.images_root,
    )

    runner = BenchmarkRunner(adapters, dataset, args.output_dir)
    summary_path = runner.run()
    print(f"\nDone. Summary: {summary_path}")
    print(f"Per-sample detail: {args.output_dir / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
