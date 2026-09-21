from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import ADAPTER_REGISTRY
from .adapters.tesseract_adapter import TesseractAdapter
from .dataset import IAMAsciiDataset, StudentMessyHandwrittenDataset
from .runner import BenchmarkRunner


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--iam-root",
        type=Path,
        help=(
            "Root of the extracted IAM Handwriting Database "
            "(must contain ascii/lines.txt or ascii/words.txt)."
        ),
    )
    dataset_group.add_argument(
        "--smhd-root",
        type=Path,
        help=(
            "Root of the downloaded Student Messy Handwritten Dataset "
            "(must contain metadata.csv, scans/, and transcriptions/)."
        ),
    )

    parser.add_argument(
        "--split",
        choices=["lines", "words"],
        default="lines",
        help="IAM ground-truth granularity. Ignored for SMHD.",
    )
    parser.add_argument(
        "--adapters",
        default="all",
        help=(
            "Comma-separated adapter names, or 'all'. Registered: "
            f"{', '.join(sorted(ADAPTER_REGISTRY))}"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of dataset samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--include-err-segmented",
        action="store_true",
        help=(
            "IAM only: include rows flagged 'err' instead of skipping them. "
            "Ignored for SMHD."
        ),
    )
    parser.add_argument(
        "--words-txt",
        "--lines-txt",
        dest="gt_file",
        type=Path,
        default=None,
        help=(
            "IAM only: explicit path to words.txt/lines.txt if auto-discovery "
            "finds the wrong file."
        ),
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help=(
            "IAM only: explicit directory containing images. "
            "Ignored for SMHD, which uses metadata.csv."
        ),
    )
    parser.add_argument(
        "--smhd-metadata-csv",
        type=Path,
        default=None,
        help="SMHD only: optional explicit path to metadata.csv.",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=None,
        help=(
            "Tesseract page segmentation mode. Default: 7 for IAM, "
            "6 for SMHD. Use 6 for dense handwritten pages; 3 is also "
            "worth testing."
        ),
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

    using_smhd = args.smhd_root is not None

    adapters = []
    for name in adapter_names:
        if name == "tesseract":
            # IAM's original benchmark uses line/word crops, for which PSM 7
            # is appropriate. SMHD contains full handwritten pages, so PSM 6
            # is a better default and avoids the empty-output failure caused
            # by treating a multi-line page as one text line.
            default_psm = 6 if using_smhd else 7
            psm = args.tesseract_psm if args.tesseract_psm is not None else default_psm
            adapters.append(TesseractAdapter(lang="eng", psm=psm))
        else:
            adapters.append(ADAPTER_REGISTRY[name]())

    if args.smhd_root is not None:
        if args.smhd_metadata_csv is not None:
            metadata_csv = args.smhd_metadata_csv
        else:
            metadata_csv = None

        dataset = StudentMessyHandwrittenDataset(
            root=args.smhd_root,
            max_samples=args.max_samples,
            metadata_csv=metadata_csv,
        )
    else:
        if args.smhd_metadata_csv is not None:
            raise SystemExit("--smhd-metadata-csv can only be used with --smhd-root")

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
