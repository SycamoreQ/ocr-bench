from __future__ import annotations

import argparse
import os
from pathlib import Path

from .adapters import ADAPTER_REGISTRY
from .adapters.chandra_adapter import ChandraOCRAdapter
from .adapters.tesseract_adapter import TesseractAdapter

from .dataset import (
    IAMAsciiDataset,
    StudentMessyHandwrittenDataset,
)

from .runner import BenchmarkRunner


def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    dataset_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

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
            "Root of the Student Messy Handwritten Dataset "
            "(must contain metadata.csv, scans/, and transcriptions/)."
        ),
    )

    parser.add_argument(
        "--split",
        choices=["lines", "words"],
        default="lines",
        help=(
            "IAM ground-truth granularity. "
            "Ignored for SMHD."
        ),
    )

    parser.add_argument(
        "--adapters",
        default="all",
        help=(
            "Comma-separated adapter names, or 'all'. "
            f"Registered: "
            f"{', '.join(sorted(ADAPTER_REGISTRY))}"
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples to process."
        ),
    )

    parser.add_argument(
        "--include-err-segmented",
        action="store_true",
        help=(
            "IAM only: include rows flagged 'err'."
        ),
    )

    parser.add_argument(
        "--words-txt",
        "--lines-txt",
        dest="gt_file",
        type=Path,
        default=None,
        help=(
            "IAM only: explicit path to words.txt/lines.txt."
        ),
    )

    parser.add_argument(
        "--images-root",
        type=Path,
        default=None,
        help=(
            "IAM only: explicit image directory."
        ),
    )

    parser.add_argument(
        "--smhd-metadata-csv",
        type=Path,
        default=None,
        help=(
            "SMHD only: explicit metadata.csv path."
        ),
    )

    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=None,
        help=(
            "Tesseract page segmentation mode. "
            "Default: 7 for IAM, 6 for SMHD."
        ),
    )

    # ---------------------------------------------------------
    # Chandra / vLLM options
    # ---------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for adapters implementing recognize_batch(). "
            "Currently Chandra uses this for vLLM inference. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--chandra-vllm-api-base",
        default=os.environ.get(
            "VLLM_API_BASE",
            "http://localhost:8000/v1",
        ),
        help=(
            "Chandra vLLM OpenAI-compatible endpoint. "
            "Default: $VLLM_API_BASE or "
            "http://localhost:8000/v1."
        ),
    )

    parser.add_argument(
        "--chandra-max-output-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum generated tokens per Chandra page. "
            "Default: 8192."
        ),
    )

    parser.add_argument(
        "--chandra-max-workers",
        type=int,
        default=None,
        help=(
            "Maximum concurrent Chandra client requests. "
            "Default: Chandra decides."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help=(
            "Where to write per_sample.csv and summary.csv."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:

    args = build_arg_parser().parse_args(argv)

    if args.batch_size < 1:
        raise SystemExit(
            "--batch-size must be >= 1"
        )

    if args.chandra_max_output_tokens < 1:
        raise SystemExit(
            "--chandra-max-output-tokens must be >= 1"
        )

    if (
        args.chandra_max_workers is not None
        and args.chandra_max_workers < 1
    ):
        raise SystemExit(
            "--chandra-max-workers must be >= 1"
        )

    # ---------------------------------------------------------
    # Adapter selection
    # ---------------------------------------------------------

    if args.adapters == "all":

        adapter_names = list(
            ADAPTER_REGISTRY
        )

    else:

        adapter_names = [
            a.strip()
            for a in args.adapters.split(",")
            if a.strip()
        ]

        unknown = (
            set(adapter_names)
            - set(ADAPTER_REGISTRY)
        )

        if unknown:
            raise SystemExit(
                f"Unknown adapter(s): "
                f"{', '.join(sorted(unknown))}. "
                f"Registered: "
                f"{', '.join(sorted(ADAPTER_REGISTRY))}"
            )

    using_smhd = (
        args.smhd_root is not None
    )

    adapters = []

    for name in adapter_names:

        if name == "tesseract":

            default_psm = (
                6 if using_smhd else 7
            )

            psm = (
                args.tesseract_psm
                if args.tesseract_psm is not None
                else default_psm
            )

            adapters.append(
                TesseractAdapter(
                    lang="eng",
                    psm=psm,
                )
            )

        elif name == "chandra":

            adapters.append(
                ChandraOCRAdapter(
                    prompt_type="ocr",
                    max_output_tokens=(
                        args.chandra_max_output_tokens
                    ),
                    vllm_api_base=(
                        args.chandra_vllm_api_base
                    ),
                    max_workers=(
                        args.chandra_max_workers
                    ),
                )
            )

        else:

            adapters.append(
                ADAPTER_REGISTRY[name]()
            )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    if args.smhd_root is not None:

        metadata_csv = (
            args.smhd_metadata_csv
            if args.smhd_metadata_csv is not None
            else None
        )

        dataset = StudentMessyHandwrittenDataset(
            root=args.smhd_root,
            max_samples=args.max_samples,
            metadata_csv=metadata_csv,
        )

    else:

        if args.smhd_metadata_csv is not None:
            raise SystemExit(
                "--smhd-metadata-csv can only be used "
                "with --smhd-root"
            )

        dataset = IAMAsciiDataset(
            root=args.iam_root,
            split=args.split,
            only_ok=(
                not args.include_err_segmented
            ),
            max_samples=args.max_samples,
            gt_file=args.gt_file,
            images_root=args.images_root,
        )

    # ---------------------------------------------------------
    # Runner
    # ---------------------------------------------------------

    runner = BenchmarkRunner(
        adapters=adapters,
        dataset=dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    summary_path = runner.run()

    print(
        f"\nDone. Summary: {summary_path}"
    )

    print(
        "Per-sample detail: "
        f"{args.output_dir / 'per_sample.csv'}"
    )


if __name__ == "__main__":
    main()
