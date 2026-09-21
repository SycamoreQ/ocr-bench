from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from PIL import Image

from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem

from ocr_benchmark.dataset import StudentMessyHandwrittenDataset
from ocr_benchmark.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched Chandra-vLLM benchmark on SMHD"
    )

    parser.add_argument(
        "--smhd-root",
        type=Path,
        default=Path("data/smhd"),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smhd/chandra_vllm"),
    )

    parser.add_argument(
        "--vllm-api-base",
        default="http://localhost:8000/v1",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = StudentMessyHandwrittenDataset(
        root=args.smhd_root
    )

    samples = list(dataset)

    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print("=" * 70)
    print("CHANDRA vLLM BATCH BENCHMARK")
    print("=" * 70)
    print(f"Dataset       : {args.smhd_root}")
    print(f"Samples       : {len(samples)}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Max tokens    : {args.max_output_tokens}")
    print(f"vLLM endpoint : {args.vllm_api_base}")
    print()

    manager = InferenceManager(method="vllm")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = args.output_dir / "per_sample.csv"
    summary_csv = args.output_dir / "summary.csv"

    rows = []

    inference_total = 0.0
    successful = 0
    failed = 0

    wer_values = []
    cer_values = []

    for start in range(0, len(samples), args.batch_size):

        batch_samples = samples[start:start + args.batch_size]

        images = []
        batch_inputs = []

        for sample in batch_samples:
            image = Image.open(sample.image_path).convert("RGB")

            images.append(image)

            batch_inputs.append(
                BatchInputItem(
                    image=image,
                    prompt_type="ocr",
                )
            )

        print(
            f"Batch {start // args.batch_size + 1}: "
            f"{start + 1}-{start + len(batch_samples)}"
        )

        try:
            t0 = time.perf_counter()

            results = manager.generate(
                batch_inputs,
                max_output_tokens=args.max_output_tokens,
                vllm_api_base=args.vllm_api_base,
                max_workers=args.max_workers,
            )

            batch_time = time.perf_counter() - t0
            inference_total += batch_time

            if len(results) != len(batch_samples):
                raise RuntimeError(
                    f"Expected {len(batch_samples)} results, "
                    f"got {len(results)}"
                )

            batch_latency = batch_time / len(batch_samples)

            print(
                f"  batch latency      : {batch_time:.3f}s"
            )

            print(
                f"  amortized/page     : {batch_latency:.3f}s"
            )

            print(
                f"  throughput         : "
                f"{len(batch_samples) / batch_time:.3f} pages/s"
            )

            for sample, result in zip(batch_samples, results):

                hypothesis = result.markdown or ""

                metrics = compute_metrics(
                    sample.reference_text,
                    hypothesis,
                )

                wer_values.append(metrics.wer)
                cer_values.append(metrics.cer)

                successful += 1

                rows.append(
                    {
                        "adapter": "chandra-vllm",
                        "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path),
                        "reference": sample.reference_text,
                        "hypothesis": hypothesis,
                        "wer": metrics.wer,
                        "cer": metrics.cer,
                        "mer": metrics.mer,
                        "wil": metrics.wil,
                        "latency_s": batch_latency,
                        "batch_latency_s": batch_time,
                        "error": result.error or "",
                    }
                )

        except Exception as exc:

            failed += len(batch_samples)

            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

            for sample in batch_samples:

                rows.append(
                    {
                        "adapter": "chandra-vllm",
                        "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path),
                        "reference": sample.reference_text,
                        "hypothesis": "",
                        "wer": "",
                        "cer": "",
                        "mer": "",
                        "wil": "",
                        "latency_s": "",
                        "batch_latency_s": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        finally:

            for image in images:
                image.close()

    mean_wer = (
        statistics.mean(wer_values)
        if wer_values
        else float("nan")
    )

    median_wer = (
        statistics.median(wer_values)
        if wer_values
        else float("nan")
    )

    mean_cer = (
        statistics.mean(cer_values)
        if cer_values
        else float("nan")
    )

    median_cer = (
        statistics.median(cer_values)
        if cer_values
        else float("nan")
    )

    throughput = (
        successful / inference_total
        if inference_total > 0
        else 0.0
    )

    mean_latency = (
        inference_total / successful
        if successful > 0
        else float("nan")
    )

    with open(
        per_sample_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "adapter",
                "sample_id",
                "image_path",
                "reference",
                "hypothesis",
                "wer",
                "cer",
                "mer",
                "wil",
                "latency_s",
                "batch_latency_s",
                "error",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    with open(
        summary_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "adapter",
                "status",
                "num_samples_scored",
                "num_samples_failed",
                "mean_wer",
                "median_wer",
                "mean_cer",
                "median_cer",
                "mean_latency_s",
                "throughput_pages_s",
                "total_inference_s",
                "batch_size",
            ],
        )

        writer.writeheader()

        writer.writerow(
            {
                "adapter": "chandra-vllm",
                "status": "ok" if failed == 0 else "partial",
                "num_samples_scored": successful,
                "num_samples_failed": failed,
                "mean_wer": mean_wer,
                "median_wer": median_wer,
                "mean_cer": mean_cer,
                "median_cer": median_cer,
                "mean_latency_s": mean_latency,
                "throughput_pages_s": throughput,
                "total_inference_s": inference_total,
                "batch_size": args.batch_size,
            }
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Samples scored : {successful}")
    print(f"Samples failed : {failed}")
    print(f"Mean WER       : {mean_wer:.4f}")
    print(f"Median WER     : {median_wer:.4f}")
    print(f"Mean CER       : {mean_cer:.4f}")
    print(f"Median CER     : {median_cer:.4f}")
    print(f"Mean latency   : {mean_latency:.4f} s/page")
    print(f"Throughput     : {throughput:.4f} pages/s")
    print(f"Total runtime  : {inference_total:.2f} s")

    print()
    print(f"Per-sample CSV : {per_sample_csv}")
    print(f"Summary CSV    : {summary_csv}")


if __name__ == "__main__":
    main()
