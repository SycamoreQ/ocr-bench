from __future__ import annotations

import csv
import statistics
import time
import traceback

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .adapters.base import AdapterUnavailableError, OCRAdapter
from .dataset import Sample
from .metrics import compute_metrics


PER_SAMPLE_FIELDS = [
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
    "batch_size",
    "error",
]


SUMMARY_FIELDS = [
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
    "num_batches",
    "batch_size",
    "note",
]


@dataclass
class _AdapterRunResult:
    adapter_name: str
    status: str  # ok | unavailable | setup_failed

    note: str = ""

    wers: list[float] = field(default_factory=list)
    cers: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    num_failed: int = 0

    # Total time spent inside OCR inference.
    total_inference_s: float = 0.0

    # Number of calls to recognize()/recognize_batch().
    num_batches: int = 0


class BenchmarkRunner:
    def __init__(
        self,
        adapters: list[OCRAdapter],
        dataset: Iterable[Sample],
        output_dir: Path | str,
        batch_size: int = 1,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.adapters = adapters

        # Materialize once so every adapter runs over the identical
        # sample set.
        self.samples: list[Sample] = list(dataset)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.batch_size = batch_size

    def run(self) -> Path:
        per_sample_path = self.output_dir / "per_sample.csv"
        summary_path = self.output_dir / "summary.csv"

        summaries: list[_AdapterRunResult] = []

        with open(
            per_sample_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=PER_SAMPLE_FIELDS,
            )

            writer.writeheader()

            for adapter in self.adapters:

                result = self._run_one_adapter(
                    adapter,
                    writer,
                )

                summaries.append(result)

        self._write_summary(
            summary_path,
            summaries,
        )

        return summary_path

    def _run_one_adapter(
        self,
        adapter: OCRAdapter,
        writer: csv.DictWriter,
    ) -> _AdapterRunResult:

        print(
            f"[{adapter.name}] checking availability..."
        )

        if not adapter.is_available():

            print(
                f"[{adapter.name}] not available, skipping"
            )

            return _AdapterRunResult(
                adapter.name,
                status="unavailable",
                note="is_available() returned False",
            )

        try:

            print(
                f"[{adapter.name}] setup..."
            )

            adapter.setup()

        except AdapterUnavailableError as exc:

            print(
                f"[{adapter.name}] setup declined: {exc}"
            )

            return _AdapterRunResult(
                adapter.name,
                status="unavailable",
                note=str(exc),
            )

        except Exception as exc:

            print(
                f"[{adapter.name}] setup crashed: {exc}"
            )

            return _AdapterRunResult(
                adapter.name,
                status="setup_failed",
                note=f"{type(exc).__name__}: {exc}",
            )

        result = _AdapterRunResult(
            adapter.name,
            status="ok",
        )

        try:

            # Use batch mode only if:
            #   1. batch_size > 1
            #   2. adapter implements recognize_batch()
            recognize_batch = getattr(
                adapter,
                "recognize_batch",
                None,
            )

            if (
                self.batch_size > 1
                and callable(recognize_batch)
            ):
                self._run_batched_adapter(
                    adapter,
                    recognize_batch,
                    writer,
                    result,
                )

                result.note = (
                    f"batched inference; "
                    f"batch_size={self.batch_size}; "
                    f"latency_s is amortized batch latency"
                )

            else:

                if self.batch_size > 1:
                    result.note = (
                        f"adapter does not implement recognize_batch(); "
                        f"ran serially despite batch_size={self.batch_size}"
                    )

                self._run_serial_adapter(
                    adapter,
                    writer,
                    result,
                )

        finally:

            try:
                adapter.teardown()
            except Exception:
                pass

        return result

    def _run_serial_adapter(
        self,
        adapter: OCRAdapter,
        writer: csv.DictWriter,
        result: _AdapterRunResult,
    ) -> None:

        for i, sample in enumerate(self.samples):

            self._run_one_sample(
                adapter,
                sample,
                writer,
                result,
            )

            if (i + 1) % 50 == 0:

                print(
                    f"[{adapter.name}] "
                    f"{i + 1}/{len(self.samples)} samples"
                )

    def _run_batched_adapter(
        self,
        adapter: OCRAdapter,
        recognize_batch,
        writer: csv.DictWriter,
        result: _AdapterRunResult,
    ) -> None:

        total = len(self.samples)

        for start in range(
            0,
            total,
            self.batch_size,
        ):

            batch_samples = self.samples[
                start : start + self.batch_size
            ]

            image_paths = [
                sample.image_path
                for sample in batch_samples
            ]

            batch_number = (
                start // self.batch_size
            ) + 1

            print(
                f"[{adapter.name}] "
                f"batch {batch_number}: "
                f"{start + 1}-{start + len(batch_samples)}"
                f"/{total}"
            )

            try:

                t0 = time.perf_counter()

                hypotheses = recognize_batch(
                    image_paths
                )

                batch_latency = (
                    time.perf_counter() - t0
                )

                result.total_inference_s += (
                    batch_latency
                )

                result.num_batches += 1

                if len(hypotheses) != len(batch_samples):

                    raise RuntimeError(
                        f"recognize_batch() returned "
                        f"{len(hypotheses)} hypotheses for "
                        f"{len(batch_samples)} samples"
                    )

                # This is the fair throughput metric for batched
                # inference: how much batch time is amortized per page.
                amortized_latency = (
                    batch_latency
                    / len(batch_samples)
                )

                batch_throughput = (
                    len(batch_samples)
                    / batch_latency
                    if batch_latency > 0
                    else 0.0
                )

                print(
                    f"[{adapter.name}] "
                    f"batch_time={batch_latency:.3f}s | "
                    f"amortized={amortized_latency:.3f}s/page | "
                    f"throughput={batch_throughput:.3f} pages/s"
                )

                for sample, hypothesis in zip(
                    batch_samples,
                    hypotheses,
                ):

                    metrics = compute_metrics(
                        sample.reference_text,
                        hypothesis,
                    )

                    row = {
                        "adapter": adapter.name,
                        "sample_id": sample.sample_id,
                        "image_path": str(
                            sample.image_path
                        ),
                        "reference": sample.reference_text,
                        "hypothesis": hypothesis,
                        "wer": metrics.wer,
                        "cer": metrics.cer,
                        "mer": metrics.mer,
                        "wil": metrics.wil,

                        # IMPORTANT:
                        # For batched Chandra this is NOT single-request
                        # latency. It is the batch time amortized over
                        # the number of images in the batch.
                        "latency_s": amortized_latency,

                        "batch_latency_s": batch_latency,
                        "batch_size": len(batch_samples),
                        "error": "",
                    }

                    writer.writerow(row)

                    result.latencies.append(
                        amortized_latency
                    )

                    if metrics.wer == metrics.wer:
                        result.wers.append(
                            metrics.wer
                        )
                        result.cers.append(
                            metrics.cer
                        )

                writer.dialect

            except Exception as exc:

                result.num_failed += len(
                    batch_samples
                )

                print(
                    f"[{adapter.name}] "
                    f"batch {batch_number} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                traceback.print_exc()

                for sample in batch_samples:

                    writer.writerow(
                        {
                            "adapter": adapter.name,
                            "sample_id": sample.sample_id,
                            "image_path": str(
                                sample.image_path
                            ),
                            "reference": sample.reference_text,
                            "hypothesis": "",
                            "wer": "",
                            "cer": "",
                            "mer": "",
                            "wil": "",
                            "latency_s": "",
                            "batch_latency_s": "",
                            "batch_size": len(
                                batch_samples
                            ),
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{exc}"
                            ),
                        }
                    )

    def _run_one_sample(
        self,
        adapter: OCRAdapter,
        sample: Sample,
        writer: csv.DictWriter,
        result: _AdapterRunResult,
    ) -> None:

        row = {
            "adapter": adapter.name,
            "sample_id": sample.sample_id,
            "image_path": str(
                sample.image_path
            ),
            "reference": sample.reference_text,
            "hypothesis": "",
            "wer": "",
            "cer": "",
            "mer": "",
            "wil": "",
            "latency_s": "",
            "batch_latency_s": "",
            "batch_size": 1,
            "error": "",
        }

        try:

            start = time.perf_counter()

            hypothesis = adapter.recognize(
                sample.image_path
            )

            latency = (
                time.perf_counter() - start
            )

            result.total_inference_s += latency
            result.num_batches += 1

            metrics = compute_metrics(
                sample.reference_text,
                hypothesis,
            )

            row["hypothesis"] = hypothesis
            row["wer"] = metrics.wer
            row["cer"] = metrics.cer
            row["mer"] = metrics.mer
            row["wil"] = metrics.wil
            row["latency_s"] = latency

            result.latencies.append(
                latency
            )

            if metrics.wer == metrics.wer:
                result.wers.append(
                    metrics.wer
                )
                result.cers.append(
                    metrics.cer
                )

        except Exception as exc:

            row["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            result.num_failed += 1

            traceback.print_exc()

        finally:

            writer.writerow(row)

    def _write_summary(
        self,
        path: Path,
        summaries: list[_AdapterRunResult],
    ) -> None:

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=SUMMARY_FIELDS,
            )

            writer.writeheader()

            for s in summaries:

                num_scored = len(s.wers)

                throughput = (
                    num_scored / s.total_inference_s
                    if s.total_inference_s > 0
                    else ""
                )

                writer.writerow(
                    {
                        "adapter": s.adapter_name,
                        "status": s.status,
                        "num_samples_scored": num_scored,
                        "num_samples_failed": s.num_failed,

                        "mean_wer": (
                            statistics.mean(s.wers)
                            if s.wers
                            else ""
                        ),

                        "median_wer": (
                            statistics.median(s.wers)
                            if s.wers
                            else ""
                        ),

                        "mean_cer": (
                            statistics.mean(s.cers)
                            if s.cers
                            else ""
                        ),

                        "median_cer": (
                            statistics.median(s.cers)
                            if s.cers
                            else ""
                        ),

                        "mean_latency_s": (
                            statistics.mean(
                                s.latencies
                            )
                            if s.latencies
                            else ""
                        ),

                        "throughput_pages_s": throughput,

                        "total_inference_s": (
                            s.total_inference_s
                        ),

                        "num_batches": (
                            s.num_batches
                        ),

                        "batch_size": self.batch_size,

                        "note": s.note,
                    }
                )
