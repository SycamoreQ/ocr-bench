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
    status: str
    note: str = ""

    wers: list[float] = field(default_factory=list)
    cers: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    num_failed: int = 0

    # Total time spent doing OCR inference.
    total_inference_s: float = 0.0

    # Number of inference calls/batches.
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

        # Materialize once so every adapter gets the exact same samples.
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
                adapter_name=adapter.name,
                status="unavailable",
                note="is_available() returned False",
            )

        # ----------------------------------------------------------
        # Adapter setup
        # ----------------------------------------------------------

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
                adapter_name=adapter.name,
                status="unavailable",
                note=str(exc),
            )

        except Exception as exc:

            print(
                f"[{adapter.name}] setup failed: {exc}"
            )

            traceback.print_exc()

            return _AdapterRunResult(
                adapter_name=adapter.name,
                status="setup_failed",
                note=f"{type(exc).__name__}: {exc}",
            )

        result = _AdapterRunResult(
            adapter_name=adapter.name,
            status="ok",
        )

        try:

            # ------------------------------------------------------
            # Batch-capable adapter
            # ------------------------------------------------------

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
                    adapter=adapter,
                    recognize_batch=recognize_batch,
                    writer=writer,
                    result=result,
                )

                result.note = (
                    "batched inference; "
                    f"batch_size={self.batch_size}; "
                    "latency_s is amortized batch latency"
                )

            # ------------------------------------------------------
            # Normal serial adapter
            # ------------------------------------------------------

            else:

                if self.batch_size > 1:

                    result.note = (
                        "adapter does not implement "
                        "recognize_batch(); "
                        f"ran serially despite "
                        f"batch_size={self.batch_size}"
                    )

                self._run_serial_adapter(
                    adapter=adapter,
                    writer=writer,
                    result=result,
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

        total = len(self.samples)

        for index, sample in enumerate(
            self.samples,
            start=1,
        ):

            self._run_one_sample(
                adapter=adapter,
                sample=sample,
                writer=writer,
                result=result,
            )

            if index % 50 == 0:
                print(
                    f"[{adapter.name}] "
                    f"{index}/{total} samples"
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
                f"{start + 1}-"
                f"{start + len(batch_samples)}/"
                f"{total}"
            )

            try:

                # --------------------------------------------------
                # Time the COMPLETE batch inference.
                # This is what we use for throughput.
                # --------------------------------------------------

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

                # --------------------------------------------------
                # Validate result count.
                # --------------------------------------------------

                if len(hypotheses) != len(batch_samples):

                    raise RuntimeError(
                        "recognize_batch() returned "
                        f"{len(hypotheses)} results for "
                        f"{len(batch_samples)} samples"
                    )

                # Amortized per-page latency.
                amortized_latency = (
                    batch_latency
                    / len(batch_samples)
                )

                throughput = (
                    len(batch_samples)
                    / batch_latency
                    if batch_latency > 0
                    else 0.0
                )

                print(
                    f"[{adapter.name}] "
                    f"batch_time={batch_latency:.3f}s | "
                    f"amortized="
                    f"{amortized_latency:.3f}s/page | "
                    f"throughput="
                    f"{throughput:.3f} pages/s"
                )

                # --------------------------------------------------
                # Score every item in the batch.
                # --------------------------------------------------

                for sample, hypothesis in zip(
                    batch_samples,
                    hypotheses,
                ):

                    metrics = compute_metrics(
                        sample.reference_text,
                        hypothesis,
                    )

                    writer.writerow(
                        {
                            "adapter": adapter.name,
                            "sample_id": sample.sample_id,
                            "image_path": str(
                                sample.image_path
                            ),
                            "reference": (
                                sample.reference_text
                            ),
                            "hypothesis": hypothesis,
                            "wer": metrics.wer,
                            "cer": metrics.cer,
                            "mer": metrics.mer,
                            "wil": metrics.wil,
                            "latency_s": (
                                amortized_latency
                            ),
                            "batch_latency_s": (
                                batch_latency
                            ),
                            "batch_size": (
                                len(batch_samples)
                            ),
                            "error": "",
                        }
                    )

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

                # Preserve failed samples in CSV.
                for sample in batch_samples:

                    writer.writerow(
                        {
                            "adapter": adapter.name,
                            "sample_id": sample.sample_id,
                            "image_path": str(
                                sample.image_path
                            ),
                            "reference": (
                                sample.reference_text
                            ),
                            "hypothesis": "",
                            "wer": "",
                            "cer": "",
                            "mer": "",
                            "wil": "",
                            "latency_s": "",
                            "batch_latency_s": "",
                            "batch_size": (
                                len(batch_samples)
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

            t0 = time.perf_counter()

            hypothesis = adapter.recognize(
                sample.image_path
            )

            latency = (
                time.perf_counter() - t0
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

            for summary in summaries:

                num_scored = len(
                    summary.wers
                )

                throughput = (
                    num_scored
                    / summary.total_inference_s
                    if summary.total_inference_s > 0
                    else ""
                )

                writer.writerow(
                    {
                        "adapter": (
                            summary.adapter_name
                        ),
                        "status": summary.status,

                        "num_samples_scored": (
                            num_scored
                        ),

                        "num_samples_failed": (
                            summary.num_failed
                        ),

                        "mean_wer": (
                            statistics.mean(
                                summary.wers
                            )
                            if summary.wers
                            else ""
                        ),

                        "median_wer": (
                            statistics.median(
                                summary.wers
                            )
                            if summary.wers
                            else ""
                        ),

                        "mean_cer": (
                            statistics.mean(
                                summary.cers
                            )
                            if summary.cers
                            else ""
                        ),

                        "median_cer": (
                            statistics.median(
                                summary.cers
                            )
                            if summary.cers
                            else ""
                        ),

                        "mean_latency_s": (
                            statistics.mean(
                                summary.latencies
                            )
                            if summary.latencies
                            else ""
                        ),

                        "throughput_pages_s": (
                            throughput
                        ),

                        "total_inference_s": (
                            summary.total_inference_s
                        ),

                        "num_batches": (
                            summary.num_batches
                        ),

                        "batch_size": (
                            self.batch_size
                        ),

                        "note": summary.note,
                    }
                )
