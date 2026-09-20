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
    "note",
]


@dataclass
class _AdapterRunResult:
    adapter_name: str
    status: str  # "ok" | "unavailable" | "setup_failed"
    note: str = ""
    wers: list[float] = field(default_factory=list)
    cers: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    num_failed: int = 0


class BenchmarkRunner:
    def __init__(
        self,
        adapters: list[OCRAdapter],
        dataset: Iterable[Sample],
        output_dir: Path | str,
    ):
        self.adapters = adapters
        # Materialize once so every adapter runs over the identical sample
        # set, and so a lazy dataset iterator isn't silently exhausted
        # after the first adapter.
        self.samples: list[Sample] = list(dataset)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        per_sample_path = self.output_dir / "per_sample.csv"
        summary_path = self.output_dir / "summary.csv"

        summaries: list[_AdapterRunResult] = []

        with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PER_SAMPLE_FIELDS)
            writer.writeheader()

            for adapter in self.adapters:
                result = self._run_one_adapter(adapter, writer)
                summaries.append(result)

        self._write_summary(summary_path, summaries)
        return summary_path

    def _run_one_adapter(
        self, adapter: OCRAdapter, writer: "csv.DictWriter"
    ) -> _AdapterRunResult:
        print(f"[{adapter.name}] checking availability...")
        if not adapter.is_available():
            print(f"[{adapter.name}] not available, skipping")
            return _AdapterRunResult(
                adapter.name, status="unavailable", note="is_available() returned False"
            )

        try:
            print(f"[{adapter.name}] setup...")
            adapter.setup()
        except AdapterUnavailableError as exc:
            print(f"[{adapter.name}] setup declined: {exc}")
            return _AdapterRunResult(adapter.name, status="unavailable", note=str(exc))
        except Exception as exc:  # noqa: BLE001 - want to keep the run alive
            print(f"[{adapter.name}] setup crashed: {exc}")
            return _AdapterRunResult(
                adapter.name,
                status="setup_failed",
                note=f"{type(exc).__name__}: {exc}",
            )

        result = _AdapterRunResult(adapter.name, status="ok")
        try:
            for i, sample in enumerate(self.samples):
                self._run_one_sample(adapter, sample, writer, result)
                if (i + 1) % 50 == 0:
                    print(f"[{adapter.name}] {i + 1}/{len(self.samples)} samples")
        finally:
            try:
                adapter.teardown()
            except Exception:  # noqa: BLE001
                pass

        return result

    def _run_one_sample(
        self,
        adapter: OCRAdapter,
        sample: Sample,
        writer: "csv.DictWriter",
        result: _AdapterRunResult,
    ) -> None:
        row = {
            "adapter": adapter.name,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "reference": sample.reference_text,
            "hypothesis": "",
            "wer": "",
            "cer": "",
            "mer": "",
            "wil": "",
            "latency_s": "",
            "error": "",
        }
        try:
            start = time.perf_counter()
            hypothesis = adapter.recognize(sample.image_path)
            latency = time.perf_counter() - start

            metrics = compute_metrics(sample.reference_text, hypothesis)

            row["hypothesis"] = hypothesis
            row["wer"] = metrics.wer
            row["cer"] = metrics.cer
            row["mer"] = metrics.mer
            row["wil"] = metrics.wil
            row["latency_s"] = latency

            result.latencies.append(latency)
            if metrics.wer == metrics.wer:  # not NaN
                result.wers.append(metrics.wer)
                result.cers.append(metrics.cer)
        except Exception as exc:  # noqa: BLE001 - one bad image shouldn't kill the run
            row["error"] = f"{type(exc).__name__}: {exc}"
            result.num_failed += 1
            traceback.print_exc()
        finally:
            writer.writerow(row)

    def _write_summary(
        self, path: Path, summaries: list[_AdapterRunResult]
    ) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for s in summaries:
                writer.writerow(
                    {
                        "adapter": s.adapter_name,
                        "status": s.status,
                        "num_samples_scored": len(s.wers),
                        "num_samples_failed": s.num_failed,
                        "mean_wer": statistics.mean(s.wers) if s.wers else "",
                        "median_wer": statistics.median(s.wers) if s.wers else "",
                        "mean_cer": statistics.mean(s.cers) if s.cers else "",
                        "median_cer": statistics.median(s.cers) if s.cers else "",
                        "mean_latency_s": (
                            statistics.mean(s.latencies) if s.latencies else ""
                        ),
                        "note": s.note,
                    }
                )
