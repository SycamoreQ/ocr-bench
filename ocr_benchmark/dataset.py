from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    reference_text: str


class IAMAsciiDataset:
    """Loader for the original IAM ASCII lines.txt / words.txt layout."""

    def __init__(
        self,
        root: Path | str,
        split: Literal["lines", "words"] = "lines",
        only_ok: bool = True,
        max_samples: int | None = None,
        gt_file: Path | str | None = None,
        images_root: Path | str | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.only_ok = only_ok
        self.max_samples = max_samples
        if not self.root.exists():
            raise FileNotFoundError(f"--iam-root does not exist: {self.root}")

        self.gt_file = Path(gt_file) if gt_file else self._find_gt_file()
        self._images_root = Path(images_root) if images_root else self.root
        self._image_index = self._build_image_index(self._images_root)

        print(
            f"[IAMAsciiDataset] ground truth: {self.gt_file} "
            f"| indexed {len(self._image_index)} images under {self._images_root}"
        )

    def _find_gt_file(self) -> Path:
        target_name = f"{self.split}.txt"

        fast_path = self.root / "ascii" / target_name
        if fast_path.exists():
            return fast_path

        direct_path = self.root / target_name
        if direct_path.exists():
            return direct_path

        matches = sorted(self.root.rglob(target_name))
        if not matches:
            raise FileNotFoundError(
                f"Could not find {target_name} anywhere under {self.root}. "
                f"Pass --{'lines' if self.split == 'lines' else 'words'}-txt "
                f"explicitly if your download uses a non-standard filename."
            )
        if len(matches) > 1:
            print(
                f"[IAMAsciiDataset] found {len(matches)} files named "
                f"{target_name} under {self.root}; using {matches[0]}. "
                f"Pass --{'lines' if self.split == 'lines' else 'words'}-txt "
                f"explicitly to pick a different one."
            )
        return matches[0]

    @staticmethod
    def _build_image_index(images_root: Path) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in images_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.stem, path)
        return index

    def __iter__(self) -> Iterator[Sample]:
        count = 0
        with open(self.gt_file, encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line or raw_line.startswith("#"):
                    continue

                fields = raw_line.split(" ")
                sample_id, segmentation = fields[0], fields[1]
                if self.only_ok and segmentation != "ok":
                    continue

                if self.split == "lines":
                    transcription = fields[8].replace("|", " ")
                else:
                    transcription = " ".join(fields[9:])

                image_path = self._image_index.get(sample_id)
                if image_path is None:
                    continue

                yield Sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    reference_text=transcription,
                )

                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return

    def __len__(self) -> int:
        return sum(1 for _ in self)


class StudentMessyHandwrittenDataset:
    """Loader for marfberg/Student-Messy-Handwritten-Dataset (SMHD).

    Expected local layout after downloading from Hugging Face:

        <root>/
          metadata.csv
          scans/0001.jpg
          scans/0002.jpg
          ...
          transcriptions/0001.txt
          transcriptions/0002.txt
          ...

    metadata.csv has two columns: `file_name,text`, where both values are
    repository-relative paths such as `scans/0001.jpg` and
    `transcriptions/0001.txt`.
    """

    DEFAULT_METADATA_NAME = "metadata.csv"

    def __init__(
        self,
        root: Path | str,
        max_samples: int | None = None,
        metadata_csv: Path | str | None = None,
    ):
        self.root = Path(root)
        self.max_samples = max_samples

        if not self.root.exists():
            raise FileNotFoundError(f"--smhd-root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"--smhd-root is not a directory: {self.root}")

        self.metadata_csv = (
            Path(metadata_csv)
            if metadata_csv is not None
            else self.root / self.DEFAULT_METADATA_NAME
        )
        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"SMHD metadata file does not exist: {self.metadata_csv}"
            )

        self._pairs = self._load_pairs()
        print(
            f"[StudentMessyHandwrittenDataset] metadata: {self.metadata_csv} "
            f"| valid pairs: {len(self._pairs)}"
        )

    def _load_pairs(self) -> list[tuple[str, Path, Path]]:
        pairs: list[tuple[str, Path, Path]] = []

        with open(self.metadata_csv, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"SMHD metadata has no header: {self.metadata_csv}")

            required = {"file_name", "text"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"SMHD metadata must contain columns {sorted(required)}; "
                    f"missing {sorted(missing)}"
                )

            for row_num, row in enumerate(reader, start=2):
                image_rel = (row.get("file_name") or "").strip()
                text_rel = (row.get("text") or "").strip()
                if not image_rel or not text_rel:
                    print(f"[SMHD] skipping metadata row {row_num}: empty path")
                    continue

                image_path = self.root / image_rel
                text_path = self.root / text_rel

                if not image_path.is_file():
                    print(
                        f"[SMHD] skipping {image_rel}: image does not exist at "
                        f"{image_path}"
                    )
                    continue
                if not text_path.is_file():
                    print(
                        f"[SMHD] skipping {text_rel}: transcription does not exist at "
                        f"{text_path}"
                    )
                    continue
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    print(
                        f"[SMHD] skipping {image_rel}: unsupported image extension"
                    )
                    continue

                sample_id = image_path.stem
                pairs.append((sample_id, image_path, text_path))

        return pairs

    @staticmethod
    def _read_transcription(text_path: Path) -> str:
        """Read an SMHD transcription without losing non-ASCII characters.

        Most files are UTF-8, but the dataset also contains at least some
        transcription files encoded in a Windows-1252-compatible encoding.
        Try UTF-8 first and fall back to CP1252 rather than using
        errors="replace", which would corrupt the reference used for OCR
        metrics.
        """
        raw = text_path.read_bytes()

        # Handle common Unicode BOMs explicitly. Check UTF-32 first because
        # its BOM also begins with the UTF-16 little-endian BOM bytes.
        if raw.startswith(b"\xff\xfe\x00\x00") or raw.startswith(b"\x00\x00\xfe\xff"):
            text = raw.decode("utf-32")
        elif raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
            text = raw.decode("utf-16")
        else:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                try:
                    text = raw.decode("cp1252")
                except UnicodeDecodeError:
                    # CP1252 leaves a few C1 bytes undefined. Latin-1 gives a
                    # lossless single-byte fallback for otherwise unusual files.
                    text = raw.decode("latin-1")
                print(
                    f"[SMHD] decoded non-UTF-8 transcription with CP1252/Latin-1: "
                    f"{text_path}"
                )

        # Normalize line endings, but keep the actual transcription characters.
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    def __iter__(self) -> Iterator[Sample]:
        samples = self._pairs
        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        for sample_id, image_path, text_path in samples:
            reference_text = self._read_transcription(text_path)

            yield Sample(
                sample_id=sample_id,
                image_path=image_path,
                reference_text=reference_text,
            )

    def __len__(self) -> int:
        if self.max_samples is None:
            return len(self._pairs)
        return min(self.max_samples, len(self._pairs))
