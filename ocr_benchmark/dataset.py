"""Loader for IAM Handwriting Database ground-truth files (lines.txt /
words.txt), tolerant of whatever directory layout the archive you have
actually extracted to.

The original IAM download layout is:

    <root>/ascii/words.txt
    <root>/words/a01/a01-000u/a01-000u-00-00.png

...but re-hosted mirrors (Kaggle, HF, random GitHub repos) routinely
repackage this with a different nesting — an extra wrapper folder, no
`ascii/` subdirectory, everything flattened, etc. The Kaggle mirror at
nibinv23/iam-handwriting-word-database, for instance, is commonly
extracted as `iam_words/words.txt` + `iam_words/words/a01/...` — one
level shallower than the original, with no `ascii/` folder at all, and
double-nested (`iam_words/iam_words/...`) is also a known Kaggle zip
quirk.

Rather than hard-coding one assumed path, this loader:
  1. Searches --iam-root recursively for a ground-truth file named
     `{split}.txt` (e.g. `words.txt`), wherever it is.
  2. Indexes every image file under --iam-root once, keyed by filename
     stem (IAM sample ids like "a01-000u-00-00" are unique across the
     whole dataset), so it finds each sample's image regardless of how
     deeply/differently the images are nested.

If your layout is unusual enough that this still doesn't find things,
pass --words-txt / --lines-txt and --images-root explicitly to skip
auto-discovery.
"""

from __future__ import annotations

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

        # Fast path: the layout the original IAM download uses.
        fast_path = self.root / "ascii" / target_name
        if fast_path.exists():
            return fast_path

        # Fast path: ground-truth file directly under root (several
        # mirrors, including the common Kaggle repackaging, do this).
        direct_path = self.root / target_name
        if direct_path.exists():
            return direct_path

        # General fallback: search the whole tree.
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
        """Map every image's filename stem -> path, once. IAM sample ids
        ("a01-000u-00-00") are unique dataset-wide, so a flat stem->path
        index sidesteps needing to know the exact writer-group/form
        nesting a given mirror uses."""
        index: dict[str, Path] = {}
        for path in images_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                # First match wins; duplicate stems under different
                # subtrees (e.g. a stray copy) are rare enough not to
                # warrant more machinery here.
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
                    # lines.txt: id ok graylevel num-components x y w h
                    # transcription(word1|word2|...) -> 9 fields before
                    # the transcription, "|" stands in for spaces.
                    transcription = fields[8].replace("|", " ")
                else:
                    # words.txt: id ok graylevel num-components x y w h tag
                    # transcription -> transcription starts at index 9
                    # (the tag, e.g. "AT"/"NN", is index 8 — easy to
                    # mistake for the start of the transcription if you
                    # forget the num-components field).
                    transcription = " ".join(fields[9:])

                image_path = self._image_index.get(sample_id)
                if image_path is None:
                    # Ground truth references an image that isn't in our
                    # index (partial download, damaged/missing file —
                    # IAM has a couple of known-bad ids) — skip it rather
                    # than crash the whole run.
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
