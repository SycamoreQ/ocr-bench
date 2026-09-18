# ocr_benchmark

Benchmarks nine open-source OCR engines against the IAM Handwriting
Database and reports WER/CER (and MER/WIL) per engine, as CSV.

## Layout

```
ocr_benchmark/
  adapters/
    base.py                    OCRAdapter interface
    docling_common.py          shared helper for docling-backed adapters
    tesseract_adapter.py       Tesseract, via docling
    rapidocr_adapter.py        RapidOCR, via docling
    easyocr_adapter.py         EasyOCR, via docling
    ocrmac_adapter.py          Apple Vision (ocrmac), via docling — macOS only
    granite_docling_adapter.py IBM Granite-Docling, via docling's VLM pipeline
    surya_adapter.py           Surya, direct integration
    paddleocr_vl_adapter.py    PaddleOCR-VL, direct integration
    unlimited_ocr_adapter.py   Baidu Unlimited-OCR, direct integration
  dataset.py                   IAM lines.txt/words.txt loader (layout-agnostic)
  metrics.py                   WER/CER/MER/WIL via jiwer
  runner.py                    orchestration -> per_sample.csv + summary.csv
  cli.py                       command-line entry point
tests/
  test_adapters.py             interface-conformance tests (not accuracy)
```

Every adapter implements the same three-method contract
(`is_available()` / `setup()` / `recognize()`) defined in
`adapters/base.py`. An engine that isn't installed, isn't available on
this platform, or fails during `setup()` is recorded as
`unavailable`/`setup_failed` in `summary.csv` rather than crashing the
whole run — with nine engines in play that's the normal case for most
environments, not a bug.

## Setup

```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr   # or: brew install tesseract
```

`requirements.txt` pulls in every engine; comment out the ones you don't
need (PaddleOCR-VL, Unlimited-OCR, and Granite-Docling in particular are
heavy multi-GB downloads on first run).

## Dataset: Kaggle IAM word mirror

You're using
https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database,
which is **word-level only** (no `lines.txt` — pass `--split words`) and
is commonly extracted with an `iam_words/` wrapper folder and no
`ascii/` subdirectory, e.g.:

```
<download>/iam_words/words.txt
<download>/iam_words/words/a01/a01-000u/a01-000u-00-00.png
```

`dataset.py` doesn't assume a fixed layout: it searches recursively under
`--iam-root` for `words.txt`, and indexes every image file it finds
under there by filename stem (IAM ids like `a01-000u-00-00` are unique
dataset-wide), so it finds the images regardless of the exact nesting —
including a double-wrapped `iam_words/iam_words/...` zip, which is a
known Kaggle quirk. Point `--iam-root` at wherever you extracted the
Kaggle download (e.g. the folder containing `iam_words/`, or
`iam_words/` itself — either works) and it will locate things itself; if
it still picks the wrong file, pass `--words-txt` / `--images-root`
explicitly.

## Run

```bash
python -m ocr_benchmark.cli \
  --iam-root /path/to/kaggle/download \
  --split words \
  --adapters tesseract,rapidocr,easyocr \
  --max-samples 200 \
  --output-dir results/smoke

# Once you've confirmed the fast engines work, run everything:
python -m ocr_benchmark.cli \
  --iam-root /path/to/kaggle/download \
  --split words \
  --adapters all \
  --output-dir results/full
```

Output:
- `results/.../per_sample.csv` — adapter, sample_id, reference,
  hypothesis, wer, cer, mer, wil, latency_s, error (one row per image
  per adapter)
- `results/.../summary.csv` — one row per adapter: status (ok /
  unavailable / setup_failed), mean/median WER & CER, mean latency,
  failure count

## Tests

```bash
pytest tests/test_adapters.py -v                # everything installed
pytest tests/test_adapters.py -v -m "not slow"   # skip the VLM-backed ones
```

These only check interface conformance (`setup()`/`recognize()` run
without violating the contract and return a string) against a tiny
synthetic image — a fast sanity check before spending time on a full IAM
run, not a substitute for looking at the actual WER/CER numbers.

## Notes on the harder adapters

- **Surya**: its Python API has changed across versions (pre-0.17 builds
  construct predictors with no arguments; 0.17+ requires a shared
  `FoundationPredictor`). `setup()`/`recognize()` try the modern call
  shape first and fall back on `TypeError`, so this should work across
  versions without edits — but if you hit a genuinely new signature,
  that's the place to look.
- **Granite-Docling** and **PaddleOCR-VL**: both are VLM-based parsers,
  much slower per image than the classical OCR engines and with
  multi-GB first-run downloads. Use `--max-samples` while sanity-checking.
- **Unlimited-OCR** (`baidu/Unlimited-OCR`): loaded via `transformers`
  with `trust_remote_code=True`. Its `.infer()` call is documented as
  writing results to disk (`save_results=True`); the model's own
  published evaluation snippet also treats its return value as the raw
  generated string, so the adapter uses the return value when it's a
  string and only falls back to reading a file out of the output
  directory otherwise.
