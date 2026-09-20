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

## Setup

```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr   
```

## Run

```bash
python -m ocr_benchmark.cli \
  --iam-root /path/to/kaggle/download \
  --split words \
  --adapters tesseract,rapidocr,easyocr \
  --max-samples 200 \
  --output-dir results/smoke


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
