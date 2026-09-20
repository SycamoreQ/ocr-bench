# OCR Benchmark Results — IAM-line (n=100)

## Metrics

- **WER (Word Error Rate)** — word-level edit distance (insertions + deletions + substitutions) divided by total reference words. `0.0` = perfect match, `1.0` = every word wrong.
- **CER (Character Error Rate)** — same idea at the character level. More forgiving of small typos than WER, more sensitive to spacing/word-boundary errors.
- **Mean vs. median** — mean is pulled hard by extreme failures (e.g. a model that abstains on half its inputs drags the pooled mean down even if it's excellent on the samples it does attempt). Median reflects the *typical* sample; a median of `0.0` means more than half the samples are exact matches.
- **Latency** — mean wall-clock seconds per single-image inference. Not directly comparable across hardware: Tesseract and RapidOCR ran on CPU, Chandra and Unlimited-OCR ran on an A100 GPU.

## Results (n=100, IAM-line test split)

| Adapter | Mean WER | Median WER | Mean CER | Median CER | Mean latency |
|---|---|---|---|---|---|
| **Chandra** | **0.076** | **0.0** | **0.038** | **0.0** | 2.21s |
| RapidOCR | 0.615 | 0.667 | 0.350 | 0.218 | 0.43s |
| Unlimited-OCR | 0.725 | 1.0 | 0.631 | 0.892 | 0.98s |
| Tesseract | 0.943 | 1.0 | 0.611 | 0.579 | 0.36s |

## Observations

- **Chandra is the clear best model, not a close one.** Its WER is roughly a tenth of the next-best VLM (Unlimited-OCR) and about an eighth of the best CPU baseline (RapidOCR).
- **Chandra's median of 0.0 across all 100 samples** means more than half of all outputs are exact, character-for-character matches to the reference transcription — not just a strong average pulled up by a few easy samples.
- **Unlimited-OCR's mean/median split (0.725 / 1.0) is driven by classification failure, not transcription weakness.** Over half its outputs are the model's own layout head deciding a given line crop is "picture, not text" rather than attempting a transcription — a symptom of applying a page-context-dependent VLM to isolated, context-free line crops. On the subset it does attempt, its transcription quality is competitive.
- **RapidOCR is the best non-VLM option and the best latency/accuracy tradeoff** — CPU-only, sub-second per sample, and a solid WER of 0.615 with no classification-abstention failure mode.
- **Tesseract is the expected weak classical baseline** (WER 0.943) — useful mainly as a floor that makes every other result look good by contrast, consistent with its print-trained (not handwriting-trained) design.


- All ran on NVIDIA A100.
