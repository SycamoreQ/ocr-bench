from __future__ import annotations

import math
from dataclasses import dataclass

import jiwer

_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
_CHAR_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


@dataclass
class SampleMetrics:
    wer: float
    cer: float
    mer: float
    wil: float
    ref_word_count: int
    ref_char_count: int


def compute_metrics(reference: str, hypothesis: str) -> SampleMetrics:
    ref_words = reference.split()
    if not ref_words:
        return SampleMetrics(
            wer=math.nan,
            cer=math.nan,
            mer=math.nan,
            wil=math.nan,
            ref_word_count=0,
            ref_char_count=len(reference),
        )

    word_out = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=_TRANSFORM,
        hypothesis_transform=_TRANSFORM,
    )
    cer = jiwer.cer(
        reference,
        hypothesis,
        reference_transform=_CHAR_TRANSFORM,
        hypothesis_transform=_CHAR_TRANSFORM,
    )

    return SampleMetrics(
        wer=word_out.wer,
        cer=cer,
        mer=word_out.mer,
        wil=word_out.wil,
        ref_word_count=len(ref_words),
        ref_char_count=len(reference),
    )
