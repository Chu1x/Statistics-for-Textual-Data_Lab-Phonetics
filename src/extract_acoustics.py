#!/usr/bin/env python3
"""Extract acoustic descriptors for phoneme tokens with Parselmouth/Praat."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import parselmouth


VOWELS = {
    "a",
    "ɑ",
    "ɑ̃",
    "e",
    "ɛ",
    "ɛ̃",
    "ə",
    "i",
    "o",
    "ɔ",
    "ɔ̃",
    "ø",
    "œ",
    "œ̃",
    "u",
    "y",
}
FRICATIVES = {"f", "v", "s", "z", "ʃ", "ʒ", "ʁ"}


def _safe_float(value: float | None) -> float:
    if value is None:
        return math.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not np.isfinite(value) or value <= 0:
        return math.nan
    return value


def _formant_at(formant: parselmouth.Formant, formant_number: int, time: float) -> float:
    return _safe_float(formant.get_value_at_time(formant_number, time))


def _mean_f0(segment: parselmouth.Sound) -> float:
    try:
        pitch = segment.to_pitch_ac(time_step=0.01)
    except Exception:
        return math.nan
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies > 0]
    if voiced.size == 0:
        return math.nan
    return float(np.mean(voiced))


def _rms_energy(segment: parselmouth.Sound) -> float:
    values = np.asarray(segment.values, dtype=float)
    if values.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(values))))


def _spectral_centre_of_gravity(segment: parselmouth.Sound) -> float:
    try:
        spectrum = segment.to_spectrum()
    except Exception:
        return math.nan

    frequencies = np.asarray(spectrum.xs(), dtype=float)
    real = np.asarray(spectrum.values[0], dtype=float)
    imaginary = np.asarray(spectrum.values[1], dtype=float)
    magnitudes = np.sqrt(real**2 + imaginary**2)

    total_magnitude = float(np.sum(magnitudes))
    if total_magnitude <= 0:
        return math.nan
    return float(np.sum(frequencies * magnitudes) / total_magnitude)


def _extract_segment(sound: parselmouth.Sound, onset: float, offset: float) -> parselmouth.Sound | None:
    if not np.isfinite(onset) or not np.isfinite(offset) or offset <= onset:
        return None
    onset = max(float(onset), sound.xmin)
    offset = min(float(offset), sound.xmax)
    if offset <= onset:
        return None
    try:
        return sound.extract_part(from_time=onset, to_time=offset, preserve_times=False)
    except Exception:
        return None


def _max_formant_for_gender(gender: str) -> int:
    return 4500 if str(gender).lower().startswith("m") else 5000


def _trajectory_times(onset: float, offset: float) -> tuple[float, float, float]:
    duration = offset - onset
    return onset + duration * 0.25, onset + duration * 0.50, onset + duration * 0.75


def _iter_groups(tokens: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for wav_path, wav_tokens in tokens.groupby("wav_path", sort=True):
        yield str(wav_path), wav_tokens


def extract_acoustic_features(tokens: pd.DataFrame) -> pd.DataFrame:
    feature_rows: list[dict[str, float | int | str]] = []

    for file_index, (wav_path, wav_tokens) in enumerate(_iter_groups(tokens), start=1):
        try:
            sound = parselmouth.Sound(wav_path)
        except Exception as exc:
            for row in wav_tokens.itertuples(index=False):
                feature_rows.append(_empty_feature_row(row, error=f"sound_load_error: {exc}"))
            continue

        gender = str(wav_tokens["gender"].iloc[0])
        max_formant = _max_formant_for_gender(gender)
        try:
            formant = sound.to_formant_burg(
                max_number_of_formants=5,
                maximum_formant=max_formant,
            )
            formant_error = ""
        except Exception as exc:
            formant = None
            formant_error = f"formant_error: {exc}"

        for row in wav_tokens.itertuples(index=False):
            feature_rows.append(_extract_row_features(row, sound, formant, max_formant, formant_error))

        if file_index % 100 == 0:
            print(f"Processed {file_index} wav files...")

    return pd.DataFrame(feature_rows)


def _empty_feature_row(row: object, error: str = "") -> dict[str, float | int | str]:
    return {
        "speaker_id": row.speaker_id,
        "sentence_id": row.sentence_id,
        "repetition_index": row.repetition_index,
        "token_index": row.token_index,
        "phoneme_label": row.phoneme_label,
        "is_vowel": row.phoneme_label in VOWELS,
        "is_fricative": row.phoneme_label in FRICATIVES,
        "duration_ms": row.duration_ms,
        "midpoint_s": math.nan,
        "f1_hz": math.nan,
        "f2_hz": math.nan,
        "f3_hz": math.nan,
        "f1_25_hz": math.nan,
        "f2_25_hz": math.nan,
        "f3_25_hz": math.nan,
        "f1_75_hz": math.nan,
        "f2_75_hz": math.nan,
        "f3_75_hz": math.nan,
        "f0_mean_hz": math.nan,
        "scg_hz": math.nan,
        "rms_energy": math.nan,
        "max_formant_hz": math.nan,
        "extraction_error": error,
    }


def _extract_row_features(
    row: object,
    sound: parselmouth.Sound,
    formant: parselmouth.Formant | None,
    max_formant: int,
    formant_error: str,
) -> dict[str, float | int | str]:
    features = _empty_feature_row(row, error=formant_error)
    onset = float(row.onset)
    offset = float(row.offset)
    duration_ms = float(row.duration_ms)
    is_vowel = str(row.phoneme_label) in VOWELS
    is_fricative = str(row.phoneme_label) in FRICATIVES
    t25, midpoint, t75 = _trajectory_times(onset, offset)

    features["midpoint_s"] = midpoint
    features["max_formant_hz"] = max_formant

    if formant is not None:
        features["f1_hz"] = _formant_at(formant, 1, midpoint)
        features["f2_hz"] = _formant_at(formant, 2, midpoint)
        if is_vowel:
            features["f3_hz"] = _formant_at(formant, 3, midpoint)
        if is_vowel and duration_ms > 80:
            features["f1_25_hz"] = _formant_at(formant, 1, t25)
            features["f2_25_hz"] = _formant_at(formant, 2, t25)
            features["f3_25_hz"] = _formant_at(formant, 3, t25)
            features["f1_75_hz"] = _formant_at(formant, 1, t75)
            features["f2_75_hz"] = _formant_at(formant, 2, t75)
            features["f3_75_hz"] = _formant_at(formant, 3, t75)

    segment = _extract_segment(sound, onset, offset)
    if segment is None:
        features["extraction_error"] = "invalid_segment"
        return features

    features["f0_mean_hz"] = _mean_f0(segment)
    features["rms_energy"] = _rms_energy(segment)
    if is_fricative:
        features["scg_hz"] = _spectral_centre_of_gravity(segment)

    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=Path,
        default=Path("data/phoneme_tokens.csv"),
        help="Input phoneme-token CSV from parse_corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features_acoustic.csv"),
        help="Output acoustic feature CSV.",
    )
    parser.add_argument(
        "--missingness-output",
        type=Path,
        default=Path("data/features_acoustic_missingness.csv"),
        help="Output CSV with missing-value proportions by phoneme and group.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick test runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = pd.read_csv(args.tokens)
    if args.limit is not None:
        tokens = tokens.head(args.limit).copy()

    features = extract_acoustic_features(tokens)
    output = tokens.merge(
        features,
        on=["speaker_id", "sentence_id", "repetition_index", "token_index", "phoneme_label", "duration_ms"],
        how="left",
        validate="one_to_one",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    missingness = (
        output.groupby(["phoneme_label", "l1_status", "gender"], dropna=False)
        .agg(
            n_tokens=("phoneme_label", "size"),
            f1_missing_prop=("f1_hz", lambda values: values.isna().mean()),
            f2_missing_prop=("f2_hz", lambda values: values.isna().mean()),
            f3_missing_prop=("f3_hz", lambda values: values.isna().mean()),
            f0_missing_prop=("f0_mean_hz", lambda values: values.isna().mean()),
            scg_missing_prop=("scg_hz", lambda values: values.isna().mean()),
        )
        .reset_index()
    )
    missingness.to_csv(args.missingness_output, index=False)

    print(f"Wrote {len(output):,} rows to {args.output}")
    print(f"Wrote missingness summary to {args.missingness_output}")
    print(f"Missing F1: {output['f1_hz'].isna().mean():.2%}")
    print(f"Missing F2: {output['f2_hz'].isna().mean():.2%}")
    print(f"Missing f0: {output['f0_mean_hz'].isna().mean():.2%}")


if __name__ == "__main__":
    main()
