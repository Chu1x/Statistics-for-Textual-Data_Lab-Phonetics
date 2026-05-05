#!/usr/bin/env python3
"""Parse RU-FR TextGrid annotations into one row per phoneme token."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<speaker>[a-z0-9]+)_(?P<speaker_l1>[a-z]+)_(?P<list_id>list\d+)_(?P<sentence_id>FRcorp\d+)\.TextGrid$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Interval:
    tier: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


def _read_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, sep=";")
    metadata.columns = [column.strip() for column in metadata.columns]
    metadata["spk"] = metadata["spk"].astype(str).str.upper()
    metadata["L1"] = metadata["L1"].astype(str).str.lower()
    metadata["Gender"] = metadata["Gender"].astype(str).str.lower()
    return metadata


def _parse_textgrid(path: Path) -> dict[str, list[Interval]]:
    tiers: dict[str, list[Interval]] = {}
    current_tier: str | None = None
    pending_start: float | None = None
    pending_end: float | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if line.startswith("name = "):
            current_tier = _unquote(line.split("=", 1)[1].strip())
            tiers.setdefault(current_tier, [])
            continue

        if current_tier is None:
            continue

        if line.startswith("xmin = "):
            pending_start = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax = "):
            pending_end = float(line.split("=", 1)[1].strip())
        elif line.startswith("text = "):
            if pending_start is None or pending_end is None:
                continue
            tiers[current_tier].append(
                Interval(
                    tier=current_tier,
                    start=pending_start,
                    end=pending_end,
                    text=_unquote(line.split("=", 1)[1].strip()),
                )
            )
            pending_start = None
            pending_end = None

    return tiers


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


def _word_at_time(words: list[Interval], time: float) -> str:
    for word in words:
        if word.start <= time <= word.end:
            return word.text
    return ""


def _clean_phone_label(label: str) -> str:
    """Keep the actual phone when an interval includes a leading annotation."""
    parts = label.strip().split()
    if len(parts) > 1:
        return parts[-1]
    return label.strip()


def _file_metadata(textgrid_path: Path) -> dict[str, object]:
    match = FILENAME_RE.match(textgrid_path.name)
    if not match:
        raise ValueError(f"Unexpected TextGrid file name: {textgrid_path}")

    groups = match.groupdict()
    sentence_id = groups["sentence_id"]
    return {
        "speaker_id": groups["speaker"].upper(),
        "speaker_l1_from_filename": groups["speaker_l1"].lower(),
        "repetition_index": int(groups["list_id"].removeprefix("list")),
        "list_id": groups["list_id"],
        "sentence_id": sentence_id,
        "sentence_number": int(sentence_id.removeprefix("FRcorp")),
    }


def build_phoneme_table(corpus_root: Path, metadata_path: Path) -> pd.DataFrame:
    metadata = _read_metadata(metadata_path)
    metadata_by_speaker = metadata.set_index("spk", drop=False)
    rows: list[dict[str, object]] = []

    textgrid_paths = sorted(corpus_root.glob("*/*.TextGrid"))
    if not textgrid_paths:
        raise FileNotFoundError(f"No TextGrid files found under {corpus_root}")

    for textgrid_path in textgrid_paths:
        file_info = _file_metadata(textgrid_path)
        speaker_id = str(file_info["speaker_id"])
        if speaker_id not in metadata_by_speaker.index:
            raise KeyError(f"Speaker {speaker_id} from {textgrid_path} missing in metadata")

        speaker_metadata = metadata_by_speaker.loc[speaker_id]
        tiers = _parse_textgrid(textgrid_path)
        phones = tiers.get("phones")
        if phones is None:
            raise KeyError(f"Missing 'phones' tier in {textgrid_path}")

        words = tiers.get("words", [])
        wav_path = textgrid_path.with_suffix(".wav")

        for token_index, phone in enumerate(phones, start=1):
            phoneme_raw = phone.text.strip()
            if not phoneme_raw:
                continue
            phoneme = _clean_phone_label(phoneme_raw)

            rows.append(
                {
                    **file_info,
                    "token_index": token_index,
                    "phoneme_label": phoneme,
                    "phoneme_label_raw": phoneme_raw,
                    "onset": phone.start,
                    "offset": phone.end,
                    "duration": phone.duration,
                    "duration_ms": phone.duration * 1000.0,
                    "word_label": _word_at_time(words, phone.midpoint),
                    "l1_status": "L1" if speaker_metadata["L1"] == "fr" else "L2",
                    "native_language": speaker_metadata["L1"],
                    "gender": speaker_metadata["Gender"],
                    "age": speaker_metadata["Age"],
                    "fr_level": speaker_metadata["FR level"],
                    "ru_level": speaker_metadata["RU level"],
                    "recording_duration": speaker_metadata["Duration"],
                    "textgrid_path": str(textgrid_path),
                    "wav_path": str(wav_path),
                }
            )

    table = pd.DataFrame(rows)
    ordered_columns = [
        "speaker_id",
        "sentence_id",
        "sentence_number",
        "repetition_index",
        "list_id",
        "token_index",
        "phoneme_label",
        "phoneme_label_raw",
        "word_label",
        "onset",
        "offset",
        "duration",
        "duration_ms",
        "l1_status",
        "native_language",
        "gender",
        "age",
        "fr_level",
        "ru_level",
        "recording_duration",
        "speaker_l1_from_filename",
        "textgrid_path",
        "wav_path",
    ]
    return table[ordered_columns].sort_values(
        ["speaker_id", "sentence_number", "repetition_index", "onset", "offset"],
        ignore_index=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only"),
        help="Directory containing one subdirectory per speaker with TextGrid/WAV files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("ru-fr_interference/2/metadata_RUFR.csv"),
        help="Speaker metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phoneme_tokens.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_phoneme_table(args.corpus_root, args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(f"Wrote {len(table):,} phoneme tokens to {args.output}")
    print(f"Speakers: {table['speaker_id'].nunique()}")
    print(f"Sentences: {table['sentence_id'].nunique()}")
    print(f"Phoneme labels: {table['phoneme_label'].nunique()}")


if __name__ == "__main__":
    main()
