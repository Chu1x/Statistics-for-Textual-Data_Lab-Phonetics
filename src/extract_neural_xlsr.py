#!/usr/bin/env python3
"""Extract phoneme-level XLS-R/wav2vec2 representations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


XLSR_SAMPLE_RATE = 16_000
XLSR_FRAME_SECONDS = 0.02


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == XLSR_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    divisor = math.gcd(sample_rate, XLSR_SAMPLE_RATE)
    up = XLSR_SAMPLE_RATE // divisor
    down = sample_rate // divisor
    return signal.resample_poly(audio, up, down).astype(np.float32, copy=False)


def _time_to_frames(onset: float, offset: float, n_frames: int) -> tuple[int, int]:
    start = max(0, int(math.floor(onset / XLSR_FRAME_SECONDS)))
    stop = max(start + 1, int(math.ceil(offset / XLSR_FRAME_SECONDS)))
    return min(start, n_frames - 1), min(stop, n_frames)


def _load_model(
    model_name: str,
    device: torch.device,
    local_files_only: bool,
) -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2Model]:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = Wav2Vec2Model.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return feature_extractor, model


def _encode_wav(
    wav_path: str,
    feature_extractor: Wav2Vec2FeatureExtractor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], int]:
    audio, sample_rate = _read_audio(wav_path)
    audio = _resample(audio, sample_rate)
    inputs = feature_extractor(
        audio,
        sampling_rate=XLSR_SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
        padding=False,
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    return tuple(hidden.squeeze(0).cpu() for hidden in outputs.hidden_states), audio.shape[0]


def extract_xlsr_features(
    tokens: pd.DataFrame,
    model_name: str,
    layers: list[int],
    device: torch.device,
    local_files_only: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, object]]:
    feature_extractor, model = _load_model(model_name, device, local_files_only)
    n_hidden_states = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size
    for layer in layers:
        if layer < 0 or layer >= n_hidden_states:
            raise ValueError(
                f"Layer {layer} is invalid for {model_name}; expected 0..{n_hidden_states - 1}. "
                "Layer 0 is the feature projection before transformer blocks."
            )

    embeddings_by_layer: dict[int, np.ndarray] = {
        layer: np.full((len(tokens), hidden_size), np.nan, dtype=np.float32)
        for layer in layers
    }
    index_rows: list[dict[str, object]] = []
    token_positions = {idx: pos for pos, idx in enumerate(tokens.index)}

    for file_index, (wav_path, wav_tokens) in enumerate(tokens.groupby("wav_path", sort=True), start=1):
        hidden_states, _ = _encode_wav(wav_path, feature_extractor, model, device)
        n_frames = hidden_states[0].shape[0]

        for row in wav_tokens.itertuples():
            token_position = token_positions[row.Index]
            start_frame, stop_frame = _time_to_frames(float(row.onset), float(row.offset), n_frames)
            for layer in layers:
                pooled = hidden_states[layer][start_frame:stop_frame].mean(dim=0).numpy()
                embeddings_by_layer[layer][token_position] = pooled

            index_rows.append(
                {
                    "row_id": token_position,
                    "speaker_id": row.speaker_id,
                    "sentence_id": row.sentence_id,
                    "repetition_index": row.repetition_index,
                    "token_index": row.token_index,
                    "phoneme_label": row.phoneme_label,
                    "wav_path": wav_path,
                    "onset": row.onset,
                    "offset": row.offset,
                    "start_frame": start_frame,
                    "stop_frame": stop_frame,
                }
            )

        if file_index % 50 == 0:
            print(f"Processed {file_index} wav files...")

    arrays = {f"embeddings_layer_{layer}": values for layer, values in embeddings_by_layer.items()}
    metadata = {
        "model_name": model_name,
        "layers": layers,
        "hidden_size": hidden_size,
        "n_tokens": len(tokens),
        "sample_rate": XLSR_SAMPLE_RATE,
        "frame_seconds": XLSR_FRAME_SECONDS,
        "device": str(device),
    }
    return arrays, pd.DataFrame(index_rows).sort_values("row_id", ignore_index=True), metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=Path("data/phoneme_tokens.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/features_xlsr.npz"))
    parser.add_argument("--index-output", type=Path, default=Path("data/features_xlsr_index.csv"))
    parser.add_argument("--model", default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[3, 9, 18],
        help="Hidden-state layers to extract: default lower/middle/upper XLS-R layers.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached Hugging Face files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = pd.read_csv(args.tokens).reset_index(drop=True)
    if args.limit is not None:
        tokens = tokens.head(args.limit).copy()

    device = _choose_device(args.device)
    arrays, index, metadata = extract_xlsr_features(
        tokens=tokens,
        model_name=args.model,
        layers=args.layers,
        device=device,
        local_files_only=args.local_files_only,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays["row_id"] = index["row_id"].to_numpy(dtype=np.int64)
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(args.output, **arrays)
    index.to_csv(args.index_output, index=False)

    print(f"Wrote {len(index):,} token embeddings to {args.output}")
    print(f"Wrote token index to {args.index_output}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
