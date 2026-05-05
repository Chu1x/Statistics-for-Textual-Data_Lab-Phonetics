#!/usr/bin/env python3
"""Normalise acoustic features and reduce neural embeddings with PCA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


FORMANT_COLUMNS = [
    "f1_hz",
    "f2_hz",
    "f3_hz",
    "f1_25_hz",
    "f2_25_hz",
    "f3_25_hz",
    "f1_75_hz",
    "f2_75_hz",
    "f3_75_hz",
]


def lobanov_normalise(acoustic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply speaker-wise Lobanov normalisation using vowel tokens only."""
    output = acoustic.copy()
    vowel_mask = output["is_vowel"].astype(bool)
    speaker_stats = (
        output.loc[vowel_mask]
        .groupby("speaker_id", dropna=False)
        .agg(
            f1_mean=("f1_hz", "mean"),
            f1_sd=("f1_hz", "std"),
            f2_mean=("f2_hz", "mean"),
            f2_sd=("f2_hz", "std"),
            f3_mean=("f3_hz", "mean"),
            f3_sd=("f3_hz", "std"),
            n_vowel_tokens=("phoneme_label", "size"),
        )
        .reset_index()
    )

    stats_by_speaker = speaker_stats.set_index("speaker_id")
    for formant in ("f1", "f2", "f3"):
        mean = output["speaker_id"].map(stats_by_speaker[f"{formant}_mean"])
        sd = output["speaker_id"].map(stats_by_speaker[f"{formant}_sd"])
        for column in [col for col in FORMANT_COLUMNS if col.startswith(formant)]:
            output[f"{column.removesuffix('_hz')}_lobanov"] = (output[column] - mean) / sd

    return output, speaker_stats


def _embedding_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(key for key in npz.files if key.startswith("embeddings_layer_"))


def pca_reduce_embeddings(
    input_path: Path,
    n_components: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    with np.load(input_path) as data:
        arrays: dict[str, np.ndarray] = {}
        metadata = json.loads(str(data["metadata_json"]))
        layer_metadata = {}

        for key in _embedding_keys(data):
            layer = key.removeprefix("embeddings_layer_")
            embeddings = data[key].astype(np.float32, copy=False)
            if np.isnan(embeddings).any():
                raise ValueError(f"{input_path}:{key} contains NaN values")

            pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
            transformed = pca.fit_transform(embeddings).astype(np.float32)
            arrays[f"pca50_layer_{layer}"] = transformed
            arrays[f"pca2_layer_{layer}"] = transformed[:, :2]
            arrays[f"explained_variance_ratio_layer_{layer}"] = pca.explained_variance_ratio_.astype(
                np.float32
            )
            arrays[f"singular_values_layer_{layer}"] = pca.singular_values_.astype(np.float32)
            layer_metadata[layer] = {
                "source_key": key,
                "n_components": n_components,
                "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            }

        arrays["row_id"] = data["row_id"].astype(np.int64)
        metadata["pca"] = {
            "n_components": n_components,
            "random_state": random_state,
            "svd_solver": "randomized",
            "layers": layer_metadata,
        }
        arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
        return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic.csv"))
    parser.add_argument("--whisper", type=Path, default=Path("data/features_whisper.npz"))
    parser.add_argument("--xlsr", type=Path, default=Path("data/features_xlsr.npz"))
    parser.add_argument(
        "--acoustic-output",
        type=Path,
        default=Path("data/features_acoustic_norm.csv"),
    )
    parser.add_argument(
        "--lobanov-stats-output",
        type=Path,
        default=Path("data/lobanov_speaker_stats.csv"),
    )
    parser.add_argument(
        "--whisper-output",
        type=Path,
        default=Path("data/features_whisper_pca.npz"),
    )
    parser.add_argument("--xlsr-output", type=Path, default=Path("data/features_xlsr_pca.npz"))
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    acoustic = pd.read_csv(args.acoustic, low_memory=False)
    acoustic_norm, lobanov_stats = lobanov_normalise(acoustic)
    args.acoustic_output.parent.mkdir(parents=True, exist_ok=True)
    acoustic_norm.to_csv(args.acoustic_output, index=False)
    lobanov_stats.to_csv(args.lobanov_stats_output, index=False)
    print(f"Wrote acoustic normalised features to {args.acoustic_output}")
    print(f"Wrote Lobanov speaker stats to {args.lobanov_stats_output}")

    whisper_arrays = pca_reduce_embeddings(args.whisper, args.pca_components, args.random_state)
    np.savez_compressed(args.whisper_output, **whisper_arrays)
    print(f"Wrote Whisper PCA features to {args.whisper_output}")

    xlsr_arrays = pca_reduce_embeddings(args.xlsr, args.pca_components, args.random_state)
    np.savez_compressed(args.xlsr_output, **xlsr_arrays)
    print(f"Wrote XLS-R PCA features to {args.xlsr_output}")

    for output_path in (args.whisper_output, args.xlsr_output):
        with np.load(output_path) as data:
            metadata = json.loads(str(data["metadata_json"]))
            print(json.dumps({"file": str(output_path), "pca": metadata["pca"]}, indent=2))


if __name__ == "__main__":
    main()
