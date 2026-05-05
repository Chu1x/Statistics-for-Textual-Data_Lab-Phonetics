#!/usr/bin/env python3
"""Fit mixed-effects models for acoustic and neural representations."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


ORAL_VOWELS = ["i", "y", "u", "e", "ø", "o", "ɛ", "œ", "ə", "a", "ɑ"]
VOWEL_HEIGHT = {
    "i": "high",
    "y": "high",
    "u": "high",
    "e": "mid",
    "ø": "mid",
    "o": "mid",
    "ɛ": "mid",
    "œ": "mid",
    "ə": "mid",
    "a": "low",
    "ɑ": "low",
}


MODEL_FORMULAS = {
    "null_model": "{response} ~ 1",
    "main_effects": "{response} ~ l2 + male",
    "full_interaction": "{response} ~ l2 * male",
    "extended_height": "{response} ~ l2 * male + C(vowel_height)",
}


def prepare_data(acoustic_path: Path, whisper_pca: Path, xlsr_pca: Path) -> pd.DataFrame:
    data = pd.read_csv(acoustic_path, low_memory=False).reset_index(drop=True)
    data = data[data["phoneme_label"].isin(ORAL_VOWELS)].copy()
    data["l2"] = data["l1_status"].eq("L2").astype(float)
    data["male"] = data["gender"].astype(str).str.lower().eq("m").astype(float)
    data["vowel_height"] = data["phoneme_label"].map(VOWEL_HEIGHT)

    with np.load(whisper_pca) as whisper, np.load(xlsr_pca) as xlsr:
        whisper_values = whisper["pca50_layer_20"]
        xlsr_values = xlsr["pca50_layer_18"]
    for pc in range(5):
        data[f"whisper_pc{pc + 1}"] = whisper_values[data.index, pc]
        data[f"xlsr_pc{pc + 1}"] = xlsr_values[data.index, pc]
    return data


def fit_model(data: pd.DataFrame, response: str, model_name: str):
    formula = MODEL_FORMULAS[model_name].format(response=response)
    model_data = data[[response, "speaker_id", "l2", "male", "vowel_height"]].dropna().copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, model_data, groups=model_data["speaker_id"])
        return model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)


def variance_parts(result) -> tuple[float, float, float, float, float]:
    fixed = np.asarray(result.model.exog @ result.fe_params)
    var_fixed = float(np.var(fixed, ddof=1))
    var_random = float(result.cov_re.iloc[0, 0]) if result.cov_re.size else 0.0
    var_residual = float(result.scale)
    total = var_fixed + var_random + var_residual
    marginal_r2 = var_fixed / total if total else np.nan
    conditional_r2 = (var_fixed + var_random) / total if total else np.nan
    icc = var_random / (var_random + var_residual) if (var_random + var_residual) else np.nan
    return var_fixed, var_random, var_residual, marginal_r2, conditional_r2, icc


def model_sequence(data: pd.DataFrame, response: str, representation: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    fitted = {}
    rows = []
    for model_name in MODEL_FORMULAS:
        try:
            result = fit_model(data, response, model_name)
            fitted[model_name] = result
            var_fixed, var_random, var_residual, marginal_r2, conditional_r2, icc = variance_parts(result)
            rows.append(
                {
                    "representation": representation,
                    "response": response,
                    "model": model_name,
                    "n_obs": int(result.nobs),
                    "converged": bool(result.converged),
                    "log_likelihood": float(result.llf),
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "df_modelwc": float(result.df_modelwc),
                    "var_fixed": var_fixed,
                    "var_random_speaker": var_random,
                    "var_residual": var_residual,
                    "icc": icc,
                    "marginal_r2": marginal_r2,
                    "conditional_r2": conditional_r2,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "representation": representation,
                    "response": response,
                    "model": model_name,
                    "n_obs": int(data[response].notna().sum()),
                    "converged": False,
                    "error": str(exc),
                }
            )

    lrt_rows = []
    comparisons = [
        ("null_model", "main_effects"),
        ("main_effects", "full_interaction"),
        ("full_interaction", "extended_height"),
    ]
    for simpler, richer in comparisons:
        if simpler not in fitted or richer not in fitted:
            continue
        ll0 = fitted[simpler].llf
        ll1 = fitted[richer].llf
        df = fitted[richer].df_modelwc - fitted[simpler].df_modelwc
        raw_stat = 2 * (ll1 - ll0)
        stat = max(0.0, raw_stat)
        lrt_rows.append(
            {
                "representation": representation,
                "response": response,
                "comparison": f"{simpler}_vs_{richer}",
                "lr_statistic": float(stat),
                "raw_lr_statistic": float(raw_stat),
                "lrt_note": "optimizer_nonmonotonic" if raw_stat < 0 else "",
                "df": float(df),
                "p_value": float(chi2.sf(stat, df)) if df > 0 else np.nan,
            }
        )

    coefficients = []
    if "extended_height" in fitted:
        result = fitted["extended_height"]
        for term, estimate in result.fe_params.items():
            coefficients.append(
                {
                    "representation": representation,
                    "response": response,
                    "model": "extended_height",
                    "term": term,
                    "estimate": float(estimate),
                    "std_error": float(result.bse_fe[term]),
                    "z_value": float(result.tvalues[term]),
                    "p_value": float(result.pvalues[term]),
                }
            )
    return rows + lrt_rows, {"fitted": fitted, "coefficients": coefficients}


def fit_all_models(data: pd.DataFrame, tables_dir: Path) -> None:
    model_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    response_specs = [
        ("acoustic", "f1_lobanov"),
        ("acoustic", "f2_lobanov"),
    ]
    response_specs += [("whisper_layer20", f"whisper_pc{i}") for i in range(1, 6)]
    response_specs += [("xlsr_layer18", f"xlsr_pc{i}") for i in range(1, 6)]

    for representation, response in response_specs:
        rows, extra = model_sequence(data, response, representation)
        model_rows.extend(rows)
        coefficient_rows.extend(extra["coefficients"])
        print(f"Fitted {representation}:{response}")

    pd.DataFrame(model_rows).to_csv(tables_dir / "mixed_model_comparisons.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(tables_dir / "mixed_model_fixed_effects.csv", index=False)

    random_slope_note = pd.DataFrame(
        [
            {
                "model": "random_slope_l1_by_speaker",
                "status": "not_identifiable",
                "reason": "L1/L2 status is constant within speaker, so a by-speaker random slope for L1 has no within-speaker variation.",
            }
        ]
    )
    random_slope_note.to_csv(tables_dir / "mixed_model_random_slope_note.csv", index=False)


def fit_a_icc_models(data: pd.DataFrame, tables_dir: Path) -> None:
    rows = []
    subset = data[data["phoneme_label"] == "a"].copy()
    for representation, response in [
        ("acoustic", "f1_lobanov"),
        ("whisper_layer20", "whisper_pc1"),
        ("xlsr_layer18", "xlsr_pc1"),
    ]:
        result = fit_model(subset, response, "null_model")
        _, var_random, var_residual, _, _, icc = variance_parts(result)
        rows.append(
            {
                "phoneme_label": "a",
                "representation": representation,
                "response": response,
                "n_obs": int(result.nobs),
                "speaker_random_variance": var_random,
                "residual_variance": var_residual,
                "icc": icc,
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "mixed_model_icc_a.csv", index=False)


def summarise_representation_r2(tables_dir: Path) -> None:
    comparisons = pd.read_csv(tables_dir / "mixed_model_comparisons.csv")
    model_rows = comparisons[comparisons["model"].eq("extended_height")].copy()
    summary = (
        model_rows.groupby("representation", dropna=False)
        .agg(
            mean_marginal_r2=("marginal_r2", "mean"),
            max_marginal_r2=("marginal_r2", "max"),
            mean_conditional_r2=("conditional_r2", "mean"),
            n_responses=("response", "count"),
        )
        .reset_index()
        .sort_values("mean_marginal_r2", ascending=False)
    )
    summary.to_csv(tables_dir / "mixed_model_representation_r2_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic", type=Path, default=Path("data/features_acoustic_norm.csv"))
    parser.add_argument("--whisper-pca", type=Path, default=Path("data/features_whisper_pca.npz"))
    parser.add_argument("--xlsr-pca", type=Path, default=Path("data/features_xlsr_pca.npz"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args.acoustic, args.whisper_pca, args.xlsr_pca)
    fit_a_icc_models(data, args.tables_dir)
    fit_all_models(data, args.tables_dir)
    summarise_representation_r2(args.tables_dir)

    outputs = sorted(path.name for path in args.tables_dir.glob("mixed_model*.csv"))
    print(json.dumps({"mixed_model_tables": outputs}, indent=2))


if __name__ == "__main__":
    main()
