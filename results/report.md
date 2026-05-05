# Acoustic and Neural Representations in a Phonetically Aligned Corpus

## Overview

This report summarises the reproducible pipeline for the Russian-French Interference Corpus. The pipeline parsed TextGrid annotations, extracted acoustic features, extracted Whisper and XLS-R hidden-state representations, normalised features, ran statistical tests, fitted mixed-effects models, evaluated ROPE classifications, and performed hierarchical clustering.

## Pipeline Outputs

Main data products:

- `data/phoneme_tokens.csv`
- `data/features_acoustic.csv`
- `data/features_acoustic_norm.csv`
- `data/features_whisper.npz`
- `data/features_whisper_pca.npz`
- `data/features_xlsr.npz`
- `data/features_xlsr_pca.npz`

Main result folders:

- `results/tables/`
- `results/figures/`

## Methods and Data Quality

The parsed corpus contains 22919 phoneme tokens from 19 speakers and 78 sentence IDs. Acoustic extraction used midpoint formants and, for vowels longer than 80 ms, additional 25% and 75% formant measurements. Vowels were Lobanov-normalised within speaker using vowel tokens only.

Whisper embeddings used `openai/whisper-medium` layers 6, 20. XLS-R embeddings used `facebook/wav2vec2-large-xlsr-53` layers 3, 9, 18. Neural representations were reduced to 50 principal components per selected layer for downstream analyses. Statistical tests used 5000 permutations where applicable and ROPE/bootstrap summaries used 2000 bootstrap samples where applicable.

Missing-value summary by analysis-relevant token set:

- **missingness 1**
  - `feature`: f1_hz
  - `n_tokens_considered`: 8277
  - `missing_prop`: 0.000
- **missingness 2**
  - `feature`: f2_hz
  - `n_tokens_considered`: 8277
  - `missing_prop`: 0.000
- **missingness 3**
  - `feature`: f3_hz
  - `n_tokens_considered`: 8277
  - `missing_prop`: 0.000
- **missingness 4**
  - `feature`: f0_mean_hz
  - `n_tokens_considered`: 8277
  - `missing_prop`: 0.047
- **missingness 5**
  - `feature`: scg_hz
  - `n_tokens_considered`: 5701
  - `missing_prop`: 0.000
- **missingness 6**
  - `feature`: f1_25_hz
  - `n_tokens_considered`: 5688
  - `missing_prop`: 0.000
- **missingness 7**
  - `feature`: f2_25_hz
  - `n_tokens_considered`: 5688
  - `missing_prop`: 0.000
- **missingness 8**
  - `feature`: f1_75_hz
  - `n_tokens_considered`: 5688
  - `missing_prop`: 0.000
- **missingness 9**
  - `feature`: f2_75_hz
  - `n_tokens_considered`: 5688
  - `missing_prop`: 0.000

Rough acoustic range flags:

- **quality 1**
  - `subset`: all_tokens
  - `feature`: f1_hz
  - `rough_low`: 100.000
  - `rough_high`: 1200.000
  - `n_tokens`: 22919
  - `n_nonmissing`: 22919
  - `n_flagged`: 1461
  - `flagged_prop_nonmissing`: 0.064
  - `min_value`: 83.732
  - `max_value`: 2645.505
- **quality 2**
  - `subset`: all_tokens
  - `feature`: f2_hz
  - `rough_low`: 300.000
  - `rough_high`: 4000.000
  - `n_tokens`: 22919
  - `n_nonmissing`: 22919
  - `n_flagged`: 4
  - `flagged_prop_nonmissing`: 0.000
  - `min_value`: 259.598
  - `max_value`: 3455.790
- **quality 3**
  - `subset`: all_tokens
  - `feature`: f0_mean_hz
  - `rough_low`: 50.000
  - `rough_high`: 500.000
  - `n_tokens`: 22919
  - `n_nonmissing`: 15915
  - `n_flagged`: 329
  - `flagged_prop_nonmissing`: 0.021
  - `min_value`: 74.985
  - `max_value`: 598.823
- **quality 4**
  - `subset`: oral_vowels
  - `feature`: f1_hz
  - `rough_low`: 100.000
  - `rough_high`: 1200.000
  - `n_tokens`: 8277
  - `n_nonmissing`: 8277
  - `n_flagged`: 16
  - `flagged_prop_nonmissing`: 0.002
  - `min_value`: 94.951
  - `max_value`: 1535.469
- **quality 5**
  - `subset`: oral_vowels
  - `feature`: f2_hz
  - `rough_low`: 300.000
  - `rough_high`: 4000.000
  - `n_tokens`: 8277
  - `n_nonmissing`: 8277
  - `n_flagged`: 1
  - `flagged_prop_nonmissing`: 0.000
  - `min_value`: 297.691
  - `max_value`: 3051.938
- **quality 6**
  - `subset`: oral_vowels
  - `feature`: f0_mean_hz
  - `rough_low`: 50.000
  - `rough_high`: 500.000
  - `n_tokens`: 8277
  - `n_nonmissing`: 7892
  - `n_flagged`: 32
  - `flagged_prop_nonmissing`: 0.004
  - `min_value`: 75.100
  - `max_value`: 584.773

Missing acoustic values were not imputed. Analyses used pairwise exclusion for the feature being tested, so each result table reports the available sample size for that feature and contrast. Rough-range flags were retained rather than automatically removed; among oral vowels, flagged counts were F1 = 16, F2 = 1, and f0 = 32. These flags are treated as quality diagnostics, not as exclusions from the main formant analyses.

Acoustic L1/L2 sensitivity check after excluding rough-range F1/F2 flags changed 0 of 22 FDR-significance decisions. Affected or filtered contrasts:

- **sensitivity 1**
  - `phoneme_label`: y
  - `feature`: f1_lobanov
  - `n_l1_main`: 273
  - `n_l2_main`: 265
  - `difference_l2_minus_l1_main`: -0.303
  - `p_fdr_bh_main`: 0.001
  - `significant_fdr_0_05_main`: True
  - `n_l1_range_filtered`: 270
  - `n_l2_range_filtered`: 264
  - `difference_l2_minus_l1_range_filtered`: -0.254
  - `p_fdr_bh_range_filtered`: 0.001
  - `significant_fdr_0_05_range_filtered`: True
  - `n_excluded_by_range_filter`: 4
  - `conclusion_changed`: False
  - `absolute_effect_change`: 0.049
- **sensitivity 2**
  - `phoneme_label`: ə
  - `feature`: f1_lobanov
  - `n_l1_main`: 30
  - `n_l2_main`: 87
  - `difference_l2_minus_l1_main`: 0.207
  - `p_fdr_bh_main`: 0.413
  - `significant_fdr_0_05_main`: False
  - `n_l1_range_filtered`: 30
  - `n_l2_range_filtered`: 86
  - `difference_l2_minus_l1_range_filtered`: 0.169
  - `p_fdr_bh_range_filtered`: 0.485
  - `significant_fdr_0_05_range_filtered`: False
  - `n_excluded_by_range_filter`: 1
  - `conclusion_changed`: False
  - `absolute_effect_change`: 0.038
- **sensitivity 3**
  - `phoneme_label`: i
  - `feature`: f1_lobanov
  - `n_l1_main`: 910
  - `n_l2_main`: 962
  - `difference_l2_minus_l1_main`: -0.011
  - `p_fdr_bh_main`: 0.283
  - `significant_fdr_0_05_main`: False
  - `n_l1_range_filtered`: 910
  - `n_l2_range_filtered`: 960
  - `difference_l2_minus_l1_range_filtered`: -0.021
  - `p_fdr_bh_range_filtered`: 0.244
  - `significant_fdr_0_05_range_filtered`: False
  - `n_excluded_by_range_filter`: 2
  - `conclusion_changed`: False
  - `absolute_effect_change`: 0.010
- **sensitivity 4**
  - `phoneme_label`: ɑ
  - `feature`: f1_lobanov
  - `n_l1_main`: 608
  - `n_l2_main`: 399
  - `difference_l2_minus_l1_main`: 0.074
  - `p_fdr_bh_main`: 0.007
  - `significant_fdr_0_05_main`: True
  - `n_l1_range_filtered`: 608
  - `n_l2_range_filtered`: 398
  - `difference_l2_minus_l1_range_filtered`: 0.064
  - `p_fdr_bh_range_filtered`: 0.008
  - `significant_fdr_0_05_range_filtered`: True
  - `n_excluded_by_range_filter`: 1
  - `conclusion_changed`: False
  - `absolute_effect_change`: 0.009
- **sensitivity 5**
  - `phoneme_label`: a
  - `feature`: f1_lobanov
  - `n_l1_main`: 1336
  - `n_l2_main`: 1763
  - `difference_l2_minus_l1_main`: -0.003
  - `p_fdr_bh_main`: 0.074
  - `significant_fdr_0_05_main`: False
  - `n_l1_range_filtered`: 1336
  - `n_l2_range_filtered`: 1755
  - `difference_l2_minus_l1_range_filtered`: -0.013
  - `p_fdr_bh_range_filtered`: 0.107
  - `significant_fdr_0_05_range_filtered`: False
  - `n_excluded_by_range_filter`: 8
  - `conclusion_changed`: False
  - `absolute_effect_change`: 0.009
- **sensitivity 6**
  - `phoneme_label`: i
  - `feature`: f2_lobanov
  - `n_l1_main`: 910
  - `n_l2_main`: 962
  - `difference_l2_minus_l1_main`: 0.032
  - `p_fdr_bh_main`: 0.000
  - `significant_fdr_0_05_main`: True
  - `n_l1_range_filtered`: 910
  - `n_l2_range_filtered`: 961
  - `difference_l2_minus_l1_range_filtered`: 0.035
  - `p_fdr_bh_range_filtered`: 0.000
  - `significant_fdr_0_05_range_filtered`: True
  - `n_excluded_by_range_filter`: 1
  - `conclusion_changed`: False
  - `absolute_effect_change`: 0.004

## Descriptive Statistics

The largest 2D between-phoneme variance ratios were observed for:

- **projection 1**
  - `model`: xlsr
  - `layer`: 18
  - `method`: pca
  - `between_phoneme_variance_ratio_2d`: 0.849
  - `within_phoneme_cosine_mean`: 0.825
  - `between_phoneme_cosine_mean`: 0.150
  - `within_between_similarity_ratio`: 5.500
  - `similarity_sample_pairs`: 20000
- **projection 2**
  - `model`: whisper
  - `layer`: 6
  - `method`: pca
  - `between_phoneme_variance_ratio_2d`: 0.818
  - `within_phoneme_cosine_mean`: 0.770
  - `between_phoneme_cosine_mean`: 0.104
  - `within_between_similarity_ratio`: 7.424
  - `similarity_sample_pairs`: 20000
- **projection 3**
  - `model`: xlsr
  - `layer`: 18
  - `method`: umap
  - `between_phoneme_variance_ratio_2d`: 0.816
  - `within_phoneme_cosine_mean`: 0.933
  - `between_phoneme_cosine_mean`: 0.765
  - `within_between_similarity_ratio`: 1.220
  - `similarity_sample_pairs`: 20000
- **projection 4**
  - `model`: xlsr
  - `layer`: 3
  - `method`: pca
  - `between_phoneme_variance_ratio_2d`: 0.805
  - `within_phoneme_cosine_mean`: 0.804
  - `between_phoneme_cosine_mean`: 0.120
  - `within_between_similarity_ratio`: 6.676
  - `similarity_sample_pairs`: 20000
- **projection 5**
  - `model`: xlsr
  - `layer`: 3
  - `method`: umap
  - `between_phoneme_variance_ratio_2d`: 0.769
  - `within_phoneme_cosine_mean`: 0.979
  - `between_phoneme_cosine_mean`: 0.878
  - `within_between_similarity_ratio`: 1.116
  - `similarity_sample_pairs`: 20000

Sampled RSM correlations:

- **rsm 1**
  - `representation_a`: acoustic
  - `representation_b`: whisper_layer20
  - `spearman_mantel_r`: 0.189
  - `p_value_asymptotic`: 0.000
  - `n_tokens_sampled`: 3000
  - `n_pairwise_values`: 4498500
- **rsm 2**
  - `representation_a`: acoustic
  - `representation_b`: xlsr_layer18
  - `spearman_mantel_r`: 0.355
  - `p_value_asymptotic`: 0.000
  - `n_tokens_sampled`: 3000
  - `n_pairwise_values`: 4498500
- **rsm 3**
  - `representation_a`: whisper_layer20
  - `representation_b`: xlsr_layer18
  - `spearman_mantel_r`: 0.676
  - `p_value_asymptotic`: 0.000
  - `n_tokens_sampled`: 3000
  - `n_pairwise_values`: 4498500

## Statistical Tests

After BH-FDR correction, acoustic L1/L2 differences persisted for: /i/ f2, /y/ f1, /y/ f2, /u/ f1, /u/ f2, /ø/ f1, /ɛ/ f1, /ɑ/ f1, /ɑ/ f2.

Phoneme-centroid distance Mantel correlations:

- **distance 1**
  - `distance_a`: acoustic_euclidean
  - `distance_b`: acoustic_mahalanobis
  - `mantel_spearman_r`: 0.984
  - `p_value_asymptotic`: 0.000
  - `n_phoneme_pairs`: 55
- **distance 2**
  - `distance_a`: acoustic_euclidean
  - `distance_b`: whisper_cosine
  - `mantel_spearman_r`: 0.674
  - `p_value_asymptotic`: 0.000
  - `n_phoneme_pairs`: 55
- **distance 3**
  - `distance_a`: acoustic_euclidean
  - `distance_b`: xlsr_cosine
  - `mantel_spearman_r`: 0.678
  - `p_value_asymptotic`: 0.000
  - `n_phoneme_pairs`: 55
- **distance 4**
  - `distance_a`: acoustic_mahalanobis
  - `distance_b`: whisper_cosine
  - `mantel_spearman_r`: 0.672
  - `p_value_asymptotic`: 0.000
  - `n_phoneme_pairs`: 55
- **distance 5**
  - `distance_a`: acoustic_mahalanobis
  - `distance_b`: xlsr_cosine
  - `mantel_spearman_r`: 0.676
  - `p_value_asymptotic`: 0.000
  - `n_phoneme_pairs`: 55
- **distance 6**
  - `distance_a`: whisper_cosine
  - `distance_b`: xlsr_cosine
  - `mantel_spearman_r`: 0.842
  - `p_value_asymptotic`: 0.000
  - `n_phoneme_pairs`: 55

Residual gender-effect tests after Lobanov normalisation found 0 FDR-significant contrasts at alpha = 0.05.

## Midpoint vs Trajectory

For long vowels with available 25% and 75% formant measurements, midpoint-based L1/L2 conclusions were compared with trajectory-mean conclusions on the same subset of tokens. 6 of 22 FDR-significance decisions changed.

Largest changes in L2-L1 effect size:

- **trajectory 1**
  - `phoneme_label`: ə
  - `axis`: F1
  - `n_l1`: 16
  - `n_l2`: 60
  - `midpoint_difference_l2_minus_l1`: 0.557
  - `trajectory_difference_l2_minus_l1`: 0.223
  - `absolute_difference_change`: 0.334
  - `midpoint_p_fdr_bh`: 0.085
  - `trajectory_p_fdr_bh`: 0.574
  - `midpoint_significant_fdr_0_05`: False
  - `trajectory_significant_fdr_0_05`: False
  - `conclusion_changed`: False
- **trajectory 2**
  - `phoneme_label`: a
  - `axis`: F1
  - `n_l1`: 887
  - `n_l2`: 1562
  - `midpoint_difference_l2_minus_l1`: -0.018
  - `trajectory_difference_l2_minus_l1`: -0.299
  - `absolute_difference_change`: 0.281
  - `midpoint_p_fdr_bh`: 1.000
  - `trajectory_p_fdr_bh`: 0.000
  - `midpoint_significant_fdr_0_05`: False
  - `trajectory_significant_fdr_0_05`: True
  - `conclusion_changed`: True
- **trajectory 3**
  - `phoneme_label`: ɑ
  - `axis`: F1
  - `n_l1`: 15
  - `n_l2`: 106
  - `midpoint_difference_l2_minus_l1`: 0.128
  - `trajectory_difference_l2_minus_l1`: 0.304
  - `absolute_difference_change`: 0.176
  - `midpoint_p_fdr_bh`: 0.811
  - `trajectory_p_fdr_bh`: 0.043
  - `midpoint_significant_fdr_0_05`: False
  - `trajectory_significant_fdr_0_05`: True
  - `conclusion_changed`: True
- **trajectory 4**
  - `phoneme_label`: ə
  - `axis`: F2
  - `n_l1`: 16
  - `n_l2`: 60
  - `midpoint_difference_l2_minus_l1`: 0.023
  - `trajectory_difference_l2_minus_l1`: -0.094
  - `absolute_difference_change`: 0.117
  - `midpoint_p_fdr_bh`: 0.866
  - `trajectory_p_fdr_bh`: 0.487
  - `midpoint_significant_fdr_0_05`: False
  - `trajectory_significant_fdr_0_05`: False
  - `conclusion_changed`: False
- **trajectory 5**
  - `phoneme_label`: u
  - `axis`: F1
  - `n_l1`: 210
  - `n_l2`: 308
  - `midpoint_difference_l2_minus_l1`: -0.180
  - `trajectory_difference_l2_minus_l1`: -0.067
  - `absolute_difference_change`: 0.113
  - `midpoint_p_fdr_bh`: 0.000
  - `trajectory_p_fdr_bh`: 0.000
  - `midpoint_significant_fdr_0_05`: True
  - `trajectory_significant_fdr_0_05`: True
  - `conclusion_changed`: False

Nearest-centroid classification:

- **classifier 1**
  - `representation`: acoustic
  - `overall_accuracy`: 0.687
  - `macro_f1`: 0.467
  - `n_tokens`: 8277
  - `group`: overall
- **classifier 2**
  - `representation`: acoustic
  - `overall_accuracy`: 0.710
  - `macro_f1`: 0.502
  - `n_tokens`: 3913
  - `group`: L1
- **classifier 3**
  - `representation`: acoustic
  - `overall_accuracy`: 0.667
  - `macro_f1`: 0.428
  - `n_tokens`: 4364
  - `group`: L2
- **classifier 4**
  - `representation`: whisper_layer20
  - `overall_accuracy`: 0.823
  - `macro_f1`: 0.735
  - `n_tokens`: 8277
  - `group`: overall
- **classifier 5**
  - `representation`: whisper_layer20
  - `overall_accuracy`: 0.870
  - `macro_f1`: 0.772
  - `n_tokens`: 3913
  - `group`: L1
- **classifier 6**
  - `representation`: whisper_layer20
  - `overall_accuracy`: 0.780
  - `macro_f1`: 0.708
  - `n_tokens`: 4364
  - `group`: L2
- **classifier 7**
  - `representation`: xlsr_layer18
  - `overall_accuracy`: 0.790
  - `macro_f1`: 0.708
  - `n_tokens`: 8277
  - `group`: overall
- **classifier 8**
  - `representation`: xlsr_layer18
  - `overall_accuracy`: 0.799
  - `macro_f1`: 0.688
  - `n_tokens`: 3913
  - `group`: L1
- **classifier 9**
  - `representation`: xlsr_layer18
  - `overall_accuracy`: 0.782
  - `macro_f1`: 0.721
  - `n_tokens`: 4364
  - `group`: L2

The best phoneme identification accuracy was obtained by `whisper_layer20` with accuracy 0.823.

## Mixed-Effects Models

ICC for /a/:

- **icc 1**
  - `phoneme_label`: a
  - `representation`: acoustic
  - `response`: f1_lobanov
  - `n_obs`: 3099
  - `speaker_random_variance`: 0.023
  - `residual_variance`: 0.580
  - `icc`: 0.038
- **icc 2**
  - `phoneme_label`: a
  - `representation`: whisper_layer20
  - `response`: whisper_pc1
  - `n_obs`: 3099
  - `speaker_random_variance`: 0.567
  - `residual_variance`: 12.774
  - `icc`: 0.042
- **icc 3**
  - `phoneme_label`: a
  - `representation`: xlsr_layer18
  - `response`: xlsr_pc1
  - `n_obs`: 3099
  - `speaker_random_variance`: 19.351
  - `residual_variance`: 602.057
  - `icc`: 0.031

Marginal and conditional R2 summary:

- **r2 1**
  - `representation`: acoustic
  - `mean_marginal_r2`: 0.430
  - `max_marginal_r2`: 0.600
  - `mean_conditional_r2`: 0.430
  - `n_responses`: 2
- **r2 2**
  - `representation`: xlsr_layer18
  - `mean_marginal_r2`: 0.410
  - `max_marginal_r2`: 0.694
  - `mean_conditional_r2`: 0.425
  - `n_responses`: 5
- **r2 3**
  - `representation`: whisper_layer20
  - `mean_marginal_r2`: 0.310
  - `max_marginal_r2`: 0.534
  - `mean_conditional_r2`: 0.316
  - `n_responses`: 5

The highest mean marginal R2 was obtained by `acoustic` with mean marginal R2 0.430.

## Confidence Intervals and ROPE

ROPE classification counts:

- acoustic_f1: Indeterminate = 8
- acoustic_f1: Insufficient data = 1
- acoustic_f1: Non-equivalent = 2
- whisper_layer20: Equivalent = 10
- whisper_layer20: Insufficient data = 1
- xlsr_layer18: Equivalent = 6
- xlsr_layer18: Indeterminate = 4
- xlsr_layer18: Insufficient data = 1

Acoustic F1 used the default [-20, +20] Hz ROPE. Neural ROPEs used the intra-speaker cosine-distance noise floor. The acoustic CI implementation is a speaker-level interval approximation rather than a strict profile-likelihood interval, so ROPE classifications should be interpreted as transparent robustness summaries rather than exact profile-likelihood decisions.

## Hierarchical Clustering

Vowel clustering:

- **vowel 1**
  - `analysis`: vowels
  - `representation`: acoustic
  - `metric`: euclidean
  - `linkage`: ward
  - `best_k_silhouette`: 3
  - `best_silhouette`: 0.458
  - `ari_height_k3`: 0.311
  - `ari_front_back_central_k4`: 0.120
  - `n_phonemes`: 11
- **vowel 2**
  - `analysis`: vowels
  - `representation`: whisper_layer20
  - `metric`: cosine
  - `linkage`: average
  - `best_k_silhouette`: 6
  - `best_silhouette`: 0.296
  - `ari_height_k3`: 0.499
  - `ari_front_back_central_k4`: -0.005
  - `n_phonemes`: 11
- **vowel 3**
  - `analysis`: vowels
  - `representation`: xlsr_layer18
  - `metric`: cosine
  - `linkage`: average
  - `best_k_silhouette`: 2
  - `best_silhouette`: 0.309
  - `ari_height_k3`: 0.115
  - `ari_front_back_central_k4`: 0.305
  - `n_phonemes`: 11

Consonant/vowel clustering:

- **cv 1**
  - `analysis`: consonant_vowel
  - `representation`: acoustic
  - `metric`: euclidean
  - `linkage`: ward
  - `k`: 2
  - `ari_consonant_vowel`: 0.487
  - `silhouette`: 0.402
  - `n_phonemes`: 21
- **cv 2**
  - `analysis`: consonant_vowel
  - `representation`: whisper_layer20
  - `metric`: cosine
  - `linkage`: average
  - `k`: 2
  - `ari_consonant_vowel`: -0.024
  - `silhouette`: 0.248
  - `n_phonemes`: 21
- **cv 3**
  - `analysis`: consonant_vowel
  - `representation`: xlsr_layer18
  - `metric`: cosine
  - `linkage`: average
  - `k`: 2
  - `ari_consonant_vowel`: 0.638
  - `silhouette`: 0.306
  - `n_phonemes`: 21

Speaker clustering:

- **speaker 1**
  - `analysis`: speakers
  - `representation`: acoustic
  - `metric`: euclidean
  - `linkage`: ward
  - `best_k_silhouette`: 3
  - `best_silhouette`: 0.117
  - `ari_l1_l2_k2`: 0.085
  - `ari_gender_k2`: 0.296
  - `n_speakers`: 19
- **speaker 2**
  - `analysis`: speakers
  - `representation`: whisper_layer20
  - `metric`: cosine
  - `linkage`: average
  - `best_k_silhouette`: 2
  - `best_silhouette`: 0.254
  - `ari_l1_l2_k2`: 1.000
  - `ari_gender_k2`: 0.296
  - `n_speakers`: 19
- **speaker 3**
  - `analysis`: speakers
  - `representation`: xlsr_layer18
  - `metric`: cosine
  - `linkage`: average
  - `best_k_silhouette`: 5
  - `best_silhouette`: 0.214
  - `ari_l1_l2_k2`: 0.296
  - `ari_gender_k2`: 1.000
  - `n_speakers`: 19

Best vowel-height recovery: `whisper_layer20` with ARI 0.499.
Best front/back/central recovery: `xlsr_layer18` with ARI 0.305.
Best consonant/vowel boundary recovery: `xlsr_layer18` with ARI 0.638.

Systematically difficult consonants included /l/, /n/, /ʁ/, consistent with sonorants behaving as acoustically intermediate categories.

## Answers to the 16 Questions

1. PCA and UMAP answered different parts of the projection question. PCA gave a linear, variance-preserving baseline whose 2D between-phoneme variance ratios were high for several layers, especially XLS-R layer 18. UMAP was more useful as a neighbourhood visualisation, but it is less directly interpretable because distances and axes are not linear acoustic or articulatory dimensions.
2. The most variable vowels in the acoustic summaries were concentrated among low/back and high rounded categories, especially /ɑ/, /u/, /y/, and /ɛ/. This partly matches the neural spaces, but the neural embeddings also preserve contextual, speaker, and language-background information, so acoustic dispersion and neural dispersion are related rather than identical.
3. The UMAP plots recover phoneme neighbourhoods, especially for XLS-R, but they do not perfectly reconstruct the IPA vowel trapezoid. This is expected because the embeddings are learned from speech signal prediction/recognition objectives rather than from explicit F1/F2 geometry.
4. RSM/Mantel correlations showed stronger acoustic alignment for XLS-R than Whisper: acoustic-Whisper r = 0.189, acoustic-XLS-R r = 0.355, and Whisper-XLS-R r = 0.676 on the sampled token similarity matrices. The two neural models are therefore more similar to each other than either is to the acoustic F1/F2 space.
5. Acoustic L1/L2 tests found 9 FDR-significant contrasts across /i/, /y/, /u/, /ø/, /ɛ/, and /ɑ/. Whisper and XLS-R each found 9 significant vowel-level neural contrasts; /œ/ had insufficient data and /ə/ was not significant. On long-vowel subsets, replacing midpoint formants with 25%/75% trajectory means changed 6 of 22 FDR decisions, so trajectory information matters for some vowels.
6. Acoustic Euclidean and Mahalanobis centroid distances were nearly identical in rank structure, r = 0.984. Acoustic-neural centroid correlations were moderate and similar for Whisper and XLS-R, r = 0.674 and r = 0.678 respectively for Euclidean acoustic distance, while Whisper-XLS-R centroid distance correlation was stronger, r = 0.842.
7. Leave-one-speaker-out nearest-centroid classification was best for Whisper layer 20, with overall accuracy 0.823 and macro-F1 0.735. XLS-R layer 18 followed with accuracy 0.790, and acoustic F1/F2 reached 0.687. McNemar tests showed all pairwise classifier differences were significant.
8. For /a/, speaker-specificity was modest in all representations: ICC = 0.038 for acoustic F1, 0.042 for Whisper PC1, and 0.031 for XLS-R PC1. This indicates that most /a/ variation is token-level or residual rather than speaker-level in these fitted models.
9. The L1 x Gender interaction was not significant for acoustic F1/F2 or Whisper PCs 1-5. In XLS-R, PC2 showed a significant interaction, estimate = 6.649, p = 0.031, so gender-related structure appears in one neural dimension but is not a broad cross-representation pattern.
10. Mixed-effects model summaries gave the highest mean marginal R2 to acoustic features, 0.430, followed by XLS-R layer 18 at 0.410 and Whisper layer 20 at 0.310. Acoustic F1/F2 therefore captured the fixed phonetic predictors most directly, while neural PCs retained additional non-phoneme variance.
11. No statistically significant acoustic F1 contrast had a 95% speaker-level CI fully inside the acoustic ROPE. The current acoustic ROPE result is therefore not a case of "statistically significant but practically equivalent"; instead, several contrasts were indeterminate and two acoustic F1 contrasts were non-equivalent.
12. ROPE classifications differed by representation: acoustic F1 had 2 non-equivalent, 8 indeterminate, and 1 insufficient contrast; Whisper had 10 equivalent and 1 insufficient contrast; XLS-R had 6 equivalent, 4 indeterminate, and 1 insufficient contrast. Whisper therefore produced the strongest practical-equivalence pattern under the chosen neural ROPE.
13. Acoustic and neural ROPE disagreement is interpretable because the acoustic tests target narrow formant shifts, while neural cosine distances average broader phonetic, contextual, and speaker-conditioned information. A small formant change can be robust acoustically while remaining within the neural within-speaker noise floor.
14. Clustering results separated different phonological structures by representation. Whisper layer 20 best recovered vowel height, ARI = 0.499; XLS-R layer 18 best recovered front/back/central grouping, ARI = 0.305; acoustic F1/F2 gave the clearest silhouette for vowel clustering but a lower height ARI than Whisper.
15. Speaker clustering showed strong representation-specific biases. Whisper layer 20 perfectly separated L1/L2 at k = 2, ARI = 1.000, whereas XLS-R layer 18 perfectly separated gender, ARI = 1.000. These results warn that neural embeddings encode speaker and group information in addition to phonetic identity.
16. Consonant/vowel clustering was best for XLS-R layer 18, ARI = 0.638, followed by acoustic features at 0.487; Whisper layer 20 was near chance, ARI = -0.024. The systematically difficult consonants were /l/, /n/, and /ʁ/, consistent with sonorants and rhotics behaving as acoustically intermediate categories.

## Reproducibility

Run the full workflow with:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake --cores 1
```

Dry-run the workflow with:

```bash
XDG_CACHE_HOME=/private/tmp/snakemake-cache .venv/bin/snakemake -n --cores 1
```

The workflow parameters are stored in `config/config.yaml`.

## References

- Lobanov, B. M. (1971). Classification of Russian vowels spoken by different speakers.
- Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations.
- Conneau, A. et al. (2021). Unsupervised cross-lingual representation learning for speech recognition.
- Radford, A. et al. (2023). Robust speech recognition via large-scale weak supervision.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection.
- Bates, D., Maechler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects models using lme4.
- Kruschke, J. K. (2018). Rejecting or accepting parameter values in Bayesian estimation.

\clearpage
\begingroup
\setlength{\intextsep}{0.25em}
\setlength{\textfloatsep}{0.25em}
\setlength{\floatsep}{0.25em}
\setlength{\abovecaptionskip}{0.15em}
\setlength{\belowcaptionskip}{0.15em}

## Figures

Figures are placed at the end to avoid interrupting long result listings in the PDF layout.

\Needspace{0.42\textheight}
**Figure 1. F1 by vowel and L1/L2 group**

![](figures/f1_lobanov_boxplot_by_group.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 2. F2 by vowel and L1/L2 group**

![](figures/f2_lobanov_boxplot_by_group.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 3. Lobanov vowel chart**

![](figures/vowel_chart_lobanov.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 4. XLS-R layer 18 UMAP projection**

![](figures/xlsr_layer_18_umap2.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 5. Midpoint vs 25/75 trajectory mean**

![](figures/trajectory_midpoint_vs_trajectory.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 6. Whisper layer 20 confusion matrix**

![](figures/confusion_whisper_layer20.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 7. Acoustic F1 ROPE forest plot**

![](figures/forest_acoustic_f1_rope.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 8. XLS-R consonant/vowel dendrogram**

![](figures/dendrogram_consonant_vowel_xlsr_layer18.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 9. Whisper layer 20 speaker dendrogram**

![](figures/dendrogram_speakers_whisper_layer20.png){width=65%}
\vspace{-0.8em}

\Needspace{0.42\textheight}
**Figure 10. XLS-R layer 18 speaker dendrogram**

![](figures/dendrogram_speakers_xlsr_layer18.png){width=65%}

\endgroup
