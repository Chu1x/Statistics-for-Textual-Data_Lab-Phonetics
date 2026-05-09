# Acoustic and Neural Representations in a Phonetically Aligned Corpus

## Overview

This project asks how far acoustic measurements and neural speech representations tell the same story about phonetic structure in the Russian-French Interference Corpus. I treated the corpus as a phoneme-level dataset: TextGrid intervals were converted into token rows, acoustic descriptors were measured at aligned intervals, Whisper and XLS-R hidden states were averaged over the same intervals, and the resulting representations were compared with descriptive, inferential, and clustering analyses.

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

After parsing, the working dataset contained 22919 phoneme tokens from 19 speakers and 78 sentence IDs. Acoustic extraction used midpoint formants and, for vowels longer than 80 ms, additional 25% and 75% formant measurements. Formants were Lobanov-normalised within speaker, using vowel tokens only, so that speaker differences in vocal-tract scale did not dominate the vowel comparisons.

For the neural analyses, I extracted `openai/whisper-medium` layers 6, 20 and `facebook/wav2vec2-large-xlsr-53` layers 3, 9, 18. Each selected layer was reduced to 50 principal components before downstream modelling. Permutation tests used 5000 permutations, and bootstrap/ROPE summaries used 2000 resamples where applicable.

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

I did not impute missing acoustic values. Instead, each analysis uses the available tokens for the feature being tested, and the corresponding result tables report the relevant sample sizes. Rough-range values were kept in the main analysis but tracked as quality diagnostics; among oral vowels, the flagged counts were F1 = 16, F2 = 1, and f0 = 32.

As a robustness check, I reran the acoustic L1/L2 tests after excluding rough-range F1/F2 values. This changed 0 of 22 FDR-significance decisions. The affected or filtered contrasts were:

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

1. PCA and UMAP were useful in different ways. PCA gave a stable linear baseline, and the strongest 2D separation appeared in the higher XLS-R layers. UMAP made local phoneme neighbourhoods easier to see, but I do not interpret its axes as acoustic dimensions.
2. The acoustic summaries point to /ɑ/, /u/, /y/, and /ɛ/ as especially variable categories. The neural spaces show related structure, but not a one-to-one copy of the formant space, which is unsurprising because the embeddings also carry context and speaker information.
3. The UMAP plots show clear phoneme neighbourhoods, particularly for XLS-R, but they do not reproduce the IPA vowel trapezoid exactly. In other words, the neural space is phonetically organised, but not simply an F1/F2 chart in another form.
4. The RSM results support the same interpretation. Acoustic-XLS-R similarity was higher than acoustic-Whisper similarity, r = 0.355 versus r = 0.189, while Whisper and XLS-R were more similar to each other, r = 0.676.
5. In the acoustic tests, 9 contrasts survived FDR correction, involving /i/, /y/, /u/, /ø/, /ɛ/, and /ɑ/. Whisper and XLS-R also showed widespread L1/L2 separation, with 9 significant vowel-level neural contrasts each. The main exception was /œ/, where the corpus simply does not provide enough data, and /ə/, which was not significant.
6. Euclidean and Mahalanobis acoustic distances gave almost the same vowel-distance ranking, r = 0.984. The acoustic-neural correlations were moderate and very similar for Whisper and XLS-R, while the two neural distance matrices were more closely aligned with each other, r = 0.842.
7. Whisper layer 20 was the best phoneme classifier in the leave-one-speaker-out test, with accuracy 0.823 and macro-F1 0.735. XLS-R layer 18 followed at 0.790 accuracy, and the acoustic baseline reached 0.687. The McNemar comparisons suggest that these differences are not just noise.
8. The /a/ models show only modest speaker-specificity: ICC = 0.038 for acoustic F1, 0.042 for Whisper PC1, and 0.031 for XLS-R PC1. Most of the remaining variation is therefore token-level or residual rather than stable between-speaker variation.
9. I did not find a broad L1-by-gender pattern. The interaction was not significant for acoustic F1/F2 or for Whisper PCs 1-5. The one exception was XLS-R PC2, estimate = 6.649, p = 0.031, which suggests a local gender-related effect in that neural dimension.
10. The mixed-effects models gave the highest mean marginal R2 to the acoustic features, 0.430. XLS-R layer 18 was close behind at 0.410, while Whisper layer 20 was lower at 0.310. This fits the idea that formants are more directly tied to the fixed vowel predictors, whereas neural PCs contain additional information.
11. I did not find a case where a statistically significant acoustic F1 effect was also clearly inside the acoustic ROPE. The acoustic ROPE results are therefore better read as a mix of non-equivalent and indeterminate contrasts, rather than as evidence for practically negligible formant differences.
12. The ROPE results differ quite sharply across representations. Acoustic F1 produced 2 non-equivalent, 8 indeterminate, and 1 insufficient contrast. Whisper produced 10 equivalent contrasts and 1 insufficient case. XLS-R sat between them, with 6 equivalent, 4 indeterminate, and 1 insufficient contrast.
13. This disagreement is meaningful rather than contradictory. Formants isolate narrow articulatory differences, while neural cosine distances pool phonetic detail with context, speaker, and model-specific information. A contrast can therefore be acoustically robust while still falling within the neural noise floor.
14. The clustering results split the phonological structure across models. Whisper layer 20 recovered vowel height best, ARI = 0.499. XLS-R layer 18 recovered front/back/central grouping best, ARI = 0.305. The acoustic representation had the clearest vowel-clustering silhouette, but not the strongest height ARI.
15. Speaker clustering revealed strong group information in the neural spaces. Whisper layer 20 separated L1/L2 perfectly at k = 2, ARI = 1.000, while XLS-R layer 18 separated gender perfectly, ARI = 1.000. This is useful, but it also means the neural embeddings should not be treated as purely phonetic.
16. For consonant/vowel clustering, XLS-R layer 18 performed best, ARI = 0.638, followed by the acoustic representation at 0.487. Whisper layer 20 was near chance, ARI = -0.024. The recurrently difficult consonants were /l/, /n/, and /ʁ/, which makes phonetic sense because sonorants and rhotics sit closer to vowels than obstruents do.

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

Figures are grouped here so that they do not interrupt the long result listings in the main text.

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
