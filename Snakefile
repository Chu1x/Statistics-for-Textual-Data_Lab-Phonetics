configfile: "config/config.yaml"

PYTHON = config["python"]
LOCAL_FILES_FLAG = "--local-files-only" if config["runtime"].get("local_files_only", False) else ""
WHISPER_LAYERS = " ".join(str(layer) for layer in config["layers"]["whisper"])
XLSR_LAYERS = " ".join(str(layer) for layer in config["layers"]["xlsr"])


rule all:
    input:
        "data/phoneme_tokens.csv",
        "data/features_acoustic.csv",
        "data/features_acoustic_norm.csv",
        "data/features_whisper.npz",
        "data/features_whisper_pca.npz",
        "data/features_xlsr.npz",
        "data/features_xlsr_pca.npz",
        "results/tables/acoustic_missingness_by_phoneme_group.csv",
        "results/report.md",


rule parse_corpus:
    input:
        metadata=config["paths"]["metadata"],
    output:
        "data/phoneme_tokens.csv",
    params:
        corpus_root=config["paths"]["corpus_root"],
    shell:
        "{PYTHON} src/parse_corpus.py --corpus-root '{params.corpus_root}' --metadata '{input.metadata}' --output '{output}'"


rule extract_acoustics:
    input:
        "data/phoneme_tokens.csv",
    output:
        features="data/features_acoustic.csv",
        missingness="data/features_acoustic_missingness.csv",
    shell:
        "{PYTHON} src/extract_acoustics.py --tokens '{input}' --output '{output.features}' --missingness-output '{output.missingness}'"


rule extract_neural_whisper:
    input:
        "data/phoneme_tokens.csv",
    output:
        embeddings="data/features_whisper.npz",
        index="data/features_whisper_index.csv",
    params:
        model=config["models"]["whisper"],
        layers=WHISPER_LAYERS,
        local_files_flag=LOCAL_FILES_FLAG,
    shell:
        "{PYTHON} src/extract_neural_whisper.py --tokens '{input}' --output '{output.embeddings}' --index-output '{output.index}' --model '{params.model}' --layers {params.layers} {params.local_files_flag}"


rule extract_neural_xlsr:
    input:
        "data/phoneme_tokens.csv",
    output:
        embeddings="data/features_xlsr.npz",
        index="data/features_xlsr_index.csv",
    params:
        model=config["models"]["xlsr"],
        layers=XLSR_LAYERS,
        local_files_flag=LOCAL_FILES_FLAG,
    shell:
        "{PYTHON} src/extract_neural_xlsr.py --tokens '{input}' --output '{output.embeddings}' --index-output '{output.index}' --model '{params.model}' --layers {params.layers} {params.local_files_flag}"


rule normalise:
    input:
        acoustic="data/features_acoustic.csv",
        whisper="data/features_whisper.npz",
        xlsr="data/features_xlsr.npz",
    output:
        acoustic_norm="data/features_acoustic_norm.csv",
        lobanov_stats="data/lobanov_speaker_stats.csv",
        whisper_pca="data/features_whisper_pca.npz",
        xlsr_pca="data/features_xlsr_pca.npz",
    params:
        pca_components=config["runtime"]["pca_components"],
        random_state=config["runtime"]["random_state"],
    shell:
        "{PYTHON} src/normalise.py --acoustic '{input.acoustic}' --whisper '{input.whisper}' --xlsr '{input.xlsr}' --acoustic-output '{output.acoustic_norm}' --lobanov-stats-output '{output.lobanov_stats}' --whisper-output '{output.whisper_pca}' --xlsr-output '{output.xlsr_pca}' --pca-components {params.pca_components} --random-state {params.random_state}"


rule analyse_descriptive:
    input:
        tokens="data/phoneme_tokens.csv",
        acoustic="data/features_acoustic_norm.csv",
        whisper_pca="data/features_whisper_pca.npz",
        xlsr_pca="data/features_xlsr_pca.npz",
    output:
        "results/tables/acoustic_vowel_descriptives.csv",
        "results/tables/neural_projection_metrics.csv",
        "results/tables/rsm_mantel_sample.csv",
        "results/figures/vowel_chart_lobanov.png",
    params:
        random_state=config["runtime"]["random_state"],
    shell:
        "{PYTHON} src/analyse_descriptive.py --tokens '{input.tokens}' --acoustic '{input.acoustic}' --whisper-pca '{input.whisper_pca}' --xlsr-pca '{input.xlsr_pca}' --random-state {params.random_state}"


rule analyse_statistical:
    input:
        acoustic="data/features_acoustic_norm.csv",
        whisper_pca="data/features_whisper_pca.npz",
        xlsr_pca="data/features_xlsr_pca.npz",
    output:
        "results/tables/acoustic_l1_l2_tests.csv",
        "results/tables/gender_residual_tests.csv",
        "results/tables/distance_matrix_acoustic_mahalanobis.csv",
        "results/tables/neural_l1_l2_permutation_whisper_layer20.csv",
        "results/tables/neural_l1_l2_permutation_xlsr_layer18.csv",
        "results/tables/phoneme_identification_metrics.csv",
        "results/figures/confusion_acoustic.png",
    params:
        permutations=config["runtime"]["permutations"],
        random_state=config["runtime"]["random_state"],
    shell:
        "{PYTHON} src/analyse_statistical.py --acoustic '{input.acoustic}' --whisper-pca '{input.whisper_pca}' --xlsr-pca '{input.xlsr_pca}' --permutations {params.permutations} --random-state {params.random_state}"


rule analyse_trajectory:
    input:
        acoustic="data/features_acoustic_norm.csv",
    output:
        "results/tables/trajectory_l1_l2_tests.csv",
        "results/tables/trajectory_midpoint_comparison.csv",
        "results/figures/trajectory_midpoint_vs_trajectory.png",
    shell:
        "{PYTHON} src/analyse_trajectory.py --acoustic '{input.acoustic}'"


rule analyse_quality:
    input:
        acoustic="data/features_acoustic_norm.csv",
    output:
        "results/tables/acoustic_quality_flags.csv",
    shell:
        "{PYTHON} src/analyse_quality.py --acoustic '{input.acoustic}' --output '{output}'"


rule analyse_sensitivity:
    input:
        acoustic="data/features_acoustic_norm.csv",
        acoustic_tests="results/tables/acoustic_l1_l2_tests.csv",
    output:
        filtered="results/tables/acoustic_l1_l2_tests_range_filtered.csv",
        comparison="results/tables/acoustic_l1_l2_sensitivity_range_filter.csv",
    shell:
        "{PYTHON} src/analyse_sensitivity.py --acoustic '{input.acoustic}' --main-tests '{input.acoustic_tests}' --filtered-output '{output.filtered}' --comparison-output '{output.comparison}'"


rule analyse_mixed_models:
    input:
        acoustic="data/features_acoustic_norm.csv",
        whisper_pca="data/features_whisper_pca.npz",
        xlsr_pca="data/features_xlsr_pca.npz",
    output:
        "results/tables/mixed_model_comparisons.csv",
        "results/tables/mixed_model_fixed_effects.csv",
        "results/tables/mixed_model_icc_a.csv",
        "results/tables/mixed_model_representation_r2_summary.csv",
        "results/tables/mixed_model_random_slope_note.csv",
    shell:
        "{PYTHON} src/analyse_mixed_models.py --acoustic '{input.acoustic}' --whisper-pca '{input.whisper_pca}' --xlsr-pca '{input.xlsr_pca}'"


rule analyse_rope:
    input:
        acoustic="data/features_acoustic_norm.csv",
        whisper_pca="data/features_whisper_pca.npz",
        xlsr_pca="data/features_xlsr_pca.npz",
        acoustic_tests="results/tables/acoustic_l1_l2_tests.csv",
    output:
        "results/tables/rope_acoustic_contrasts.csv",
        "results/tables/rope_neural_contrasts_whisper_layer20.csv",
        "results/tables/rope_neural_contrasts_xlsr_layer18.csv",
        "results/tables/rope_summary.csv",
        "results/figures/forest_acoustic_f1_rope.png",
    params:
        bootstrap=config["runtime"]["bootstrap"],
        random_state=config["runtime"]["random_state"],
    shell:
        "{PYTHON} src/analyse_rope.py --acoustic '{input.acoustic}' --whisper-pca '{input.whisper_pca}' --xlsr-pca '{input.xlsr_pca}' --bootstrap {params.bootstrap} --random-state {params.random_state}"


rule analyse_clustering:
    input:
        acoustic="data/features_acoustic_norm.csv",
        whisper_pca="data/features_whisper_pca.npz",
        xlsr_pca="data/features_xlsr_pca.npz",
    output:
        "results/tables/clustering_vowel_ari.csv",
        "results/tables/clustering_consonant_vowel_ari.csv",
        "results/tables/clustering_speaker_ari.csv",
        "results/figures/dendrogram_vowels_acoustic.png",
    shell:
        "{PYTHON} src/analyse_clustering.py --acoustic '{input.acoustic}' --whisper-pca '{input.whisper_pca}' --xlsr-pca '{input.xlsr_pca}'"


rule write_report:
    input:
        tokens="data/phoneme_tokens.csv",
        acoustic_norm="data/features_acoustic_norm.csv",
        acoustic_missingness="data/features_acoustic_missingness.csv",
        quality="results/tables/acoustic_quality_flags.csv",
        sensitivity="results/tables/acoustic_l1_l2_sensitivity_range_filter.csv",
        descriptive="results/tables/neural_projection_metrics.csv",
        statistics="results/tables/phoneme_identification_metrics.csv",
        trajectory="results/tables/trajectory_midpoint_comparison.csv",
        mixed="results/tables/mixed_model_representation_r2_summary.csv",
        rope="results/tables/rope_summary.csv",
        clustering="results/tables/clustering_vowel_ari.csv",
    output:
        report="results/report.md",
        missingness_by_group="results/tables/acoustic_missingness_by_phoneme_group.csv",
    shell:
        "{PYTHON} src/write_report.py --output '{output.report}'"
