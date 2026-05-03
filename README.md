# GroupAffect-4 — Analysis Code

Reproduction code for all analyses reported in:

> **GroupAffect-4: A Multimodal Dataset of Four-Person Collaborative Interaction**

Scripts are self-contained Python files organised by pipeline stage. No package installation beyond `requirements.txt` is needed.

---

## Setup

```bash
python -m pip install -r requirements.txt
```

**Python ≥ 3.10 required.**

---

## Step 0 — Download the Dataset

```bash
python download_dataset.py --out-dir /path/to/data
```

This fetches all files from the Zenodo record and verifies MD5 checksums. Already-downloaded files are skipped automatically.

For restricted (pre-publication) records, pass a Zenodo personal access token:

```bash
python download_dataset.py --out-dir /path/to/data --token <your-token>
```

After download the data directory will have the following layout:

```
<data-root>/
├── sub-01/
│   └── ses-*/
│       ├── beh/   *_stimuli_answers.tsv
│       ├── et/
│       ├── physio/
│       └── audio/
├── sub-02/ … sub-10/
└── participants.tsv
```

Derived feature TSVs (output of `pipeline/`) land under:

```
<derived-features-dir>/
├── physio_participant_task.tsv
├── audio_participant_task.tsv
├── features_pupil_participant_task.tsv
└── …
```

Benchmark-preprocessed data under:

```
results/benchmarks/
├── preprocessed_participant_task.tsv
└── preprocessed_feature_list.txt
```

---

## Folder Structure

```
├── download_dataset.py     Step 0 — Download dataset from Zenodo
├── pipeline/               Step 1 — Feature extraction from raw BIDS data
├── analysis/               Step 2 — Dataset characterisation & statistics
├── benchmarks/             Step 3 — Benchmark baselines (paper §5)
├── figures/                Figure generation scripts
├── supplementary_analyses/ Additional analyses
├── notebooks/              Jupyter notebook (stimulus QC audit)
└── run_all_paper_analyses.py
```

---

## Step 1 — Feature Extraction (`pipeline/`)

Extract per-participant, per-task features from raw BIDS files.

| Script | What it does |
|--------|-------------|
| `run_feature_pipeline.py` | **Master runner** — runs all extractors in sequence |
| `extract_physio_features.py` | EmotiBit ECG, EDA, skin temperature (participant-task + rolling window) |
| `extract_audio_features.py` | Close-talk prosody/speech features with bleed rejection |
| `extract_eyetracking_features.py` | Fixations, saccades, gaze direction (Tobii ET) |
| `extract_pupil_features.py` | Tobii pupil dilation dynamics |
| `preprocess_features.py` | Plausibility gating, winsorisation, within-person z-score, KNN imputation |
| `build_semantic_biomarkers.py` | Composite biomarker features (cognitive load, arousal, attention, fatigue) |
| `build_participant_group_comparisons.py` | Join participant features with behavioural annotations |
| `compute_group_dynamics.py` | Dyad/group synchrony metrics from window-level tables |
| `run_preprocessing_analysis.py` | Preprocessing diagnostics and coverage report |
| `common.py` | Shared utilities (session discovery, TSV I/O, argument parsing) |

**Typical run:**

```bash
python pipeline/run_feature_pipeline.py \
  --data-root /path/to/GroupAffect-4-bids \
  --out-dir /path/to/derived_features \
  --window-s 30 --step-s 15
```

---

## Step 2 — Analysis (`analysis/`)

Produce the dataset characterisation results (tables and figures in paper §3–§4 and appendices).

| Script | Paper section / figure |
|--------|----------------------|
| `analyze_physio_paper.py` | Physio task effects, usability table, temporal profiles |
| `analyze_autonomic_paper.py` | EmotiBit + Tobii pupil combined summaries (appendix) |
| `analyze_task_responses.py` | VAD (Valence-Arousal-Dominance) probe analysis |
| `analyze_task_response_qc.py` | Self-report QC and validity |
| `analyze_dataset_stats.py` | Dataset coverage, modality availability (§3, Table 1) |
| `analyze_cross_modal.py` | Cross-modal correlation heatmap (appendix) |
| `analyze_personality.py` | BFI-44 personality correlations (Appendix B) |
| `analyze_expanded_stats.py` | Expanded inferential statistics |
| `analyze_individual_audio.py` | Per-participant audio task profiles and QC |
| `analyze_multimodal_statistics.py` | Mixed models, paired task tests, synchrony |
| `01_self_report_by_task.py` | Self-report (VAD, postblock) by task |
| `02_physio_task_fingerprint.py` | Physio response profiles per task |
| `03_transcript_speaking_balance.py` | Speaking-time balance (Gini coefficient) |

Each script accepts `--data-root`, `--features-dir`, and `--out-dir` arguments. Run with `--help` for details.

---

## Step 3 — Benchmarks (`benchmarks/`)

Reproduce the 8 benchmark baselines (B0–B7) in paper §5.

| Script | What it produces |
|--------|-----------------|
| `benchmark_no_biomarkers.py` | **Main results** — B0–B7 with bootstrap 95% CIs (35 features, LOGO-CV) |
| `benchmark_additions.py` | Additional benchmark variants and ablations |
| `feature_importance.py` | Ranked feature importance figure |
| `fold_variance.py` / `fold_variance2.py` | Fold-level variance diagnostics |
| `run_neurips_benchmark_baselines.py` | End-to-end benchmark runner (preprocessing → results) |
| `run_feature_ablation.py` | Modality ablation study |

**Reproduce paper Table 2 (benchmark results):**

```bash
python benchmarks/run_neurips_benchmark_baselines.py \
  --features-dir /path/to/derived_features \
  --out-dir results/benchmarks

python benchmarks/benchmark_no_biomarkers.py
```

---

## Figure Generation (`figures/`)

| Script | Output |
|--------|--------|
| `make_demographics_figure.py` | Age, sex, education distribution |
| `plot_feature_overview.py` | Feature coverage overview |
| `plot_results_figures.py` | Model performance heatmaps and strip plots |
| `plot_rms_levels.py` | Audio RMS level diagnostics |
| `visualize_physio_features.py` | Physio feature/QC quick-view PNGs |
| `visualize_eyetracking_features.py` | Eye-tracking feature overview |

---

## Supplementary Analyses (`supplementary_analyses/`)

`supplementary_analyses.py` contains 10 additional analyses:

1. Trust trajectory (T2→T4)
2. Satisfaction by task
3. Voice/inclusion variation across participants
4. SAM probe timing fidelity
5. T4 individual contribution analysis
6. T1 hidden-profile outcome analysis
7. Group composition effects
8. BFI personality correlations
9. Post-block survey profiles
10. Familiarity distribution

```bash
python supplementary_analyses/reviewer_analyses.py
```

Update the `DATA_ROOT` variable at the top of the script to point to your local BIDS release.

---

## Jupyter Notebook (`notebooks/`)

`dataset_paper_qc_audit_analysis.ipynb` — interactive QC audit of stimulus presentation timing and response validity.

```bash
jupyter notebook notebooks/dataset_paper_qc_audit_analysis.ipynb
```

---

## Master Runner

To regenerate all paper figures and tables in sequence:

```bash
python run_all_paper_analyses.py \
  --data-root /path/to/GroupAffect-4-bids \
  --features-dir /path/to/derived_features \
  --results-dir results \
  --figures-dir figures
```

---

## Benchmark Summary (paper §5)

| ID | Target | Metric | Score |
|----|--------|--------|-------|
| B0 | Task classification (T1–T4) | Accuracy | 0.734 ± 0.112 |
| B1 | Valence (high/low) | AUC | 0.657 ± 0.081 |
| B1 | Arousal (high/low) | AUC | 0.504 ± 0.166 |
| B2 | Dominance (high/low) | AUC | 0.652 ± 0.112 |
| B3 | Mental demand | AUC | 0.715 ± 0.153 |
| B3 | Engagement | AUC | 0.580 ± 0.139 |
| B4 | BFI Extraversion | AUC | 0.365 ± 0.312 |
| B5 | T4 contribution | AUC | 0.786 ± 0.281 |
| B6 | Speaking Gini | MAE | 0.089 |
| B7 | Speech overlap | MAE | 0.063 |

All baselines use leave-one-group-out cross-validation (LOGO-CV). B0–B3 apply within-person z-scoring before the CV split and are documented as biased characterisation estimates in the paper.
