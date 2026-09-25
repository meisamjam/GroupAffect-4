# GroupAffect-4 — Analysis Code

Reproduction code for all analyses reported in:

> **GroupAffect-4: A Multimodal Dataset of Four-Person Collaborative Interaction**  
> Jamshidi Seikavandi M., Modica A., Obara A., et al.  
> NeurIPS 2026 Datasets & Benchmarks Track

Code repository: <https://github.com/meisamjam/GroupAffect-4>

---

## Dataset

Two Zenodo records are available:

| Release | Size | Zenodo | Contents |
|---------|------|--------|----------|
| **Subset** (start here) | ~3.7 GB | <https://zenodo.org/records/20037833> | Physiology, ET (7/10 sessions), transcripts, behavioural, annotations — no audio WAV |
| **Full** | ~30 GB | <https://zenodo.org/records/20037847> | Everything above + raw audio WAV (48 kHz) for all 10 sessions + ET all sessions |

Raw audio WAV files in the full release are Restricted access (Data Use Agreement). Request access on the Zenodo record page.

---

## Setup

```bash
pip install -r requirements.txt
```

**Python ≥ 3.10 required.**

---

## Step 0 — Download the Dataset

### Subset release (recommended first step, ~3.7 GB)

```bash
python download_dataset.py --dataset subset --out-dir data/ --extract
```

### Full release (~30 GB, includes audio WAV)

Audio zip files are restricted. Generate a Zenodo personal access token at  
<https://zenodo.org/account/settings/applications/>, then:

```bash
python download_dataset.py --dataset full --out-dir data/ --extract --token <your-token>
```

### Options

```
--dataset subset|full   Which Zenodo record to download (default: subset)
--out-dir PATH          Where to save zip files
--extract               Unpack zips into <out-dir>/bids_release_no_video/ after download
--token TOKEN           Zenodo personal access token (required for draft/restricted records)
--skip-download         Skip downloading; only extract already-present zips
--skip-existing         Skip files that already exist and pass checksum (default: true)
```

After extraction the BIDS root is at:

```
data/bids_release_no_video/
├── dataset_description.json
├── participants.tsv
├── croissant_metadata.json
├── README
├── sub-01/
│   └── ses-*/
│       ├── beh/        *_stimuli_answers.tsv, *_postblock.tsv
│       ├── et/         *_eyetrack.tsv.gz
│       ├── physio/     *_physio.tsv.gz
│       ├── audio/      *.wav (full release), *_transcript.tsv, *_words.tsv
│       └── annot/      sync JSON/TSV files
├── sub-02/ … sub-10/
└── …
```

---

## Step 1 — Feature Extraction (`pipeline/`)

Extract per-participant, per-task features from raw BIDS files.

```bash
python pipeline/run_feature_pipeline.py \
  --data-root data/bids_release_no_video \
  --out-dir derived_features \
  --window-s 30 --step-s 15
```

| Script | What it does |
|--------|-------------|
| `run_feature_pipeline.py` | **Master runner** — runs all extractors in sequence |
| `extract_physio_features.py` | EmotiBit PPG/EDA/skin temperature (participant-task + rolling window) |
| `extract_audio_features.py` | Close-talk prosody and speech features |
| `extract_eyetracking_features.py` | Fixations, saccades, gaze direction (Tobii ET) |
| `extract_pupil_features.py` | Tobii pupil dilation dynamics |
| `preprocess_features.py` | Plausibility gating, winsorisation, within-person z-score, KNN imputation |
| `build_semantic_biomarkers.py` | Composite biomarker features |
| `build_participant_group_comparisons.py` | Join participant features with behavioural annotations |
| `compute_group_dynamics.py` | Dyad/group synchrony metrics |
| `run_preprocessing_analysis.py` | Preprocessing diagnostics and coverage report |
| `common.py` | Shared utilities (session discovery, TSV I/O, argument parsing) |

Derived feature TSVs land under `derived_features/`:

```
derived_features/
├── physio_participant_task.tsv
├── audio_participant_task.tsv
├── features_pupil_participant_task.tsv
└── …
```

---

## Step 2 — Analysis (`analysis/`)

Produce dataset characterisation results (tables and figures in paper §3–§4 and appendices).

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

Each script accepts `--data-root`, `--features-dir`, and `--out-dir`. Run with `--help` for details.

---

## Step 3 — Benchmarks (`benchmarks/`)

Reproduce the originally published B0–B7 baselines from paper §5. The corrected Level-1 diagnostic below is a separate analysis.

```bash
python benchmarks/run_neurips_benchmark_baselines.py \
  --features-dir derived_features \
  --out-dir results/benchmarks

python benchmarks/benchmark_no_biomarkers.py
```

| Script | What it produces |
|--------|-----------------|
| `benchmark_no_biomarkers.py` | **Original paper results** — B0–B7 with bootstrap 95% CIs (35 features, LOGO-CV) |
| `rebuttal_level1_corrected.py` | Corrected Level-1 diagnostic, fold metrics, and predictions (B0–B3d only) |
| `benchmark_additions.py` | Additional benchmark variants and ablations |
| `feature_importance.py` | Ranked feature importance figure |
| `fold_variance.py` / `fold_variance2.py` | Fold-level variance diagnostics |
| `run_neurips_benchmark_baselines.py` | End-to-end benchmark runner |
| `run_feature_ablation.py` | Modality ablation study |

---

## Step 4 — Figures (`figures/`)

| Script | Output |
|--------|--------|
| `make_demographics_figure.py` | Age, sex, education distribution |
| `plot_feature_overview.py` | Feature coverage overview |
| `plot_results_figures.py` | Model performance heatmaps and strip plots |
| `plot_rms_levels.py` | Audio RMS level diagnostics |
| `visualize_physio_features.py` | Physio feature/QC quick-view PNGs |
| `visualize_eyetracking_features.py` | Eye-tracking feature overview |

---

## Supplementary Analyses

```bash
python supplementary_analyses/reviewer_analyses.py
```

Contains 10 additional analyses: trust trajectory, satisfaction by task, voice/inclusion variation, SAM probe timing, T4 contribution, T1 hidden-profile outcome, group composition effects, BFI correlations, post-block survey profiles, and familiarity distribution.

---

## Jupyter Notebook

```bash
jupyter notebook notebooks/dataset_paper_qc_audit_analysis.ipynb
```

Interactive QC audit of stimulus presentation timing and response validity.

---

## Master Runner — All Analyses in One Command

```bash
python run_all_paper_analyses.py \
  --bids-root data/bids_release_no_video \
  --features-dir derived_features \
  --results-base results \
  --figures-base figures
```

Add `--skip-slow` to skip the long-running mixed-model analyses.

---

## Full Workflow (from scratch)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and extract the subset dataset (~3.7 GB)
python download_dataset.py --dataset subset --out-dir data/ --extract

# 3. Extract features from raw BIDS data
python pipeline/run_feature_pipeline.py \
  --data-root data/bids_release_no_video \
  --out-dir derived_features

# 4. Run all paper analyses and generate figures
python run_all_paper_analyses.py \
  --bids-root data/bids_release_no_video \
  --features-dir derived_features \
  --results-base results \
  --figures-base figures
```

---

## Benchmark results and provenance

### Corrected Level-1 diagnostic (PR #2)

The following rerun uses T0-delta physiology/pupil measures and absolute current-task audio features. It splits by group, then fits feature filtering, clipping, KNN imputation, scaling, and binary target thresholds on training groups within each fold. The model is fixed logistic regression (`C=1`); no hyperparameter search is reported. Scores are unweighted means across held-out groups, with across-fold SD.

| ID | Target | Metric | Mean ± fold SD | Valid folds |
|----|--------|--------|----------------|-------------|
| B0 | Task classification (T1–T4) | Accuracy | 0.450 ± 0.134 | 10/10 |
| B1a | Valence (high/low) | AUC | 0.595 ± 0.100 | 10/10 |
| B1b | Arousal (high/low) | AUC | 0.686 ± 0.170 | 10/10 |
| B2 | Dominance (high/low) | AUC | 0.656 ± 0.265 | 10/10 |
| B3a | Mental demand (high/low) | AUC | 0.801 ± 0.125 | 8/10 |
| B3b | Engagement (high/low) | AUC | 0.631 ± 0.119 | 8/10 |
| B3c | Satisfaction (T2/T3; high/low) | AUC | 0.743 ± 0.174 | 7/10 |
| B3d | Mean seat-directed trust (T2/T4; high/low) | AUC | 0.727 ± 0.269 | 8/10 |

Ten group splits were attempted for each target. B3a, B3b, and B3d have eight eligible groups; B3c has eight eligible groups but one additional held-out fold with a single target class, where AUC is undefined. The [summary](results/rebuttal_level1/corrected_level1_summary.tsv), [fold metrics](results/rebuttal_level1/corrected_level1_folds.tsv), and [runner](benchmarks/rebuttal_level1_corrected.py) specify the exact calculation. The fold-bootstrap intervals in the summary are descriptive resampling intervals over only 7–10 evaluable groups; they should not be read as precise population-level uncertainty.

Run the diagnostic after generating its input tables (the repository does not include the raw analysis tables):

```bash
python benchmarks/rebuttal_level1_corrected.py --help
python benchmarks/rebuttal_level1_corrected.py
```

These are local pilot diagnostics, not a replacement for all of the paper's B0–B7 results. The feature set, eligible rows, and protocol differ from the original analysis, so differences between individual scores do not isolate the effect of the preprocessing correction. B3a is the strongest result in this small suite; B2, B3c, and B3d show substantial fold variability or reduced coverage. No external validation is reported.

### Originally published benchmark summary (paper §5; archival)

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

These are the values originally displayed in this README at main commit `c6d96dcf7a957e8b4761ef5b40ee5e813a4da492`; they are retained for provenance, not offered as leakage-free estimates. The original B0–B3 preprocessing uses within-person z-scoring across active tasks before the group split, so earlier tasks can depend on a participant's later tasks. Global winsorization and feature selection also precede cross-validation, although the baseline runner fits KNN imputation inside each training fold. This is a future-looking/transductive concern within participants; it does not imply that training groups used held-out participants' normalization statistics.

The initial PR's [comparison table](results/rebuttal_level1/corrected_level1_comparison.tsv) records B0 as 0.641 in its historical comparator, whereas this README's originally published B0 is 0.734. The repository does not establish why those historical values differ. Do not treat that comparison table as a direct before/after estimate of leakage or silently substitute its historical values for the published table.
