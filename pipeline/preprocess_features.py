"""Standard multimodal preprocessing for benchmark evaluation.

Applies per-modality data cleaning, physiological plausibility checks,
within-person robust normalisation, feature selection, and quality gating.
Produces a cleaned participant-task feature table ready for LOGO-CV benchmarks.
The saved benchmark table intentionally retains residual missing values; the
baseline runner fits KNN imputers inside each leave-one-group-out training fold.

Preprocessing philosophy
------------------------
All steps are modality-aware and standard in affective computing:
  1. Plausibility gating  – replace physiologically impossible values with NaN.
  2. Quality gating       – honour per-row QC flags (missing_frac, gaze validity).
  3. Outlier winsorisation – clip at ±3 SD across the full sample per feature.
  4. Within-person z-score – robust (median/MAD) normalisation per participant
                             across their T1–T4 rows; removes individual set-point
                             differences that dominate between-person variance.
  5. Feature selection    – drop features with >50% missing or near-zero variance.
  6. Fold-local imputation – k=5 nearest-neighbour imputation is fitted by the
                             benchmark runner inside each LOGO-CV fold.

Why within-person normalisation matters
----------------------------------------
ICC for HR-delta across participants ≈ 0.01: virtually all variance is
WITHIN-person across tasks, not between persons. Global imputation followed
by between-person regression treats individual set-points as noise.
Robust within-person z-scoring converts each feature to "how unusual was this
task for this person", which is the correct representation for affective and
cognitive-state prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ─── Physiological plausibility bounds ───────────────────────────────────────
# Absolute values
PLAUSIBILITY_ABS: dict[str, tuple[float, float]] = {
    "hr_mean_bpm":            (40.0,  180.0),   # resting-to-vigorous HR
    "hrv_rmssd_ms":           (10.0,  300.0),   # 25 Hz floor ~40 ms; cap at 300
    "eda_tonic_mean":         (0.0,   25.0),    # µS; >25 is motion artefact
    "eda_phasic_rate_hz":     (0.0,   3.0),     # SCR peaks/s; >3 unreliable
    "eda_phasic_mean":        (0.0,   5.0),
    "eda_scr_amplitude_mean": (0.0,   10.0),
    "temp_mean":              (28.0,  40.0),    # °C wrist skin temperature
    "thermopile_mean":        (28.0,  40.0),
    "pupil_mean":             (1.5,   9.0),     # mm indoor pupil diameter
    "pupil_left_mean":        (1.5,   9.0),
    "pupil_right_mean":       (1.5,   9.0),
    # audio pitch in semitones from 27.5 Hz (OpenSMILE F0semitoneFrom27.5Hz)
    "audio_pitch_mean":       (5.0,   55.0),    # 5 st = 36 Hz, 55 st = 843 Hz
    "audio_pitch_sd":         (0.0,   20.0),
    "audio_hnr_mean":         (-10.0, 25.0),    # dB; <-10 = noise dominated
    "audio_jitter_mean":      (0.0,   0.10),    # fraction; >10% pathological
    "audio_shimmer_mean":     (0.0,   5.0),     # dB
    "audio_speech_rate_proxy":(0.0,   20.0),    # syllables/s upper bound
    "audio_speaking_fraction":(0.0,   1.0),
    "audio_overlap_fraction": (0.0,   1.0),
}

# Delta-from-baseline features: tighter physiologically plausible windows
PLAUSIBILITY_DELTA: dict[str, tuple[float, float]] = {
    "hr_mean_bpm_delta_t0":            (-40.0,  40.0),
    "hrv_rmssd_ms_delta_t0":           (-150.0, 150.0),
    "eda_tonic_mean_delta_t0":         (-3.0,   8.0),
    "eda_phasic_rate_hz_delta_t0":     (-1.5,   2.5),
    "eda_scr_amplitude_mean_delta_t0": (-5.0,   8.0),
    "temp_mean_delta_t0":              (-3.0,   5.0),
    "thermopile_mean_delta_t0":        (-3.0,   5.0),
    "pupil_mean_delta_t0":             (-3.0,   3.0),
    "pupil_left_mean_delta_t0":        (-3.0,   3.0),
    "pupil_right_mean_delta_t0":       (-3.0,   3.0),
}

# ET quality thresholds
ET_PUPIL_MISSING_MAX   = 0.50   # rows with >50% pupil missing → NaN pupil features
ET_GAZE_VALID_MIN      = 0.50   # rows with <50% valid gaze → NaN gaze features

# Imputation k-neighbours
KNN_K = 5

# Within-person normalisation: minimum non-NaN rows to compute MAD
WITHIN_PERSON_MIN_ROWS = 2


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _robust_zscore_series(s: pd.Series) -> pd.Series:
    """Robust z-score using median and MAD (×1.4826 ≈ σ for normal)."""
    valid = s.dropna()
    if len(valid) < WITHIN_PERSON_MIN_ROWS:
        return pd.Series(np.nan, index=s.index)
    med = float(valid.median())
    mad = float((valid - med).abs().median()) * 1.4826
    if mad < 1e-9:
        # constant feature for this person: centre but don't scale
        return (s - med)
    return (s - med) / mad


def _winsorise(df: pd.DataFrame, cols: list[str], n_sd: float = 3.0) -> pd.DataFrame:
    """Winsorise at ±n_sd across the whole sample (per column)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 4:
            continue
        mu, sigma = vals.mean(), vals.std()
        lo, hi = mu - n_sd * sigma, mu + n_sd * sigma
        n_clipped = (df[col] < lo).sum() + (df[col] > hi).sum()
        if n_clipped > 0:
            LOGGER.debug("Winsorise %s: clip %d rows to [%.3f, %.3f]", col, n_clipped, lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def _apply_plausibility(df: pd.DataFrame,
                         bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Replace values outside physiological bounds with NaN."""
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        bad = (df[col] < lo) | (df[col] > hi)
        n_bad = int(bad.sum())
        if n_bad > 0:
            LOGGER.debug("Plausibility gate %s: %d rows set to NaN (outside [%.2f, %.2f])",
                         col, n_bad, lo, hi)
            df.loc[bad, col] = np.nan
    return df


def _within_person_zscore(df: pd.DataFrame,
                           feature_cols: list[str],
                           person_col: str) -> pd.DataFrame:
    """Robust within-person z-score for each feature column."""
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby(person_col)[col].transform(_robust_zscore_series)
    return df


def _knn_impute(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """KNN imputation on a block of feature columns."""
    from sklearn.impute import KNNImputer
    avail = [c for c in feature_cols if c in df.columns]
    if not avail:
        return df
    block = df[avail].copy()
    if block.isnull().any().any():
        imp = KNNImputer(n_neighbors=KNN_K, weights="distance")
        block[avail] = imp.fit_transform(block[avail])
        df = df.copy()
        df[avail] = block[avail]
    return df


def _drop_low_quality_features(df: pd.DataFrame,
                                 feature_cols: list[str],
                                 missing_threshold: float = 0.50,
                                 variance_threshold: float = 1e-4) -> list[str]:
    """Return feature_cols excluding those with too much missing or near-zero variance."""
    kept = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        miss_rate = df[col].isna().mean()
        if miss_rate > missing_threshold:
            LOGGER.info("Drop %s: %.1f%% missing", col, 100 * miss_rate)
            continue
        variance = df[col].dropna().var()
        if variance < variance_threshold:
            LOGGER.info("Drop %s: near-zero variance (%.2e)", col, variance)
            continue
        kept.append(col)
    return kept


def _drop_correlated_features(df: pd.DataFrame,
                                feature_cols: list[str],
                                threshold: float = 0.95) -> list[str]:
    """Greedy removal of highly correlated features (|r| > threshold)."""
    if len(feature_cols) < 2:
        return feature_cols
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = {col for col in upper.columns if any(upper[col] > threshold)}
    if to_drop:
        LOGGER.info("Drop %d highly correlated features (|r|>%.2f): %s",
                    len(to_drop), threshold, sorted(to_drop)[:5])
    return [c for c in feature_cols if c not in to_drop]


# ─── Modality-specific preprocessing ─────────────────────────────────────────

PHYSIO_DELTA_COLS = [
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_tonic_mean_delta_t0",
    "eda_mean_delta_t0",
    "eda_phasic_rate_hz_delta_t0",
    "eda_scr_amplitude_mean_delta_t0",
    "temp_mean_delta_t0",
    "thermopile_mean_delta_t0",
    "temp_aux_mean_delta_t0",
]

PHYSIO_ABS_COLS = [
    "hr_mean_bpm",
    "hrv_rmssd_ms",
    "eda_tonic_mean",
    "eda_phasic_rate_hz",
    "eda_phasic_mean",
    "temp_mean",
]

ET_PUPIL_COLS = [
    "pupil_mean_delta_t0",
    "pupil_left_mean_delta_t0",
    "pupil_right_mean_delta_t0",
    "pupil_mean",
    "pupil_left_mean",
    "pupil_right_mean",
]

ET_GAZE_COLS = [
    "gaze_x_mean_delta_t0",
    "gaze_y_mean_delta_t0",
    "gaze_dispersion",
    "gaze_velocity_mean",
]

AUDIO_COLS = [
    "audio_speaking_fraction",
    "audio_overlap_fraction",
    "audio_energy_mean",
    "audio_energy_sd",
    "audio_pitch_mean",
    "audio_pitch_sd",
    "audio_hnr_mean",
    "audio_jitter_mean",
    "audio_shimmer_mean",
    "audio_voiced_segments_per_sec",
    "audio_mean_voiced_segment_s",
    "audio_speech_rate_proxy",
]


def preprocess_physio(df: pd.DataFrame) -> pd.DataFrame:
    """Apply physiological plausibility gates and winsorisation to physio features."""
    df = _apply_plausibility(df, PLAUSIBILITY_ABS)
    df = _apply_plausibility(df, PLAUSIBILITY_DELTA)
    # Extra: absolute HR > 180 for a seated lab task is almost certainly artefact
    if "hr_mean_bpm" in df.columns:
        df.loc[df["hr_mean_bpm"] > 180, "hr_mean_bpm"] = np.nan
    # Winsorise at ±3 SD across all participants (after plausibility gate)
    physio_feats = [c for c in PHYSIO_DELTA_COLS + PHYSIO_ABS_COLS if c in df.columns]
    df = _winsorise(df, physio_feats)
    return df


def preprocess_et(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ET quality gates and plausibility checks."""
    df = df.copy()
    # Gate high-missing-fraction rows
    if "pupil_missing_frac" in df.columns:
        bad_pupil = df["pupil_missing_frac"] > ET_PUPIL_MISSING_MAX
        n_bad = int(bad_pupil.sum())
        if n_bad > 0:
            LOGGER.info("ET: NaN pupil features for %d rows (missing_frac > %.2f)",
                        n_bad, ET_PUPIL_MISSING_MAX)
            df.loc[bad_pupil, [c for c in ET_PUPIL_COLS if c in df.columns]] = np.nan
    # Gate low-validity gaze rows
    if "gaze_valid_frac" in df.columns:
        bad_gaze = df["gaze_valid_frac"] < ET_GAZE_VALID_MIN
        n_bad = int(bad_gaze.sum())
        if n_bad > 0:
            LOGGER.info("ET: NaN gaze features for %d rows (valid_frac < %.2f)",
                        n_bad, ET_GAZE_VALID_MIN)
            df.loc[bad_gaze, [c for c in ET_GAZE_COLS if c in df.columns]] = np.nan
    # Plausibility
    df = _apply_plausibility(df, PLAUSIBILITY_ABS)
    df = _apply_plausibility(df, PLAUSIBILITY_DELTA)
    et_feats = [c for c in ET_PUPIL_COLS + ET_GAZE_COLS if c in df.columns]
    df = _winsorise(df, et_feats)
    return df


def preprocess_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Apply audio plausibility gates.

    Note: pitch_mean is F0semitoneFrom27.5Hz (OpenSMILE), not Hz.
    Valid range 5–55 semitones ≈ 36–843 Hz fundamental.
    """
    df = _apply_plausibility(df, PLAUSIBILITY_ABS)
    audio_feats = [c for c in AUDIO_COLS if c in df.columns]
    df = _winsorise(df, audio_feats)
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def preprocess_participant_task(
    df: pd.DataFrame,
    person_col: str = "participant_uid",
    active_tasks: tuple[str, ...] = ("T1", "T2", "T3", "T4"),
    apply_within_person_norm: bool = True,
    apply_knn_imputation: bool = False,
    apply_feature_selection: bool = True,
    correlation_threshold: float = 0.95,
    missing_threshold: float = 0.50,
) -> tuple[pd.DataFrame, list[str]]:
    """Full preprocessing pipeline for the participant-task feature table.

    Parameters
    ----------
    df : merged participant-task dataframe with physio, ET, audio, and label columns.
    person_col : column identifying each unique participant.
    active_tasks : task subset to use for within-person normalisation reference.
    apply_within_person_norm : if True, apply robust within-person z-scoring.
    apply_knn_imputation : if True, apply KNN imputation after normalisation.
        For benchmark exports this should remain False so missing values are
        imputed inside each LOGO-CV training fold by the baseline runner.
    apply_feature_selection : if True, drop high-missing and high-correlation features.
    correlation_threshold : drop one feature from pairs with |r| > this threshold.
    missing_threshold : drop features with missingness rate > this threshold.

    Returns
    -------
    df_clean : cleaned dataframe.
    clean_feature_cols : list of feature columns after selection.
    """
    df = df.copy()

    # --- 1. Per-modality plausibility and winsorisation -----------------------
    df = preprocess_physio(df)
    df = preprocess_et(df)
    df = preprocess_audio(df)

    # --- 2. Identify feature columns (exclude labels and IDs) -----------------
    non_feature_patterns = [
        "session_id", "participant", "group_id", "task", "seat", "sex", "age",
        "handedness", "education", "english", "session_core", "sub_",
        "vad_", "ans_", "bfi44_", "qc_", "qc_flag", "qc_notes",
        "source_file", "missing_frac", "valid_frac", "available",
        "duration_s", "sample_rate", "coverage", "task_index", "task_events",
        "ppg_", "recording", "_uid", "_global", "_index",
    ]

    def _is_feature(col: str) -> bool:
        return not any(col.startswith(p) or p in col for p in non_feature_patterns)

    candidate_features = [c for c in df.select_dtypes(include=[np.number]).columns
                          if _is_feature(c)]

    # Focus on well-understood modality features
    explicit_features = [c for c in (
        PHYSIO_DELTA_COLS + PHYSIO_ABS_COLS + ET_PUPIL_COLS + ET_GAZE_COLS + AUDIO_COLS
    ) if c in df.columns]
    all_features = list(dict.fromkeys(explicit_features + candidate_features))

    LOGGER.info("Candidate features before selection: %d", len(all_features))

    # --- 3. Within-person robust z-score (active tasks only for computing stats)
    if apply_within_person_norm and person_col in df.columns:
        # Compute z-score params from active task rows, apply to all rows
        active_mask = df["task"].isin(active_tasks) if "task" in df.columns else pd.Series(True, index=df.index)
        active_df = df[active_mask].copy()
        for col in all_features:
            if col not in df.columns:
                continue
            def _person_robust_z(s: pd.Series) -> pd.Series:
                return _robust_zscore_series(s)
            # Compute stats on active tasks, apply to whole df via person mapping
            person_stats: dict[str, tuple[float, float]] = {}
            for pid, grp in active_df.groupby(person_col):
                vals = grp[col].dropna()
                if len(vals) < WITHIN_PERSON_MIN_ROWS:
                    continue
                med = float(vals.median())
                mad = float((vals - med).abs().median()) * 1.4826
                person_stats[str(pid)] = (med, mad)
            if person_stats:
                def _apply_stats(row: pd.Series) -> float:
                    pid = str(row[person_col])
                    v = row[col]
                    if pd.isna(v) or pid not in person_stats:
                        return v
                    med, mad = person_stats[pid]
                    if mad < 1e-9:
                        return float(v - med)
                    return float((v - med) / mad)
                df[col] = df[[person_col, col]].apply(_apply_stats, axis=1)
        LOGGER.info("Applied within-person robust z-score normalisation.")

    # --- 4. Feature selection -------------------------------------------------
    if apply_feature_selection:
        all_features = _drop_low_quality_features(df, all_features,
                                                   missing_threshold=missing_threshold)
        all_features = _drop_correlated_features(df, all_features,
                                                  threshold=correlation_threshold)

    LOGGER.info("Features after selection: %d", len(all_features))

    # --- 5. Optional descriptive KNN imputation per modality block ------------
    if apply_knn_imputation:
        physio_feats = [c for c in PHYSIO_DELTA_COLS + PHYSIO_ABS_COLS if c in all_features]
        et_feats = [c for c in ET_PUPIL_COLS + ET_GAZE_COLS if c in all_features]
        audio_feats = [c for c in AUDIO_COLS if c in all_features]
        remaining = [c for c in all_features
                     if c not in physio_feats + et_feats + audio_feats]
        df = _knn_impute(df, physio_feats)
        df = _knn_impute(df, et_feats)
        df = _knn_impute(df, audio_feats)
        if remaining:
            df = _knn_impute(df, remaining)
        LOGGER.info("Applied KNN imputation (k=%d) per modality block.", KNN_K)

    return df, all_features


def run_and_save(
    stats_pt_path: Path = Path("results/statistics/analysis_dataset_participant_task.tsv"),
    audio_path: Path = Path("results/audio/individual_audio_task.tsv"),
    vad_path: Path = Path("results/task_responses/vad_participant_task.tsv"),
    out_path: Path = Path("results/benchmarks/preprocessed_participant_task.tsv"),
    out_features_path: Path = Path("results/benchmarks/preprocessed_feature_list.txt"),
) -> tuple[pd.DataFrame, list[str]]:
    """Load, preprocess, and save the cleaned participant-task table."""
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    stats_pt = pd.read_csv(stats_pt_path, sep="\t")
    audio = pd.read_csv(audio_path, sep="\t")
    vad = pd.read_csv(vad_path, sep="\t")

    # Normalise task/participant column names
    if "task_id" in stats_pt.columns and "task" not in stats_pt.columns:
        stats_pt = stats_pt.rename(columns={"task_id": "task"})

    # Merge audio
    audio2 = audio.copy()
    if "task_id" in audio2.columns:
        if "task" in audio2.columns:
            audio2 = audio2.drop(columns=["task"])
        audio2 = audio2.rename(columns={"task_id": "task"})
    src_cols = [
        "speaking_fraction", "overlap_fraction", "energy_mean", "energy_sd",
        "pitch_mean", "pitch_sd", "hnr_mean", "jitter_mean", "shimmer_mean",
        "voiced_segments_per_sec", "mean_voiced_segment_s", "speech_rate_proxy",
    ]
    avail = [c for c in src_cols if c in audio2.columns]
    audio_key = [c for c in ["session_id", "participant_id", "task"] if c in audio2.columns]
    audio_sel = audio2[audio_key + avail].rename(columns={c: f"audio_{c}" for c in avail})
    merge_on = [c for c in ["session_id", "participant_id", "task"]
                if c in stats_pt.columns and c in audio_sel.columns]
    df = stats_pt.merge(audio_sel, on=merge_on, how="left")

    # Merge VAD labels
    vad2 = vad.rename(columns={
        "participant": "participant_id",
        "arousal": "vad_arousal",
        "valence": "vad_valence",
        "dominance": "vad_dominance",
    }).copy()
    for col in ["vad_arousal", "vad_valence", "vad_dominance"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    vad_on = [c for c in ["session_id", "participant_id", "task"]
               if c in df.columns and c in vad2.columns]
    vad_cols = vad_on + [c for c in ["vad_arousal", "vad_valence", "vad_dominance"] if c in vad2.columns]
    df = df.merge(vad2[vad_cols], on=vad_on, how="left")

    # Active tasks only for benchmarks
    df = df[df["task"].isin(("T1", "T2", "T3", "T4"))].copy().reset_index(drop=True)

    # Determine person identifier
    person_col = "participant_uid" if "participant_uid" in df.columns else "participant_global_id"
    if person_col not in df.columns:
        person_col = "participant_id"

    df_clean, feature_cols = preprocess_participant_task(
        df, person_col=person_col,
        apply_within_person_norm=True,
        apply_knn_imputation=False,
        apply_feature_selection=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, sep="\t", index=False)
    out_features_path.write_text("\n".join(feature_cols), encoding="utf-8")

    LOGGER.info(
        "Wrote cleaned table with residual NaNs retained for fold-local KNN: "
        "%s (%d rows, %d features)",
        out_path,
        len(df_clean),
        len(feature_cols),
    )
    return df_clean, feature_cols


if __name__ == "__main__":
    run_and_save()
