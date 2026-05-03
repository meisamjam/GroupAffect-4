"""Preprocessing quality control and statistics recomputation for GroupAffect-4.

Applies the full modality-aware preprocessing pipeline to
analysis_dataset_participant_task.tsv, reports step-by-step changes,
recomputes task-effect statistics and cross-modal correlations on the
cleaned data, and writes:

  results/statistics/preprocessed/participant_task_preprocessed.tsv
  results/statistics/preprocessed/task_effect_stats.tsv
  results/statistics/preprocessed/feature_qc_summary.tsv
  results/statistics/preprocessed/cross_modal_correlations.tsv
  results/PREPROCESSING_REPORT.md

Usage
-----
  cd GroupAffect-4-data-processing
  py -3 tools/features/run_preprocessing_analysis.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Import preprocessing functions from sibling module
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_features import (
    PLAUSIBILITY_ABS,
    PLAUSIBILITY_DELTA,
    ET_PUPIL_MISSING_MAX,
    ET_GAZE_VALID_MIN,
    ET_PUPIL_COLS,
    ET_GAZE_COLS,
    PHYSIO_ABS_COLS,
    PHYSIO_DELTA_COLS,
    AUDIO_COLS,
    KNN_K,
    _apply_plausibility,
    _winsorise,
    _knn_impute,
    preprocess_participant_task,
)

LOG = logging.getLogger(__name__)

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

STATS_PT_PATH = Path("results/statistics/analysis_dataset_participant_task.tsv")
AUDIO_PATH    = Path("results/audio/individual_audio_task.tsv")
OUT_DIR       = Path("results/statistics/preprocessed")
REPORT_PATH   = Path("results/PREPROCESSING_REPORT.md")

ACTIVE_TASKS = ("T1", "T2", "T3", "T4")

# Key feature groups for statistics reporting
PHYSIO_KEY = [
    "hr_mean_bpm",
    "hrv_rmssd_ms",
    "eda_tonic_mean",
    "eda_phasic_rate_hz",
    "temp_mean",
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_tonic_mean_delta_t0",
    "temp_mean_delta_t0",
]
ET_KEY = [
    "pupil_mean",
    "pupil_mean_delta_t0",
]
AUDIO_KEY = [
    "audio_speaking_fraction",
    "audio_energy_mean",
    "audio_pitch_mean",
    "audio_speech_rate_proxy",
    "audio_hnr_mean",
]
LABEL_KEY = [
    "ans_engagement",
    "ans_mental_demand",
    "ans_valence",
    "ans_arousal",
    "vad_valence",
    "vad_arousal",
    "vad_dominance",
]


# â”€â”€â”€ Step-by-step preprocessing with audit trail â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class StepAudit:
    """Tracks value-level changes at each preprocessing step."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        self.steps: list[dict[str, Any]] = []
        self._initial_shape = df.shape
        self._feature_cols = feature_cols

    def record(self, name: str, df_before: pd.DataFrame, df_after: pd.DataFrame,
               feature_cols: list[str]) -> None:
        avail = [c for c in feature_cols if c in df_before.columns and c in df_after.columns]
        total_vals = len(avail) * len(df_before)
        nan_before = int(df_before[avail].isna().sum().sum())
        nan_after  = int(df_after[avail].isna().sum().sum())
        new_nans   = max(0, nan_after - nan_before)
        # Values clipped (changed but not NaN'd)
        clipped = 0
        for c in avail:
            b = df_before[c].dropna()
            a = df_after[c].dropna()
            idx = b.index.intersection(a.index)
            clipped += int((b.loc[idx] != a.loc[idx]).sum())
        self.steps.append({
            "step": name,
            "total_feature_values": total_vals,
            "nan_before": nan_before,
            "nan_after": nan_after,
            "new_nans": new_nans,
            "values_clipped": clipped,
        })

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def run_preprocessing_with_audit(
    df: pd.DataFrame,
    person_col: str,
) -> tuple[pd.DataFrame, list[str], StepAudit]:
    """Run preprocessing step-by-step and return (cleaned_df, feature_cols, audit)."""
    all_feature_cols = [c for c in (PHYSIO_KEY + ET_KEY + AUDIO_KEY) if c in df.columns]
    # Expand to all numeric non-label columns
    non_feat = {
        "session_id", "participant_id", "group_id", "task", "seat", "sex", "age",
        "handedness", "education", "english", "task_index", person_col,
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    expanded = [c for c in numeric_cols if c not in non_feat
                and not any(c.startswith(p) for p in ["ans_", "vad_", "bfi44_", "qc_"])
                and "available" not in c and "coverage" not in c
                and "duration_s" not in c and "_t0" not in c.replace("_delta_t0", "")]
    # Merge explicit + expanded without duplication
    feat_cols = list(dict.fromkeys(all_feature_cols + expanded))
    feat_cols = [c for c in feat_cols if c in df.columns]

    audit = StepAudit(df, feat_cols)
    df_cur = df.copy()

    # Step 1: ET quality gating
    df_step = df_cur.copy()
    if "pupil_missing_frac" in df_step.columns:
        bad_pupil = df_step["pupil_missing_frac"] > ET_PUPIL_MISSING_MAX
        pupil_feats = [c for c in ET_PUPIL_COLS if c in df_step.columns]
        df_step.loc[bad_pupil, pupil_feats] = np.nan
    if "gaze_valid_frac" in df_step.columns:
        bad_gaze = df_step["gaze_valid_frac"] < ET_GAZE_VALID_MIN
        gaze_feats = [c for c in ET_GAZE_COLS if c in df_step.columns]
        df_step.loc[bad_gaze, gaze_feats] = np.nan
    audit.record("ET quality gating", df_cur, df_step, feat_cols)
    df_cur = df_step

    # Step 2: Physiological plausibility gating
    df_step = _apply_plausibility(df_cur, PLAUSIBILITY_ABS)
    df_step = _apply_plausibility(df_step, PLAUSIBILITY_DELTA)
    audit.record("Plausibility gating", df_cur, df_step, feat_cols)
    df_cur = df_step

    # Step 3: Winsorisation
    all_win_cols = [c for c in (PHYSIO_DELTA_COLS + PHYSIO_ABS_COLS + ET_PUPIL_COLS + AUDIO_COLS)
                    if c in df_cur.columns]
    df_step = _winsorise(df_cur, all_win_cols, n_sd=3.0)
    audit.record("Winsorisation (3 SD)", df_cur, df_step, feat_cols)
    df_cur = df_step

    # Step 4: Within-person robust z-score
    df_step = df_cur.copy()
    active = df_step[df_step["task"].isin(ACTIVE_TASKS)] if "task" in df_step.columns else df_step
    for col in feat_cols:
        if col not in df_step.columns:
            continue
        person_stats: dict[str, tuple[float, float]] = {}
        for pid, grp in active.groupby(person_col):
            vals = grp[col].dropna()
            if len(vals) < 2:
                continue
            med = float(vals.median())
            mad = float((vals - med).abs().median()) * 1.4826
            person_stats[str(pid)] = (med, mad)
        if person_stats:
            def _apply(row: pd.Series) -> float:
                pid = str(row[person_col])
                v = row[col]
                if pd.isna(v) or pid not in person_stats:
                    return v
                med, mad = person_stats[pid]
                if mad < 1e-9:
                    return float(v - med)
                return float((v - med) / mad)
            df_step[col] = df_step[[person_col, col]].apply(_apply, axis=1)
    audit.record("Within-person z-score", df_cur, df_step, feat_cols)
    df_cur = df_step

    # Step 5: descriptive KNN imputation audit per modality block
    df_step = df_cur.copy()
    physio_feats = [c for c in PHYSIO_DELTA_COLS + PHYSIO_ABS_COLS if c in feat_cols]
    et_feats     = [c for c in ET_PUPIL_COLS + ET_GAZE_COLS if c in feat_cols]
    audio_feats  = [c for c in AUDIO_COLS if c in feat_cols]
    df_step = _knn_impute(df_step, physio_feats)
    df_step = _knn_impute(df_step, et_feats)
    df_step = _knn_impute(df_step, audio_feats)
    audit.record("KNN imputation audit (k=5)", df_cur, df_step, feat_cols)
    df_cur = df_step

    return df_cur, feat_cols, audit


# â”€â”€â”€ Statistics helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled_sd = math.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_sd < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def task_effect_stats(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """For each feature, compute Wilcoxon signed-rank T2 vs T0 and T2 vs T1,
    plus Cohen's d and missingness rate."""
    rows = []
    active = df[df["task"].isin(ACTIVE_TASKS)].copy() if "task" in df.columns else df
    for col in feat_cols:
        if col not in df.columns:
            continue
        miss_rate = active[col].isna().mean()
        row: dict[str, Any] = {
            "feature": col,
            "n_active_rows": len(active),
            "n_valid": int(active[col].notna().sum()),
            "missing_rate": round(miss_rate, 4),
            "mean": round(active[col].mean(), 4) if active[col].notna().any() else math.nan,
            "sd": round(active[col].std(), 4) if active[col].notna().any() else math.nan,
            "median": round(active[col].median(), 4) if active[col].notna().any() else math.nan,
        }
        # T2 vs T1 effect
        t2 = df[df["task"] == "T2"][col].dropna().to_numpy() if "task" in df.columns else np.array([])
        t1 = df[df["task"] == "T1"][col].dropna().to_numpy() if "task" in df.columns else np.array([])
        if len(t2) >= 5 and len(t1) >= 5:
            try:
                res = stats.wilcoxon(t2[:len(t1)], t1[:len(t1)]) if len(t2) == len(t1) else stats.mannwhitneyu(t2, t1)
                row["T2_vs_T1_p"] = round(float(res.pvalue), 5)
            except Exception:
                row["T2_vs_T1_p"] = math.nan
            row["T2_vs_T1_d"] = round(cohens_d(t2, t1), 4)
        else:
            row["T2_vs_T1_p"] = math.nan
            row["T2_vs_T1_d"] = math.nan
        # T3 vs T1 effect
        t3 = df[df["task"] == "T3"][col].dropna().to_numpy() if "task" in df.columns else np.array([])
        if len(t3) >= 5 and len(t1) >= 5:
            try:
                res = stats.mannwhitneyu(t3, t1)
                row["T3_vs_T1_p"] = round(float(res.pvalue), 5)
            except Exception:
                row["T3_vs_T1_p"] = math.nan
            row["T3_vs_T1_d"] = round(cohens_d(t3, t1), 4)
        else:
            row["T3_vs_T1_p"] = math.nan
            row["T3_vs_T1_d"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cross_modal_correlations(df: pd.DataFrame, feat_cols: list[str],
                              label_cols: list[str]) -> pd.DataFrame:
    """Pearson r between each (feature, label) pair in active tasks."""
    active = df[df["task"].isin(ACTIVE_TASKS)].copy() if "task" in df.columns else df
    rows = []
    for feat in feat_cols:
        if feat not in active.columns:
            continue
        for lbl in label_cols:
            if lbl not in active.columns:
                continue
            pair = active[[feat, lbl]].dropna()
            if len(pair) < 10:
                rows.append({"feature": feat, "label": lbl, "n": len(pair), "r": math.nan, "p": math.nan})
                continue
            r, p = stats.pearsonr(pair[feat], pair[lbl])
            rows.append({"feature": feat, "label": lbl, "n": len(pair),
                         "r": round(float(r), 4), "p": round(float(p), 5)})
    df_out = pd.DataFrame(rows)
    # BH-FDR correction
    if not df_out.empty and df_out["p"].notna().any():
        from statsmodels.stats.multitest import multipletests
        mask = df_out["p"].notna()
        _, q, _, _ = multipletests(df_out.loc[mask, "p"], method="fdr_bh")
        df_out.loc[mask, "q_fdr"] = np.round(q, 5)
    return df_out


def feature_qc_summary(df_raw: pd.DataFrame, df_pre: pd.DataFrame,
                        feat_cols: list[str]) -> pd.DataFrame:
    """Compare missingness and distribution before/after preprocessing."""
    rows = []
    for col in feat_cols:
        raw_miss = df_raw[col].isna().mean() if col in df_raw.columns else math.nan
        pre_miss = df_pre[col].isna().mean() if col in df_pre.columns else math.nan
        raw_mean = df_raw[col].mean() if col in df_raw.columns else math.nan
        pre_mean = df_pre[col].mean() if col in df_pre.columns else math.nan
        raw_sd   = df_raw[col].std()  if col in df_raw.columns else math.nan
        pre_sd   = df_pre[col].std()  if col in df_pre.columns else math.nan
        rows.append({
            "feature": col,
            "missing_raw": round(raw_miss, 4),
            "missing_preprocessed": round(pre_miss, 4),
            "mean_raw": round(raw_mean, 4),
            "mean_preprocessed": round(pre_mean, 4),
            "sd_raw": round(raw_sd, 4),
            "sd_preprocessed": round(pre_sd, 4),
        })
    return pd.DataFrame(rows)


# â”€â”€â”€ Markdown report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def write_markdown_report(
    audit: StepAudit,
    qc_df: pd.DataFrame,
    task_df_raw: pd.DataFrame,
    task_df_pre: pd.DataFrame,
    corr_df_raw: pd.DataFrame,
    corr_df_pre: pd.DataFrame,
    n_raw_rows: int,
    n_feat_raw: int,
    n_feat_pre: int,
    out_path: Path,
) -> None:
    lines: list[str] = []

    def h1(t): lines.extend([f"# {t}", ""])
    def h2(t): lines.extend([f"## {t}", ""])
    def h3(t): lines.extend([f"### {t}", ""])
    def p(t=""):  lines.extend([t, ""])
    def tbl(*row): lines.append("| " + " | ".join(str(x) for x in row) + " |")
    def tbl_sep(*widths): lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    def code(t):  lines.extend(["```", t, "```", ""])

    h1("GroupAffect-4 â€” Multimodal Preprocessing Report")
    p(f"**Generated:** 2026-05-01  \n**Dataset:** GroupAffect-4 (N=40, 10 groups of 4, 5 tasks)  \n"
      f"**Script:** `tools/features/run_preprocessing_analysis.py`")
    p("---")

    h2("1. Overview")
    p(f"This report documents the step-by-step descriptive preprocessing audit "
      f"for the GroupAffect-4 multimodal participant-task feature table "
      f"(`analysis_dataset_participant_task.tsv`, {n_raw_rows} rows Ã— {n_feat_raw} candidate features). "
      f"The benchmark-ready export applies stricter missingness and collinearity "
      f"filters in `tools/features/preprocess_features.py` and stores its final "
      f"feature list in `results/benchmarks/preprocessed_feature_list.txt`.")
    p("The pipeline follows standard affective computing practice:")
    lines.append("1. **ET quality gating** â€” null out pupil features when >50% of samples are missing; null gaze features when valid fraction <50%.")
    lines.append("2. **Physiological plausibility gating** â€” replace out-of-range absolute and delta values with NaN.")
    lines.append("3. **Winsorisation** â€” clip at Â±3 SD across the full sample per feature.")
    lines.append("4. **Within-person robust z-score** â€” normalise each feature per participant using median/MAD (Ã—1.4826) computed on T1â€“T4 rows.")
    lines.append("5. **KNN imputation audit** â€” fill remaining NaN values with k=5 nearest-neighbour imputation per modality block for descriptive coverage summaries.")
    p()
    p("**Why within-person normalisation matters:**  ")
    p("ICC for HR delta across participants ~0.01 -- virtually all physiological variance "
      "is within-person, not between persons. Standard z-scoring across people treats "
      "individual set-points as signal; robust within-person normalisation converts each "
      "row to 'how unusual was this task for this person', which is the correct "
      "representation for state prediction.")
    p("**Preprocessing strategy per benchmark type:**  ")
    p("Within-person z-scoring is appropriate for state/task classification targets "
      "(which task am I in? am I more dominant than my baseline?). For between-person "
      "regression targets (raw valence ratings, BFI personality scores, contribution tokens), "
      "within-person normalisation destroys the between-person variance that constitutes the "
      "prediction signal. Those benchmarks use sample-level features without within-person z-score. "
      "For benchmark evaluation, residual missing values are imputed inside each LOGO-CV training "
      "fold rather than in the saved feature table.")
    p("---")

    h2("2. Preprocessing Step Audit")
    tbl("Step", "Total featureÃ—row cells", "NaN before", "NaN after", "New NaNs", "Values clipped")
    tbl_sep(35, 24, 10, 10, 9, 15)
    for _, row in audit.summary().iterrows():
        tbl(row['step'], f"{row['total_feature_values']:,}",
            f"{row['nan_before']:,}", f"{row['nan_after']:,}",
            f"{row['new_nans']:,}", f"{row['values_clipped']:,}")
    p()

    h2("3. Feature Coverage Before vs After Preprocessing")
    p("Features with >20% missing in either raw or preprocessed data (physiology/ET/audio only):")
    # Filter to only show domain-relevant features with high missing rates
    notable_feats = [c for c in (PHYSIO_KEY + ET_KEY + AUDIO_KEY) if c in qc_df["feature"].values]
    notable = qc_df[qc_df["feature"].isin(notable_feats)].sort_values("missing_raw", ascending=False).head(15)
    tbl("Feature", "Missing (raw)", "Missing (preprocessed)", "Mean raw", "Mean preprocessed", "SD raw", "SD preprocessed")
    tbl_sep(40, 13, 22, 9, 17, 7, 15)
    for _, r in notable.iterrows():
        tbl(r['feature'], f"{r['missing_raw']:.1%}", f"{r['missing_preprocessed']:.1%}",
            f"{r['mean_raw']:.3f}", f"{r['mean_preprocessed']:.3f}",
            f"{r['sd_raw']:.3f}", f"{r['sd_preprocessed']:.3f}")
    p()
    p("Note: The table above reports the fully cleaned descriptive preprocessing pass, including "
      "KNN imputation for audit statistics. The benchmark-ready table retains residual NaNs so "
      "KNN can be fit separately inside each LOGO-CV fold.")

    h2("4. Task Effect Statistics: Raw vs Preprocessed")
    p("Cohen's d (Mann-Whitney U) for T2 vs T1 (negotiation vs reading) and T3 vs T1 "
      "(ideas/creativity vs reading). Larger |d| = stronger task separation.  \n"
      "Missing rate shown before and after the descriptive preprocessing pass.")
    tbl("Feature", "d(T2,T1) raw", "d(T2,T1) prep", "d(T3,T1) raw", "d(T3,T1) prep", "Miss raw", "Miss prep")
    tbl_sep(40, 12, 13, 12, 13, 9, 10)
    key_feats = [c for c in (PHYSIO_KEY + ET_KEY + AUDIO_KEY) if c in task_df_raw["feature"].values]
    raw_idx = task_df_raw.set_index("feature")
    pre_idx = task_df_pre.set_index("feature")
    for col in key_feats:
        if col not in raw_idx.index or col not in pre_idx.index:
            continue
        r = raw_idx.loc[col]
        p_ = pre_idx.loc[col]
        d21r = r.get("T2_vs_T1_d", math.nan)
        d21p = p_.get("T2_vs_T1_d", math.nan)
        d31r = r.get("T3_vs_T1_d", math.nan)
        d31p = p_.get("T3_vs_T1_d", math.nan)
        mr   = r.get("missing_rate", math.nan)
        mp   = p_.get("missing_rate", math.nan)
        fmt_d = lambda v: f"{v:+.3f}" if not math.isnan(v) else "--"
        tbl(col, fmt_d(d21r), fmt_d(d21p), fmt_d(d31r), fmt_d(d31p),
            f"{mr:.1%}" if not math.isnan(mr) else "--",
            f"{mp:.1%}" if not math.isnan(mp) else "--")
    p()

    h2("5. Cross-Modal Correlations: Raw vs Preprocessed")
    p("Pearson r between key physiological/audio features and self-report labels.  \n"
      "q_fdr = BH-corrected false discovery rate. ** = q<0.05 after FDR correction.")
    tbl("Feature", "Label", "r raw", "r prep", "q raw", "q prep", "n")
    tbl_sep(35, 22, 6, 7, 6, 7, 4)
    feat_sub = [c for c in (PHYSIO_KEY[:5] + ET_KEY + AUDIO_KEY[:3]) if c in corr_df_raw["feature"].values]
    lbl_sub  = [c for c in LABEL_KEY if c in corr_df_raw["label"].values]
    raw_corr = corr_df_raw.set_index(["feature", "label"])
    pre_corr = corr_df_pre.set_index(["feature", "label"])
    for feat in feat_sub:
        for lbl in lbl_sub:
            rr = raw_corr.loc[(feat, lbl)] if (feat, lbl) in raw_corr.index else None
            pr = pre_corr.loc[(feat, lbl)] if (feat, lbl) in pre_corr.index else None
            if rr is None and pr is None:
                continue
            r_raw = float(rr["r"]) if rr is not None else math.nan
            r_pre = float(pr["r"]) if pr is not None else math.nan
            q_raw = float(rr.get("q_fdr", math.nan)) if rr is not None else math.nan
            q_pre = float(pr.get("q_fdr", math.nan)) if pr is not None else math.nan
            n     = int(rr["n"]) if rr is not None else (int(pr["n"]) if pr is not None else 0)
            sig_r = "**" if not math.isnan(q_raw) and q_raw < 0.05 else ""
            sig_p = "**" if not math.isnan(q_pre) and q_pre < 0.05 else ""
            r_raw_s = f"{r_raw:+.3f}{sig_r}" if not math.isnan(r_raw) else "--"
            r_pre_s = f"{r_pre:+.3f}{sig_p}" if not math.isnan(r_pre) else "--"
            q_raw_s = f"{q_raw:.3f}" if not math.isnan(q_raw) else "--"
            q_pre_s = f"{q_pre:.3f}" if not math.isnan(q_pre) else "--"
            tbl(feat, lbl, r_raw_s, r_pre_s, q_raw_s, q_pre_s, n)
    p()

    h2("6. Plausibility Gate Thresholds")
    p("Absolute feature bounds:")
    tbl("Feature", "Lower bound", "Upper bound", "Unit")
    tbl_sep(35, 12, 12, 28)
    units = {
        "hr_mean_bpm": "bpm", "hrv_rmssd_ms": "ms", "eda_tonic_mean": "uS",
        "eda_phasic_rate_hz": "Hz", "temp_mean": "deg C", "pupil_mean": "mm",
        "audio_pitch_mean": "semitones from 27.5 Hz", "audio_hnr_mean": "dB",
        "audio_jitter_mean": "fraction", "audio_speaking_fraction": "fraction",
    }
    for feat, (lo, hi) in list(PLAUSIBILITY_ABS.items())[:12]:
        tbl(feat, lo, hi, units.get(feat, "--"))
    p()
    p("Delta bounds (change from T0 baseline):")
    tbl("Feature", "Lower bound", "Upper bound")
    tbl_sep(45, 12, 12)
    for feat, (lo, hi) in PLAUSIBILITY_DELTA.items():
        tbl(feat, lo, hi)
    p()

    h2("7. Output Files")
    p("| File | Description |")
    p("|------|-------------|")
    p("| `results/statistics/preprocessed/participant_task_preprocessed.tsv` | "
      "Full participant-task table with preprocessed features + original labels |")
    p("| `results/statistics/preprocessed/feature_qc_summary.tsv` | "
      "Per-feature missingness and distribution statistics before/after preprocessing |")
    p("| `results/statistics/preprocessed/task_effect_stats.tsv` | "
      "Task effect Cohen's d and p-values on preprocessed features |")
    p("| `results/statistics/preprocessed/cross_modal_correlations.tsv` | "
      "Pearson r between preprocessed features and self-report labels (BH-FDR corrected) |")
    p("| `results/benchmarks/preprocessed_participant_task.tsv` | "
      "Feature-selected version used for LOGO-CV benchmarks; residual NaNs are retained for fold-local KNN imputation |")
    p()

    h2("8. Known Limitations and Caveats")
    lines.append("- **HRV RMSSD quantisation floor**: At 25 Hz PPG sampling, the minimum")
    lines.append("  detectable RR interval difference is 40 ms, setting a noise floor on RMSSD.")
    lines.append("  The plausibility lower bound (10 ms) retains physiologically plausible values")
    lines.append("  while the 25 Hz floor reduces resolution for participants with fast HR.")
    lines.append("- **Audio T0 artefact**: The resting baseline recording (T0) used lapel")
    lines.append("  microphones that misclassify ambient noise during silence, inflating")
    lines.append("  speaking fraction and energy metrics. All T0 audio delta features should be")
    lines.append("  treated as unreliable; they are dropped from benchmark feature sets.")
    lines.append("- **Within-person z-score and between-person regression**: Within-person")
    lines.append("  normalisation is the correct choice for state/task benchmarks but destroys")
    lines.append("  between-person signal needed for trait/affective regression. Benchmarks are")
    lines.append("  split accordingly â€” see Section 7 of the paper.")
    lines.append("- **Small sample (n=40)**: All results are exploratory. Effect sizes and")
    lines.append("  confidence intervals should be interpreted alongside missingness rates.")
    lines.append("- **Fold-local benchmark imputation**: The descriptive preprocessing audit")
    lines.append("  includes a KNN-imputed pass for coverage and effect-size summaries, but")
    lines.append("  the benchmark-ready table retains residual NaNs. `run_neurips_benchmark_baselines.py`")
    lines.append("  fits KNN imputers inside each LOGO-CV training fold and applies them only")
    lines.append("  to the held-out group, avoiding missing-data leakage across folds.")
    p()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", out_path)


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> int:
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    LOG.info("Loading participant-task table: %s", STATS_PT_PATH)
    df_raw = pd.read_csv(STATS_PT_PATH, sep="\t")

    # Normalise task column name
    if "task_id" in df_raw.columns and "task" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"task_id": "task"})

    # Merge audio features if available
    if AUDIO_PATH.exists():
        audio = pd.read_csv(AUDIO_PATH, sep="\t")
        if "task_id" in audio.columns:
            if "task" in audio.columns:
                audio = audio.drop(columns=["task"])
            audio = audio.rename(columns={"task_id": "task"})
        src_cols = [
            "speaking_fraction", "overlap_fraction", "energy_mean", "energy_sd",
            "pitch_mean", "pitch_sd", "hnr_mean", "jitter_mean", "shimmer_mean",
            "voiced_segments_per_sec", "mean_voiced_segment_s", "speech_rate_proxy",
        ]
        avail = [c for c in src_cols if c in audio.columns]
        merge_key = [c for c in ["session_id", "participant_id", "task"] if c in audio.columns and c in df_raw.columns]
        audio_sel = audio[merge_key + avail].rename(columns={c: f"audio_{c}" for c in avail})
        # Drop duplicate audio cols already in df_raw
        dup = [c for c in audio_sel.columns if c in df_raw.columns and c not in merge_key]
        if dup:
            audio_sel = audio_sel.drop(columns=dup)
        df_raw = df_raw.merge(audio_sel, on=merge_key, how="left")
        LOG.info("Merged audio features: %d new columns", len(avail))

    # Active tasks only for preprocessing statistics
    df_active_raw = df_raw[df_raw["task"].isin(ACTIVE_TASKS)].copy().reset_index(drop=True)

    # Determine person identifier
    person_col = "participant_uid" if "participant_uid" in df_raw.columns else "participant_id"

    LOG.info("Raw table: %d rows Ã— %d columns", len(df_raw), len(df_raw.columns))

    # ---- Run preprocessing with audit ----
    LOG.info("Running preprocessing pipeline...")
    df_pre, feat_cols, audit = run_preprocessing_with_audit(df_active_raw, person_col)

    LOG.info("Preprocessing complete: %d features retained", len(feat_cols))
    for _, row in audit.summary().iterrows():
        LOG.info("  %-35s new NaNs: %d  clipped: %d",
                 row["step"], row["new_nans"], row["values_clipped"])

    # ---- Feature QC summary ----
    qc_df = feature_qc_summary(df_active_raw, df_pre, feat_cols)

    # ---- Task effect statistics: raw vs preprocessed ----
    LOG.info("Computing task effect statistics...")
    task_stats_raw = task_effect_stats(df_active_raw, feat_cols)
    task_stats_pre = task_effect_stats(df_pre, feat_cols)

    # ---- Cross-modal correlations: raw vs preprocessed ----
    avail_labels = [c for c in LABEL_KEY if c in df_active_raw.columns]
    LOG.info("Computing cross-modal correlations (%d labels)...", len(avail_labels))
    corr_raw = cross_modal_correlations(df_active_raw, feat_cols, avail_labels)
    corr_pre = cross_modal_correlations(df_pre, feat_cols, avail_labels)

    # ---- Save outputs ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pt_out = OUT_DIR / "participant_task_preprocessed.tsv"
    df_pre.to_csv(pt_out, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows Ã— %d cols)", pt_out, len(df_pre), len(df_pre.columns))

    qc_out = OUT_DIR / "feature_qc_summary.tsv"
    qc_df.to_csv(qc_out, sep="\t", index=False)
    LOG.info("Wrote %s", qc_out)

    task_out = OUT_DIR / "task_effect_stats.tsv"
    task_stats_pre.to_csv(task_out, sep="\t", index=False)
    LOG.info("Wrote %s", task_out)

    corr_out = OUT_DIR / "cross_modal_correlations.tsv"
    corr_pre.to_csv(corr_out, sep="\t", index=False)
    LOG.info("Wrote %s", corr_out)

    # Also save raw correlation for comparison
    corr_raw_out = OUT_DIR / "cross_modal_correlations_raw.tsv"
    corr_raw.to_csv(corr_raw_out, sep="\t", index=False)

    # ---- Markdown report ----
    LOG.info("Writing markdown report...")
    write_markdown_report(
        audit=audit,
        qc_df=qc_df,
        task_df_raw=task_stats_raw,
        task_df_pre=task_stats_pre,
        corr_df_raw=corr_raw,
        corr_df_pre=corr_pre,
        n_raw_rows=len(df_active_raw),
        n_feat_raw=len(feat_cols),
        n_feat_pre=len(feat_cols),
        out_path=REPORT_PATH,
    )

    # ---- Console summary ----
    print("\n=== PREPROCESSING AUDIT ===")
    print(audit.summary().to_string(index=False))
    print("\n=== TASK EFFECT: T2 vs T1 Cohen's d (key features) ===")
    key_f = [c for c in PHYSIO_KEY[:6] + ET_KEY + AUDIO_KEY[:3] if c in task_stats_raw["feature"].values]
    raw_i = task_stats_raw.set_index("feature")
    pre_i = task_stats_pre.set_index("feature")
    for c in key_f:
        if c not in raw_i.index:
            continue
        d_raw = raw_i.loc[c, "T2_vs_T1_d"]
        d_pre = pre_i.loc[c, "T2_vs_T1_d"] if c in pre_i.index else math.nan
        direction = ("UP" if abs(d_pre) > abs(d_raw) else "down") if not math.isnan(d_raw) and not math.isnan(d_pre) else "?"
        print(f"  {c:40s} raw d={d_raw:+.3f}  pre d={d_pre:+.3f}  {direction}")

    print("\n=== TOP CROSS-MODAL CORRELATIONS (preprocessed, |r|>0.15) ===")
    top_corr = corr_pre[corr_pre["r"].abs() > 0.15].sort_values("r", key=abs, ascending=False).head(15)
    for _, r in top_corr.iterrows():
        q_str = f"q={r.get('q_fdr', float('nan')):.3f}" if "q_fdr" in r and not math.isnan(r.get("q_fdr", float("nan"))) else ""
        print(f"  {r['feature']:35s} Ã— {r['label']:25s} r={r['r']:+.3f}  n={int(r['n'])}  {q_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

