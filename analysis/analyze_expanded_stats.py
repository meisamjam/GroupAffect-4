"""Expanded multi-modal statistics for GroupAffect-4 dataset.

Builds on the outputs of analyze_multimodal_statistics, analyze_individual_audio,
and analyze_cross_modal to produce:

  1. Group-level speech dynamics (Gini, speaking balance, BFI Ã— audio)
  2. Task-stratified BFI Ã— audio/physio/VAD correlations
  3. Cross-modal affect convergence (physio â†” VAD â†” audio congruence)
  4. Multi-modal task profiles (what distinguishes each task)
  5. Paper-quality summary figures

Inputs (auto-discovered from default paths):
  - results/statistics/analysis_dataset_participant_task.tsv  (215 cols)
  - results/audio/individual_audio_task.tsv                   (68 cols, with group share)
  - results/personality/group_trait_stats.tsv
  - data/derived_features/features_group_dynamics_task.tsv
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("analyze_expanded_stats")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
ACTIVE_TASKS = ["T1", "T2", "T3", "T4"]

BFI_TRAITS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]
BFI_LABELS = {
    "bfi44_e": "Extraversion",
    "bfi44_a": "Agreeableness",
    "bfi44_c": "Conscientiousness",
    "bfi44_n": "Neuroticism",
    "bfi44_o": "Openness",
}

AUDIO_FEATURES = [
    "audio_speaking_fraction",
    # audio_turn_count: not available â€” requires transcripts (transcript_available=False for all sessions)
    "audio_overlap_fraction",
    "audio_energy_mean",
    "audio_pitch_mean",
    "audio_pitch_sd",
    "audio_speech_rate_proxy",
    "audio_hnr_mean",
    "audio_jitter_mean",
    "audio_shimmer_mean",
]

# Absolute audio features (active tasks only; avoid T0-delta artefact)
AUDIO_ABS_FEATURES = [
    "audio_speaking_fraction",
    "audio_overlap_fraction",
    "audio_energy_mean",
    "audio_pitch_mean",
    "audio_pitch_sd",
    "audio_speech_rate_proxy",
    "audio_hnr_mean",
    "audio_jitter_mean",
    "audio_shimmer_mean",
]

# Keep delta list for legacy reference but flag as artefactual for T0-delta use
AUDIO_DELTA_FEATURES = [f + "_delta_t0" for f in AUDIO_FEATURES]

PHYSIO_DELTA_FEATURES = [
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_mean_delta_t0",
    "eda_phasic_rate_hz_delta_t0",
    "scr_rate_hz_delta_t0",
    "temp_mean_delta_t0",
    "pupil_mean_delta_t0",
]

VAD_FEATURES = ["vad_arousal_self_num", "vad_valence_self_num", "vad_dominance_self_num"]

POSTBLOCK_FEATURES = [
    "ans_engagement",
    "ans_mental_demand",
    "ans_voice_inclusion",
    "ans_team_coordination",
    "ans_fairness",
    "ans_satisfaction",
    "ans_regret",
    "ans_psych_safety",
    "ans_overall_valence",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required table missing: {path}")
        LOG.warning("Optional table not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    LOG.info("Loaded %s (%d rows, %d cols)", path.name, len(df), len(df.columns))
    return df


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    LOG.info("Wrote %s (%d rows)", path, len(df))


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> tuple[float, float]:
    try:
        from scipy import stats as sp
    except ImportError:
        return float("nan"), float("nan")
    pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                         "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(pair) < 5 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn = sp.pearsonr if method == "pearson" else sp.spearmanr
        res = fn(pair["a"], pair["b"])
    return float(res.statistic), float(res.pvalue)


def _bh_fdr(pvals: list[float]) -> list[float]:
    raw = np.array([float(p) if p is not None else np.nan for p in pvals])
    q = np.full(raw.shape, np.nan)
    valid = np.where(np.isfinite(raw))[0]
    if len(valid) == 0:
        return q.tolist()
    ordered = valid[np.argsort(raw[valid])]
    n = len(ordered)
    ranked = raw[ordered] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ordered] = np.clip(adj, 0.0, 1.0)
    return q.tolist()


# ---------------------------------------------------------------------------
# 1. Group-level speech dynamics
# ---------------------------------------------------------------------------

def compute_group_speech_dynamics(audio: pd.DataFrame) -> pd.DataFrame:
    """Compute per-group-task Gini coefficient and speaking balance metrics."""
    rows: list[dict] = []
    for (grp, task), gdf in audio.groupby(["group_id", "task_id"]):
        vals = pd.to_numeric(gdf["speaking_fraction"], errors="coerce").dropna()
        n = len(vals)
        if n < 2:
            continue
        total = vals.sum()
        gini = float("nan")
        if total > 0 and n > 1:
            sorted_vals = np.sort(vals.values)
            idx = np.arange(1, n + 1)
            gini = float((2 * (idx * sorted_vals).sum() - (n + 1) * sorted_vals.sum())
                         / (n * sorted_vals.sum()))
        share_vals = pd.to_numeric(gdf["speaking_share_group"], errors="coerce").dropna()
        dominant_share = float(share_vals.max()) if len(share_vals) else float("nan")
        rows.append({
            "group_id": grp,
            "task_id": task,
            "n_speakers": n,
            "gini_speaking": gini,
            "dominant_speaker_share": dominant_share,
            "mean_speaking_fraction": float(vals.mean()),
            "sd_speaking_fraction": float(vals.std(ddof=1)) if n > 1 else float("nan"),
            "max_speaking_fraction": float(vals.max()),
            "min_speaking_fraction": float(vals.min()),
        })
    return pd.DataFrame(rows)


def merge_group_personality(group_speech: pd.DataFrame, group_traits: pd.DataFrame) -> pd.DataFrame:
    gt = group_traits.copy()
    return group_speech.merge(gt, on="group_id", how="left")


def compute_group_bfi_audio_correlations(
    group_speech_pers: pd.DataFrame, min_n: int = 6
) -> pd.DataFrame:
    """Correlate group BFI means with group speech Gini and balance, per task.

    NOTE: n = 8â€“10 groups. All results are EXPLORATORY. Power is insufficient for
    conventional significance testing (|r|=0.5 requires nâ‰¥26 for 80% power).
    P-values are reported descriptively only.
    """
    speech_features = ["gini_speaking", "dominant_speaker_share", "mean_speaking_fraction"]
    bfi_group_cols = [f"{t}_mean" for t in BFI_TRAITS]
    rows: list[dict] = []
    for task in ACTIVE_TASKS:
        tdf = group_speech_pers[group_speech_pers["task_id"] == task].copy()
        if len(tdf) < min_n:
            continue
        for bf in bfi_group_cols:
            if bf not in tdf.columns:
                continue
            for sf in speech_features:
                if sf not in tdf.columns:
                    continue
                n = int(tdf[[bf, sf]].dropna().shape[0])
                r, p = _safe_corr(tdf[bf], tdf[sf])
                rows.append({
                    "task": task, "bfi_group": bf, "speech_feature": sf,
                    "n": n,
                    "pearson_r": r, "p_pearson": p,
                    "exploratory": True,
                    "power_note": (
                        f"EXPLORATORY: n={n} groups; 80% power for |r|=0.5 requires nâ‰¥26. "
                        "P-values are descriptive only."
                    ),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_pearson"] = _bh_fdr(out["p_pearson"].tolist())
    return out


# ---------------------------------------------------------------------------
# 2. Task-stratified BFI Ã— audio/physio correlations
# ---------------------------------------------------------------------------

def compute_bfi_feature_by_task(
    df: pd.DataFrame,
    feature_cols: list[str],
    bfi_z_cols: list[str],
    min_n: int = 8,
    scope_prefix: str = "by_task",
) -> pd.DataFrame:
    """Correlate BFI traits with features within each task separately.

    Each task slice has one row per participant (n â‰ˆ n_unique_participants),
    so this is an approximately-correct between-person analysis.
    Label: analysis_level='within_task_between_person'.
    """
    rows: list[dict] = []
    for task in TASK_ORDER:
        tdf = df[df["task"] == task].copy() if "task" in df.columns else df.copy()
        # Deduplicate to one row per participant (safeguard against duplicate rows)
        if "participant_uid" in tdf.columns:
            tdf = tdf.drop_duplicates(subset=["participant_uid"])
        elif "participant_id" in tdf.columns:
            tdf = tdf.drop_duplicates(subset=["participant_id"])
        for trait in bfi_z_cols:
            if trait not in tdf.columns:
                continue
            for feat in feature_cols:
                if feat not in tdf.columns:
                    continue
                pair = tdf[[trait, feat]].apply(pd.to_numeric, errors="coerce").dropna()
                n = len(pair)
                if n < min_n:
                    continue
                r, p = _safe_corr(pair[trait], pair[feat])
                rho, prho = _safe_corr(pair[trait], pair[feat], "spearman")
                rows.append({
                    "scope": f"{scope_prefix}_{task}",
                    "task": task,
                    "trait": trait,
                    "feature": feat,
                    "n": n,
                    "analysis_level": "within_task_between_person",
                    "pearson_r": r,
                    "p_pearson": p,
                    "spearman_rho": rho,
                    "p_spearman": prho,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_pearson"] = _bh_fdr(out["p_pearson"].tolist())
        out["q_spearman"] = _bh_fdr(out["p_spearman"].tolist())
    return out


def compute_bfi_participant_mean_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    bfi_z_cols: list[str],
    min_n: int = 5,
    active_tasks_only: bool = True,
) -> pd.DataFrame:
    """True between-person BFI correlations at the participant level.

    Aggregates each participant's features to their mean across active tasks,
    then correlates with BFI traits (which are constant per participant).
    This eliminates pseudo-replication: n = n_unique_participants (22â€“32).

    Label: analysis_level='between_person_participant_mean'.
    """
    uid_col = "participant_uid" if "participant_uid" in df.columns else "participant_id"
    tasks_subset = df[df["task"].isin(ACTIVE_TASKS)] if active_tasks_only and "task" in df.columns else df

    # One row per participant: mean of features across tasks, keep BFI (constant)
    agg_cols = [c for c in feature_cols + bfi_z_cols if c in tasks_subset.columns]
    if uid_col not in tasks_subset.columns or not agg_cols:
        return pd.DataFrame()

    for col in agg_cols:
        tasks_subset = tasks_subset.copy()
        tasks_subset[col] = pd.to_numeric(tasks_subset[col], errors="coerce")

    participant_means = (
        tasks_subset[[uid_col] + agg_cols]
        .groupby(uid_col)[agg_cols]
        .mean()
        .reset_index()
    )

    n_participants = participant_means[uid_col].nunique()
    rows: list[dict] = []
    for trait in bfi_z_cols:
        if trait not in participant_means.columns:
            continue
        for feat in feature_cols:
            if feat not in participant_means.columns:
                continue
            pair = participant_means[[trait, feat]].dropna()
            n = len(pair)
            if n < min_n:
                continue
            r, p = _safe_corr(pair[trait], pair[feat])
            rho, prho = _safe_corr(pair[trait], pair[feat], "spearman")
            rows.append({
                "scope": "participant_mean_active_tasks",
                "task": "all_active",
                "trait": trait,
                "feature": feat,
                "n": n,
                "n_unique_participants": n_participants,
                "analysis_level": "between_person_participant_mean",
                "note": "participant mean across T1â€“T4; n=unique participants, not task rows",
                "pearson_r": r,
                "p_pearson": p,
                "spearman_rho": rho,
                "p_spearman": prho,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_pearson"] = _bh_fdr(out["p_pearson"].tolist())
        out["q_spearman"] = _bh_fdr(out["p_spearman"].tolist())
    return out


def compare_pooled_vs_participant_level(
    by_task_df: pd.DataFrame,
    participant_level_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge by-task and participant-level BFI correlations for side-by-side comparison.

    Allows auditing whether pooled (inflated N) and participant-level (correct N)
    correlations agree in direction and magnitude.
    """
    if by_task_df.empty or participant_level_df.empty:
        return pd.DataFrame()

    # Pool the by-task rows to one row per trait Ã— feature (mean r across tasks)
    pooled_summary = (
        by_task_df.groupby(["trait", "feature"])
        .agg(
            n_task_rows_total=("n", "sum"),
            n_tasks=("task", "nunique"),
            pearson_r_mean_across_tasks=("pearson_r", "mean"),
            p_pearson_min=("p_pearson", "min"),
        )
        .reset_index()
    )
    pooled_summary["analysis_level"] = "pooled_by_task_summary"

    part_sub = participant_level_df[
        ["trait", "feature", "n", "n_unique_participants",
         "pearson_r", "p_pearson", "spearman_rho", "p_spearman",
         "q_pearson", "analysis_level"]
    ].copy()
    part_sub = part_sub.rename(columns={
        "n": "n_participant_level",
        "pearson_r": "pearson_r_participant_level",
        "p_pearson": "p_pearson_participant_level",
        "spearman_rho": "spearman_rho_participant_level",
        "p_spearman": "p_spearman_participant_level",
        "q_pearson": "q_pearson_participant_level",
    })

    merged = pooled_summary.merge(part_sub, on=["trait", "feature"], how="outer")
    merged["direction_agrees"] = (
        (merged["pearson_r_mean_across_tasks"] * merged["pearson_r_participant_level"]) > 0
    )
    merged = merged.sort_values("pearson_r_participant_level", key=abs, ascending=False)
    return merged


# ---------------------------------------------------------------------------
# 3. Cross-modal affect convergence
# ---------------------------------------------------------------------------

def compute_affect_convergence(df: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    """
    Cross-modal convergence: do physio/audio deltas agree with VAD self-reports?
    Computes Pearson r between each (physio/audio delta) Ã— VAD pairing, pooled and by task.
    """
    physio_del = _present(df, PHYSIO_DELTA_FEATURES)
    # Use absolute audio features (T0 delta baseline is artefactual)
    audio_abs = _present(df, AUDIO_ABS_FEATURES)
    vad_present = _present(df, VAD_FEATURES)
    rows: list[dict] = []

    for scope, subset in [("all_tasks", df)] + [
        (f"task_{t}", df[df["task"] == t]) for t in ACTIVE_TASKS if "task" in df.columns
    ]:
        for pred in physio_del + audio_abs:
            for vad in vad_present:
                pair = subset[[pred, vad]].apply(pd.to_numeric, errors="coerce").dropna()
                n = len(pair)
                if n < min_n:
                    continue
                r, p = _safe_corr(pair[pred], pair[vad])
                rows.append({
                    "scope": scope,
                    "predictor": pred,
                    "vad": vad,
                    "modality": "physio" if pred in physio_del else "audio_abs",
                    "n": n,
                    "pearson_r": r,
                    "p_pearson": p,
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_pearson"] = _bh_fdr(out["p_pearson"].tolist())
    return out


def compute_affect_discordance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where physio arousal direction disagrees with VAD arousal.
    High EDA/HR but low VAD arousal = discordant.
    Returns participant-task rows with discordance flag and z-scores.
    """
    out = df.copy()
    # Use corrected physio columns (hr_mean_bpm_delta_t0 from new pipeline)
    physio_arousal_cols = ["eda_mean_delta_t0", "hr_mean_bpm_delta_t0"]
    for col in physio_arousal_cols + ["vad_arousal_self_num"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    has_physio = any(c in out.columns for c in physio_arousal_cols)
    has_vad = "vad_arousal_self_num" in out.columns

    if not (has_physio and has_vad):
        return pd.DataFrame()

    # Composite physiological arousal (z-scored average of available arousal-proxy deltas)
    for col in physio_arousal_cols:
        if col not in out.columns:
            continue
        mu, sd = out[col].mean(), out[col].std(ddof=1)
        if sd > 0:
            out[f"{col}_z"] = (out[col] - mu) / sd
        else:
            out[f"{col}_z"] = 0.0

    z_cols = [f"{c}_z" for c in physio_arousal_cols if f"{c}_z" in out.columns]
    out["physio_arousal_z"] = out[z_cols].mean(axis=1)

    vad_mu = out["vad_arousal_self_num"].mean()
    vad_sd = out["vad_arousal_self_num"].std(ddof=1)
    if vad_sd > 0:
        out["vad_arousal_z"] = (out["vad_arousal_self_num"] - vad_mu) / vad_sd
    else:
        out["vad_arousal_z"] = 0.0

    out["affect_discordance"] = (out["physio_arousal_z"] - out["vad_arousal_z"]).abs()
    out["physio_high_vad_low"] = (out["physio_arousal_z"] > 0.5) & (out["vad_arousal_z"] < -0.5)
    out["physio_low_vad_high"] = (out["physio_arousal_z"] < -0.5) & (out["vad_arousal_z"] > 0.5)

    keep_cols = ["session_id", "task", "participant_id", "group_id",
                 "physio_arousal_z", "vad_arousal_z", "affect_discordance",
                 "physio_high_vad_low", "physio_low_vad_high",
                 "eda_mean_delta_t0_z", "hr_mean_bpm_delta_t0_z"]
    return out[[c for c in keep_cols if c in out.columns]].dropna(subset=["physio_arousal_z", "vad_arousal_z"])


# ---------------------------------------------------------------------------
# 4. Multi-modal task profiles
# ---------------------------------------------------------------------------

def compute_task_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-task mean and 95%CI for key features across all modalities."""
    feature_set = (
        _present(df, VAD_FEATURES)
        + _present(df, POSTBLOCK_FEATURES)
        + _present(df, PHYSIO_DELTA_FEATURES)
        + _present(df, AUDIO_DELTA_FEATURES)
    )
    rows: list[dict] = []
    for task in TASK_ORDER:
        tdf = df[df["task"] == task] if "task" in df.columns else df
        for feat in feature_set:
            vals = pd.to_numeric(tdf[feat], errors="coerce").dropna()
            n = len(vals)
            if n < 3:
                continue
            se = vals.std(ddof=1) / np.sqrt(n)
            rows.append({
                "task": task,
                "feature": feat,
                "n": n,
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)),
                "se": float(se),
                "ci95_low": float(vals.mean() - 1.96 * se),
                "ci95_high": float(vals.mean() + 1.96 * se),
                "median": float(vals.median()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------

def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        return plt, mticker
    except ImportError:
        return None, None


def fig_group_speech_balance(group_speech: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    plt, mticker = _mpl()
    if plt is None:
        return

    active = group_speech[group_speech["task_id"].isin(ACTIVE_TASKS)].copy()
    if active.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Group Speech Balance by Task", fontsize=13, fontweight="bold")

    # Gini by task (violin)
    ax = axes[0]
    task_ginis = [
        active[active["task_id"] == t]["gini_speaking"].dropna().values
        for t in ACTIVE_TASKS
    ]
    valid_idx = [i for i, v in enumerate(task_ginis) if len(v) >= 2]
    if valid_idx:
        parts = ax.violinplot(
            [task_ginis[i] for i in valid_idx],
            positions=valid_idx, showmedians=True, showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
    ax.set_xticks(range(len(ACTIVE_TASKS)))
    ax.set_xticklabels(ACTIVE_TASKS)
    ax.set_xlabel("Task")
    ax.set_ylabel("Gini coefficient (speaking time)")
    ax.set_title("Speaking Inequality by Task")
    ax.axhline(0.4, ls="--", color="gray", lw=0.8, label="Gini=0.4")
    ax.legend(fontsize=9)

    # Dominant speaker share by task
    ax = axes[1]
    colors = plt.cm.Set2(np.linspace(0, 1, len(ACTIVE_TASKS)))
    for i, task in enumerate(ACTIVE_TASKS):
        tdf = active[active["task_id"] == task]
        vals = tdf["dominant_speaker_share"].dropna()
        ax.scatter(
            [i] * len(vals) + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals)),
            vals.values, alpha=0.5, color=colors[i], s=40,
        )
        if len(vals) >= 2:
            ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1),
                        fmt="o", color=colors[i], capsize=4, markersize=8)
    ax.set_xticks(range(len(ACTIVE_TASKS)))
    ax.set_xticklabels(ACTIVE_TASKS)
    ax.set_xlabel("Task")
    ax.set_ylabel("Dominant speaker share")
    ax.set_title("Dominant Speaker Fraction by Task")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_speech_balance.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_bfi_audio_heatmap(corr_df: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    plt, _ = _mpl()
    if plt is None or corr_df.empty:
        return

    sig = corr_df[(corr_df["p_pearson"] < 0.05)].copy()
    if sig.empty:
        sig = corr_df.copy()

    # Pivot: traits x audio features, faceted by task
    tasks = [t for t in ACTIVE_TASKS if t in sig["task"].values]
    if not tasks:
        return

    bfi_z = [f"{t}_z" for t in BFI_TRAITS]
    feat_short = {f: f.replace("audio_", "").replace("_delta_t0", "Î”").replace("_mean", "_Î¼") for f in AUDIO_DELTA_FEATURES}

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 5), squeeze=False)
    fig.suptitle("BFI Traits Ã— Audio Features (Pearson r, per task)", fontsize=12, fontweight="bold")

    all_r = corr_df["pearson_r"].replace([np.inf, -np.inf], np.nan).dropna()
    vmax = float(all_r.abs().quantile(0.95)) if len(all_r) else 0.4

    for col_i, task in enumerate(tasks):
        ax = axes[0, col_i]
        tdf = sig[sig["task"] == task]

        audio_feats = [f for f in AUDIO_DELTA_FEATURES if f in tdf["feature"].values]
        bfi_present = [b for b in bfi_z if b in tdf["trait"].values]
        if not audio_feats or not bfi_present:
            ax.set_visible(False)
            continue

        mat = np.full((len(bfi_present), len(audio_feats)), np.nan)
        for ri, trait in enumerate(bfi_present):
            for ci, feat in enumerate(audio_feats):
                row = tdf[(tdf["trait"] == trait) & (tdf["feature"] == feat)]
                if not row.empty:
                    mat[ri, ci] = float(row["pearson_r"].iloc[0])

        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Task {task}", fontsize=10)
        ax.set_yticks(range(len(bfi_present)))
        ax.set_yticklabels([BFI_LABELS.get(b.replace("_z", ""), b) for b in bfi_present], fontsize=8)
        ax.set_xticks(range(len(audio_feats)))
        ax.set_xticklabels([feat_short.get(f, f) for f in audio_feats], rotation=45, ha="right", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ri in range(len(bfi_present)):
            for ci in range(len(audio_feats)):
                trait, feat = bfi_present[ri], audio_feats[ci]
                p_row = corr_df[(corr_df["task"] == task) & (corr_df["trait"] == trait) & (corr_df["feature"] == feat)]
                if not p_row.empty and float(p_row["p_pearson"].iloc[0]) < 0.05:
                    ax.text(ci, ri, "*", ha="center", va="center", fontsize=10, color="black")

    plt.tight_layout()
    path = out_dir / "bfi_audio_correlation_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_bfi_physio_audio_combined(
    bfi_physio: pd.DataFrame, bfi_audio: pd.DataFrame, out_dir: Path, dpi: int = 150
) -> None:
    """Forest plot of strongest BFI correlations with physio and audio features."""
    plt, _ = _mpl()
    if plt is None:
        return

    combined = pd.concat([
        bfi_physio.assign(modality="physio"),
        bfi_audio.assign(modality="audio"),
    ], ignore_index=True)
    if combined.empty:
        return

    sig = combined[(combined["p_pearson"] < 0.05)].copy()
    if len(sig) < 3:
        sig = combined.copy()

    sig = sig.sort_values("pearson_r", key=abs, ascending=False).head(40)
    sig["label"] = sig.apply(
        lambda r: f"{BFI_LABELS.get(r['trait'].replace('_z',''), r['trait'])} Ã— "
                  f"{r['feature'].replace('_delta_t0','Î”').replace('ppg_','').replace('audio_','')} [{r['task']}]",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(sig) * 0.3)))
    colors = {"physio": "#2196F3", "audio": "#FF9800"}
    for i, (_, row) in enumerate(sig.iterrows()):
        color = colors.get(row.get("modality", "physio"), "gray")
        ax.barh(i, row["pearson_r"], color=color, alpha=0.7, height=0.6)
        if row["p_pearson"] < 0.01:
            ax.text(row["pearson_r"] + (0.01 if row["pearson_r"] >= 0 else -0.01),
                    i, "**", va="center", ha="left" if row["pearson_r"] >= 0 else "right",
                    fontsize=8)
        elif row["p_pearson"] < 0.05:
            ax.text(row["pearson_r"] + (0.01 if row["pearson_r"] >= 0 else -0.01),
                    i, "*", va="center", ha="left" if row["pearson_r"] >= 0 else "right",
                    fontsize=8)

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["label"].tolist(), fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title("BFI Ã— Physio/Audio: Top Correlations (p < 0.05)", fontsize=11, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["physio"], label="Physio"), Patch(color=colors["audio"], label="Audio")],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    path = out_dir / "bfi_physio_audio_forest.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_cross_modal_convergence(conv_df: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    plt, _ = _mpl()
    if plt is None or conv_df.empty:
        return

    all_scope = conv_df[conv_df["scope"] == "all_tasks"].copy()
    if all_scope.empty:
        return

    # Pivot: predictors Ã— VAD dimensions
    vad_dims = [v for v in VAD_FEATURES if v in all_scope["vad"].values]
    predictors = all_scope["predictor"].unique().tolist()

    mat = np.full((len(predictors), len(vad_dims)), np.nan)
    pmat = np.full_like(mat, np.nan)
    for ri, pred in enumerate(predictors):
        for ci, vad in enumerate(vad_dims):
            row = all_scope[(all_scope["predictor"] == pred) & (all_scope["vad"] == vad)]
            if not row.empty:
                mat[ri, ci] = float(row["pearson_r"].iloc[0])
                pmat[ri, ci] = float(row["p_pearson"].iloc[0])

    # Order by absolute max correlation
    row_max = np.nanmax(np.abs(mat), axis=1)
    order = np.argsort(row_max)[::-1]
    mat, pmat, predictors_ordered = mat[order], pmat[order], [predictors[i] for i in order]

    vmax = float(np.nanquantile(np.abs(mat), 0.95)) if not np.all(np.isnan(mat)) else 0.4
    fig, ax = plt.subplots(figsize=(6, max(5, len(predictors_ordered) * 0.35)))

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Pearson r")

    ax.set_xticks(range(len(vad_dims)))
    ax.set_xticklabels([v.replace("vad_", "").replace("_self_num", "").capitalize() for v in vad_dims], fontsize=9)
    ax.set_yticks(range(len(predictors_ordered)))
    ax.set_yticklabels(
        [p.replace("_delta_t0", "Î”").replace("ppg_", "").replace("audio_", "").replace("_mean", "_Î¼")
         for p in predictors_ordered],
        fontsize=8,
    )
    for ri in range(len(predictors_ordered)):
        for ci in range(len(vad_dims)):
            if np.isfinite(pmat[ri, ci]) and pmat[ri, ci] < 0.05:
                ax.text(ci, ri, "*", ha="center", va="center", fontsize=10)

    ax.set_title("Physio/Audio Î” vs. Self-Reported VAD (All Tasks)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "cross_modal_affect_convergence.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_task_profiles(profiles: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    """Multi-panel radar/bar charts showing task-by-task multi-modal profiles."""
    plt, _ = _mpl()
    if plt is None or profiles.empty:
        return

    highlight_features = {
        "VAD": ["vad_arousal_self_num", "vad_valence_self_num", "vad_dominance_self_num"],
        "Postblock": ["ans_engagement", "ans_mental_demand", "ans_satisfaction", "ans_voice_inclusion"],
        "Physio Î”": ["ppg_rate_proxy_bpm_delta_t0", "eda_mean_delta_t0", "pupil_mean_delta_t0"],
        "Audio Î”": ["audio_speaking_fraction_delta_t0", "audio_overlap_fraction_delta_t0",
                    "audio_energy_mean_delta_t0", "audio_speech_rate_proxy_delta_t0"],
    }

    n_panels = len(highlight_features)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6), squeeze=False)
    fig.suptitle("Multi-Modal Task Profiles (Mean Â± 95%CI)", fontsize=13, fontweight="bold")

    colors = plt.cm.Set1(np.linspace(0, 1, len(TASK_ORDER)))
    task_colors = {t: colors[i] for i, t in enumerate(TASK_ORDER)}

    for col_i, (panel_name, feats) in enumerate(highlight_features.items()):
        ax = axes[0, col_i]
        feats_present = [f for f in feats if f in profiles["feature"].values]
        if not feats_present:
            ax.set_visible(False)
            continue

        short_labels = [
            f.replace("vad_", "").replace("_self_num", "")
             .replace("ans_", "").replace("_delta_t0", "Î”")
             .replace("ppg_rate_proxy_bpm", "HR")
             .replace("eda_mean", "EDA")
             .replace("pupil_mean", "Pupil")
             .replace("audio_speaking_fraction", "Speech frac")
             .replace("audio_overlap_fraction", "Overlap")
             .replace("audio_energy_mean", "Energy")
             .replace("audio_speech_rate_proxy", "Speech rate")
            for f in feats_present
        ]

        x = np.arange(len(feats_present))
        width = 0.15
        n_tasks = len(TASK_ORDER)
        offsets = np.linspace(-(n_tasks - 1) * width / 2, (n_tasks - 1) * width / 2, n_tasks)

        for ti, task in enumerate(TASK_ORDER):
            tdf = profiles[profiles["task"] == task]
            means, ci_lows, ci_highs = [], [], []
            for feat in feats_present:
                row = tdf[tdf["feature"] == feat]
                if row.empty:
                    means.append(np.nan); ci_lows.append(np.nan); ci_highs.append(np.nan)
                else:
                    means.append(float(row["mean"].iloc[0]))
                    ci_lows.append(float(row["ci95_low"].iloc[0]))
                    ci_highs.append(float(row["ci95_high"].iloc[0]))

            means = np.array(means)
            valid = np.isfinite(means)
            if not valid.any():
                continue

            ax.bar(x[valid] + offsets[ti], means[valid], width, color=task_colors[task],
                   alpha=0.7, label=task)
            yerr_low = means[valid] - np.array(ci_lows)[valid]
            yerr_high = np.array(ci_highs)[valid] - means[valid]
            ax.errorbar(x[valid] + offsets[ti], means[valid],
                        yerr=[np.clip(yerr_low, 0, None), np.clip(yerr_high, 0, None)],
                        fmt="none", color="black", capsize=2, lw=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(panel_name, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        if col_i == 0:
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = out_dir / "multimodal_task_profiles.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_gini_bfi_scatter(merged: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    """Scatter: group Extraversion mean vs. speaking Gini per task."""
    plt, _ = _mpl()
    if plt is None or merged.empty:
        return

    if "bfi44_e_mean" not in merged.columns or "gini_speaking" not in merged.columns:
        return

    active = merged[merged["task_id"].isin(ACTIVE_TASKS)].dropna(subset=["bfi44_e_mean", "gini_speaking"])
    if len(active) < 4:
        return

    tasks = [t for t in ACTIVE_TASKS if t in active["task_id"].values]
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), squeeze=False)
    fig.suptitle("Group Extraversion vs. Speaking Inequality (Gini)", fontsize=11, fontweight="bold")

    for col_i, task in enumerate(tasks):
        ax = axes[0, col_i]
        tdf = active[active["task_id"] == task]
        if len(tdf) < 3:
            ax.set_visible(False)
            continue
        x = tdf["bfi44_e_mean"].values
        y = tdf["gini_speaking"].values
        ax.scatter(x, y, color="#5C6BC0", alpha=0.8, s=60, edgecolors="white", lw=0.5)
        if len(tdf) >= 4:
            from numpy.polynomial.polynomial import polyfit as pfit
            try:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 50)
                ax.plot(xline, m * xline + b, "--", color="#E53935", lw=1.5)
                r, p = _safe_corr(pd.Series(x), pd.Series(y))
                ax.text(0.05, 0.95, f"r={r:.2f}, p={p:.3f}", transform=ax.transAxes,
                        fontsize=9, va="top", color="#E53935")
            except Exception:
                pass
        ax.set_title(f"Task {task}", fontsize=10)
        ax.set_xlabel("Group Extraversion (mean)")
        ax.set_ylabel("Gini (speaking time)")

    plt.tight_layout()
    path = out_dir / "gini_extraversion_scatter.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_discordance_by_task(discord: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    """Show physio vs VAD arousal scatter per task with discordance highlighted."""
    plt, _ = _mpl()
    if plt is None or discord.empty:
        return

    tasks = [t for t in ACTIVE_TASKS if "task" in discord.columns and t in discord["task"].values]
    if not tasks:
        return

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), squeeze=False)
    fig.suptitle("Physio Arousal (Î”, z-scored) vs. VAD Self-Reported Arousal", fontsize=11, fontweight="bold")

    for col_i, task in enumerate(tasks):
        ax = axes[0, col_i]
        tdf = discord[discord["task"] == task].dropna(subset=["physio_arousal_z", "vad_arousal_z"])
        if len(tdf) < 3:
            ax.set_visible(False)
            continue

        colors_pts = np.where(tdf.get("physio_high_vad_low", pd.Series([False] * len(tdf))).values, "#E53935",
                    np.where(tdf.get("physio_low_vad_high", pd.Series([False] * len(tdf))).values, "#1E88E5", "#757575"))
        ax.scatter(tdf["vad_arousal_z"], tdf["physio_arousal_z"], c=colors_pts, alpha=0.7, s=40)
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.axvline(0, color="gray", lw=0.6, ls="--")
        r, p = _safe_corr(tdf["vad_arousal_z"], tdf["physio_arousal_z"])
        ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes, fontsize=9, va="top")
        ax.set_title(f"Task {task}", fontsize=10)
        ax.set_xlabel("VAD Arousal (z)")
        ax.set_ylabel("Physio Arousal Î” (z)")

    plt.tight_layout()
    path = out_dir / "physio_vad_arousal_concordance.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_summary_panel(
    participant: pd.DataFrame,
    group_speech_pers: pd.DataFrame,
    out_dir: Path, dpi: int = 150,
) -> None:
    """Paper-quality 3Ã—3 summary panel combining key cross-modal findings."""
    plt, _ = _mpl()
    if plt is None:
        return

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("GroupAffect-4 Multi-Modal Summary", fontsize=14, fontweight="bold", y=1.01)

    # Panel A: VAD arousal by task
    ax_a = fig.add_subplot(3, 3, 1)
    vad_col = "vad_arousal_self_num"
    if vad_col in participant.columns and "task" in participant.columns:
        for i, task in enumerate(ACTIVE_TASKS):
            vals = pd.to_numeric(participant[participant["task"] == task][vad_col], errors="coerce").dropna()
            if len(vals) >= 2:
                ax_a.boxplot(vals.values, positions=[i], widths=0.5, patch_artist=True,
                             boxprops=dict(facecolor=f"C{i}", alpha=0.6),
                             medianprops=dict(color="black", lw=2), showfliers=False)
    ax_a.set_xticks(range(len(ACTIVE_TASKS)))
    ax_a.set_xticklabels(ACTIVE_TASKS)
    ax_a.set_ylabel("VAD Arousal (1â€“9)")
    ax_a.set_title("A: Arousal by Task", fontsize=10, fontweight="bold")

    # Panel B: EDA delta by task
    ax_b = fig.add_subplot(3, 3, 2)
    eda_col = "eda_mean_delta_t0"
    if eda_col in participant.columns and "task" in participant.columns:
        for i, task in enumerate(ACTIVE_TASKS):
            vals = pd.to_numeric(participant[participant["task"] == task][eda_col], errors="coerce").dropna()
            if len(vals) >= 2:
                ax_b.boxplot(vals.values, positions=[i], widths=0.5, patch_artist=True,
                             boxprops=dict(facecolor=f"C{i}", alpha=0.6),
                             medianprops=dict(color="black", lw=2), showfliers=False)
    ax_b.set_xticks(range(len(ACTIVE_TASKS)))
    ax_b.set_xticklabels(ACTIVE_TASKS)
    ax_b.set_ylabel("EDA Î” vs T0")
    ax_b.set_title("B: Skin Conductance Î”", fontsize=10, fontweight="bold")

    # Panel C: Speaking fraction by task
    ax_c = fig.add_subplot(3, 3, 3)
    spk_col = "audio_speaking_fraction_delta_t0"
    if spk_col in participant.columns and "task" in participant.columns:
        for i, task in enumerate(ACTIVE_TASKS):
            vals = pd.to_numeric(participant[participant["task"] == task][spk_col], errors="coerce").dropna()
            if len(vals) >= 2:
                ax_c.boxplot(vals.values, positions=[i], widths=0.5, patch_artist=True,
                             boxprops=dict(facecolor=f"C{i}", alpha=0.6),
                             medianprops=dict(color="black", lw=2), showfliers=False)
    ax_c.set_xticks(range(len(ACTIVE_TASKS)))
    ax_c.set_xticklabels(ACTIVE_TASKS)
    ax_c.set_ylabel("Speaking fraction Î” vs T0")
    ax_c.set_title("C: Speaking Activity Î”", fontsize=10, fontweight="bold")

    # Panel D: BFI Agreeableness vs HRV scatter
    ax_d = fig.add_subplot(3, 3, 4)
    trait_col, physio_col = "bfi44_a", "ppg_rmssd_ms_delta_t0"
    if trait_col in participant.columns and physio_col in participant.columns:
        sub = participant[["task", trait_col, physio_col]].copy()
        for col in [trait_col, physio_col]:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()
        ax_d.scatter(sub[trait_col], sub[physio_col], alpha=0.4, s=15, color="#5C6BC0")
        if len(sub) >= 5:
            try:
                m, b = np.polyfit(sub[trait_col], sub[physio_col], 1)
                xl = np.linspace(sub[trait_col].min(), sub[trait_col].max(), 50)
                ax_d.plot(xl, m * xl + b, "--", color="#E53935", lw=1.5)
                r, p = _safe_corr(sub[trait_col], sub[physio_col])
                ax_d.text(0.05, 0.95, f"r={r:.2f}, p={p:.3f}", transform=ax_d.transAxes,
                          fontsize=9, va="top", color="#E53935")
            except Exception:
                pass
    ax_d.set_xlabel("Agreeableness")
    ax_d.set_ylabel("HRV RMSSD Î” (ms)")
    ax_d.set_title("D: Agreeableness Ã— HRV", fontsize=10, fontweight="bold")

    # Panel E: Extraversion vs speaking fraction (absolute, active tasks only â€” T0 baseline artefact excluded)
    ax_e = fig.add_subplot(3, 3, 5)
    trait_col2, audio_col = "bfi44_e", "audio_speaking_fraction"
    if trait_col2 in participant.columns and audio_col in participant.columns:
        sub2 = participant[participant["task"].isin(ACTIVE_TASKS)][[trait_col2, audio_col, "task"]].copy()
        for col in [trait_col2, audio_col]:
            sub2[col] = pd.to_numeric(sub2[col], errors="coerce")
        sub2 = sub2.dropna()
        colors_map = {t: f"C{i}" for i, t in enumerate(TASK_ORDER)}
        for task, tdf in sub2.groupby("task"):
            ax_e.scatter(tdf[trait_col2], tdf[audio_col], alpha=0.5, s=20,
                         color=colors_map.get(task, "gray"), label=task)
        if len(sub2) >= 5:
            try:
                m, b = np.polyfit(sub2[trait_col2], sub2[audio_col], 1)
                xl = np.linspace(sub2[trait_col2].min(), sub2[trait_col2].max(), 50)
                ax_e.plot(xl, m * xl + b, "--", color="black", lw=1.5)
                r, p = _safe_corr(sub2[trait_col2], sub2[audio_col])
                ax_e.text(0.05, 0.95, f"r={r:.2f}, p={p:.3f}", transform=ax_e.transAxes,
                          fontsize=9, va="top")
            except Exception:
                pass
        ax_e.legend(fontsize=7, ncol=2)
    ax_e.set_xlabel("Extraversion")
    ax_e.set_ylabel("Speaking fraction (active tasks)")
    ax_e.set_title("E: Extraversion Ã— Speech (abs)", fontsize=10, fontweight="bold")

    # Panel F: Conscientiousness vs EDA
    ax_f = fig.add_subplot(3, 3, 6)
    trait_col3, eda_col2 = "bfi44_c", "eda_phasic_rate_hz_delta_t0"
    if trait_col3 in participant.columns and eda_col2 in participant.columns:
        sub3 = participant[[trait_col3, eda_col2]].apply(pd.to_numeric, errors="coerce").dropna()
        ax_f.scatter(sub3[trait_col3], sub3[eda_col2], alpha=0.4, s=15, color="#43A047")
        if len(sub3) >= 5:
            try:
                m, b = np.polyfit(sub3[trait_col3], sub3[eda_col2], 1)
                xl = np.linspace(sub3[trait_col3].min(), sub3[trait_col3].max(), 50)
                ax_f.plot(xl, m * xl + b, "--", color="#E53935", lw=1.5)
                r, p = _safe_corr(sub3[trait_col3], sub3[eda_col2])
                ax_f.text(0.05, 0.95, f"r={r:.2f}, p={p:.3f}", transform=ax_f.transAxes,
                          fontsize=9, va="top", color="#E53935")
            except Exception:
                pass
    ax_f.set_xlabel("Conscientiousness")
    ax_f.set_ylabel("SCR rate Î” (Hz)")
    ax_f.set_title("F: Conscientiousness Ã— SCR", fontsize=10, fontweight="bold")

    # Panel G: Group Gini by task (if available)
    ax_g = fig.add_subplot(3, 3, 7)
    if not group_speech_pers.empty and "task_id" in group_speech_pers.columns:
        active_gs = group_speech_pers[group_speech_pers["task_id"].isin(ACTIVE_TASKS)]
        task_means = active_gs.groupby("task_id")["gini_speaking"].agg(["mean", "std", "count"]).reindex(ACTIVE_TASKS).dropna()
        ax_g.bar(range(len(task_means)), task_means["mean"], color="steelblue", alpha=0.7)
        ax_g.errorbar(range(len(task_means)), task_means["mean"],
                      yerr=task_means["std"] / np.sqrt(task_means["count"]),
                      fmt="none", color="black", capsize=4)
        ax_g.set_xticks(range(len(task_means)))
        ax_g.set_xticklabels(task_means.index.tolist())
    ax_g.set_ylabel("Gini (speaking time)")
    ax_g.set_title("G: Speaking Inequality by Task", fontsize=10, fontweight="bold")

    # Panel H: Engagement by task
    ax_h = fig.add_subplot(3, 3, 8)
    eng_col = "ans_engagement"
    if eng_col in participant.columns and "task" in participant.columns:
        for i, task in enumerate(ACTIVE_TASKS):
            vals = pd.to_numeric(participant[participant["task"] == task][eng_col], errors="coerce").dropna()
            if len(vals) >= 2:
                ax_h.boxplot(vals.values, positions=[i], widths=0.5, patch_artist=True,
                             boxprops=dict(facecolor=f"C{i}", alpha=0.6),
                             medianprops=dict(color="black", lw=2), showfliers=False)
    ax_h.set_xticks(range(len(ACTIVE_TASKS)))
    ax_h.set_xticklabels(ACTIVE_TASKS)
    ax_h.set_ylabel("Engagement rating (1â€“7)")
    ax_h.set_title("H: Self-Reported Engagement", fontsize=10, fontweight="bold")

    # Panel I: Pupil delta by task
    ax_i = fig.add_subplot(3, 3, 9)
    pup_col = "pupil_mean_delta_t0"
    if pup_col in participant.columns and "task" in participant.columns:
        for i, task in enumerate(ACTIVE_TASKS):
            vals = pd.to_numeric(participant[participant["task"] == task][pup_col], errors="coerce").dropna()
            if len(vals) >= 2:
                ax_i.boxplot(vals.values, positions=[i], widths=0.5, patch_artist=True,
                             boxprops=dict(facecolor=f"C{i}", alpha=0.6),
                             medianprops=dict(color="black", lw=2), showfliers=False)
    ax_i.set_xticks(range(len(ACTIVE_TASKS)))
    ax_i.set_xticklabels(ACTIVE_TASKS)
    ax_i.set_ylabel("Pupil Î” vs T0 (mm)")
    ax_i.set_title("I: Pupil Dilation Î”", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "summary_panel_9.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


def fig_audio_physio_vad_correlation_heatmap(conv_df: pd.DataFrame, out_dir: Path, dpi: int = 150) -> None:
    """Per-task heatmaps: physio+audio Î” vs. VAD and postblock."""
    plt, _ = _mpl()
    if plt is None or conv_df.empty:
        return

    tasks = [t for t in ACTIVE_TASKS if f"task_{t}" in conv_df["scope"].values]
    if not tasks:
        return

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 7), squeeze=False)
    fig.suptitle("Physio/Audio Î” vs VAD â€” Per Task (Pearson r)", fontsize=12, fontweight="bold")

    predictors_all = conv_df["predictor"].unique().tolist()
    vad_dims = [v for v in VAD_FEATURES if v in conv_df["vad"].values]

    for col_i, task in enumerate(tasks):
        ax = axes[0, col_i]
        tdf = conv_df[conv_df["scope"] == f"task_{task}"]
        preds = [p for p in predictors_all if p in tdf["predictor"].values]
        if not preds:
            ax.set_visible(False)
            continue

        mat = np.full((len(preds), len(vad_dims)), np.nan)
        pmat = np.full_like(mat, np.nan)
        for ri, pred in enumerate(preds):
            for ci, vad in enumerate(vad_dims):
                row = tdf[(tdf["predictor"] == pred) & (tdf["vad"] == vad)]
                if not row.empty:
                    mat[ri, ci] = float(row["pearson_r"].iloc[0])
                    pmat[ri, ci] = float(row["p_pearson"].iloc[0])

        vmax = float(np.nanquantile(np.abs(mat), 0.95)) if not np.all(np.isnan(mat)) else 0.4
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        ax.set_title(f"Task {task}", fontsize=10)
        ax.set_yticks(range(len(preds)))
        ax.set_yticklabels(
            [p.replace("_delta_t0", "Î”").replace("ppg_", "").replace("audio_", "").replace("_mean", "_Î¼")
             for p in preds],
            fontsize=7,
        )
        ax.set_xticks(range(len(vad_dims)))
        ax.set_xticklabels([v.replace("vad_", "").replace("_self_num", "").capitalize() for v in vad_dims], fontsize=9)
        for ri in range(len(preds)):
            for ci in range(len(vad_dims)):
                if np.isfinite(pmat[ri, ci]) and pmat[ri, ci] < 0.05:
                    ax.text(ci, ri, "*", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    path = out_dir / "physio_audio_vad_by_task.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Expanded multi-modal statistics for GroupAffect-4.")
    p.add_argument("--features-dir", type=Path, default=Path("data") / "derived_features")
    p.add_argument("--stats-dir", type=Path, default=Path("results") / "statistics")
    p.add_argument("--audio-dir", type=Path, default=Path("results") / "audio")
    p.add_argument("--personality-dir", type=Path, default=Path("results") / "personality")
    p.add_argument("--out-dir", type=Path, default=Path("results") / "expanded_stats")
    p.add_argument("--figures-dir", type=Path, default=None)
    p.add_argument("--min-n", type=int, default=8)
    p.add_argument("--min-groups", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = args.out_dir.resolve()
    fig_dir = (args.figures_dir or out_dir / "figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load core tables -----
    participant = _read(args.stats_dir / "analysis_dataset_participant_task.tsv")
    audio_ind = _read(args.audio_dir / "individual_audio_task.tsv", required=False)
    group_traits = _read(args.personality_dir / "group_trait_stats.tsv", required=False)

    # Normalize task column name
    if "task_id" in participant.columns and "task" not in participant.columns:
        participant["task"] = participant["task_id"]

    BFI_Z = [f"{t}_z" for t in BFI_TRAITS if f"{t}_z" in participant.columns]
    all_audio_cols = AUDIO_ABS_FEATURES + AUDIO_DELTA_FEATURES
    _numeric(participant, BFI_TRAITS + BFI_Z + PHYSIO_DELTA_FEATURES + all_audio_cols + VAD_FEATURES + POSTBLOCK_FEATURES)

    # For BFI Ã— audio: use absolute audio features restricted to active tasks (T0 baseline is artefactual)
    participant_active = participant[participant["task"].isin(ACTIVE_TASKS)].copy()

    # ----- 1. Group speech dynamics -----
    LOG.info("Computing group speech dynamics ...")
    group_speech = pd.DataFrame()
    group_speech_pers = pd.DataFrame()
    if not audio_ind.empty:
        group_speech = compute_group_speech_dynamics(audio_ind)
        _write(group_speech, out_dir / "group_speech_dynamics.tsv")
        if not group_traits.empty:
            group_speech_pers = merge_group_personality(group_speech, group_traits)
            _write(group_speech_pers, out_dir / "group_speech_with_personality.tsv")
            bfi_group_audio_corr = compute_group_bfi_audio_correlations(group_speech_pers, min_n=args.min_groups)
            _write(bfi_group_audio_corr, out_dir / "group_bfi_speech_correlations.tsv")

    # ----- 2. Task-stratified BFI Ã— audio/physio correlations -----
    LOG.info("Computing task-stratified BFI correlations ...")
    # Audio: use absolute features, active tasks only (T0 baseline is contaminated by ambient noise)
    bfi_audio_task = compute_bfi_feature_by_task(
        participant_active, _present(participant_active, AUDIO_ABS_FEATURES), BFI_Z,
        min_n=args.min_n, scope_prefix="bfi_audio_abs",
    )
    _write(bfi_audio_task, out_dir / "bfi_audio_by_task_correlations.tsv")

    bfi_physio_task = compute_bfi_feature_by_task(
        participant, _present(participant, PHYSIO_DELTA_FEATURES), BFI_Z,
        min_n=args.min_n, scope_prefix="bfi_physio",
    )
    _write(bfi_physio_task, out_dir / "bfi_physio_by_task_correlations.tsv")

    bfi_vad_task = compute_bfi_feature_by_task(
        participant, _present(participant, VAD_FEATURES), BFI_Z,
        min_n=args.min_n, scope_prefix="bfi_vad",
    )
    _write(bfi_vad_task, out_dir / "bfi_vad_by_task_correlations.tsv")

    # ----- 2b. Participant-level BFI correlations (Rec 4: correct between-person analysis) -----
    LOG.info("Computing participant-level BFI correlations (between-person, n=unique participants) ...")
    all_bfi_features = (
        _present(participant_active, AUDIO_ABS_FEATURES)
        + _present(participant, PHYSIO_DELTA_FEATURES)
        + _present(participant, VAD_FEATURES)
    )
    bfi_participant_level = compute_bfi_participant_mean_correlations(
        participant, all_bfi_features, BFI_Z, min_n=5, active_tasks_only=True,
    )
    _write(bfi_participant_level, out_dir / "bfi_participant_level_correlations.tsv")

    # Comparison: pooled by-task summary vs participant-level â€” for audit/validation
    all_by_task = pd.concat(
        [df for df in [bfi_audio_task, bfi_physio_task, bfi_vad_task] if not df.empty],
        ignore_index=True,
    )
    bfi_level_comparison = compare_pooled_vs_participant_level(all_by_task, bfi_participant_level)
    _write(bfi_level_comparison, out_dir / "bfi_pooled_vs_participant_level_comparison.tsv")

    n_part = int(bfi_participant_level["n_unique_participants"].iloc[0]) if not bfi_participant_level.empty else 0
    n_sig_part = int((bfi_participant_level["p_pearson"] < 0.05).sum()) if not bfi_participant_level.empty else 0
    n_sig_task = int((all_by_task["p_pearson"] < 0.05).sum()) if not all_by_task.empty else 0
    LOG.info(
        "BFI participant-level: %d pairs, %d significant (p<0.05) | "
        "by-task significant: %d (inflation check)",
        len(bfi_participant_level), n_sig_part, n_sig_task,
    )

    # ----- 3. Cross-modal affect convergence -----
    # Use absolute audio + physio deltas vs VAD; restrict audio to active tasks
    LOG.info("Computing cross-modal affect convergence ...")
    conv_df = compute_affect_convergence(participant_active, min_n=args.min_n)
    _write(conv_df, out_dir / "affect_convergence_correlations.tsv")

    discord = compute_affect_discordance(participant)
    if not discord.empty:
        _write(discord, out_dir / "affect_discordance_rows.tsv")
        discord_summary = discord.groupby("task").agg(
            n=("affect_discordance", "count"),
            mean_discordance=("affect_discordance", "mean"),
            frac_physio_high_vad_low=("physio_high_vad_low", "mean"),
            frac_physio_low_vad_high=("physio_low_vad_high", "mean"),
        ).reset_index()
        _write(discord_summary, out_dir / "affect_discordance_by_task.tsv")

    # ----- 4. Multi-modal task profiles -----
    LOG.info("Computing task profiles ...")
    profiles = compute_task_profiles(participant)
    _write(profiles, out_dir / "multimodal_task_profiles.tsv")

    # ----- 5. Figures -----
    if not args.no_figures:
        LOG.info("Generating figures ...")
        if not group_speech.empty:
            fig_group_speech_balance(group_speech, fig_dir, dpi=args.dpi)
        if not group_speech_pers.empty:
            fig_gini_bfi_scatter(group_speech_pers, fig_dir, dpi=args.dpi)

        if not bfi_audio_task.empty:
            fig_bfi_audio_heatmap(bfi_audio_task, fig_dir, dpi=args.dpi)
            fig_bfi_physio_audio_combined(bfi_physio_task, bfi_audio_task, fig_dir, dpi=args.dpi)

        if not conv_df.empty:
            fig_cross_modal_convergence(conv_df, fig_dir, dpi=args.dpi)
            fig_audio_physio_vad_correlation_heatmap(conv_df, fig_dir, dpi=args.dpi)

        if not discord.empty:
            fig_discordance_by_task(discord, fig_dir, dpi=args.dpi)

        if not profiles.empty:
            fig_task_profiles(profiles, fig_dir, dpi=args.dpi)

        fig_summary_panel(participant, group_speech_pers, fig_dir, dpi=args.dpi)

    LOG.info("Expanded statistics complete -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

