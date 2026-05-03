"""Feature-modality ablation study for GroupAffect-4 benchmarks.

Conditions
----------
  cardiac     HR + HRV (delta and absolute)
  eda         EDA tonic/phasic + skin temperature (delta and absolute)
  pupil       Pupil dilation (ET)
  audio       Prosodic / speech-activity features
  bfi         BFI-44 personality traits
  card+eda    Cardiac + EDA  (â‰ˆ classic physio)
  card+pup    Cardiac + Pupil
  eda+pup     EDA + Pupil
  sensor      Cardiac + EDA + Pupil + Audio
  all         Sensor + BFI

Benchmarks
----------
  B0   Task-label classification (T1-T4)
  B1   Valence / arousal regression
  B2   Dominance binary classification
  B3   Engagement / mental demand binary classification
  B4   Personality trait regression (person-level)
  B5   T4 contribution regression
  B6   Speech-overlap balance (group-task level)
  B7   Speaking floor prediction (dominant speaker per task)
  Bs   Speaking Gini (group-task level)

Outputs
-------
  results/benchmarks/ablation_results.tsv
  paper/tables/ablation_table.tex
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
STATS_PT   = Path("results/statistics/analysis_dataset_participant_task.tsv")
AUDIO_PATH = Path("results/audio/individual_audio_task.tsv")
VAD_PATH   = Path("results/task_responses/vad_participant_task.tsv")
PARTS_PATH = Path("data/bids_release_no_video/participants.tsv")
OUT_DIR    = Path("results/benchmarks")
OUT_TSV    = OUT_DIR / "ablation_results.tsv"
OUT_TEX    = Path("paper/tables/ablation_table.tex")

ACTIVE_TASKS = ("T1", "T2", "T3", "T4")
KNN_K = 5

# â”€â”€â”€ Canonical modality groups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

CARDIAC = [
    "hr_mean_bpm_delta_t0", "hrv_rmssd_ms_delta_t0",
    "hr_mean_bpm",          "hrv_rmssd_ms",
]

EDA = [
    "eda_tonic_mean_delta_t0", "eda_phasic_rate_hz_delta_t0", "temp_mean_delta_t0",
    "eda_tonic_mean",          "eda_phasic_rate_hz",           "temp_mean",
]

PUPIL = [
    "pupil_mean_delta_t0",
    "pupil_mean", "pupil_std", "pupil_slope_per_s",
]

# Prosodic/speech features (no speaking_fraction/overlap â€” leakage risk for B6/B7)
AUDIO_PROSODIC = [
    "audio_energy_mean", "audio_energy_sd",
    "audio_pitch_mean",  "audio_pitch_sd",
    "audio_hnr_mean",    "audio_jitter_mean",  "audio_shimmer_mean",
    "audio_voiced_segments_per_sec", "audio_mean_voiced_segment_s",
    "audio_speech_rate_proxy",
]

# Full audio including social speech-activity features
AUDIO = AUDIO_PROSODIC + [
    "audio_speaking_fraction",
    "audio_overlap_fraction",
]

BFI = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]

# All sensor features combined
SENSOR = CARDIAC + EDA + PUPIL + AUDIO

# â”€â”€â”€ Ablation conditions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each entry: (condition_id, list-of-feature-lists)
CONDITIONS: list[tuple[str, list[list[str]]]] = [
    ("cardiac",   [CARDIAC]),
    ("eda",       [EDA]),
    ("pupil",     [PUPIL]),
    ("audio",     [AUDIO]),
    ("bfi",       [BFI]),
    ("card+eda",  [CARDIAC, EDA]),
    ("card+pup",  [CARDIAC, PUPIL]),
    ("eda+pup",   [EDA, PUPIL]),
    ("sensor",    [CARDIAC, EDA, PUPIL, AUDIO]),
    ("all",       [CARDIAC, EDA, PUPIL, AUDIO, BFI]),
]

COND_LABELS: dict[str, str] = {
    "cardiac":  "Cardiac",
    "eda":      "EDA/GSR",
    "pupil":    "Pupil",
    "audio":    "Audio",
    "bfi":      "BFI",
    "card+eda": "Card+EDA",
    "card+pup": "Card+Pup",
    "eda+pup":  "EDA+Pup",
    "sensor":   "Sensor",
    "all":      "All+BFI",
}


def _avail(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# â”€â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (participant_task_df, participants_df)."""
    stats = pd.read_csv(STATS_PT, sep="\t")
    if "task_id" in stats.columns and "task" not in stats.columns:
        stats = stats.rename(columns={"task_id": "task"})

    # Merge audio
    audio = pd.read_csv(AUDIO_PATH, sep="\t")
    if "task_id" in audio.columns:
        audio = audio.drop(columns=["task"], errors="ignore")
        audio = audio.rename(columns={"task_id": "task"})
    src_cols = [
        "speaking_fraction", "overlap_fraction", "energy_mean", "energy_sd",
        "pitch_mean", "pitch_sd", "hnr_mean", "jitter_mean", "shimmer_mean",
        "voiced_segments_per_sec", "mean_voiced_segment_s", "speech_rate_proxy",
        "speaking_rank_in_group", "speaking_share_group",
    ]
    avail_audio = [c for c in src_cols if c in audio.columns]
    key = [c for c in ["session_id", "participant_id", "task"] if c in audio.columns and c in stats.columns]
    audio_sel = audio[key + avail_audio].rename(
        columns={c: f"audio_{c}" for c in avail_audio
                 if c not in ["speaking_rank_in_group", "speaking_share_group"]})
    # speaking_rank / share are kept with original names
    drop_dup = [c for c in audio_sel.columns if c in stats.columns and c not in key]
    audio_sel = audio_sel.drop(columns=drop_dup, errors="ignore")
    stats = stats.merge(audio_sel, on=key, how="left", suffixes=("", "_aud"))
    for col in list(stats.columns):
        if col.endswith("_aud"):
            base = col[:-4]
            if base in stats.columns:
                stats[base] = stats[base].combine_first(stats[col])
            stats = stats.drop(columns=[col])

    # Merge VAD labels
    vad = pd.read_csv(VAD_PATH, sep="\t")
    vad = vad.rename(columns={
        "participant": "participant_id",
        "arousal":   "vad_arousal",
        "valence":   "vad_valence",
        "dominance": "vad_dominance",
    })
    for col in ["vad_arousal", "vad_valence", "vad_dominance"]:
        if col in stats.columns:
            stats = stats.drop(columns=[col])
    vad_key = [c for c in ["session_id", "participant_id", "task"] if c in stats.columns and c in vad.columns]
    vad_cols = vad_key + [c for c in ["vad_arousal", "vad_valence", "vad_dominance"] if c in vad.columns]
    stats = stats.merge(vad[vad_cols], on=vad_key, how="left")

    # Merge BFI + group_id from participants
    parts = pd.read_csv(PARTS_PATH, sep="\t")
    bfi_src = [c for c in BFI if c in parts.columns]
    bfi_merge = parts[["participant_id"] + bfi_src].drop_duplicates(subset=["participant_id"])
    global_id = next((c for c in ["participant_global_id", "participant_id"]
                      if c in stats.columns and
                      len(set(stats[c].dropna()) & set(parts["participant_id"].dropna())) > 5),
                     None)
    if global_id:
        for col in bfi_src:
            if col in stats.columns:
                stats = stats.drop(columns=[col])
        stats = stats.merge(
            bfi_merge.rename(columns={"participant_id": global_id}),
            on=global_id, how="left")

    # Active tasks only
    pt = stats[stats["task"].isin(ACTIVE_TASKS)].copy().reset_index(drop=True)

    # Robust within-person z-score for sensor features
    person_col = next((c for c in ["participant_uid", "participant_global_id", "participant_id"]
                       if c in pt.columns), "participant_id")
    sensor_feats = _avail(pt, CARDIAC + EDA + PUPIL + AUDIO)
    for col in sensor_feats:
        stats_map: dict[str, tuple[float, float]] = {}
        for pid, grp in pt.groupby(person_col):
            vals = grp[col].dropna()
            if len(vals) < 2:
                continue
            med = float(vals.median())
            mad = float((vals - med).abs().median()) * 1.4826
            stats_map[str(pid)] = (med, mad)
        if not stats_map:
            continue
        def _z(row: pd.Series, col=col, stats_map=stats_map, person_col=person_col) -> float:
            v = row[col]
            pid = str(row[person_col])
            if pd.isna(v) or pid not in stats_map:
                return v
            med, mad = stats_map[pid]
            return float(v - med) if mad < 1e-9 else float((v - med) / mad)
        pt[col] = pt[[person_col, col]].apply(_z, axis=1)

    LOG.info("Participant-task table: %d rows, %d cols", len(pt), len(pt.columns))
    return pt, parts


# â”€â”€â”€ LOGO-CV helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fold_knn(train: pd.DataFrame, test: pd.DataFrame,
              feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    tr = train[feats].apply(pd.to_numeric, errors="coerce").copy()
    te = test[feats].apply(pd.to_numeric, errors="coerce").copy()
    for col in tr.columns[tr.isna().all()]:
        tr[col] = 0.0
        te[col] = 0.0
    if tr.isna().any().any() or te.isna().any().any():
        k = max(1, min(KNN_K, len(tr)))
        imp = KNNImputer(n_neighbors=k, weights="distance")
        X_tr = imp.fit_transform(tr)
        X_te = imp.transform(te)
    else:
        X_tr = tr.to_numpy(dtype=float)
        X_te = te.to_numpy(dtype=float)
    return X_tr, X_te


def logo_clf(df: pd.DataFrame, feats: list[str], target: str,
             binarise: bool = False) -> dict[str, float]:
    groups = sorted(df["group_id"].dropna().unique())
    y_pred_all, y_true_all, y_score_all = [], [], []
    is_binary = binarise or (df[target].nunique() == 2)

    for gid in groups:
        tr = df[df["group_id"] != gid].copy()
        te = df[df["group_id"] == gid].copy()
        if binarise:
            med = float(tr[target].median())
            tr[target] = (tr[target] >= med).astype(int)
            te[target] = (te[target] >= med).astype(int)
        avail = _avail(df, feats)
        if not avail or tr[target].nunique() < 2:
            majority = tr[target].mode(dropna=True).iloc[0]
            y_pred_all.extend([majority] * len(te))
            y_true_all.extend(te[target].tolist())
            y_score_all.extend([float("nan")] * len(te))
            continue
        X_tr, X_te = _fold_knn(tr, te, avail)
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(X_tr, tr[target])
        y_pred_all.extend(model.predict(X_te).tolist())
        y_true_all.extend(te[target].tolist())
        if is_binary and hasattr(model[-1], "classes_") and len(model[-1].classes_) == 2:
            y_score_all.extend(model.predict_proba(X_te)[:, 1].tolist())
        else:
            y_score_all.extend([float("nan")] * len(te))

    acc = accuracy_score(y_true_all, y_pred_all)
    auc = float("nan")
    if is_binary and not all(math.isnan(s) for s in y_score_all):
        try:
            auc = roc_auc_score([int(y) for y in y_true_all],
                                [0.5 if math.isnan(s) else s for s in y_score_all])
        except Exception:
            pass
    return {"accuracy": acc, "roc_auc": auc}


def logo_reg(df: pd.DataFrame, feats: list[str], target: str) -> dict[str, float]:
    groups = sorted(df["group_id"].dropna().unique())
    y_pred_all, y_true_all = [], []

    for gid in groups:
        tr = df[df["group_id"] != gid].dropna(subset=[target])
        te = df[df["group_id"] == gid]
        avail = _avail(df, feats)
        if not avail or len(tr) < 3:
            baseline_val = float(tr[target].mean()) if len(tr) > 0 else 0.0
            y_pred_all.extend([baseline_val] * len(te))
        else:
            X_tr, X_te = _fold_knn(tr, te, avail)
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X_tr, tr[target])
            y_pred_all.extend(model.predict(X_te).tolist())
        y_true_all.extend(te[target].tolist())

    mask = [not math.isnan(y) for y in y_true_all]
    yt = np.array([y for y, m in zip(y_true_all, mask) if m])
    yp = np.array([p for p, m in zip(y_pred_all, mask) if m])
    mae = float(mean_absolute_error(yt, yp))
    try:
        r = float(scipy_stats.pearsonr(yt, yp).statistic)
    except Exception:
        r = float("nan")
    return {"mae": mae, "pearson_r": r}


# â”€â”€â”€ Per-benchmark ablation runners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _run_cond(name: str, feat_lists: list[list[str]],
              df: pd.DataFrame, kind: str, target: str,
              bid: str, tgt_label: str, n: int) -> dict:
    feats = list(dict.fromkeys(f for fl in feat_lists for f in _avail(df, fl)))
    has_data = len(feats) > 0 and len(df.dropna(subset=[target])) >= 10
    if not has_data:
        return {"benchmark_id": bid, "target": tgt_label, "condition": name, "n": n,
                "primary_metric": "nan", "primary_value": float("nan"),
                "secondary_metric": "nan", "secondary_value": float("nan"),
                "n_features": 0}
    if kind == "clf":
        res = logo_clf(df, feats, target)
        return {"benchmark_id": bid, "target": tgt_label, "condition": name, "n": n,
                "primary_metric": "accuracy", "primary_value": res["accuracy"],
                "secondary_metric": "roc_auc", "secondary_value": res["roc_auc"],
                "n_features": len(feats)}
    else:
        res = logo_reg(df, feats, target)
        return {"benchmark_id": bid, "target": tgt_label, "condition": name, "n": n,
                "primary_metric": "mae", "primary_value": res["mae"],
                "secondary_metric": "pearson_r", "secondary_value": res["pearson_r"],
                "n_features": len(feats)}


def run_ablation(pt: pd.DataFrame, parts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # Person-level mean table for B4
    uid = next((c for c in ["participant_uid", "participant_global_id", "participant_id"]
                if c in pt.columns), "participant_id")
    global_id = next((c for c in ["participant_global_id", "participant_id"]
                      if c in pt.columns and
                      len(set(pt[c].dropna()) & set(parts["participant_id"].dropna())) > 5),
                     None)
    sensor_avail = _avail(pt, CARDIAC + EDA + PUPIL + AUDIO)
    grp_first = pt[[uid, "group_id"]].drop_duplicates(subset=[uid])
    p_mean = pt[[uid] + sensor_avail].groupby(uid)[sensor_avail].mean().reset_index()
    p_mean = p_mean.merge(grp_first, on=uid, how="left")
    bfi_avail = _avail(parts, BFI)
    if global_id and bfi_avail:
        pid_first = pt[[uid, global_id]].drop_duplicates(subset=[uid])
        bfi_df = parts[["participant_id"] + bfi_avail].drop_duplicates(subset=["participant_id"])
        pid_bfi = pid_first.merge(bfi_df, left_on=global_id, right_on="participant_id", how="left")
        for col in bfi_avail:
            if col in pid_bfi.columns:
                p_mean = p_mean.merge(pid_bfi[[uid, col]], on=uid, how="left")

    # â”€â”€ B0: Task classification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("B0: task classification ...")
    b0 = pt.dropna(subset=["task", "group_id"]).copy()
    for cname, flists in CONDITIONS:
        if cname == "bfi":
            continue  # BFI constant within person across tasks
        rows.append(_run_cond(cname, flists, b0, "clf", "task",
                               "B0_task", "Task label (T1-T4)", len(b0)))

    # â”€â”€ B1: Valence & arousal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("B1: valence / arousal ...")
    for target, label in [("vad_valence", "Valence"), ("vad_arousal", "Arousal")]:
        if target not in pt.columns:
            continue
        sub = pt.dropna(subset=[target, "group_id"]).copy()
        for cname, flists in CONDITIONS:
            rows.append(_run_cond(cname, flists, sub, "reg", target,
                                   "B1_affective", label, len(sub)))

    # â”€â”€ B2: Dominance binary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("B2: dominance ...")
    if "vad_dominance" in pt.columns:
        sub = pt[pt["task"].isin(("T1", "T2", "T3"))].dropna(
            subset=["vad_dominance", "group_id"]).copy()
        med = float(sub["vad_dominance"].median())
        sub["dom_high"] = (sub["vad_dominance"] >= med).astype(int)
        for cname, flists in CONDITIONS:
            rows.append(_run_cond(cname, flists, sub, "clf", "dom_high",
                                   "B2_dominance", "Dominance (high/low)", len(sub)))

    # â”€â”€ B3: Cognitive states â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("B3: engagement / mental demand ...")
    for target, label in [("ans_engagement", "Engagement"), ("ans_mental_demand", "Mental demand")]:
        if target not in pt.columns:
            continue
        sub = pt.dropna(subset=[target, "group_id"]).copy()
        if len(sub) < 20:
            continue
        med = float(sub[target].median())
        sub[f"{target}_h"] = (sub[target] >= med).astype(int)
        for cname, flists in CONDITIONS:
            rows.append(_run_cond(cname, flists, sub, "clf", f"{target}_h",
                                   "B3_cognitive", label, len(sub)))

    # â”€â”€ B4: Personality (person-level) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("B4: personality ...")
    for trait, label in [("bfi44_e", "BFI Extraversion"), ("bfi44_o", "BFI Openness")]:
        if trait not in p_mean.columns:
            continue
        sub = p_mean.dropna(subset=[trait, "group_id"]).copy()
        if len(sub) < 10:
            continue
        b4_conds = [(cn, [fl for fl in flists if fl is not BFI])
                    for cn, flists in CONDITIONS if cn != "bfi" and cn != "all"]
        for cname, flists in b4_conds:
            clean = [fl for fl in flists if fl]
            if not clean:
                continue
            rows.append(_run_cond(cname, clean, sub, "reg", trait,
                                   "B4_personality", label, len(sub)))

    # â”€â”€ B5: Meeting performance (T4 only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("B5: contribution ...")
    if "ans_contribution" in pt.columns:
        sub = pt[pt["task"] == "T4"].dropna(subset=["ans_contribution", "group_id"]).copy()
        if len(sub) >= 10:
            for cname, flists in CONDITIONS:
                rows.append(_run_cond(cname, flists, sub, "reg", "ans_contribution",
                                       "B5_performance", "T4 Contribution", len(sub)))

    # â”€â”€ B7: Speaking floor prediction (who is rank-1 speaker per task) â”€â”€â”€â”€â”€â”€â”€â”€
    # Uses physio/pupil/BFI only (no speaking_fraction/overlap = leakage)
    # Predicts whether a participant will be the top floor-holder in their group
    LOG.info("B7: speaking floor prediction ...")
    if "speaking_rank_in_group" in pt.columns:
        sub7 = pt.dropna(subset=["speaking_rank_in_group", "group_id"]).copy()
        sub7["floor_dominant"] = (sub7["speaking_rank_in_group"] == 1).astype(int)
        # Feature sets: exclude speaking/overlap features
        B7_CONDS: list[tuple[str, list[list[str]]]] = [
            ("cardiac",  [CARDIAC]),
            ("eda",      [EDA]),
            ("pupil",    [PUPIL]),
            ("bfi",      [BFI]),
            ("card+eda", [CARDIAC, EDA]),
            ("card+pup", [CARDIAC, PUPIL]),
            ("eda+pup",  [EDA, PUPIL]),
            ("sensor",   [CARDIAC, EDA, PUPIL]),       # no audio (leakage)
            ("all",      [CARDIAC, EDA, PUPIL, BFI]),  # sensor + BFI
        ]
        for cname, flists in B7_CONDS:
            rows.append(_run_cond(cname, flists, sub7, "clf", "floor_dominant",
                                   "B7_floor", "Floor dominance", len(sub7)))

    # â”€â”€ Bs & B6: Group-task level benchmarks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    LOG.info("Bs / B6: group-task level ...")
    audio2 = pd.read_csv(AUDIO_PATH, sep="\t")
    if "task_id" in audio2.columns:
        audio2 = audio2.drop(columns=["task"], errors="ignore")
        audio2 = audio2.rename(columns={"task_id": "task"})

    avail_a = (audio2["audio_available"].astype(bool)
               if "audio_available" in audio2.columns
               else pd.Series(True, index=audio2.index))
    sub_a = audio2[audio2["task"].isin(ACTIVE_TASKS) & avail_a].copy()

    # Group-mean BFI
    bfi_grp = parts.groupby("group_id", as_index=False)[_avail(parts, BFI)].mean()

    # Group-mean physio + pupil from participant-task table
    physio_pupil_cols = _avail(pt, CARDIAC + EDA + PUPIL)
    grp_physio = (pt[pt["task"].isin(ACTIVE_TASKS)]
                  .groupby(["group_id", "task"], as_index=False)[physio_pupil_cols]
                  .mean())

    # â”€â”€ Bs: Speaking Gini â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "speaking_fraction" in sub_a.columns and "group_id" in sub_a.columns:
        gini_vals = (sub_a.groupby(["group_id", "task"], as_index=False)
                     .apply(lambda g: pd.Series({"gini_speaking": _gini(g["speaking_fraction"])}))
                     .dropna(subset=["gini_speaking"])
                     .reset_index(drop=True))
        # Merge group-mean BFI, physio/pupil, and task dummies
        bs_design = gini_vals.merge(bfi_grp, on="group_id", how="left")
        bs_design = bs_design.merge(grp_physio, on=["group_id", "task"], how="left")
        dummies_bs = pd.get_dummies(bs_design["task"], prefix="task", dtype=float)
        bs_design = pd.concat([bs_design, dummies_bs], axis=1)
        task_cols = dummies_bs.columns.tolist()
        bfi_feats_bs = _avail(bs_design, BFI)
        card_feats_bs = _avail(bs_design, CARDIAC)
        eda_feats_bs  = _avail(bs_design, EDA)
        pup_feats_bs  = _avail(bs_design, PUPIL)
        bs_conds = [
            ("cardiac",  [card_feats_bs]),
            ("eda",      [eda_feats_bs]),
            ("pupil",    [pup_feats_bs]),
            ("bfi",      [bfi_feats_bs]),
            ("card+eda", [card_feats_bs, eda_feats_bs]),
            ("sensor",   [card_feats_bs, eda_feats_bs, pup_feats_bs]),
            ("all",      [card_feats_bs, eda_feats_bs, pup_feats_bs, bfi_feats_bs]),
            ("bfi+task", [bfi_feats_bs, task_cols]),
        ]
        for cname, flists in bs_conds:
            rows.append(_run_cond(cname, flists, bs_design, "reg", "gini_speaking",
                                   "Bs_speaking", "Speaking Gini", len(bs_design)))

    # â”€â”€ B6: Speech-overlap balance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if "overlap_fraction" in sub_a.columns and "group_id" in sub_a.columns:
        # Target: group-task mean overlap fraction
        overlap_grp = (sub_a.groupby(["group_id", "task"], as_index=False)
                       ["overlap_fraction"].mean())
        # Prosodic features only (no speaking/overlap = leakage)
        pros_src = [c for c in [
            "energy_mean", "energy_sd", "pitch_mean", "pitch_sd",
            "hnr_mean", "jitter_mean", "shimmer_mean",
            "voiced_segments_per_sec", "mean_voiced_segment_s", "speech_rate_proxy",
        ] if c in sub_a.columns]
        pros_grp = (sub_a.groupby(["group_id", "task"], as_index=False)[pros_src]
                    .mean()
                    .rename(columns={c: f"audio_{c}" for c in pros_src}))
        b6_design = overlap_grp.merge(pros_grp, on=["group_id", "task"], how="left")
        b6_design = b6_design.merge(grp_physio, on=["group_id", "task"], how="left")
        b6_design = b6_design.merge(bfi_grp, on="group_id", how="left")
        dummies_b6 = pd.get_dummies(b6_design["task"], prefix="task", dtype=float)
        b6_design = pd.concat([b6_design, dummies_b6], axis=1)
        task_cols_b6 = dummies_b6.columns.tolist()
        b6_audio = [f"audio_{c}" for c in pros_src if f"audio_{c}" in b6_design.columns]
        b6_bfi   = _avail(b6_design, BFI)
        b6_card  = _avail(b6_design, CARDIAC)
        b6_eda   = _avail(b6_design, EDA)
        b6_pup   = _avail(b6_design, PUPIL)
        b6_conds = [
            ("cardiac",  [b6_card]),
            ("eda",      [b6_eda]),
            ("pupil",    [b6_pup]),
            ("audio",    [b6_audio]),
            ("bfi",      [b6_bfi]),
            ("card+eda", [b6_card, b6_eda]),
            ("sensor",   [b6_card, b6_eda, b6_pup, b6_audio]),
            ("all",      [b6_card, b6_eda, b6_pup, b6_audio, b6_bfi]),
            ("bfi+task", [b6_bfi, task_cols_b6]),
        ]
        for cname, flists in b6_conds:
            rows.append(_run_cond(cname, flists, b6_design, "reg", "overlap_fraction",
                                   "B6_overlap", "Speech overlap", len(b6_design)))

    return pd.DataFrame(rows)


def _gini(s: pd.Series) -> float:
    a = s.dropna().to_numpy(dtype=float)
    if a.size == 0 or np.isclose(a.mean(), 0.0):
        return math.nan
    diffs = np.abs(a[:, None] - a[None, :]).sum()
    return float(diffs / (2 * a.size ** 2 * a.mean()))


# â”€â”€â”€ LaTeX output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fmt(v: float, decimals: int = 3) -> str:
    return "--" if (v is None or math.isnan(v)) else f"{v:.{decimals}f}"


def write_latex(df: pd.DataFrame, path: Path) -> None:
    cond_order = [c for c, _ in CONDITIONS]
    cond_labels_list = [COND_LABELS[c] for c in cond_order]

    pivot_rows: list[dict] = []
    for (bid, tgt), grp in df.groupby(["benchmark_id", "target"], sort=False):
        is_higher_better = grp["primary_metric"].iloc[0] in ("accuracy", "roc_auc", "pearson_r")
        row: dict = {"bid": bid, "target": tgt,
                     "metric": grp["primary_metric"].iloc[0],
                     "n": int(grp["n"].iloc[0]),
                     "higher_better": is_higher_better}
        for _, r in grp.iterrows():
            row[r["condition"]] = r["primary_value"]
        pivot_rows.append(row)

    bid_short = {
        "B0_task": "B0", "B1_affective": "B1", "B2_dominance": "B2",
        "B3_cognitive": "B3", "B4_personality": "B4", "B5_performance": "B5",
        "B6_overlap": "B6", "B7_floor": "B7", "Bs_speaking": r"B$_s$",
    }
    metric_short = {"accuracy": "Acc.", "roc_auc": "AUC", "mae": "MAE", "pearson_r": "$r$"}

    col_spec = "ll" + "r" * len(cond_order)
    lines = [
        r"% Ablation table â€” generated by tools/features/run_feature_ablation.py",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Feature-modality ablation: LOGO-CV primary metric under ten",
        r"    feature-subset conditions.",
        r"    \textbf{Cardiac}: HR mean and HRV RMSSD (delta + absolute).",
        r"    \textbf{EDA/GSR}: EDA tonic/phasic and skin temperature.",
        r"    \textbf{Pupil}: pupil dilation delta and absolute.",
        r"    \textbf{Audio}: speaking fraction, overlap, energy, pitch, prosody.",
        r"    \textbf{BFI}: Big Five personality traits.",
        r"    \textbf{Sensor}: all four sensor modalities combined.",
        r"    Bold = best condition per row.",
        r"    B7 audio conditions omitted (speaking features would be leakage).",
        r"    B6/B$_s$: group-task level; physio/pupil are group means.}",
        r"  \label{tab:ablation}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]
    header_conds = " & ".join(f"\\textbf{{{l}}}" for l in cond_labels_list)
    lines.append(f"    \\textbf{{Benchmark}} & \\textbf{{Metric}} & {header_conds} \\\\")
    lines.append(r"    \midrule")

    prev_bid = None
    for row in pivot_rows:
        bid = row["bid"]
        if prev_bid is not None and bid != prev_bid:
            lines.append(r"    \addlinespace[2pt]")
        prev_bid = bid
        prefix = bid_short.get(bid, bid)
        tgt_short = row["target"][:18]
        metric_disp = metric_short.get(row["metric"], row["metric"])

        vals = [row.get(c, float("nan")) for c in cond_order]
        fmts = [_fmt(v) for v in vals]
        valid_fmts = [f for f in fmts if f != "--"]
        if valid_fmts:
            best_fmt = (max if row["higher_better"] else min)(valid_fmts, key=float)
        else:
            best_fmt = None

        cells = []
        for f in fmts:
            if best_fmt is not None and f != "--" and f == best_fmt:
                cells.append(f"\\textbf{{{f}}}")
            else:
                cells.append(f)
        lines.append(f"    {prefix}: {tgt_short[:16]} & {metric_disp} & {' & '.join(cells)} \\\\")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", path)


def print_ablation(df: pd.DataFrame) -> None:
    cond_order = [c for c, _ in CONDITIONS]
    w = 100
    print("\n" + "=" * w)
    print(f"{'Benchmark':25s} {'Target':22s} {'Metric':8s} "
          + "  ".join(f"{COND_LABELS[c]:8s}" for c in cond_order))
    print("=" * w)
    prev_bid = None
    for (bid, tgt), grp in df.groupby(["benchmark_id", "target"], sort=False):
        if prev_bid is not None and bid != prev_bid:
            print("-" * w)
        prev_bid = bid
        metric = grp["primary_metric"].iloc[0]
        vals = {r["condition"]: r["primary_value"] for _, r in grp.iterrows()}
        is_higher = metric in ("accuracy", "roc_auc", "pearson_r")
        valid = {k: v for k, v in vals.items() if not math.isnan(v)}
        print(f"{bid[:25]:25s} {tgt[:22]:22s} {metric[:8]:8s} "
              + "  ".join(f"{vals.get(c, float('nan')):8.3f}"
                          if not math.isnan(vals.get(c, float('nan'))) else "      --"
                          for c in cond_order))


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> int:
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    LOG.info("Loading data ...")
    pt, parts = load_data()

    LOG.info("Running ablation (%d conditions) ...", len(CONDITIONS))
    results = run_ablation(pt, parts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_TSV, sep="\t", index=False)
    LOG.info("Wrote %s", OUT_TSV)

    write_latex(results, OUT_TEX)
    print_ablation(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

