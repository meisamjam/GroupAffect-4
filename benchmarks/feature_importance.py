"""GroupAffect-4 -- Feature Importance Visualisation (NeurIPS 2026)

Produces three publication-quality figures saved to paper/figure/benchmarks/:

  feature_importance_ranked.pdf   -- Per-benchmark horizontal bar chart of
                                     mean |coefficient| across LOGO-CV folds,
                                     coloured by modality group.

  feature_importance_heatmap.pdf  -- Cross-benchmark heatmap: rows = 31 features
                                     (ranked by overall importance), cols = benchmarks.
                                     Cell colour = mean normalised |coeff|.

  modality_fold_strip.pdf         -- Per-benchmark strip + box of fold AUC/ACC,
                                     with a coloured band showing the chance baseline
                                     and points coloured by modality group.

Feature set: 31 sensor/behavioural features (40 total minus 5 biomarker composites
minus 4 annotation process-metadata features excluded for target-leakage reasons).
The annotation features -- answers_n, ann_total_events_n, ann_response_postblock_n,
ann_event_span_s -- encode annotation-data completeness, which is structurally
correlated with the annotation-derived benchmark targets (VAD probes → B1/B2;
post-block forms → B3/B_trust), and are therefore excluded from this analysis.

Run from the repo root:
    python paper/analysis/feature_importance.py
"""

import warnings
warnings.filterwarnings("ignore")

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------ #
# Paths                                                               #
# ------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_TSV  = ROOT / "results/benchmarks/preprocessed_participant_task.tsv"
FEATURE_LIST_TXT  = ROOT / "results/benchmarks/preprocessed_feature_list.txt"
OUT_DIR           = ROOT / "paper/figure/benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIOMARKER_COLS = [
    "biomarker_cognitive_load", "biomarker_arousal_stress",
    "biomarker_attention", "biomarker_decision_pressure",
    "biomarker_fatigue_depletion",
]

# ------------------------------------------------------------------ #
# Modality colour palette                                             #
# ------------------------------------------------------------------ #
MODALITY_MAP = {
    "hr_mean_bpm_delta_t0":        "Physiology",
    "hr_sd_bpm":                   "Physiology",
    "hrv_rmssd_ms":                "Physiology",
    "hrv_quality_score":           "Physiology",
    "eda_tonic_mean_delta_t0":     "Physiology",
    "eda_phasic_rate_hz_delta_t0": "Physiology",
    "eda_phasic_mean_delta_t0":    "Physiology",
    "eda_scr_count":               "Physiology",
    "temp_mean_delta_t0":          "Motion/Temp",
    "accel_motion_mean":           "Motion/Temp",
    "motion_high_fraction":        "Motion/Temp",
    "pupil_left_mean":             "Eye-tracking",
    "pupil_right_mean":            "Eye-tracking",
    "pupil_std":                   "Eye-tracking",
    "pupil_slope_per_s":           "Eye-tracking",
    "pupil_mean_delta_t0":         "Eye-tracking",
    # NOTE: annotation process-metadata features (answers_n, ann_total_events_n,
    # ann_response_postblock_n, ann_event_span_s) are intentionally excluded.
    # They encode data-completeness for the same annotations that produce the
    # benchmark targets (VAD probes → B1/B2; post-block forms → B3/B_trust),
    # so including them would constitute target-leakage.
    "audio_energy_mean_x":         "Audio",
    "audio_energy_sd_x":           "Audio",
    "audio_hnr_mean_x":            "Audio",
    "audio_jitter_mean_x":         "Audio",
    "audio_mean_unvoiced_segment_s": "Audio",
    "audio_mean_voiced_segment_s_x": "Audio",
    "audio_overlap_fraction_x":    "Audio",
    "audio_pause_count":           "Audio",
    "audio_pitch_mean_x":          "Audio",
    "audio_pitch_sd_x":            "Audio",
    "audio_shimmer_mean_x":        "Audio",
    "audio_speaking_fraction_x":   "Audio",
    "audio_speaking_time_s":       "Audio",
    "audio_speech_rate_proxy_x":   "Audio",
    "audio_voiced_segments_per_sec_x": "Audio",
}

MODALITY_COLORS = {
    "Physiology":   "#2171b5",
    "Eye-tracking": "#238b45",
    "Audio":        "#d94801",
    "Motion/Temp":  "#756bb1",
}

FEATURE_LABELS = {
    "hr_mean_bpm_delta_t0":           "HR mean Δbaseline",
    "hr_sd_bpm":                      "HR SD",
    "hrv_rmssd_ms":                   "HRV RMSSD",
    "hrv_quality_score":              "HRV quality",
    "eda_tonic_mean_delta_t0":        "EDA tonic Δbaseline",
    "eda_phasic_rate_hz_delta_t0":    "EDA phasic rate Δbaseline",
    "eda_phasic_mean_delta_t0":       "EDA phasic mean Δbaseline",
    "eda_scr_count":                  "SCR count",
    "temp_mean_delta_t0":             "Skin temp Δbaseline",
    "accel_motion_mean":              "Accel motion mean",
    "motion_high_fraction":           "High-motion fraction",
    "pupil_left_mean":                "Pupil left mean",
    "pupil_right_mean":               "Pupil right mean",
    "pupil_std":                      "Pupil SD",
    "pupil_slope_per_s":              "Pupil slope",
    "pupil_mean_delta_t0":            "Pupil mean Δbaseline",
    # Annotation process-metadata features excluded (target-leakage risk)
    "audio_energy_mean_x":            "Audio energy mean",
    "audio_energy_sd_x":              "Audio energy SD",
    "audio_hnr_mean_x":               "HNR mean",
    "audio_jitter_mean_x":            "Jitter mean",
    "audio_mean_unvoiced_segment_s":  "Unvoiced seg. mean (s)",
    "audio_mean_voiced_segment_s_x":  "Voiced seg. mean (s)",
    "audio_overlap_fraction_x":       "Overlap fraction ★",
    "audio_pause_count":              "Pause count",
    "audio_pitch_mean_x":             "Pitch mean",
    "audio_pitch_sd_x":               "Pitch SD",
    "audio_shimmer_mean_x":           "Shimmer mean",
    "audio_speaking_fraction_x":      "Speaking fraction",
    "audio_speaking_time_s":          "Speaking time (s)",
    "audio_speech_rate_proxy_x":      "Speech rate proxy",
    "audio_voiced_segments_per_sec_x":"Voiced segs/s",
}

ANNOTATION_LEAKAGE_COLS = {
    "answers_n", "ann_total_events_n", "ann_response_postblock_n", "ann_event_span_s",
}

# ------------------------------------------------------------------ #
# Load data                                                           #
# ------------------------------------------------------------------ #
df = pd.read_csv(PREPROCESSED_TSV, sep="\t")
active = df[df["task"].isin(["T1", "T2", "T3", "T4"])].copy()

with open(FEATURE_LIST_TXT) as fh:
    all_features = [ln.strip() for ln in fh if ln.strip()]

feats_35 = [
    f for f in all_features
    if f not in BIOMARKER_COLS and f not in ANNOTATION_LEAKAGE_COLS
]
avail    = [f for f in feats_35 if f in active.columns]
print(f"Available features (annotation leakage excluded): {len(avail)}")

# ------------------------------------------------------------------ #
# LOGO-CV with coefficient extraction                                 #
# ------------------------------------------------------------------ #

def logo_coefs_binary(data, feats, label_col, group_col="group_id", C=0.5):
    """Return (fold_aucs, coef_matrix [n_folds × n_feats])."""
    groups   = sorted(data[group_col].dropna().unique())
    fold_auc = []
    coef_mat = []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test  = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        thr  = float(train[label_col].median())
        y_tr = (train[label_col] >= thr).astype(int).values
        y_te = (test[label_col]  >= thr).astype(int).values
        X_tr = train[feats].values.astype(float)
        X_te = test[feats].values.astype(float)
        imp  = KNNImputer(n_neighbors=min(5, len(train)))
        X_tr = imp.fit_transform(X_tr); X_te = imp.transform(X_te)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr);  X_te = sc.transform(X_te)
        clf  = LogisticRegression(C=C, max_iter=1000, random_state=42,
                                  class_weight="balanced")
        try:
            clf.fit(X_tr, y_tr)
            coef_mat.append(np.abs(clf.coef_[0]))
            if len(np.unique(y_te)) > 1:
                fold_auc.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        except Exception:
            pass
    coef_arr = np.array(coef_mat) if coef_mat else np.zeros((1, len(feats)))
    return np.array(fold_auc), coef_arr


def logo_coefs_multiclass(data, feats, label_col, group_col="group_id"):
    """Return (fold_accs, coef_matrix [n_folds × n_feats])."""
    groups   = sorted(data[group_col].dropna().unique())
    fold_acc = []
    coef_mat = []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test  = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        X_tr = train[feats].values.astype(float)
        X_te = test[feats].values.astype(float)
        imp  = KNNImputer(n_neighbors=5)
        X_tr = imp.fit_transform(X_tr); X_te = imp.transform(X_te)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr);  X_te = sc.transform(X_te)
        clf  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        try:
            clf.fit(X_tr, train[label_col].values)
            fold_acc.append(accuracy_score(test[label_col].values, clf.predict(X_te)))
            # mean |coef| over classes
            coef_mat.append(np.abs(clf.coef_).mean(axis=0))
        except Exception:
            pass
    coef_arr = np.array(coef_mat) if coef_mat else np.zeros((1, len(feats)))
    return np.array(fold_acc), coef_arr


# ------------------------------------------------------------------ #
# Run all benchmarks and collect coefficients                         #
# ------------------------------------------------------------------ #
np.random.seed(42)

# Compute trust_mean from per-seat items if available
for trust_col in ["ans_trust_front", "ans_trust_next", "ans_trust_angle"]:
    if trust_col not in active.columns:
        active[trust_col] = np.nan
active["trust_mean"] = active[["ans_trust_front", "ans_trust_next", "ans_trust_angle"]].mean(axis=1)

BENCHMARKS = {}

# B0 task label — multiclass
b0_data = active.dropna(subset=["task"]).copy()
b0_aucs, b0_coefs = logo_coefs_multiclass(b0_data, avail, "task")
BENCHMARKS["B0\nTask label"] = {"metric": "Acc", "perf": b0_aucs, "coefs": b0_coefs,
                                 "chance": 0.265, "feats": avail}

# B1a valence
if "vad_valence" in active.columns:
    b1a_aucs, b1a_coefs = logo_coefs_binary(active.dropna(subset=["vad_valence"]), avail, "vad_valence")
    BENCHMARKS["B1a\nValence"] = {"metric": "AUC", "perf": b1a_aucs, "coefs": b1a_coefs,
                                   "chance": 0.5, "feats": avail}

# B1b arousal
if "vad_arousal" in active.columns:
    b1b_aucs, b1b_coefs = logo_coefs_binary(active.dropna(subset=["vad_arousal"]), avail, "vad_arousal")
    BENCHMARKS["B1b\nArousal"] = {"metric": "AUC", "perf": b1b_aucs, "coefs": b1b_coefs,
                                   "chance": 0.5, "feats": avail}

# B2 dominance
if "vad_dominance" in active.columns:
    b2_aucs, b2_coefs = logo_coefs_binary(active.dropna(subset=["vad_dominance"]), avail, "vad_dominance")
    BENCHMARKS["B2\nDominance"] = {"metric": "AUC", "perf": b2_aucs, "coefs": b2_coefs,
                                    "chance": 0.5, "feats": avail}

# B3a mental demand
if "ans_mental_demand" in active.columns:
    b3a_aucs, b3a_coefs = logo_coefs_binary(active.dropna(subset=["ans_mental_demand"]), avail, "ans_mental_demand")
    BENCHMARKS["B3a\nMental demand"] = {"metric": "AUC", "perf": b3a_aucs, "coefs": b3a_coefs,
                                         "chance": 0.5, "feats": avail}

# B3b engagement
if "ans_engagement" in active.columns:
    b3b_aucs, b3b_coefs = logo_coefs_binary(active.dropna(subset=["ans_engagement"]), avail, "ans_engagement")
    BENCHMARKS["B3b\nEngagement"] = {"metric": "AUC", "perf": b3b_aucs, "coefs": b3b_coefs,
                                      "chance": 0.5, "feats": avail}

# B_sat satisfaction (T2+T3 only)
sat_tasks = active[active["task"].isin(["T2", "T3"])].copy()
if "ans_satisfaction" in sat_tasks.columns:
    bsat_aucs, bsat_coefs = logo_coefs_binary(sat_tasks.dropna(subset=["ans_satisfaction"]), avail, "ans_satisfaction")
    BENCHMARKS["B_sat\nSatisfaction"] = {"metric": "AUC", "perf": bsat_aucs, "coefs": bsat_coefs,
                                          "chance": 0.5, "feats": avail}

# B_trust (T2+T4)
trust_tasks = active[active["task"].isin(["T2", "T4"])].copy()
if trust_tasks["trust_mean"].notna().sum() >= 10:
    btrust_aucs, btrust_coefs = logo_coefs_binary(trust_tasks.dropna(subset=["trust_mean"]), avail, "trust_mean")
    BENCHMARKS["B_trust\nTrust"] = {"metric": "AUC", "perf": btrust_aucs, "coefs": btrust_coefs,
                                     "chance": 0.5, "feats": avail}

print(f"Benchmarks computed: {list(BENCHMARKS.keys())}")

if not BENCHMARKS:
    print("ERROR: No benchmark columns found in data. Check column names in TSV.")
    import sys; sys.exit(1)

# ------------------------------------------------------------------ #
# Helper: normalise coefficients across features per benchmark        #
# ------------------------------------------------------------------ #

def mean_norm_coefs(coef_arr, feats):
    """Mean |coef| per feature, min-max normalised to [0,1]."""
    mean_c = coef_arr.mean(axis=0)
    mn, mx = mean_c.min(), mean_c.max()
    if mx - mn < 1e-9:
        return np.zeros_like(mean_c)
    return (mean_c - mn) / (mx - mn)


# ------------------------------------------------------------------ #
# Figure 1: Ranked horizontal bar charts (one panel per benchmark)   #
# ------------------------------------------------------------------ #
N_BENCH = len(BENCHMARKS)
N_TOP   = 15          # top-N features shown per benchmark
NCOLS   = min(4, N_BENCH)
NROWS   = math.ceil(N_BENCH / NCOLS)

fig1, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4.2, NROWS * 3.8))
axes = np.array(axes).flatten()

for ax_i, (bname, bdata) in enumerate(BENCHMARKS.items()):
    ax   = axes[ax_i]
    nf   = mean_norm_coefs(bdata["coefs"], bdata["feats"])
    ranked_idx = np.argsort(nf)[::-1][:N_TOP]
    ranked_idx = ranked_idx[::-1]   # ascending for horizontal bar (bottom = most important)

    bar_labels  = [FEATURE_LABELS.get(bdata["feats"][i], bdata["feats"][i]) for i in ranked_idx]
    bar_vals    = nf[ranked_idx]
    bar_colors  = [MODALITY_COLORS.get(MODALITY_MAP.get(bdata["feats"][i], ""), "#aaaaaa")
                   for i in ranked_idx]

    ax.barh(range(len(ranked_idx)), bar_vals, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(ranked_idx)))
    ax.set_yticklabels(bar_labels, fontsize=7)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Norm. |coef|", fontsize=7)
    ax.set_title(bname.replace("\n", " — "), fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate mean AUC/Acc in corner
    if len(bdata["perf"]) > 0:
        ax.text(0.98, 0.02, f'{bdata["metric"]} {bdata["perf"].mean():.3f}',
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="dimgray")

# hide surplus axes
for ax in axes[N_BENCH:]:
    ax.set_visible(False)

# shared legend
legend_patches = [mpatches.Patch(color=c, label=m)
                  for m, c in MODALITY_COLORS.items()]
fig1.legend(handles=legend_patches, loc="lower center", ncol=len(MODALITY_COLORS),
            fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

fig1.suptitle("Feature importance per benchmark (mean |coefficient|, LOGO-CV folds)",
              fontsize=10, fontweight="bold", y=1.01)
fig1.tight_layout(rect=[0, 0.03, 1, 1])
out1 = OUT_DIR / "feature_importance_ranked.pdf"
fig1.savefig(out1, bbox_inches="tight", dpi=200)
fig1.savefig(out1.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved: {out1}")
plt.close(fig1)


# ------------------------------------------------------------------ #
# Figure 2: Cross-benchmark importance heatmap                        #
# ------------------------------------------------------------------ #

# Build matrix: rows = features ordered by overall importance, cols = benchmarks
bnames = list(BENCHMARKS.keys())
feat_names = avail

# Collect normalised coef per benchmark
coef_matrix = np.zeros((len(feat_names), len(bnames)))
for bj, bname in enumerate(bnames):
    bdata = BENCHMARKS[bname]
    nf = mean_norm_coefs(bdata["coefs"], bdata["feats"])
    # align to global feat_names order
    for fi, fn in enumerate(feat_names):
        if fn in bdata["feats"]:
            idx = bdata["feats"].index(fn)
            coef_matrix[fi, bj] = nf[idx]

# Sort features by mean importance across benchmarks
row_order = np.argsort(coef_matrix.mean(axis=1))[::-1]
coef_sorted = coef_matrix[row_order]
row_labels  = [FEATURE_LABELS.get(feat_names[i], feat_names[i]) for i in row_order]
row_modality = [MODALITY_MAP.get(feat_names[i], "") for i in row_order]

fig2, ax2 = plt.subplots(figsize=(max(7, N_BENCH * 1.2), max(8, len(feat_names) * 0.38)))
im = ax2.imshow(coef_sorted, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                interpolation="nearest")

col_labels = [b.replace("\n", "\n") for b in bnames]
ax2.set_xticks(range(N_BENCH))
ax2.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
ax2.set_yticks(range(len(feat_names)))
ax2.set_yticklabels(row_labels, fontsize=7)

# colour y-tick labels by modality
for yi, lbl in enumerate(ax2.get_yticklabels()):
    mod = row_modality[yi]
    lbl.set_color(MODALITY_COLORS.get(mod, "black"))

plt.colorbar(im, ax=ax2, label="Normalised |coefficient|", shrink=0.6, pad=0.02)
ax2.set_title("Cross-benchmark feature importance heatmap\n"
              "(colour = mean normalised |coef| across LOGO-CV folds)",
              fontsize=9, fontweight="bold", pad=8)

# modality legend via y-tick colour key
legend_patches = [mpatches.Patch(color=c, label=m)
                  for m, c in MODALITY_COLORS.items()]
ax2.legend(handles=legend_patches, loc="upper right", fontsize=7,
           framealpha=0.9, title="Modality", title_fontsize=7,
           bbox_to_anchor=(1.22, 1))

fig2.tight_layout()
out2 = OUT_DIR / "feature_importance_heatmap.pdf"
fig2.savefig(out2, bbox_inches="tight", dpi=200)
fig2.savefig(out2.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved: {out2}")
plt.close(fig2)


# ------------------------------------------------------------------ #
# Figure 3: Per-fold strip + box plot across benchmarks              #
# ------------------------------------------------------------------ #
fig3, ax3 = plt.subplots(figsize=(max(8, N_BENCH * 1.4), 4.5))

bench_keys = list(BENCHMARKS.keys())
x_positions = np.arange(N_BENCH)
jitter_w    = 0.18

for xi, bname in enumerate(bench_keys):
    bdata  = BENCHMARKS[bname]
    perfs  = bdata["perf"]
    chance = bdata["chance"]

    # chance band
    ax3.axhspan(chance - 0.005, chance + 0.005, alpha=0.15, color="gray", zorder=0)
    ax3.hlines(chance, xi - 0.4, xi + 0.4, colors="gray", linewidths=1.0,
               linestyles="--", zorder=1)

    if len(perfs) == 0:
        continue

    # box
    bp = ax3.boxplot(perfs, positions=[xi], widths=0.35, patch_artist=True,
                     showfliers=False, zorder=2,
                     boxprops=dict(facecolor="#d0d0d0", alpha=0.5, linewidth=0.8),
                     medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8))

    # individual fold dots
    jitter = np.random.uniform(-jitter_w, jitter_w, size=len(perfs))
    ax3.scatter(xi + jitter, perfs, s=28, alpha=0.85, zorder=3,
                color="#2c7bb6", edgecolors="white", linewidths=0.5)

    # annotate mean
    ax3.text(xi, perfs.mean() + 0.025, f"{perfs.mean():.3f}",
             ha="center", va="bottom", fontsize=7, fontweight="bold", color="#2c7bb6")

ax3.set_xticks(x_positions)
ax3.set_xticklabels([b.replace("\n", "\n") for b in bench_keys], fontsize=8)
ax3.set_ylabel("AUC / Accuracy", fontsize=9)
ax3.set_title("LOGO-CV per-fold performance across benchmarks\n"
              "(grey dashed line = chance; mean annotated above median)",
              fontsize=9, fontweight="bold")
ax3.set_xlim(-0.6, N_BENCH - 0.4)
ax3.set_ylim(0.0, 1.05)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.yaxis.grid(True, linestyle=":", alpha=0.5)

# metric type annotations on x-axis
for xi, bname in enumerate(bench_keys):
    metric = BENCHMARKS[bname]["metric"]
    ax3.text(xi, -0.09, f"[{metric}]", ha="center", va="top",
             fontsize=6.5, color="dimgray",
             transform=ax3.get_xaxis_transform())

fig3.tight_layout()
out3 = OUT_DIR / "modality_fold_strip.pdf"
fig3.savefig(out3, bbox_inches="tight", dpi=200)
fig3.savefig(out3.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"Saved: {out3}")
plt.close(fig3)

print("\nAll figures written to:", OUT_DIR)
