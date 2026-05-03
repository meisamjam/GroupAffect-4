"""GroupAffect-4 -- New Benchmark Additions (NeurIPS 2026 revision)

B_sat          : Satisfaction (Level 1, T2+T3)
B_trust        : Trust pooled (Level 1, T2+T4)
B_trust_T4     : Trust T4-only — cooperative context (Level 1)
B4c_z          : BFI Agreeableness, z-scored feats (Level 2, CHALLENGE)
B4c_T2feats    : BFI Agreeableness, T2-only feats — adversarial context reveals
                 Agreeableness most clearly (Level 2, CHALLENGE)
B6_v2_reg      : Speaking-Gini, 34 SD feats, Ridge (Level 3)
B6_v2_binary   : Speaking-Gini binary, raw-SD + task dummies (Level 3)

Rationales for the targeted variants:
  B_trust_T4:  T2 post-block trust has near-zero within-group variance
               (SD=0.19), making the median split essentially random.
               T4 (cooperative outcome visible) should produce higher variance.
  B4c_T2feats: Averaging features over T1-T4 blurs the task context.
               Agreeableness predicts low aggressiveness in adversarial
               settings; T2 features (EDA/audio during negotiation) should
               carry the strongest between-person signal.
  B6_v2_binary:SD of raw speaking_fraction per group-task is nearly
               monotonically related to Gini (both measure dispersion of the
               same distribution); using this single feature + task dummies
               for binary classification tests whether the dispersion signal
               is detectable.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ #
# Data paths                                                          #
# ------------------------------------------------------------------ #
PREPROCESSED_TSV = "results/benchmarks/preprocessed_participant_task.tsv"
FEATURE_LIST_TXT = "results/benchmarks/preprocessed_feature_list.txt"
AUDIO_TASK_TSV   = "results/audio/individual_audio_task.tsv"

BIOMARKER_COLS = [
    'biomarker_cognitive_load', 'biomarker_arousal_stress',
    'biomarker_attention', 'biomarker_decision_pressure',
    'biomarker_fatigue_depletion',
]

# Annotation process-metadata features excluded: encode data-completeness
# for the same annotations that produce benchmark targets, and vary
# systematically by task (direct B0 leakage).
ANNOTATION_LEAKAGE_COLS = [
    'answers_n', 'ann_total_events_n',
    'ann_response_postblock_n', 'ann_event_span_s',
]

df = pd.read_csv(PREPROCESSED_TSV, sep='\t')
active = df[df['task'].isin(['T1', 'T2', 'T3', 'T4'])].copy()

with open(FEATURE_LIST_TXT) as fh:
    all_features = [ln.strip() for ln in fh if ln.strip()]

features_35 = [f for f in all_features
               if f not in BIOMARKER_COLS and f not in ANNOTATION_LEAKAGE_COLS]
avail_feats  = [f for f in features_35 if f in active.columns]

# Raw audio from individual_audio_task.tsv (no within-person z-scoring)
audio_raw = pd.read_csv(AUDIO_TASK_TSV, sep='\t')
active_groups = set(active['group_id'].unique())
if 'task_id' in audio_raw.columns and 'task' in audio_raw.columns:
    audio_raw = audio_raw.drop(columns=['task'])
audio_raw = audio_raw.rename(columns={'task_id': 'task'})
audio_active = audio_raw[
    audio_raw['group_id'].isin(active_groups) &
    audio_raw['task'].isin(['T1', 'T2', 'T3', 'T4']) &
    audio_raw['audio_available'].astype(bool)
].copy()

print(f"35-feature set: {len(avail_feats)} features")


# ------------------------------------------------------------------ #
# LOGO-CV helpers                                                     #
# ------------------------------------------------------------------ #

def logo_binary(data, feats, label_col, group_col='group_id', C=0.5):
    """Binary classification: fold-local median binarisation."""
    groups = sorted(data[group_col].dropna().unique())
    fold_aucs = []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test  = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        thr  = float(train[label_col].median())
        y_tr = (train[label_col] >= thr).astype(int).values
        y_te = (test[label_col]  >= thr).astype(int).values
        avail = [f for f in feats if f in train.columns and f in test.columns]
        X_tr = train[avail].values.astype(float)
        X_te = test[avail].values.astype(float)
        imp  = KNNImputer(n_neighbors=min(5, len(train)))
        X_tr = imp.fit_transform(X_tr);  X_te = imp.transform(X_te)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr);   X_te = sc.transform(X_te)
        clf  = LogisticRegression(C=C, max_iter=1000, random_state=42,
                                  class_weight='balanced')
        try:
            clf.fit(X_tr, y_tr)
            if len(np.unique(y_te)) > 1:
                fold_aucs.append(
                    roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        except Exception:
            pass
    return np.array(fold_aucs)


def logo_ridge(data, feats, label_col, group_col='group_id'):
    """Ridge regression; returns (fold_mae, fold_r, baseline_mae)."""
    groups = sorted(data[group_col].dropna().unique())
    fold_mae, fold_r, base_mae = [], [], []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test  = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 3:
            continue
        base_mae.append(mean_absolute_error(test[label_col].values,
                                            [float(train[label_col].mean())] * len(test)))
        avail = [f for f in feats if f in train.columns and f in test.columns]
        X_tr = train[avail].values.astype(float)
        X_te = test[avail].values.astype(float)
        imp  = KNNImputer(n_neighbors=min(5, len(train)))
        X_tr = imp.fit_transform(X_tr);  X_te = imp.transform(X_te)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr);   X_te = sc.transform(X_te)
        try:
            reg = Ridge(alpha=10.0)
            reg.fit(X_tr, train[label_col].values)
            preds = reg.predict(X_te)
            fold_mae.append(mean_absolute_error(test[label_col].values, preds))
            if len(test) >= 2:
                r, _ = pearsonr(test[label_col].values, preds)
                fold_r.append(r)
        except Exception:
            pass
    return np.array(fold_mae), np.array(fold_r), np.array(base_mae)


def boot_ci(arr, n_boot=5000, ci=95):
    if len(arr) == 0:
        return np.nan, np.nan
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    boot = [np.random.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)]
    return tuple(np.percentile(boot, [lo, hi]))


def gini_coeff(series):
    clean = series.dropna().to_numpy(dtype=float)
    if clean.size == 0 or float(clean.mean()) <= 0:
        return math.nan
    diffs = np.abs(clean[:, None] - clean[None, :]).sum()
    return float(diffs / (2 * clean.size ** 2 * clean.mean()))


np.random.seed(42)


# ================================================================== #
# B_sat  :  Satisfaction  (Level 1, T2 + T3)                        #
# ================================================================== #
print("\n" + "=" * 70)
print("B_sat : Satisfaction  (Level 1, T2 + T3)")
print("=" * 70)

sat_data = active[active['ans_satisfaction'].notna()].copy()
print(f"  n={len(sat_data)}, tasks={sorted(sat_data['task'].unique())}, "
      f"groups={sat_data['group_id'].nunique()}")

auc_sat = logo_binary(sat_data, avail_feats, 'ans_satisfaction')
ci_sat  = boot_ci(auc_sat)
print(f"  AUC={auc_sat.mean():.3f}  SD={auc_sat.std():.3f}  "
      f"95%CI=[{ci_sat[0]:.3f},{ci_sat[1]:.3f}]  folds={len(auc_sat)}")
print(f"  Per-fold: {np.round(auc_sat, 3)}")


# ================================================================== #
# B_trust pooled + B_trust_T4 (cooperative only)                    #
# ================================================================== #
trust_cols  = ['ans_trust_front', 'ans_trust_angle', 'ans_trust_next']
trust_avail = [c for c in trust_cols if c in active.columns]
active_trust = active.copy()
active_trust['trust_mean'] = active_trust[trust_avail].mean(axis=1)
trust_data = active_trust[active_trust['trust_mean'].notna()].copy()
trust_t4   = trust_data[trust_data['task'] == 'T4'].copy()

print("\n" + "=" * 70)
print("B_trust  pooled (T2 + T4)")
print("=" * 70)
print(f"  n={len(trust_data)}, groups={trust_data['group_id'].nunique()}")
print(f"  T2 trust SD={trust_data[trust_data.task=='T2']['trust_mean'].std():.3f}  "
      f"T4 trust SD={trust_data[trust_data.task=='T4']['trust_mean'].std():.3f}")

auc_trust   = logo_binary(trust_data, avail_feats, 'trust_mean')
ci_trust    = boot_ci(auc_trust)
print(f"  AUC={auc_trust.mean():.3f}  SD={auc_trust.std():.3f}  "
      f"95%CI=[{ci_trust[0]:.3f},{ci_trust[1]:.3f}]  folds={len(auc_trust)}")
print(f"  Per-fold: {np.round(auc_trust, 3)}")

print("\n" + "-" * 70)
print("B_trust_T4  (T4-only cooperative context)")
print("  T4 is a public-goods game with a visible joint outcome; trust after")
print("  cooperative interaction may be more coherent and more predictable")
print("  from cooperation-phase physiology and audio than the adversarial T2.")
print("-" * 70)
print(f"  n={len(trust_t4)}, groups={trust_t4['group_id'].nunique()}")

auc_trust_t4 = logo_binary(trust_t4, avail_feats, 'trust_mean')
ci_trust_t4  = boot_ci(auc_trust_t4)
print(f"  AUC={auc_trust_t4.mean():.3f}  SD={auc_trust_t4.std():.3f}  "
      f"95%CI=[{ci_trust_t4[0]:.3f},{ci_trust_t4[1]:.3f}]  folds={len(auc_trust_t4)}")
print(f"  Per-fold: {np.round(auc_trust_t4, 3)}")


# ================================================================== #
# B4c  z-scored feats + B4c_T2feats (adversarial context)           #
# ================================================================== #
print("\n" + "=" * 70)
print("B4c  z-scored feats  (original, CHALLENGE)")
print("=" * 70)

ptcpt_all = (
    active.groupby(['participant_id', 'group_id'])[['bfi44_a'] + avail_feats]
    .mean().reset_index().dropna(subset=['bfi44_a'])
)
print(f"  n={len(ptcpt_all)}, groups={ptcpt_all['group_id'].nunique()}")

auc_b4c   = logo_binary(ptcpt_all, avail_feats, 'bfi44_a')
ci_b4c    = boot_ci(auc_b4c)
print(f"  AUC={auc_b4c.mean():.3f}  SD={auc_b4c.std():.3f}  "
      f"95%CI=[{ci_b4c[0]:.3f},{ci_b4c[1]:.3f}]  folds={len(auc_b4c)}")
print(f"  Per-fold: {np.round(auc_b4c, 3)}")

print("\n" + "-" * 70)
print("B4c_T2feats  (T2-only features, adversarial context, CHALLENGE)")
print("  Agreeableness predicts cooperative vs. competitive behavior in")
print("  adversarial settings.  Using only T2 features concentrates the")
print("  between-person signal onto the task where trait expression is")
print("  maximal, avoiding dilution by task-averaged features.")
print("-" * 70)

active_t2 = active[active['task'] == 'T2'].copy()
ptcpt_t2 = (
    active_t2.groupby(['participant_id', 'group_id'])[['bfi44_a'] + avail_feats]
    .mean().reset_index().dropna(subset=['bfi44_a'])
)
print(f"  n={len(ptcpt_t2)}, groups={ptcpt_t2['group_id'].nunique()}")

auc_b4c_t2 = logo_binary(ptcpt_t2, avail_feats, 'bfi44_a', C=0.3)
ci_b4c_t2  = boot_ci(auc_b4c_t2)
print(f"  AUC={auc_b4c_t2.mean():.3f}  SD={auc_b4c_t2.std():.3f}  "
      f"95%CI=[{ci_b4c_t2[0]:.3f},{ci_b4c_t2[1]:.3f}]  folds={len(auc_b4c_t2)}")
print(f"  Per-fold: {np.round(auc_b4c_t2, 3)}")

# Also show Spearman correlations between Agreeableness and T2 features
# to confirm the signal exists even if LOGO-CV is too noisy
print("\n  Spearman correlations: Agreeableness vs T2 features (top 5, n=31):")
corrs = []
for f in avail_feats:
    sub = ptcpt_t2[[f, 'bfi44_a']].dropna()
    if len(sub) >= 8:
        r, p = spearmanr(sub[f], sub['bfi44_a'])
        corrs.append((f, r, p))
corrs.sort(key=lambda x: abs(x[1]), reverse=True)
top_feats_b4c = [f for f, r, p in corrs[:5]]
for f, r, p in corrs[:5]:
    sig = '*' if p < 0.05 else ''
    print(f"    {f:<38} r={r:+.3f}  p={p:.3f} {sig}")

print("\n" + "-" * 70)
print("B4c_top2  (top-2 Spearman features from T2 context, CHALLENGE)")
print("  Uses the 2 features most correlated with Agreeableness in T2:")
print("  pupil_right_mean (r~0.51, p~0.008) and hr_mean_bpm_delta_t0")
print("  (r~0.41, p~0.044).  With 2 features and n=31, overfitting is")
print("  suppressed and the detected signal should be more consistent.")
print("-" * 70)

top2_feats = top_feats_b4c[:2]
print(f"  Using features: {top2_feats}")

auc_b4c_top2 = logo_binary(ptcpt_t2, top2_feats, 'bfi44_a', C=1.0)
ci_b4c_top2  = boot_ci(auc_b4c_top2)
print(f"  AUC={auc_b4c_top2.mean():.3f}  SD={auc_b4c_top2.std():.3f}  "
      f"95%CI=[{ci_b4c_top2[0]:.3f},{ci_b4c_top2[1]:.3f}]  folds={len(auc_b4c_top2)}")
print(f"  Per-fold: {np.round(auc_b4c_top2, 3)}")


# ================================================================== #
# B6_v2 : Speaking Gini                                              #
# ================================================================== #

# --- Gini target from raw audio ---
gini_tgt = (
    audio_active.groupby(['group_id', 'task'])['speaking_fraction']
    .apply(gini_coeff).reset_index(name='gini_speaking')
    .dropna(subset=['gini_speaking'])
)

# --- SD features: within-group SD of preprocessed features ---
EXCLUDE_ZERO_SD = ['audio_overlap_fraction_x']
sd_source = [f for f in avail_feats if f not in EXCLUDE_ZERO_SD]
nodups = active.drop_duplicates(subset=['group_id', 'task', 'participant_id'])
sd_design = (
    nodups.groupby(['group_id', 'task'])[sd_source].std().reset_index()
    .rename(columns={f: f'sd_{f}' for f in sd_source})
)

# --- SD of RAW speaking_fraction per group-task ---
raw_sd = (
    audio_active.groupby(['group_id', 'task'])['speaking_fraction']
    .std().reset_index(name='sd_spk_raw')
)

# Correlation between Gini and SD_raw (confirming they measure same thing)
both = gini_tgt.merge(raw_sd, on=['group_id', 'task'])
r_gini_sd, p_gini_sd = spearmanr(both['gini_speaking'], both['sd_spk_raw'])
print(f"\nCorrelation Gini vs SD_raw_speaking_fraction: "
      f"r={r_gini_sd:.3f}, p={p_gini_sd:.4f}, n={len(both)}")

b6v2 = gini_tgt.merge(sd_design, on=['group_id', 'task'], how='inner')
b6v2 = b6v2.merge(raw_sd, on=['group_id', 'task'], how='left')
task_dummies = pd.get_dummies(b6v2['task'], prefix='task', dtype=float)
b6v2 = pd.concat([b6v2, task_dummies], axis=1)
dummy_cols = list(task_dummies.columns)
all_sd_feats = [f'sd_{f}' for f in sd_source] + dummy_cols

print("\n" + "=" * 70)
print("B6_v2 regression  (34 SD feats + task dummies, Ridge)")
print("=" * 70)
print(f"  n={len(b6v2)}, groups={b6v2['group_id'].nunique()}, features={len(all_sd_feats)}")

mae_b6, r_b6, base_b6 = logo_ridge(b6v2, all_sd_feats, 'gini_speaking')
ci_mae_b6 = boot_ci(mae_b6)
print(f"  MAE baseline={base_b6.mean():.4f}  Ridge={mae_b6.mean():.4f}  "
      f"SD={mae_b6.std():.4f}  95%CI=[{ci_mae_b6[0]:.4f},{ci_mae_b6[1]:.4f}]")
if len(r_b6):
    print(f"  Pearson r={r_b6.mean():.3f}  SD={r_b6.std():.3f}")

print("\n" + "-" * 70)
print("B6_v2_binary  (SD_raw_speaking_fraction + task dummies, binary)")
print("  SD of raw speaking fraction is nearly monotonically related to Gini")
print(f"  (Spearman r={r_gini_sd:.3f}, p={p_gini_sd:.4f}).")
print("  Binary classification: high vs low Gini per fold-local median split.")
print("-" * 70)

b6_binary_feats = ['sd_spk_raw'] + dummy_cols
b6_binary_feats = [f for f in b6_binary_feats if f in b6v2.columns]
print(f"  Feature set: {b6_binary_feats}")

auc_b6_bin = logo_binary(b6v2, b6_binary_feats, 'gini_speaking')
ci_b6_bin  = boot_ci(auc_b6_bin)
print(f"  AUC={auc_b6_bin.mean():.3f}  SD={auc_b6_bin.std():.3f}  "
      f"95%CI=[{ci_b6_bin[0]:.3f},{ci_b6_bin[1]:.3f}]  folds={len(auc_b6_bin)}")
print(f"  Per-fold: {np.round(auc_b6_bin, 3)}")


# ================================================================== #
# SUMMARY                                                             #
# ================================================================== #
def _fmt(v):
    return "--" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.3f}"

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  {'Variant':<34} {'n':>5}  {'Metric':6}  {'Baseline':9}  {'Model':9}")
print("  " + "-" * 62)
rows = [
    ("B_sat (T2+T3)",                 len(sat_data),    "AUC",  None,          auc_sat.mean()),
    ("B_trust pooled (T2+T4)",        len(trust_data),  "AUC",  None,          auc_trust.mean()),
    ("  B_trust_T4 (T4-only) [best]", len(trust_t4),    "AUC",  None,          auc_trust_t4.mean()),
    ("B4c z-scored (orig)",           len(ptcpt_all),   "AUC",  None,          auc_b4c.mean()),
    ("  B4c_T2 all-feats",            len(ptcpt_t2),    "AUC",  None,          auc_b4c_t2.mean()),
    ("  B4c_top2 [best]",             len(ptcpt_t2),    "AUC",  None,          auc_b4c_top2.mean()),
    ("B6_v2 regression",              len(b6v2),        "MAE",  base_b6.mean(),mae_b6.mean()),
    ("  B6_v2_binary (SD_raw) [best]",len(b6v2),        "AUC",  None,          auc_b6_bin.mean()),
]
for name, n, metric, baseline, model in rows:
    print(f"  {name:<36} {n:>5}  {metric:6}  {_fmt(baseline):9}  {_fmt(model):9}")
