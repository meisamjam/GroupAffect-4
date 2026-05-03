"""
Benchmarks without biomarker composite features.
Reports:
  - All B0-B7 results with 35 features (biomarker_* excluded)
  - Bootstrap 95% CIs on mean AUC/ACC for every benchmark (B0-B5)
  - Single-feature B3 baseline using audio_overlap_fraction_x only
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BIOMARKER_COLS = [
    'biomarker_cognitive_load',
    'biomarker_arousal_stress',
    'biomarker_attention',
    'biomarker_decision_pressure',
    'biomarker_fatigue_depletion',
]

# Annotation process-metadata features are excluded because they encode
# data-completeness for the same annotations that produce the benchmark
# targets (VAD probes -> B1/B2; post-block forms -> B3/B_trust).
# ann_event_span_s and answers_n also vary systematically by task,
# creating direct leakage into the B0 task-classification sanity check.
ANNOTATION_LEAKAGE_COLS = [
    'answers_n',
    'ann_total_events_n',
    'ann_response_postblock_n',
    'ann_event_span_s',
]

df = pd.read_csv("results/benchmarks/preprocessed_participant_task.tsv", sep='\t')
active = df[df['task'].isin(['T1', 'T2', 'T3', 'T4'])].copy()

with open("results/benchmarks/preprocessed_feature_list.txt") as f:
    all_features = [l.strip() for l in f if l.strip()]

features = [f for f in all_features
            if f not in BIOMARKER_COLS and f not in ANNOTATION_LEAKAGE_COLS]
available_feats = [f for f in features if f in active.columns]

print(f"Original features: {len(all_features)}")
print(f"After removing biomarker_* and annotation leakage: {len(features)}")
print(f"Available in data: {len(available_feats)}")
print(f"Removed biomarkers: {[f for f in BIOMARKER_COLS if f in all_features]}")
print(f"Removed annotation leakage: {ANNOTATION_LEAKAGE_COLS}")


def logo_multiclass(data, feats, label_col, group_col='group_id'):
    groups = sorted(data[group_col].unique())
    fold_accs = []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        X_tr = train[feats].values.astype(float)
        X_te = test[feats].values.astype(float)
        imp = KNNImputer(n_neighbors=5)
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        try:
            clf.fit(X_tr, train[label_col].values)
            fold_accs.append(accuracy_score(test[label_col].values, clf.predict(X_te)))
        except:
            pass
    return np.array(fold_accs)


def logo_binary(data, feats, label_col, group_col='group_id'):
    groups = sorted(data[group_col].unique())
    fold_accs, fold_aucs = [], []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        thr = train[label_col].median()
        y_tr = (train[label_col] >= thr).astype(int).values
        y_te = (test[label_col] >= thr).astype(int).values
        X_tr = train[feats].values.astype(float)
        X_te = test[feats].values.astype(float)
        imp = KNNImputer(n_neighbors=5)
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        try:
            clf.fit(X_tr, y_tr)
            fold_accs.append(accuracy_score(y_te, clf.predict(X_te)))
            if len(np.unique(y_te)) > 1:
                fold_aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        except:
            pass
    return np.array(fold_accs), np.array(fold_aucs)


def boot_ci(arr, n_boot=5000, ci=95):
    if len(arr) == 0:
        return np.nan, np.nan
    lo = (100 - ci) / 2
    hi = 100 - lo
    boot = [np.random.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_boot)]
    return np.percentile(boot, [lo, hi])


np.random.seed(42)

print("\n" + "=" * 70)
print("BENCHMARKS WITHOUT BIOMARKER FEATURES (35-feature set)")
print("=" * 70)

# B0
b0 = logo_multiclass(active, available_feats, 'task')
b0_ci = boot_ci(b0)
print(f"\nB0 Task classification:  acc={b0.mean():.3f}  SD={b0.std():.3f}  "
      f"95%CI=[{b0_ci[0]:.3f},{b0_ci[1]:.3f}]  n={len(b0)}")
print(f"  per-fold: {np.round(b0, 3)}")

# B1 Valence
_, b1v = logo_binary(active, available_feats, 'vad_valence')
b1v_ci = boot_ci(b1v)
print(f"\nB1 Valence:  AUC={b1v.mean():.3f}  SD={b1v.std():.3f}  "
      f"95%CI=[{b1v_ci[0]:.3f},{b1v_ci[1]:.3f}]  n={len(b1v)}")

# B1 Arousal
_, b1a = logo_binary(active, available_feats, 'vad_arousal')
b1a_ci = boot_ci(b1a)
print(f"B1 Arousal:  AUC={b1a.mean():.3f}  SD={b1a.std():.3f}  "
      f"95%CI=[{b1a_ci[0]:.3f},{b1a_ci[1]:.3f}]  n={len(b1a)}")

# B2 Dominance
_, b2 = logo_binary(active, available_feats, 'vad_dominance')
b2_ci = boot_ci(b2)
print(f"\nB2 Dominance:  AUC={b2.mean():.3f}  SD={b2.std():.3f}  "
      f"95%CI=[{b2_ci[0]:.3f},{b2_ci[1]:.3f}]  n={len(b2)}")

# B3
_, b3m = logo_binary(active, available_feats, 'ans_mental_demand')
b3m_ci = boot_ci(b3m)
_, b3e = logo_binary(active, available_feats, 'ans_engagement')
b3e_ci = boot_ci(b3e)
print(f"\nB3 Mental demand:  AUC={b3m.mean():.3f}  SD={b3m.std():.3f}  "
      f"95%CI=[{b3m_ci[0]:.3f},{b3m_ci[1]:.3f}]  n={len(b3m)}")
print(f"B3 Engagement:    AUC={b3e.mean():.3f}  SD={b3e.std():.3f}  "
      f"95%CI=[{b3e_ci[0]:.3f},{b3e_ci[1]:.3f}]  n={len(b3e)}")

# B4
ptcpt = active.groupby(['participant_id', 'group_id'])[
    ['bfi44_e', 'bfi44_o'] + available_feats].mean().reset_index()
_, b4e = logo_binary(ptcpt, available_feats, 'bfi44_e')
b4e_ci = boot_ci(b4e)
_, b4o = logo_binary(ptcpt, available_feats, 'bfi44_o')
b4o_ci = boot_ci(b4o)
print(f"\nB4 Extraversion:  AUC={b4e.mean():.3f}  SD={b4e.std():.3f}  "
      f"95%CI=[{b4e_ci[0]:.3f},{b4e_ci[1]:.3f}]  n={len(b4e)}")
print(f"B4 Openness:      AUC={b4o.mean():.3f}  SD={b4o.std():.3f}  "
      f"95%CI=[{b4o_ci[0]:.3f},{b4o_ci[1]:.3f}]  n={len(b4o)}")

# B5
t4 = active[active['task'] == 'T4'].dropna(subset=['ans_contribution']).copy()
_, b5 = logo_binary(t4, available_feats, 'ans_contribution')
b5_ci = boot_ci(b5)
print(f"\nB5 T4 Contribution:  AUC={b5.mean():.3f}  SD={b5.std():.3f}  "
      f"95%CI=[{b5_ci[0]:.3f},{b5_ci[1]:.3f}]  n={len(b5)}")

# ---- Single-feature B3 baseline: audio_overlap_fraction_x only ----
print("\n" + "=" * 70)
print("SINGLE-FEATURE B3 BASELINE: audio_overlap_fraction_x")
print("=" * 70)
single_feat = 'audio_overlap_fraction_x'
if single_feat in active.columns:
    _, b3m_single = logo_binary(active, [single_feat], 'ans_mental_demand')
    _, b3e_single = logo_binary(active, [single_feat], 'ans_engagement')
    print(f"B3 Mental demand (single feat):  AUC={b3m_single.mean():.3f}  "
          f"SD={b3m_single.std():.3f}  n={len(b3m_single)}")
    print(f"B3 Engagement (single feat):     AUC={b3e_single.mean():.3f}  "
          f"SD={b3e_single.std():.3f}  n={len(b3e_single)}")
    print(f"\nFull-feature mental demand gain over single feat: "
          f"+{b3m.mean()-b3m_single.mean():.3f} AUC")
else:
    print(f"Feature {single_feat} not found in active columns")

# ---- Summary ----
print("\n" + "=" * 70)
print("SUMMARY TABLE (35-feature set, biomarker_* removed)")
print("=" * 70)
print(f"{'Benchmark':<30} {'Metric':<6} {'Mean':>7} {'SD':>7} "
      f"{'95%CI_lo':>9} {'95%CI_hi':>9} {'N':>5}")
print("-" * 75)
rows = [
    ('B0 Task classification',  'acc', b0,  b0_ci),
    ('B1 Valence',              'auc', b1v, b1v_ci),
    ('B1 Arousal',              'auc', b1a, b1a_ci),
    ('B2 Dominance',            'auc', b2,  b2_ci),
    ('B3 Mental demand',        'auc', b3m, b3m_ci),
    ('B3 Engagement',           'auc', b3e, b3e_ci),
    ('B4 Extraversion',         'auc', b4e, b4e_ci),
    ('B4 Openness',             'auc', b4o, b4o_ci),
    ('B5 T4 Contribution',      'auc', b5,  b5_ci),
]
for name, metric, arr, ci in rows:
    print(f"{name:<30} {metric:<6} {arr.mean():>7.3f} {arr.std():>7.3f} "
          f"{ci[0]:>9.3f} {ci[1]:>9.3f} {len(arr):>5}")
