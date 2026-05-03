import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("results/benchmarks/preprocessed_participant_task.tsv", sep='\t')
active = df[df['task'].isin(['T1', 'T2', 'T3', 'T4'])].copy()

with open("results/benchmarks/preprocessed_feature_list.txt") as f:
    features = [l.strip() for l in f if l.strip()]
available_feats = [f for f in features if f in active.columns]

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
            preds = clf.predict(X_te)
            fold_accs.append(accuracy_score(y_te, preds))
            if len(np.unique(y_te)) > 1:
                fold_aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
        except:
            pass
    return np.array(fold_accs), np.array(fold_aucs)

def logo_regress(data, feats, label_col, group_col='group_id'):
    groups = sorted(data[group_col].unique())
    fold_maes, naive_maes = [], []
    for g in groups:
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        y_tr = train[label_col].values
        y_te = test[label_col].values
        naive_maes.append(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean())))
        X_tr = train[feats].values.astype(float)
        X_te = test[feats].values.astype(float)
        imp = KNNImputer(n_neighbors=5)
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        reg = Ridge(alpha=10.0)
        try:
            reg.fit(X_tr, y_tr)
            fold_maes.append(mean_absolute_error(y_te, reg.predict(X_te)))
        except:
            pass
    return np.array(fold_maes), np.array(naive_maes)

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

def fmt(arr, metric='acc'):
    if len(arr) == 0:
        return "no folds"
    return f"per-fold: {np.round(arr,3)}  =>  Mean={arr.mean():.3f}  SD={arr.std():.3f}  n={len(arr)}"

print("=" * 70)
print("FOLD-LEVEL VARIANCE — ALL BENCHMARKS")
print("=" * 70)

# B0
print("\nB0: Task classification")
accs = logo_multiclass(active, available_feats, 'task')
print("  ACC:", fmt(accs))

# B1 Valence / Arousal
print("\nB1: Valence")
accs, aucs = logo_binary(active, available_feats, 'vad_valence')
print("  ACC:", fmt(accs))
print("  AUC:", fmt(aucs))

print("\nB1: Arousal")
accs, aucs = logo_binary(active, available_feats, 'vad_arousal')
print("  ACC:", fmt(accs))
print("  AUC:", fmt(aucs))

# B2 Dominance
print("\nB2: Dominance")
dom_data = active.dropna(subset=['vad_dominance'])
accs, aucs = logo_binary(dom_data, available_feats, 'vad_dominance')
print("  ACC:", fmt(accs))
print("  AUC:", fmt(aucs))

# B3 Mental demand / Engagement
print("\nB3: Mental demand")
accs, aucs = logo_binary(active, available_feats, 'ans_mental_demand')
print("  ACC:", fmt(accs))
print("  AUC:", fmt(aucs))

print("\nB3: Engagement")
accs, aucs = logo_binary(active, available_feats, 'ans_engagement')
print("  ACC:", fmt(accs))
print("  AUC:", fmt(aucs))

# B4 Personality — participant-level (use T1-only mean per participant as proxy)
print("\nB4: BFI Extraversion (participant-level)")
ptcpt = active.groupby(['participant_id','group_id'])[['bfi44_e','bfi44_o'] + available_feats].mean().reset_index()
accs_e, aucs_e = logo_binary(ptcpt, available_feats, 'bfi44_e')
print("  AUC:", fmt(aucs_e))
# Bootstrap CI for Openness
accs_o, aucs_o = logo_binary(ptcpt, available_feats, 'bfi44_o')
print("\nB4: BFI Openness")
print("  AUC:", fmt(aucs_o))
# Bootstrap CI on overall AUC
if len(aucs_o) > 0:
    n_boot = 5000
    boot = [np.random.choice(aucs_o, size=len(aucs_o), replace=True).mean() for _ in range(n_boot)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"  Bootstrap 95% CI on mean AUC: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Overlap with 0.500: {ci_lo <= 0.5 <= ci_hi}")

# B5 Contribution
print("\nB5: T4 contribution")
t4 = active[active['task'] == 'T4'].dropna(subset=['ans_contribution']).copy()
accs_b5, aucs_b5 = logo_binary(t4, available_feats, 'ans_contribution')
print("  ACC:", fmt(accs_b5))
print("  AUC:", fmt(aucs_b5))

# B6/B7 group level
print("\nB6/B7: Loading group-level data")
try:
    grp_df = pd.read_csv("results/benchmarks/preprocessed_participant_task.tsv", sep='\t')
    grp_feats_b6 = [c for c in grp_df.columns if 'gini' in c.lower()]
    grp_feats_b7 = [c for c in grp_df.columns if 'overlap' in c.lower() and 'group' in c.lower()]
    print(f"  Gini cols: {grp_feats_b6}")
    print(f"  Overlap cols: {grp_feats_b7}")
    # Check for group-level Gini column
    all_cols = [c for c in grp_df.columns if 'speaking' in c.lower() and 'gini' in c.lower()]
    print(f"  Speaking Gini cols: {all_cols}")
except Exception as e:
    print(f"  Error: {e}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY — MEAN ± SD ACROSS FOLDS")
print("=" * 70)
print(f"{'Benchmark':<30} {'Metric':<8} {'Mean':>7} {'SD':>7} {'N folds':>8}")
print("-" * 65)

b0_accs = logo_multiclass(active, available_feats, 'task')
print(f"{'B0 Task classification':<30} {'acc':<8} {b0_accs.mean():>7.3f} {b0_accs.std():>7.3f} {len(b0_accs):>8}")

_, b1v_aucs = logo_binary(active, available_feats, 'vad_valence')
print(f"{'B1 Valence':<30} {'auc':<8} {b1v_aucs.mean():>7.3f} {b1v_aucs.std():>7.3f} {len(b1v_aucs):>8}")

_, b1a_aucs = logo_binary(active, available_feats, 'vad_arousal')
print(f"{'B1 Arousal':<30} {'auc':<8} {b1a_aucs.mean():>7.3f} {b1a_aucs.std():>7.3f} {len(b1a_aucs):>8}")

_, b2_aucs = logo_binary(active, available_feats, 'vad_dominance')
print(f"{'B2 Dominance':<30} {'auc':<8} {b2_aucs.mean():>7.3f} {b2_aucs.std():>7.3f} {len(b2_aucs):>8}")

_, b3m_aucs = logo_binary(active, available_feats, 'ans_mental_demand')
print(f"{'B3 Mental demand':<30} {'auc':<8} {b3m_aucs.mean():>7.3f} {b3m_aucs.std():>7.3f} {len(b3m_aucs):>8}")

_, b3e_aucs = logo_binary(active, available_feats, 'ans_engagement')
print(f"{'B3 Engagement':<30} {'auc':<8} {b3e_aucs.mean():>7.3f} {b3e_aucs.std():>7.3f} {len(b3e_aucs):>8}")

ptcpt = active.groupby(['participant_id', 'group_id'])[['bfi44_e', 'bfi44_o'] + available_feats].mean().reset_index()
_, b4e_aucs = logo_binary(ptcpt, available_feats, 'bfi44_e')
print(f"{'B4 Extraversion':<30} {'auc':<8} {b4e_aucs.mean():>7.3f} {b4e_aucs.std():>7.3f} {len(b4e_aucs):>8}")
_, b4o_aucs = logo_binary(ptcpt, available_feats, 'bfi44_o')
print(f"{'B4 Openness':<30} {'auc':<8} {b4o_aucs.mean():>7.3f} {b4o_aucs.std():>7.3f} {len(b4o_aucs):>8}")
n_boot = 5000
boot_o = [np.random.choice(b4o_aucs, size=len(b4o_aucs), replace=True).mean() for _ in range(n_boot)]
ci_lo_o, ci_hi_o = np.percentile(boot_o, [2.5, 97.5])
print(f"  B4 Openness bootstrap 95% CI: [{ci_lo_o:.3f}, {ci_hi_o:.3f}]  (contains 0.5: {ci_lo_o <= 0.5 <= ci_hi_o})")

t4 = active[active['task'] == 'T4'].dropna(subset=['ans_contribution']).copy()
_, b5_aucs = logo_binary(t4, available_feats, 'ans_contribution')
print(f"{'B5 T4 contribution':<30} {'auc':<8} {b5_aucs.mean():>7.3f} {b5_aucs.std():>7.3f} {len(b5_aucs):>8}")

print("\nNote: B6/B7 group-level benchmarks use group-aggregated data (n=38, 10 groups); fold SD not computed here.")
