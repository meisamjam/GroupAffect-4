import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("results/benchmarks/preprocessed_participant_task.tsv", sep='\t')
print(f"Feature table: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"Groups: {sorted(df['group_id'].unique())}")
print(f"Tasks: {sorted(df['task'].unique())}")

with open("results/benchmarks/preprocessed_feature_list.txt") as f:
    features = [l.strip() for l in f if l.strip()]
print(f"Features ({len(features)}): {features}")

# Show all columns to find label names
print("\nAll columns:")
for c in df.columns:
    print(" ", c)

def logo_cv_multiclass(data, feats, label_col, group_col='group_id'):
    groups = sorted(data[group_col].unique())
    fold_accs = []
    for g in groups:
        test = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        train = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        if len(test) == 0 or len(train) < 5:
            continue
        available = [f for f in feats if f in data.columns]
        X_tr = train[available].values.astype(float)
        X_te = test[available].values.astype(float)
        y_tr = train[label_col].values
        y_te = test[label_col].values
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
        except Exception as e:
            print(f"  fold {g} failed: {e}")
    return np.array(fold_accs)

def logo_cv_binary(data, feats, label_col, group_col='group_id'):
    groups = sorted(data[group_col].unique())
    fold_accs, fold_aucs = [], []
    for g in groups:
        train_all = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test_all = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test_all) == 0 or len(train_all) < 5:
            continue
        thr = train_all[label_col].median()
        y_tr = (train_all[label_col] >= thr).astype(int).values
        y_te = (test_all[label_col] >= thr).astype(int).values
        available = [f for f in feats if f in data.columns]
        X_tr = train_all[available].values.astype(float)
        X_te = test_all[available].values.astype(float)
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
                proba = clf.predict_proba(X_te)[:, 1]
                fold_aucs.append(roc_auc_score(y_te, proba))
        except Exception as e:
            print(f"  fold {g} failed: {e}")
    return np.array(fold_accs), np.array(fold_aucs)

def logo_cv_regress(data, feats, label_col, group_col='group_id'):
    groups = sorted(data[group_col].unique())
    fold_maes, naive_maes = [], []
    for g in groups:
        train_all = data[data[group_col] != g].dropna(subset=[label_col]).copy()
        test_all = data[data[group_col] == g].dropna(subset=[label_col]).copy()
        if len(test_all) == 0 or len(train_all) < 5:
            continue
        y_tr = train_all[label_col].values
        y_te = test_all[label_col].values
        naive_maes.append(mean_absolute_error(y_te, np.full_like(y_te, y_tr.mean(), dtype=float)))
        available = [f for f in feats if f in data.columns]
        X_tr = train_all[available].values.astype(float)
        X_te = test_all[available].values.astype(float)
        imp = KNNImputer(n_neighbors=5)
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        reg = Ridge(alpha=10.0)
        try:
            reg.fit(X_tr, y_tr)
            preds = reg.predict(X_te)
            fold_maes.append(mean_absolute_error(y_te, preds))
        except Exception as e:
            print(f"  fold {g} failed: {e}")
    return np.array(fold_maes), np.array(naive_maes)

active = df[df['task'].isin(['T1', 'T2', 'T3', 'T4'])].copy()
print(f"\nActive rows: {len(active)}")

print("\n" + "="*70)
print("FOLD-LEVEL VARIANCE ACROSS LOGO-CV BENCHMARKS")
print("="*70)

# B0
print("\nB0: Task classification (T1-T4)")
available_feats = [f for f in features if f in active.columns]
accs = logo_cv_multiclass(active, available_feats, 'task')
print(f"  Per-fold acc: {np.round(accs, 3)}")
print(f"  Mean={accs.mean():.3f}  SD={accs.std():.3f}  n_folds={len(accs)}")

# Find label columns
label_candidates = {
    'valence': [c for c in active.columns if 'valence' in c.lower()],
    'arousal': [c for c in active.columns if 'arousal' in c.lower()],
    'dominance': [c for c in active.columns if 'dominance' in c.lower()],
    'engagement': [c for c in active.columns if 'engagement' in c.lower()],
    'mental_demand': [c for c in active.columns if 'mental_demand' in c.lower() or 'mental demand' in c.lower()],
    'extraversion': [c for c in active.columns if 'extraver' in c.lower() or 'bfi44_e' in c.lower()],
    'openness': [c for c in active.columns if 'openness' in c.lower() or 'bfi44_o' in c.lower()],
    'contribution': [c for c in active.columns if 'contribution' in c.lower()],
    'gini': [c for c in active.columns if 'gini' in c.lower() or 'speaking' in c.lower()],
    'overlap': [c for c in active.columns if 'overlap' in c.lower()],
}
print("\nLabel column candidates:")
for k, v in label_candidates.items():
    print(f"  {k}: {v}")
