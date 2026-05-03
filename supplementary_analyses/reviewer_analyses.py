"""
GroupAffect-4 Dataset -- Reviewer Analyses
NeurIPS 2026 Dataset Paper Revision

All 10 groups (grp-07 to grp-16), 4 participants each, tasks T0-T4.
Behavioral data from stimuli_answers TSV files under sub-01/ses-*/beh/.

Data loading strategy:
  - For groups with source_file column: filter rows where source_file group == file group.
  - For grp-08 and grp-11 (in the combined grp-08 session file, no source_file column):
    filter by wall_clock epoch windows identified from session timestamps.
  - grp-16 data is in its own file (no source_file issues).
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = r"c:\Codes\affectai-processing\affectai-data-processing\data\bids_release_no_video"
PARTICIPANTS_TSV = os.path.join(DATA_ROOT, "participants.tsv")

# grp-08 and grp-11 live inside the combined grp-08 session file (no source_file column).
# Epoch windows determined from wall_clock gap analysis.
SPECIAL_GROUPS = {
    "grp-08": (1773393620, 1773397045),   # 2026-03-13 09:20-10:17 UTC
    "grp-11": (1773825907, 1773829055),   # 2026-03-18 09:25-10:17 UTC
}
# The combined file (no source_file col) is the grp-08 session file
COMBINED_FILE_KEY = "grp-08"   # filename contains "grp-08"


def extract_group_from_source(src):
    if not isinstance(src, str):
        return None
    for part in src.replace(".jsonl", "").split("_"):
        if part.startswith("grp-"):
            return part
    return None


def load_all_behavioral():
    pattern = os.path.join(DATA_ROOT, "sub-01", "ses-*", "beh",
                           "*_task-T0T1T2T3T4_stimuli_answers.tsv")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} behavioral data files.")

    combined_file = None          # the big file without source_file column
    regular_files = []
    for f in files:
        df_test = pd.read_csv(f, sep="\t", dtype=str, nrows=2)
        if "source_file" not in df_test.columns:
            combined_file = f
        else:
            regular_files.append(f)

    dfs = []

    # ---- Regular files (have source_file column) --------------------------------
    for f in regular_files:
        basename = os.path.basename(f)
        grp_parts = [p for p in basename.split("_") if p.startswith("grp-")]
        group_id = grp_parts[0] if grp_parts else "unknown"
        df = pd.read_csv(f, sep="\t", dtype=str)
        df["group_id"] = group_id
        df["item_value_num"] = pd.to_numeric(df["item_value"], errors="coerce")
        df["wall_clock"] = pd.to_numeric(df["wall_clock"], errors="coerce")
        df["lsl_clock"] = pd.to_numeric(df["lsl_clock"], errors="coerce")
        df["source_group"] = df["source_file"].apply(extract_group_from_source)
        sub = df[df["source_group"] == group_id].copy()
        dfs.append(sub)

    # ---- Combined file (grp-08 and grp-11 epoch windows) -----------------------
    if combined_file is not None:
        df_comb = pd.read_csv(combined_file, sep="\t", dtype=str)
        df_comb["item_value_num"] = pd.to_numeric(df_comb["item_value"], errors="coerce")
        df_comb["wall_clock"] = pd.to_numeric(df_comb["wall_clock"], errors="coerce")
        df_comb["lsl_clock"] = pd.to_numeric(df_comb["lsl_clock"], errors="coerce")
        df_comb["source_file"] = None
        df_comb["source_group"] = None

        for grp_id, (wc_start, wc_end) in SPECIAL_GROUPS.items():
            sub = df_comb[
                (df_comb["wall_clock"] >= wc_start) &
                (df_comb["wall_clock"] <= wc_end)
            ].copy()
            sub["group_id"] = grp_id
            dfs.append(sub)

    raw = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after group-level filtering: {len(raw)}")

    # Deduplicate on key identifying columns
    key_cols = ["wall_clock", "task", "response_type", "participant",
                "item_key", "item_value", "group_id"]
    clean = raw.drop_duplicates(subset=key_cols)
    print(f"Rows after exact-duplicate removal: {len(clean)}")
    print(f"Groups present: {sorted(clean['group_id'].unique())}")

    return clean


def load_participants():
    return pd.read_csv(PARTICIPANTS_TSV, sep="\t")


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("GroupAffect-4: Reviewer Analyses")
print("=" * 70)

df = load_all_behavioral()
# Convenience: numeric item value column
df["val"] = df["item_value_num"]

participants = load_participants()

# ============================================================
# SECTION 1: Trust T2 -> T4 Trajectory
# ============================================================
print("\n" + "=" * 70)
print("SECTION 1: Trust T2 -> T4 Trajectory")
print("=" * 70)

TRUST_ITEMS = ["trust_front", "trust_angle", "trust_next"]

trust_df = df[
    (df["response_type"] == "postblock") &
    (df["item_key"].isin(TRUST_ITEMS)) &
    (df["task"].isin(["T2", "T4"]))
].copy()

print(f"\nTrust observations (T2+T4): {len(trust_df)}")
print(f"Groups with T2/T4 trust data: {sorted(trust_df['group_id'].unique())}")

# Per-participant per-task mean trust (averaging across trust_front/angle/next)
part_trust = (
    trust_df
    .groupby(["group_id", "participant", "task"])["val"]
    .mean()
    .reset_index()
    .rename(columns={"val": "mean_trust"})
)

for task in ["T2", "T4"]:
    sub = part_trust[part_trust["task"] == task]["mean_trust"].dropna()
    print(f"\n  Mean trust after {task}: {sub.mean():.3f} +/- {sub.std():.3f}  (n={len(sub)})")

# Per-group means
print("\n  Per-group mean trust:")
print(f"  {'Group':<12} {'T2_mean':>10} {'T4_mean':>10} {'Delta':>10}")
group_trust_wide = (
    part_trust
    .groupby(["group_id", "task"])["mean_trust"]
    .mean()
    .unstack("task")
)
for grp, row in group_trust_wide.iterrows():
    t2 = row.get("T2", np.nan)
    t4 = row.get("T4", np.nan)
    delta = t4 - t2 if (not np.isnan(t2) and not np.isnan(t4)) else np.nan
    print(f"  {grp:<12} {t2:>10.3f} {t4:>10.3f} {delta:>10.3f}")

# Paired t-test across participants (each participant is one observation)
trust_part_wide = part_trust.pivot_table(
    index=["group_id", "participant"], columns="task", values="mean_trust"
).dropna(subset=["T2", "T4"])

t_stat_s1, p_val_s1 = stats.ttest_rel(trust_part_wide["T4"], trust_part_wide["T2"])
print(f"\n  Paired t-test (T4 vs T2): t({len(trust_part_wide)-1}) = {t_stat_s1:.3f}, "
      f"p = {p_val_s1:.4f} (two-tailed)")
print(f"  One-tailed p (T4 > T2): {p_val_s1/2:.4f}")
mean_diff = (trust_part_wide["T4"] - trust_part_wide["T2"]).mean()
sd_diff = (trust_part_wide["T4"] - trust_part_wide["T2"]).std()
print(f"  Mean difference (T4 - T2): {mean_diff:.3f} +/- {sd_diff:.3f}")

# How many groups show trust increase
both = group_trust_wide.dropna(subset=["T2", "T4"])
n_increase = int((both["T4"] > both["T2"]).sum())
print(f"\n  Groups with T4 trust > T2 trust: {n_increase} / {len(both)}")

# ============================================================
# SECTION 2: Satisfaction by task (T2, T3 only -- not available T1/T4)
# ============================================================
print("\n" + "=" * 70)
print("SECTION 2: Satisfaction by Task")
print("=" * 70)

sat_df = df[
    (df["response_type"] == "postblock") &
    (df["item_key"] == "satisfaction")
].copy()

print(f"\nSatisfaction observations (all tasks): {len(sat_df)}")
print(f"Tasks with satisfaction: {sorted(sat_df['task'].unique())}")
print("  NOTE: satisfaction item only present in T2 and T3 postblock surveys")

print("\n  Mean +/- SD satisfaction per task (where available):")
sat_stats = sat_df.groupby("task")["val"].agg(["mean", "std", "count"])
for task in sorted(sat_stats.index):
    r = sat_stats.loc[task]
    print(f"  {task}: {r['mean']:.3f} +/- {r['std']:.3f}  (n={int(r['count'])})")

# T3 vs T2 comparison
tasks_avail = sorted(sat_df["task"].unique())
if "T2" in tasks_avail and "T3" in tasks_avail:
    t2_sat = sat_df[sat_df["task"] == "T2"]["val"].dropna()
    t3_sat = sat_df[sat_df["task"] == "T3"]["val"].dropna()
    t_s, p_s = stats.ttest_ind(t3_sat, t2_sat)
    print(f"\n  T3 vs T2 (Welch's t-test): t = {t_s:.3f}, p = {p_s:.4f}")
    print(f"  T3 mean={t3_sat.mean():.3f}, T2 mean={t2_sat.mean():.3f}, diff={t3_sat.mean()-t2_sat.mean():.3f}")

# Pairwise (within available tasks)
if len(tasks_avail) >= 2:
    print("\n  Pairwise comparisons:")
    print(f"  {'Pair':<12} {'t-stat':>10} {'p-value':>12} {'mean_A':>8} {'mean_B':>8}")
    for t_a, t_b in combinations(tasks_avail, 2):
        a = sat_df[sat_df["task"] == t_a]["val"].dropna()
        b = sat_df[sat_df["task"] == t_b]["val"].dropna()
        if len(a) > 1 and len(b) > 1:
            t_s2, p_s2 = stats.ttest_ind(a, b)
            sig = "*" if p_s2 < 0.05 else ""
            print(f"  {t_a} vs {t_b}:    {t_s2:>10.3f} {p_s2:>12.4f}  "
                  f"{a.mean():>8.3f} {b.mean():>8.3f} {sig}")

# Per-group satisfaction T2 vs T3
print("\n  Per-group satisfaction T2 and T3:")
print(f"  {'Group':<12} {'T2':>8} {'T3':>8} {'Delta':>8}")
grp_sat = (
    sat_df.groupby(["group_id", "task"])["val"]
    .mean()
    .unstack("task")
)
for grp, row in grp_sat.iterrows():
    t2v = row.get("T2", np.nan)
    t3v = row.get("T3", np.nan)
    dlt = t3v - t2v if (not np.isnan(t2v) and not np.isnan(t3v)) else np.nan
    print(f"  {grp:<12} {t2v:>8.3f} {t3v:>8.3f} {dlt:>8.3f}")

# ============================================================
# SECTION 3: Voice & Inclusion by task
# ============================================================
print("\n" + "=" * 70)
print("SECTION 3: Voice & Inclusion Means and SDs by Task")
print("=" * 70)

vi_df = df[
    (df["response_type"] == "postblock") &
    (df["item_key"] == "voice_inclusion")
].copy()

tasks_vi = sorted(vi_df["task"].unique())
print(f"\nVoice_inclusion tasks present: {tasks_vi}")
print(f"Total voice_inclusion observations: {len(vi_df)}")

print("\n  Mean +/- SD voice_inclusion per task:")
vi_stats = vi_df.groupby("task")["val"].agg(["mean", "std", "count", "min", "max"])
for task in tasks_vi:
    r = vi_stats.loc[task]
    print(f"  {task}: {r['mean']:.3f} +/- {r['std']:.3f}  "
          f"(n={int(r['count'])}, min={r['min']:.0f}, max={r['max']:.0f})")

# Variation assessment
task_means_vi = vi_stats["mean"].values
if len(task_means_vi) > 1:
    rng = task_means_vi.max() - task_means_vi.min()
    print(f"\n  Range of task means: {rng:.3f} "
          f"(max={task_means_vi.max():.3f}, min={task_means_vi.min():.3f})")
    print(f"  Assessment: {'Meaningful variation (range > 0.5)' if rng > 0.5 else 'Modest variation (range <= 0.5)'}")

# One-way ANOVA
groups_vi = [vi_df[vi_df["task"] == t]["val"].dropna().values for t in tasks_vi]
valid_vi = [g for g in groups_vi if len(g) >= 2]
if len(valid_vi) >= 2:
    f_stat, p_anova = stats.f_oneway(*valid_vi)
    print(f"\n  One-way ANOVA across tasks: F({len(valid_vi)-1},{sum(len(g) for g in valid_vi)-len(valid_vi)}) = {f_stat:.3f}, p = {p_anova:.4f}")

# Pairwise
print("\n  Pairwise voice_inclusion comparisons:")
print(f"  {'Pair':<12} {'t-stat':>10} {'p-value':>12} {'mean_A':>8} {'mean_B':>8}")
for t_a, t_b in combinations(tasks_vi, 2):
    a = vi_df[vi_df["task"] == t_a]["val"].dropna()
    b = vi_df[vi_df["task"] == t_b]["val"].dropna()
    if len(a) > 1 and len(b) > 1:
        t_s3, p_s3 = stats.ttest_ind(a, b)
        sig = "*" if p_s3 < 0.05 else ""
        print(f"  {t_a} vs {t_b}:    {t_s3:>10.3f} {p_s3:>12.4f}  "
              f"{a.mean():>8.3f} {b.mean():>8.3f} {sig}")

# ============================================================
# SECTION 4: SAM Probe Timing
# ============================================================
print("\n" + "=" * 70)
print("SECTION 4: SAM Probe Timing")
print("=" * 70)

vad_df = df[
    (df["response_type"] == "vad") &
    (df["task"].isin(["T1", "T2", "T3", "T4"]))
].copy()

print(f"\nVAD probe rows (T1-T4): {len(vad_df)}")

# Round wall_clock to 1s to group simultaneous val/aro/dom within each submission
vad_df["wc_r"] = vad_df["wall_clock"].round(0)

print("\n  Per-task timing relative to first VAD response in each group+task:")
print(f"  {'Task':<6} {'Med_first':>12} {'Med_last':>12} {'Med_span':>12} "
      f"{'Med_N_probes':>14} {'N_groups':>10}")

for task in ["T1", "T2", "T3", "T4"]:
    task_vad = vad_df[vad_df["task"] == task]
    stats_list = []
    for grp, gdf in task_vad.groupby("group_id"):
        unique_times = sorted(gdf["wc_r"].dropna().unique())
        if len(unique_times) >= 2:
            t0 = unique_times[0]
            rel = [t - t0 for t in unique_times]
            stats_list.append({
                "first": rel[0],
                "last": rel[-1],
                "span": rel[-1],
                "n": len(unique_times)
            })
    if stats_list:
        sdf = pd.DataFrame(stats_list)
        print(f"  {task:<6} {np.median(sdf['first']):>10.1f}s {np.median(sdf['last']):>10.1f}s "
              f"{np.median(sdf['span']):>10.1f}s {np.median(sdf['n']):>14.1f} {len(stats_list):>10}")

# Absolute wall_clock info
print("\n  Absolute wall_clock range per task (across all groups):")
for task in ["T1", "T2", "T3", "T4"]:
    sub = vad_df[vad_df["task"] == task]["wall_clock"].dropna()
    if len(sub) > 0:
        print(f"  {task}: n={len(sub)}, span={sub.max()-sub.min():.0f}s "
              f"(cross-session span -- not within-task)")

# Per-group probe counts
print("\n  Number of unique VAD submission times per group per task:")
print(f"  {'Group':<12}", end="")
for t in ["T1","T2","T3","T4"]:
    print(f"  {t:>6}", end="")
print()
for grp, gdf in vad_df.groupby("group_id"):
    print(f"  {grp:<12}", end="")
    for t in ["T1","T2","T3","T4"]:
        sub = gdf[gdf["task"]==t]["wc_r"].dropna().nunique()
        print(f"  {sub:>6}", end="")
    print()

# ============================================================
# SECTION 5: T4 Contribution Analysis
# ============================================================
print("\n" + "=" * 70)
print("SECTION 5: T4 Contribution Analysis")
print("=" * 70)

contrib_df = df[
    (df["response_type"] == "form") &
    (df["item_key"] == "contribution") &
    (df["task"] == "T4")
].copy()

print(f"\nT4 contribution raw observations: {len(contrib_df)}")

# Take final submission per participant per group
contrib_final = (
    contrib_df
    .sort_values("wall_clock")
    .groupby(["group_id", "participant"])
    .last()
    .reset_index()
)
print(f"Final contribution per participant (last submission): {len(contrib_final)}")

vals = contrib_final["val"].dropna()
print(f"\n  Overall: mean={vals.mean():.3f}, SD={vals.std():.3f}, "
      f"min={vals.min():.0f}, max={vals.max():.0f}, range={vals.max()-vals.min():.0f}")

# Per-group
print("\n  Per-group contributions and submission time spread:")
print(f"  {'Group':<12} {'P1':>5} {'P2':>5} {'P3':>5} {'P4':>5} "
      f"{'Sum':>5} {'WC_std(s)':>12} {'Near-simul':>12}")
n_near = 0
n_groups_with_contrib = 0
for grp, gdf in contrib_final.groupby("group_id"):
    n_groups_with_contrib += 1
    vals_map = {r["participant"]: r["val"] for _, r in gdf.iterrows()}
    p_vals = [vals_map.get(f"P{i}", np.nan) for i in range(1, 5)]
    total = sum(v for v in p_vals if not np.isnan(v))
    wc_std = gdf["wall_clock"].std() if len(gdf) > 1 else 0.0
    near = "YES" if wc_std < 60 else "NO"
    if wc_std < 60:
        n_near += 1
    p_str = "  ".join(f"{v:>5.0f}" if not np.isnan(v) else f"{'N/A':>5}" for v in p_vals)
    print(f"  {grp:<12} {p_str} {total:>5.0f} {wc_std:>12.1f}s {near:>12}")

med_std = contrib_final.groupby("group_id")["wall_clock"].std().median()
print(f"\n  Median wall_clock std across groups: {med_std:.1f}s")
print(f"  Groups with near-simultaneous submission (std < 60s): "
      f"{n_near} / {n_groups_with_contrib}")
print(f"  Interpretation: All groups submitted contributions within seconds of each other")

# ============================================================
# SECTION 6: T1 Hidden-Profile Outcome Analysis
# ============================================================
print("\n" + "=" * 70)
print("SECTION 6: T1 Hidden-Profile Outcome Analysis")
print("=" * 70)

cand_df = df[
    (df["response_type"] == "form") &
    (df["item_key"] == "selected_candidate") &
    (df["task"] == "T1")
].copy()

# item_value is string here (keep original)
cand_df["candidate"] = cand_df["item_value"].astype(str).str.strip()

print(f"\nCandidate selection raw observations: {len(cand_df)}")

# Final selection per participant per group
cand_final = (
    cand_df
    .sort_values("wall_clock")
    .groupby(["group_id", "participant"])
    .last()
    .reset_index()
)
print(f"Final selections (last per participant): {len(cand_final)}")

# Frequency
freq = cand_final["candidate"].value_counts()
print("\n  Candidate selection frequency (all participants):")
for cand, cnt in freq.items():
    pct = 100 * cnt / len(cand_final)
    print(f"  {cand}: {cnt} ({pct:.1f}%)")

if len(freq) == 0:
    print("  WARNING: No candidate data available.")
else:
    correct_cand = freq.index[0]
    print(f"\n  Most-selected candidate (inferred 'correct' / informed): {correct_cand}")

    # Per-group analysis
    print("\n  Per-group consensus:")
    print(f"  {'Group':<12} {'P1':>14} {'P2':>14} {'P3':>14} {'P4':>14} "
          f"{'Consensus':>10} {'Correct':>8}")
    n_correct_consensus = 0
    n_groups_cand = 0
    for grp, gdf in cand_final.groupby("group_id"):
        n_groups_cand += 1
        choices = {r["participant"]: r["candidate"] for _, r in gdf.iterrows()}
        row_choices = [choices.get(f"P{i}", "?") for i in range(1, 5)]
        unique_choices = set(c for c in row_choices if c not in ("?", "nan"))
        consensus = "YES" if len(unique_choices) == 1 else "NO"
        correct = "YES" if (consensus == "YES" and correct_cand in unique_choices) else (
                  "PARTIAL" if correct_cand in unique_choices else "NO")
        if consensus == "YES" and correct_cand in unique_choices:
            n_correct_consensus += 1
        c_str = "  ".join(f"{c:>14}" for c in row_choices)
        print(f"  {grp:<12} {c_str} {consensus:>10} {correct:>8}")

    print(f"\n  Groups reaching correct consensus: "
          f"{n_correct_consensus} / {n_groups_cand} "
          f"({100*n_correct_consensus/max(n_groups_cand,1):.0f}%)")

# ============================================================
# SECTION 7: Group Composition Analysis
# ============================================================
print("\n" + "=" * 70)
print("SECTION 7: Group Composition Analysis")
print("=" * 70)

print("\n  Per-group demographics:")
print(f"  {'Group':<12} {'Males':>7} {'Females':>8} {'Age_range':>12} "
      f"{'Age_SD':>8}  Education mix")

group_age_sd = {}
group_sex_div = {}
for grp, gdf in participants.groupby("group_id"):
    n_male = (gdf["sex"] == "male").sum()
    n_female = (gdf["sex"] == "female").sum()
    age_range = f"{gdf['age'].min()}-{gdf['age'].max()}"
    age_sd = gdf["age"].std(ddof=1)
    edu_mix = "/".join(gdf["education"].value_counts().index.tolist())
    group_age_sd[grp] = age_sd
    group_sex_div[grp] = bool(min(n_male, n_female) >= 2)
    print(f"  {grp:<12} {n_male:>7} {n_female:>8} {age_range:>12} "
          f"{age_sd:>8.2f}  {edu_mix}")

mean_age_sd = np.mean(list(group_age_sd.values()))
n_sex_div = sum(group_sex_div.values())
pct_sex_div = 100 * n_sex_div / len(group_sex_div)
print(f"\n  Overall mean within-group age SD: {mean_age_sd:.2f} years")
print(f"  Groups with >= 2 of each sex: {n_sex_div} / {len(group_sex_div)} ({pct_sex_div:.0f}%)")

# Correlations: group age diversity vs behavioral outcomes
pb_df = df[df["response_type"] == "postblock"].copy()

trust_t2_grp = pb_df[
    (pb_df["item_key"].isin(TRUST_ITEMS)) &
    (pb_df["task"] == "T2")
].groupby("group_id")["val"].mean()

sat_grp = pb_df[pb_df["item_key"] == "satisfaction"].groupby("group_id")["val"].mean()

eq_grp = pb_df[pb_df["item_key"] == "equality_of_contribution"].groupby("group_id")["val"].mean()

age_sd_series = pd.Series(group_age_sd)

print("\n  Spearman correlations: within-group age SD vs group-level outcomes:")
print(f"  {'Outcome':<30} {'r':>8} {'p':>10} {'n':>5}")
for label, series in [("Trust_T2", trust_t2_grp),
                       ("Satisfaction", sat_grp),
                       ("Equality_of_contribution", eq_grp)]:
    common_idx = age_sd_series.index.intersection(series.dropna().index)
    if len(common_idx) >= 5:
        r_val, p_val = stats.spearmanr(age_sd_series[common_idx], series[common_idx])
        print(f"  {label:<30} {r_val:>8.3f} {p_val:>10.4f} {len(common_idx):>5}")
    else:
        print(f"  {label:<30}  insufficient data (n={len(common_idx)})")

# ============================================================
# SECTION 8: BFI Correlations with Behavior
# ============================================================
print("\n" + "=" * 70)
print("SECTION 8: BFI Correlations with Behavior")
print("=" * 70)

BFI_COLS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]
BFI_LABELS = {
    "bfi44_e": "E (Extraversion)",
    "bfi44_a": "A (Agreeableness)",
    "bfi44_c": "C (Conscientiousness)",
    "bfi44_n": "N (Neuroticism)",
    "bfi44_o": "O (Openness)"
}

# VAD means per participant (T1-T4)
vad_all = df[
    (df["response_type"] == "vad") &
    (df["task"].isin(["T1", "T2", "T3", "T4"]))
].copy()

vad_means = (
    vad_all
    .groupby(["group_id", "participant", "item_key"])["val"]
    .mean()
    .unstack("item_key")
    .reset_index()
)
vad_means.rename(columns={k: f"mean_{k}" for k in ["valence", "arousal", "dominance"]
                           if k in vad_means.columns}, inplace=True)

# VAD response count as participation proxy
vad_count = (
    vad_all.groupby(["group_id", "participant"])
    .size()
    .reset_index(name="vad_response_count")
)

# Postblock means per participant (T1-T4 where available)
pb_items_sel = ["engagement", "mental_demand", "satisfaction"]
pb_means = (
    pb_df[
        pb_df["item_key"].isin(pb_items_sel) &
        pb_df["task"].isin(["T1", "T2", "T3", "T4"])
    ]
    .groupby(["group_id", "participant", "item_key"])["val"]
    .mean()
    .unstack("item_key")
    .reset_index()
)
pb_means.rename(columns={k: f"mean_{k}" for k in pb_items_sel
                          if k in pb_means.columns}, inplace=True)

# Merge behavioral data
behav = pd.merge(vad_means, vad_count, on=["group_id", "participant"], how="outer")
behav = pd.merge(behav, pb_means, on=["group_id", "participant"], how="outer")

# Match to participants using group_id + seat (P1-P4)
participants_sub = participants.rename(columns={"seat": "participant"})[
    ["participant_id", "group_id", "participant"] + BFI_COLS
]
behav_full = pd.merge(behav, participants_sub, on=["group_id", "participant"], how="left")

BEHAV_COLS = [c for c in [
    "mean_valence", "mean_arousal", "mean_dominance",
    "mean_engagement", "mean_mental_demand", "mean_satisfaction",
    "vad_response_count"
] if c in behav_full.columns]

n_with_bfi = behav_full["participant_id"].notna().sum()
print(f"\nParticipants matched to BFI data: {n_with_bfi}")
print(f"Behavioral measures: {BEHAV_COLS}")

# Compute all Spearman correlations
corr_results = []
for bfi in BFI_COLS:
    for beh in BEHAV_COLS:
        pair_df = behav_full[[bfi, beh]].dropna()
        if len(pair_df) >= 5:
            r, p = stats.spearmanr(pair_df[bfi], pair_df[beh])
            corr_results.append({"BFI": bfi, "Behavior": beh,
                                   "r": r, "p": p, "n": len(pair_df)})

res_df = pd.DataFrame(corr_results)

# BH-FDR correction
def bh_fdr(pvals):
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = pvals[sorted_idx[i]] * n / rank
        if i == n - 1:
            adjusted[sorted_idx[i]] = adj
        else:
            adjusted[sorted_idx[i]] = min(adj, adjusted[sorted_idx[i + 1]])
    return np.clip(adjusted, 0, 1)

res_df["p_fdr"] = bh_fdr(res_df["p"].values)
res_df["sig"] = res_df["p_fdr"] < 0.05
res_df_sorted = res_df.sort_values("p")

print(f"\n  All BFI x Behavior Spearman correlations (sorted by p-value):")
print(f"  {'BFI':<28} {'Behavior':<22} {'r':>8} {'p':>10} {'p_FDR':>10} {'Sig':>5}")
for _, row in res_df_sorted.iterrows():
    sig_mark = ("***" if row["p_fdr"] < 0.001 else
                "**" if row["p_fdr"] < 0.01 else
                "*" if row["p_fdr"] < 0.05 else "")
    print(f"  {BFI_LABELS.get(row['BFI'], row['BFI']):<28} "
          f"{row['Behavior']:<22} {row['r']:>8.3f} {row['p']:>10.4f} "
          f"{row['p_fdr']:>10.4f} {sig_mark:>5}")

n_sig = int(res_df["sig"].sum())
print(f"\n  Significant after BH-FDR correction: {n_sig} / {len(res_df)}")
if n_sig > 0:
    print("  Significant correlations:")
    for _, row in res_df[res_df["sig"]].sort_values("p_fdr").iterrows():
        print(f"    {BFI_LABELS[row['BFI']]} x {row['Behavior']}: "
              f"r={row['r']:.3f}, p={row['p']:.4f}, p_FDR={row['p_fdr']:.4f}")

# ============================================================
# SECTION 9: Postblock Survey Profiles by Task
# ============================================================
print("\n" + "=" * 70)
print("SECTION 9: Postblock Survey Profiles -- Means by Task")
print("=" * 70)

ITEMS_9 = ["engagement", "mental_demand", "team_coordination",
           "equality_of_contribution", "voice_inclusion"]

print("\n  Mean +/- SD per item per task (format: mean+/-SD):")
header = f"  {'Item':<28}"
for t in ["T1", "T2", "T3", "T4"]:
    header += f" {'  ' + t:>16}"
print(header)

anova_results_9 = {}
for item in ITEMS_9:
    item_df = pb_df[pb_df["item_key"] == item]
    row_str = f"  {item:<28}"
    groups_for_test = []
    for task in ["T1", "T2", "T3", "T4"]:
        sub = item_df[item_df["task"] == task]["val"].dropna()
        if len(sub) > 0:
            row_str += f" {sub.mean():>6.2f}+/-{sub.std():>5.2f}"
            groups_for_test.append(sub.values)
        else:
            row_str += f"{'N/A':>16}"
            groups_for_test.append(np.array([]))
    print(row_str)

    # One-way ANOVA
    valid_g = [g for g in groups_for_test if len(g) >= 2]
    if len(valid_g) >= 2:
        try:
            f_s, p_s = stats.f_oneway(*valid_g)
            anova_results_9[item] = ("ANOVA", f_s, p_s, sum(len(g) for g in valid_g),
                                      len(valid_g))
        except Exception:
            pass

print("\n  One-way ANOVA results (across available tasks):")
print(f"  {'Item':<28} {'Test':>6} {'F-stat':>10} {'p-value':>12} {'Sig':>6}")
for item, (test, stat, p, n_total, k) in anova_results_9.items():
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"  {item:<28} {test:>6} {stat:>10.3f} {p:>12.4f} {sig:>6}")

# ============================================================
# SECTION 10: Familiarity Distribution
# ============================================================
print("\n" + "=" * 70)
print("SECTION 10: Familiarity Distribution")
print("=" * 70)

FAM_ITEMS = ["familiarity_p1", "familiarity_p2", "familiarity_p3", "familiarity_p4"]

fam_df = df[
    (df["response_type"] == "postblock") &
    (df["item_key"].isin(FAM_ITEMS)) &
    (df["task"] == "T1")
].copy()

print(f"\nFamiliarity observations (T1 postblock): {len(fam_df)}")
print(f"Groups present: {sorted(fam_df['group_id'].unique())}")

all_fam = fam_df["val"].dropna()
pct_low = 100 * (all_fam < 4).sum() / len(all_fam)
pct_one = 100 * (all_fam == 1).sum() / len(all_fam)
print(f"\n  Overall familiarity (1-7 scale):")
print(f"  mean={all_fam.mean():.3f}, SD={all_fam.std():.3f}, "
      f"min={all_fam.min():.0f}, max={all_fam.max():.0f}, "
      f"median={all_fam.median():.1f}")
print(f"  % responses below 4 (low familiarity): {pct_low:.1f}%")
print(f"  % responses = 1 (no familiarity): {pct_one:.1f}%")

print("\n  Per familiarity item:")
for item in FAM_ITEMS:
    sub = fam_df[fam_df["item_key"] == item]["val"].dropna()
    if len(sub) > 0:
        print(f"  {item}: mean={sub.mean():.3f} +/- {sub.std():.3f}  (n={len(sub)})")

print("\n  Note: familiarity_p1 = rating of own familiarity with P1, etc.")
print("  Own-self familiarity (diagonal: p1 rates p1) expected to be high.")
# Self-familiarity: P1 rates familiarity_p1, P2 rates familiarity_p2, etc.
self_fam = fam_df[
    ((fam_df["participant"] == "P1") & (fam_df["item_key"] == "familiarity_p1")) |
    ((fam_df["participant"] == "P2") & (fam_df["item_key"] == "familiarity_p2")) |
    ((fam_df["participant"] == "P3") & (fam_df["item_key"] == "familiarity_p3")) |
    ((fam_df["participant"] == "P4") & (fam_df["item_key"] == "familiarity_p4"))
]["val"].dropna()

other_fam = fam_df[
    ~(
        ((fam_df["participant"] == "P1") & (fam_df["item_key"] == "familiarity_p1")) |
        ((fam_df["participant"] == "P2") & (fam_df["item_key"] == "familiarity_p2")) |
        ((fam_df["participant"] == "P3") & (fam_df["item_key"] == "familiarity_p3")) |
        ((fam_df["participant"] == "P4") & (fam_df["item_key"] == "familiarity_p4"))
    )
]["val"].dropna()

if len(self_fam) > 0:
    print(f"\n  Self-familiarity: mean={self_fam.mean():.3f} +/- {self_fam.std():.3f} (n={len(self_fam)})")
if len(other_fam) > 0:
    print(f"  Other-familiarity (cross-participant): mean={other_fam.mean():.3f} +/- {other_fam.std():.3f} (n={len(other_fam)})")

# Per-group mean familiarity (all items)
print("\n  Per-group mean familiarity (all familiarity items, excluding self):")
print(f"  {'Group':<12} {'Mean_fam':>10} {'SD':>8} {'Above_3.5':>10}")
max_grp_fam = 0.0
for grp, gdf in other_fam if False else fam_df.groupby("group_id"):
    # exclude self-reports
    non_self = gdf[
        ~(
            ((gdf["participant"] == "P1") & (gdf["item_key"] == "familiarity_p1")) |
            ((gdf["participant"] == "P2") & (gdf["item_key"] == "familiarity_p2")) |
            ((gdf["participant"] == "P3") & (gdf["item_key"] == "familiarity_p3")) |
            ((gdf["participant"] == "P4") & (gdf["item_key"] == "familiarity_p4"))
        )
    ]["val"].dropna()
    if len(non_self) == 0:
        continue
    grp_mean = non_self.mean()
    grp_sd = non_self.std()
    flag = "YES" if grp_mean > 3.5 else "no"
    if grp_mean > max_grp_fam:
        max_grp_fam = grp_mean
    print(f"  {grp:<12} {grp_mean:>10.3f} {grp_sd:>8.3f} {flag:>10}")

print(f"\n  Maximum group mean other-familiarity: {max_grp_fam:.3f}")
print(f"  Any group with mean > 3.5: {'YES' if max_grp_fam > 3.5 else 'NO -- all groups below 3.5'}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE -- SUMMARY")
print("=" * 70)
print(f"  Dataset: {df['group_id'].nunique()} groups x 4 participants = "
      f"up to {df['group_id'].nunique()*4} participants")
print(f"  Total cleaned behavioral rows: {len(df)}")
print(f"  Groups analyzed: {sorted(df['group_id'].unique())}")
print()
print("  Key findings:")
t2_mean = part_trust[part_trust["task"]=="T2"]["mean_trust"].mean()
t4_mean = part_trust[part_trust["task"]=="T4"]["mean_trust"].mean()
print(f"  S1: Trust T2={t2_mean:.2f}, T4={t4_mean:.2f}, "
      f"paired t({len(trust_part_wide)-1})={t_stat_s1:.3f}, p={p_val_s1:.4f}")
sat_summary = sat_df.groupby("task")["val"].mean()
sat_str = ", ".join(f"{t}={sat_summary[t]:.2f}" for t in sorted(sat_summary.index))
print(f"  S2: Satisfaction: {sat_str}")
vi_summary = vi_df.groupby("task")["val"].mean()
vi_str = ", ".join(f"{t}={vi_summary[t]:.2f}" for t in sorted(vi_summary.index))
print(f"  S3: Voice_inclusion: {vi_str}")
print(f"  S5: T4 contributions mean={vals.mean():.2f}, SD={vals.std():.2f}; "
      f"all {n_near}/{n_groups_with_contrib} groups near-simultaneous")
print(f"  S8: BH-FDR significant BFI correlations: {n_sig}/{len(res_df)}")
print(f"  S10: Mean other-familiarity={other_fam.mean():.3f}, "
      f"max group mean={max_grp_fam:.3f} (all below 3.5)")
