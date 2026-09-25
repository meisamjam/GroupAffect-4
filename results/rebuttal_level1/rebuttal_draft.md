> **Archive note (2026-09-25).** The text below is the original July 2026 rebuttal draft, retained as written for provenance. Its B0 “submitted” comparator of 0.641 conflicts with the 0.734 originally published in this repository's README; their relationship is unverified. Its final paragraph describes the state at the time of drafting: this PR now updates repository documentation. Use the corrected results and caveats in the root README for the current public summary.

We thank the reviewers and AC for identifying the preprocessing concern. We reran all
Level-1 benchmarks locally under a corrected group-disjoint protocol. The submitted
within-participant normalization did not mix training participants with held-out-group
statistics. The actual issue was future-looking/transductive preprocessing within a
participant: an earlier task row could use that participant's later T1--T4 observations.
Global winsorization and feature selection were additional non-nested steps.

The corrected analysis uses a fixed, interpretable pool of T0-delta physiology/pupil
features and absolute current-task audio features. Within every LOGO fold, feature
filtering, winsorization, KNN imputation, scaling, and the binary target median are fitted
using the nine training groups only. The model is a fixed unweighted logistic regression
(`C=1`, seed 42), without tuning. We report mean held-out-group performance, across-fold
SD, a 5,000-resample fold bootstrap interval, and the number of folds with a defined
metric:

| ID | Target | Corrected result | Valid folds |
|---|---|---:|---:|
| B0 | Task label | Accuracy 0.450 (SD 0.134), 95% CI [0.375, 0.525] | 10/10 |
| B1a | Valence | AUC 0.595 (SD 0.100), 95% CI [0.529, 0.645] | 10/10 |
| B1b | Arousal | AUC 0.686 (SD 0.170), 95% CI [0.584, 0.783] | 10/10 |
| B2 | Dominance | AUC 0.656 (SD 0.265), 95% CI [0.504, 0.812] | 10/10 |
| B3a | Mental demand | AUC 0.801 (SD 0.125), 95% CI [0.718, 0.878] | 8/10 |
| B3b | Engagement | AUC 0.631 (SD 0.119), 95% CI [0.554, 0.708] | 8/10 |
| B3c | Satisfaction (T2/T3) | AUC 0.743 (SD 0.174), 95% CI [0.610, 0.848] | 7/10 |
| B3d | Mean seat-directed trust (T2/T4) | AUC 0.727 (SD 0.269), 95% CI [0.547, 0.883] | 8/10 |

The source physiology/pupil analysis table contains eight groups; audio and VAD tables
contain all ten. Consequently, B3a/B3b/B3d have eight eligible held-out groups, and B3c
has seven folds with both classes. We attempted all ten group splits and explicitly
recorded undefined folds.

Mental demand remains the clearest supported conclusion. Valence and engagement retain
more modest feasibility evidence. Arousal improves under the corrected specification,
while dominance, satisfaction, and trust have substantial fold variability and/or fewer
valid folds; we therefore treat them as tentative pilot characterization rather than
stable predictive claims. B0 weakens substantially (0.641 submitted to 0.450 corrected),
so we withdraw the stronger task-classification characterization. More generally, we
describe this suite as pilot characterization and sanity checks, not a stable
leaderboard.

This diagnostic was performed locally for the rebuttal. No submitted PDF,
supplementary material, dataset, code URL, repository, or hosted artifact was modified,
and these results are not presented as replacing the original submission.
