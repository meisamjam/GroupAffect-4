"""Run leakage-free Level-1 rebuttal benchmarks from raw participant-task tables.

All data-estimated preprocessing is fitted separately inside each leave-one-group-out
training fold. Outputs include summary, fold, and row-level prediction TSV files.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
SEED = 42
ACTIVE_TASKS = ("T1", "T2", "T3", "T4")
PHYSIO_PUPIL_DELTA = [
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_mean_delta_t0",
    "eda_phasic_rate_hz_delta_t0",
    "temp_mean_delta_t0",
    "pupil_mean_delta_t0",
]
AUDIO_ABS_FEATURES = [
    "audio_speaking_fraction",
    "audio_overlap_fraction",
    "audio_energy_mean",
    "audio_energy_sd",
    "audio_pitch_mean",
    "audio_pitch_sd",
    "audio_hnr_mean",
    "audio_jitter_mean",
    "audio_shimmer_mean",
    "audio_voiced_segments_per_sec",
    "audio_mean_voiced_segment_s",
    "audio_speech_rate_proxy",
]
CANDIDATE_FEATURES = PHYSIO_PUPIL_DELTA + AUDIO_ABS_FEATURES


@dataclass(frozen=True)
class Benchmark:
    benchmark: str
    target: str
    tasks: tuple[str, ...]
    multiclass: bool = False


BENCHMARKS = (
    Benchmark("B0", "task", ACTIVE_TASKS, multiclass=True),
    Benchmark("B1a", "vad_valence", ACTIVE_TASKS),
    Benchmark("B1b", "vad_arousal", ACTIVE_TASKS),
    Benchmark("B2", "vad_dominance", ACTIVE_TASKS),
    Benchmark("B3a", "ans_mental_demand", ACTIVE_TASKS),
    Benchmark("B3b", "ans_engagement", ACTIVE_TASKS),
    Benchmark("B3c", "ans_satisfaction", ("T2", "T3")),
    Benchmark("B3d", "trust_mean", ("T2", "T4")),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-pt",
        type=Path,
        default=Path("results/statistics/analysis_dataset_participant_task.tsv"),
    )
    parser.add_argument(
        "--audio-task",
        type=Path,
        default=Path("results/audio/individual_audio_task.tsv"),
    )
    parser.add_argument(
        "--participants",
        type=Path,
        default=Path("data/bids_release_no_video/participants.tsv"),
    )
    parser.add_argument(
        "--vad-task",
        type=Path,
        default=Path("results/task_responses/vad_participant_task.tsv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/rebuttal_level1"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_tsv(path: Path) -> pd.DataFrame:
    """Load a required TSV input."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", low_memory=False)


def assemble_table(
    stats: pd.DataFrame,
    audio: pd.DataFrame,
    participants: pd.DataFrame,
    vad: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble raw participant-task predictors and outcomes."""
    stats = stats.rename(columns={"task_id": "task"}) if "task" not in stats else stats.copy()
    audio = audio.rename(columns={"task_id": "task"}) if "task" not in audio else audio.copy()
    vad = vad.rename(
        columns={
            "participant": "participant_id",
            "valence": "vad_valence",
            "arousal": "vad_arousal",
            "dominance": "vad_dominance",
        }
    )
    table = stats[stats["task"].isin(ACTIVE_TASKS)].copy()

    # Read and validate participant metadata even when group_id is already present.
    if not {"participant_id", "group_id"}.issubset(participants.columns):
        raise ValueError("participants.tsv must contain participant_id and group_id")
    if table["group_id"].isna().any():
        seat_groups = participants[["session_id", "seat", "group_id"]].rename(
            columns={"seat": "participant_id"}
        )
        table = table.drop(columns="group_id").merge(
            seat_groups, on=["session_id", "participant_id"], how="left", validate="many_to_one"
        )

    source_audio = [name.removeprefix("audio_") for name in AUDIO_ABS_FEATURES]
    audio_cols = ["session_id", "participant_id", "task"] + source_audio
    audio_selected = audio[audio_cols].rename(
        columns={name: f"audio_{name}" for name in source_audio}
    )
    table = table.drop(columns=[c for c in AUDIO_ABS_FEATURES if c in table.columns])
    # Outer merge retains groups that have audio and questionnaire data but no
    # physiology/pupil row; their missing modalities are handled inside folds.
    table = table.merge(
        audio_selected,
        on=["session_id", "participant_id", "task"],
        how="outer",
        validate="one_to_one",
    )
    if "group_id" not in table or table["group_id"].isna().any():
        audio_groups = audio[["session_id", "participant_id", "task", "group_id"]]
        table = table.merge(
            audio_groups,
            on=["session_id", "participant_id", "task"],
            how="left",
            suffixes=("", "_from_audio"),
            validate="one_to_one",
        )
        table["group_id"] = table["group_id"].fillna(table.pop("group_id_from_audio"))

    vad_cols = ["session_id", "participant_id", "task", "vad_valence", "vad_arousal",
                "vad_dominance"]
    table = table.drop(columns=[c for c in vad_cols[3:] if c in table.columns])
    table = table.merge(
        vad[vad_cols],
        on=["session_id", "participant_id", "task"],
        how="left",
        validate="one_to_one",
    )
    trust_cols = ["ans_trust_front", "ans_trust_angle", "ans_trust_next"]
    table["trust_mean"] = table[trust_cols].mean(axis=1)
    if table.duplicated(["session_id", "participant_id", "task"]).any():
        raise ValueError("More than one row per participant-task combination")
    return table


def select_features(train: pd.DataFrame) -> list[str]:
    """Apply missingness, variance, and correlation filters using training data only."""
    numeric = train[CANDIDATE_FEATURES].apply(pd.to_numeric, errors="coerce")
    selected = [
        col
        for col in CANDIDATE_FEATURES
        if numeric[col].notna().mean() >= 0.20 and numeric[col].nunique(dropna=True) > 1
    ]
    kept: list[str] = []
    for col in selected:
        if not kept:
            kept.append(col)
            continue
        correlations = numeric[kept].corrwith(numeric[col]).abs()
        if not (correlations >= 0.95).any():
            kept.append(col)
    return kept


def preprocess_fold(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit clipping, KNN imputation, and scaling on one training fold."""
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    x_train = train[features].apply(pd.to_numeric, errors="coerce").copy()
    x_test = test[features].apply(pd.to_numeric, errors="coerce").copy()
    lower = x_train.quantile(0.01)
    upper = x_train.quantile(0.99)
    x_train = x_train.clip(lower=lower, upper=upper, axis="columns")
    x_test = x_test.clip(lower=lower, upper=upper, axis="columns")
    imputer = KNNImputer(n_neighbors=min(5, len(x_train)))
    train_imputed = imputer.fit_transform(x_train)
    test_imputed = imputer.transform(x_test)
    scaler = StandardScaler()
    return scaler.fit_transform(train_imputed), scaler.transform(test_imputed)


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Return a deterministic 95% percentile interval over fold metrics."""
    if not values:
        return math.nan, math.nan
    rng = np.random.default_rng(SEED)
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def run_benchmark(
    table: pd.DataFrame, spec: Benchmark, all_groups: list[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run all LOGO folds for one benchmark."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    subset = table[table["task"].isin(spec.tasks)].dropna(subset=[spec.target]).copy()
    folds: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    metrics: list[float] = []
    for group_id in all_groups:
        train = subset[subset["group_id"] != group_id].copy()
        test = subset[subset["group_id"] == group_id].copy()
        assert set(train["group_id"]).isdisjoint(set(test["group_id"]))
        threshold = math.nan
        if spec.multiclass:
            y_train = train[spec.target].astype(str)
            y_test = test[spec.target].astype(str)
        else:
            threshold = float(train[spec.target].median())
            y_train = (train[spec.target] >= threshold).astype(int)
            y_test = (test[spec.target] >= threshold).astype(int)

        features = select_features(train)
        if not features:
            raise ValueError(f"{spec.benchmark}/{group_id}: no usable training features")
        if test.empty:
            folds.append(
                {
                    "benchmark": spec.benchmark,
                    "target": spec.target,
                    "held_out_group": group_id,
                    "n_train": len(train),
                    "n_test": 0,
                    "n_features": len(features),
                    "training_median": threshold,
                    "fold_metric": math.nan,
                }
            )
            continue
        x_train, x_test = preprocess_fold(train, test, features)
        model = LogisticRegression(
            C=1.0, max_iter=500, random_state=SEED, class_weight=None
        )
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        if spec.multiclass:
            scores = np.full(len(test), np.nan)
            metric = float(accuracy_score(y_test, predicted)) if len(test) else math.nan
        else:
            positive_index = list(model.classes_).index(1)
            scores = model.predict_proba(x_test)[:, positive_index]
            metric = (
                float(roc_auc_score(y_test, scores))
                if len(test) and y_test.nunique() == 2
                else math.nan
            )
        if not math.isnan(metric):
            metrics.append(metric)
        folds.append(
            {
                "benchmark": spec.benchmark,
                "target": spec.target,
                "held_out_group": group_id,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(features),
                "training_median": threshold,
                "fold_metric": metric,
            }
        )
        for position, (index, row) in enumerate(test.iterrows()):
            predictions.append(
                {
                    "benchmark": spec.benchmark,
                    "target": spec.target,
                    "held_out_group": group_id,
                    "session_id": row["session_id"],
                    "participant_id": row["participant_id"],
                    "task": row["task"],
                    "row_index": index,
                    "training_median": threshold,
                    "target_value": row[spec.target],
                    "target_class": y_test.iloc[position],
                    "predicted_class": predicted[position],
                    "predicted_score": scores[position],
                }
            )
    ci_lower, ci_upper = bootstrap_ci(metrics)
    summary = {
        "benchmark": spec.benchmark,
        "target": spec.target,
        "metric": "accuracy" if spec.multiclass else "roc_auc",
        "mean": float(np.mean(metrics)) if metrics else math.nan,
        "SD": float(np.std(metrics, ddof=1)) if len(metrics) > 1 else math.nan,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "n_valid_folds": len(metrics),
        "n_rows": len(subset),
        "n_groups": subset["group_id"].nunique(),
    }
    return folds, predictions, summary


def validate(table: pd.DataFrame, groups: list[str]) -> None:
    """Run protocol invariants before model fitting."""
    if len(groups) != 10:
        raise ValueError(f"Expected 10 groups, found {len(groups)}")
    assert "task" not in CANDIDATE_FEATURES
    assert not any(feature.startswith("vad_") for feature in CANDIDATE_FEATURES)
    assert not any(feature.startswith("ans_") for feature in CANDIDATE_FEATURES)
    assert not any(feature.startswith("bfi44_") for feature in CANDIDATE_FEATURES)
    assert not any(feature.startswith("biomarker_") for feature in CANDIDATE_FEATURES)
    assert table.loc[table["task"].isin(("T2", "T3")), "ans_satisfaction"].notna().any()
    assert table.loc[table["task"].isin(("T2", "T4")), "trust_mean"].notna().any()


def main() -> int:
    """Run the corrected suite and write required artifacts."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    table = assemble_table(
        load_tsv(args.stats_pt),
        load_tsv(args.audio_task),
        load_tsv(args.participants),
        load_tsv(args.vad_task),
    )
    groups = sorted(table["group_id"].dropna().unique().tolist())
    validate(table, groups)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for spec in BENCHMARKS:
        folds, predictions, summary = run_benchmark(table, spec, groups)
        if len(folds) != 10:
            raise AssertionError(f"{spec.benchmark}: expected 10 attempted folds")
        fold_rows.extend(folds)
        prediction_rows.extend(predictions)
        summary_rows.append(summary)
        LOGGER.info(
            "%s %s mean=%.3f SD=%.3f valid=%d/10",
            spec.benchmark,
            summary["metric"],
            summary["mean"],
            summary["SD"],
            summary["n_valid_folds"],
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.out_dir / "corrected_level1_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        args.out_dir / "corrected_level1_folds.tsv", sep="\t", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        args.out_dir / "corrected_level1_predictions.tsv", sep="\t", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
