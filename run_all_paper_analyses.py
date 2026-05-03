"""
Orchestrate all paper-relevant analyses and figure generation.

This is the master script that runs all visualization and statistical analyses
needed for the NeurIPS 2026 dataset paper, reusing existing analysis pipelines.

Runs in order:
  1. Task responses analysis (VAD, postblock, decisions)
  2. Task-response QC and appendix summaries
  3. Autonomic paper analysis (HR, EDA, pupil, temperature, motion)
  4. Physio paper analysis (detailed physio QC and task effects)
  5. Personality analysis (BFI-44 trait distributions and group profiles)
  6. Cross-modal analysis (inter-modality correlations)
  7. Multimodal statistics (mixed models, group effects)
  8. Expanded stats (additional exploratory analyses)
  9. Individual audio analysis (prosodic features)
  10. Eye-tracking visualizations

Output structure:
  results/task_responses/      â† VAD, postblock, decisions tables + plots
  results/task_response_qc/    â† completeness, trust/familiarity, QC tables
  results/autonomic/           â† task-wise autonomic changes + heatmaps
  results/physio/              â† physiology feature correlations + QC
  results/personality/         â† BFI trait distributions, group profiles
  results/cross_modal/         â† cross-modality correlations
  results/statistics/          â† mixed models, significance tests
  results/expanded_stats/      â† exploratory analyses
  results/audio/               â† prosodic features and task effects
  
  figures/task_responses/      â† VAD by task, postblock heatmaps
  figures/task_response_qc/    â† missingness and appendix-style QC figures
  figures/autonomic/           â† autonomic task fingerprints, coverage
  figures/physio/              â† physio QC, task effects, correlations
  figures/personality/         â† trait distributions, group heatmaps
  figures/cross_modal/         â† cross-modal scatter, correlations
  figures/et/                  â† eye-tracking QC, gaze, pupil
  figures/dataset_stats/       â† coverage, demographics, usability

Usage
-----
    python tools/features/run_all_paper_analyses.py \\
        --features-dir features \\
        --bids-root data/bids_release_no_video \\
        --results-base results \\
        --figures-base figures \\
        --verbose
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("run_all_paper_analyses")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all paper analyses and figure generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features"),
        help="Directory containing physio, pupil, and audio feature TSV files.",
    )
    p.add_argument(
        "--bids-root",
        type=Path,
        default=Path("data") / "bids_release_no_video",
        help="BIDS dataset root (for participants.tsv, events.tsv).",
    )
    p.add_argument(
        "--results-base",
        type=Path,
        default=Path("results"),
        help="Base directory for all results/ subdirectories.",
    )
    p.add_argument(
        "--figures-base",
        type=Path,
        default=Path("figures"),
        help="Base directory for all figures/ subdirectories.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure output DPI.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    p.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip long-running analyses (multimodal stats, expanded stats).",
    )
    return p


def _run(cmd: list[str]) -> None:
    """Run a subprocess command and log it."""
    cmd_str = " ".join(str(c) for c in cmd)
    LOG.info("Running: %s", cmd_str)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOG.error("Command failed with exit code %d: %s", result.returncode, cmd_str)
        return
    LOG.info("âœ“ Command succeeded")


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    LOG.info("=" * 80)
    LOG.info("GroupAffect-4 Paper Analysis Pipeline")
    LOG.info("=" * 80)
    LOG.info("Features dir:   %s", args.features_dir.resolve())
    LOG.info("BIDS root:      %s", args.bids_root.resolve())
    LOG.info("Results base:   %s", args.results_base.resolve())
    LOG.info("Figures base:   %s", args.figures_base.resolve())
    LOG.info("DPI:            %d", args.dpi)
    LOG.info("=" * 80)

    py = sys.executable
    verbose_flag = ["--verbose"] if args.verbose else []
    dpi_flag = ["--dpi", str(args.dpi)]

    # 1. Task responses (VAD, postblock, decisions)
    # Note: analyze_task_responses.py does not support --verbose flag
    LOG.info("\n[1/9] Task Responses Analysis...")
    _run([
        py, "tools/features/analyze_task_responses.py",
        "--bids-root", str(args.bids_root),
        "--results-dir", str(args.results_base / "task_responses"),
        "--figures-dir", str(args.figures_base / "task_responses"),
        *dpi_flag,
    ])

    # 2. Task-response QC and appendix summaries
    LOG.info("\n[2/10] Task Response QC...")
    _run([
        py, "tools/features/analyze_task_response_qc.py",
        "--bids-root", str(args.bids_root),
        "--results-dir", str(args.results_base / "task_response_qc"),
        "--figures-dir", str(args.figures_base / "task_response_qc"),
        *dpi_flag,
        *verbose_flag,
    ])

    # 3. Autonomic paper analysis (HR, EDA, pupil, temp, motion)
    LOG.info("\n[3/10] Autonomic Paper Analysis...")
    _run([
        py, "tools/features/analyze_autonomic_paper.py",
        "--features-dir", str(args.features_dir),
        "--results-dir", str(args.results_base / "autonomic"),
        "--figures-dir", str(args.figures_base / "autonomic"),
        *dpi_flag,
        *verbose_flag,
    ])

    # 4. Physio paper analysis (detailed QC and task effects)
    LOG.info("\n[4/10] Physiology Paper Analysis...")
    _run([
        py, "tools/features/analyze_physio_paper.py",
        "--features-dir", str(args.features_dir),
        "--results-dir", str(args.results_base / "physio"),
        "--figures-dir", str(args.figures_base / "physio"),
        *dpi_flag,
        *verbose_flag,
    ])

    # 5. Personality analysis (BFI-44 distributions, group profiles)
    LOG.info("\n[5/10] Personality Analysis...")
    _run([
        py, "tools/features/analyze_personality.py",
        "--bids-root", str(args.bids_root),
        "--results-dir", str(args.results_base / "personality"),
        "--figures-dir", str(args.figures_base / "personality"),
        *dpi_flag,
        *verbose_flag,
    ])

    # 6. Cross-modal analysis (inter-modality correlations)
    LOG.info("\n[6/10] Cross-Modal Analysis...")
    _run([
        py, "tools/features/analyze_cross_modal.py",
        "--features-dir", str(args.features_dir),
        "--results-dir", str(args.results_base / "cross_modal"),
        "--figures-dir", str(args.figures_base / "cross_modal"),
        *dpi_flag,
        *verbose_flag,
    ])

    # 7. Multimodal statistics (mixed models, significance tests)
    if not args.skip_slow:
        LOG.info("\n[7/10] Multimodal Statistics (inferential)...")
        _run([
            py, "tools/features/analyze_multimodal_statistics.py",
            "--features-dir", str(args.features_dir),
            "--results-dir", str(args.results_base / "statistics"),
            *verbose_flag,
        ])
    else:
        LOG.warning("[7/10] Skipping Multimodal Statistics (--skip-slow)")

    # 8. Expanded stats (exploratory analyses)
    if not args.skip_slow:
        LOG.info("\n[8/10] Expanded Statistics (exploratory)...")
        _run([
            py, "tools/features/analyze_expanded_stats.py",
            "--features-dir", str(args.features_dir),
            "--results-dir", str(args.results_base / "expanded_stats"),
            *verbose_flag,
        ])
    else:
        LOG.warning("[8/10] Skipping Expanded Statistics (--skip-slow)")

    # 9. Individual audio analysis (prosodic features + transcript turn-taking)
    LOG.info("\n[9/10] Individual Audio Analysis...")
    _run([
        py, "tools/features/analyze_individual_audio.py",
        "--features-dir", str(args.features_dir),
        "--results-dir", str(args.results_base / "audio"),
        "--figures-dir", str(args.figures_base / "audio"),
        "--transcripts-root", str(args.results_base / "audio" / "transcripts"),
        "--paper-table", str(Path("paper/tables/audio_turn_taking_summary.tex")),
        *dpi_flag,
        *verbose_flag,
    ])

    # 10. Eye-tracking visualization and QC
    LOG.info("\n[10/10] Eye-Tracking Visualizations...")
    _run([
        py, "tools/features/visualize_eyetracking_features.py",
        "--features-dir", str(args.features_dir),
        "--figures-dir", str(args.figures_base / "et"),
        *dpi_flag,
        *verbose_flag,
    ])

    LOG.info("\n" + "=" * 80)
    LOG.info("Paper Analysis Pipeline Complete!")
    LOG.info("=" * 80)
    LOG.info("Results available in:")
    for subdir in [
        "task_responses",
        "task_response_qc",
        "autonomic",
        "physio",
        "personality",
        "cross_modal",
        "statistics",
        "expanded_stats",
        "audio",
    ]:
        print(f"  {args.results_base / subdir}")
    LOG.info("\nFigures available in:")
    for subdir in [
        "task_responses",
        "task_response_qc",
        "autonomic",
        "physio",
        "personality",
        "cross_modal",
        "et",
        "audio",
    ]:
        print(f"  {args.figures_base / subdir}")
    LOG.info("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

