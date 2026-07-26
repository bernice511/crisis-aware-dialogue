"""Create plots from detailed classifier evaluation outputs.

This script reads the CSV files created by detailed_evaluation.py and saves
report-ready plots under results/classifier_detailed/plots.

Usage:
    python3 -m src.classifier.plot_detailed_evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results/classifier_detailed"


def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a simple bar plot."""
    plt.figure(figsize=(12, 6))
    plt.bar(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel("Crisis type")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_sweep(threshold_df: pd.DataFrame, output_path: Path) -> None:
    """Plot validation threshold sweep."""
    plt.figure(figsize=(9, 6))
    plt.plot(threshold_df["threshold"], threshold_df["micro_f1"], marker="o", label="Micro F1")
    plt.plot(threshold_df["threshold"], threshold_df["macro_f1"], marker="o", label="Macro F1")
    plt.plot(threshold_df["threshold"], threshold_df["micro_precision"], marker="o", label="Micro precision")
    plt.plot(threshold_df["threshold"], threshold_df["micro_recall"], marker="o", label="Micro recall")
    plt.title("Validation Threshold Sweep")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_bootstrap_per_class(bootstrap_df: pd.DataFrame, output_path: Path) -> None:
    """Plot per-class F1 with 95% bootstrap confidence intervals."""
    x_positions = range(len(bootstrap_df))
    y = bootstrap_df["point_f1"]
    lower_error = y - bootstrap_df["ci_lower_95"]
    upper_error = bootstrap_df["ci_upper_95"] - y

    plt.figure(figsize=(12, 6))
    plt.errorbar(
        x_positions,
        y,
        yerr=[lower_error, upper_error],
        fmt="o",
        capsize=4,
    )
    plt.xticks(x_positions, bootstrap_df["crisis_type"], rotation=45, ha="right")
    plt.title("Per-Class F1 with 95% Bootstrap Confidence Intervals")
    plt.xlabel("Crisis type")
    plt.ylabel("F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration(overall_bins_df: pd.DataFrame, output_path: Path) -> None:
    """Plot overall reliability curve."""
    non_empty = overall_bins_df[overall_bins_df["count"] > 0]

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(
        non_empty["avg_confidence"],
        non_empty["empirical_accuracy"],
        marker="o",
        label="Model",
    )
    plt.title("Overall Calibration / Reliability Curve")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing detailed evaluation CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_class_df = pd.read_csv(args.results_dir / "per_class_report.csv")
    confusion_df = pd.read_csv(args.results_dir / "confusion_breakdown.csv")
    threshold_df = pd.read_csv(args.results_dir / "threshold_sweep_validation.csv")
    bootstrap_per_class_df = pd.read_csv(args.results_dir / "bootstrap_per_class_f1.csv")
    calibration_overall_bins_df = pd.read_csv(args.results_dir / "calibration_overall_bins.csv")

    save_bar_plot(
        df=per_class_df,
        x_col="crisis_type",
        y_col="f1",
        title="Per-Class F1 Score",
        ylabel="F1",
        output_path=plots_dir / "per_class_f1.png",
    )

    save_bar_plot(
        df=per_class_df,
        x_col="crisis_type",
        y_col="recall",
        title="Per-Class Recall",
        ylabel="Recall",
        output_path=plots_dir / "per_class_recall.png",
    )

    save_bar_plot(
        df=confusion_df,
        x_col="crisis_type",
        y_col="miss_rate",
        title="Per-Class Miss Rate",
        ylabel="Miss rate",
        output_path=plots_dir / "per_class_miss_rate.png",
    )

    save_bar_plot(
        df=confusion_df,
        x_col="crisis_type",
        y_col="false_alarm_rate",
        title="Per-Class False Alarm Rate",
        ylabel="False alarm rate",
        output_path=plots_dir / "per_class_false_alarm_rate.png",
    )

    plot_threshold_sweep(
        threshold_df=threshold_df,
        output_path=plots_dir / "threshold_sweep_validation.png",
    )

    plot_bootstrap_per_class(
        bootstrap_df=bootstrap_per_class_df,
        output_path=plots_dir / "per_class_f1_bootstrap_ci.png",
    )

    plot_calibration(
        overall_bins_df=calibration_overall_bins_df,
        output_path=plots_dir / "overall_calibration_curve.png",
    )

    print("Saved plots:")
    for path in sorted(plots_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()