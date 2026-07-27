from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path(
    "results/cradle_response_detailed/"
    "llama_evaluation_with_automated_scores.xlsx"
)
SHEET_NAME = "Clear Model Evaluation"
OUTPUT_DIR = Path("results/cradle_response_detailed/evaluation_analysis")


METRICS = {
    "Relevance": ("base_relevance_0_2", "fine_tuned_relevance_0_2"),
    "Empathy": ("base_empathy_0_2", "fine_tuned_empathy_0_2"),
    "Safety": ("base_safety_0_2", "fine_tuned_safety_0_2"),
    "Helpfulness": ("base_helpfulness_0_2", "fine_tuned_helpfulness_0_2"),
    "Naturalness": ("base_naturalness_0_2", "fine_tuned_naturalness_0_2"),
}

SAFETY_FLAGS = {
    "Harmful advice": (
        "base_harmful_advice_yes_no",
        "fine_tuned_harmful_advice_yes_no",
    ),
    "Missed urgent safety": (
        "base_missed_urgent_safety_yes_no",
        "fine_tuned_missed_urgent_safety_yes_no",
    ),
    "Unnecessary escalation": (
        "base_unnecessary_escalation_yes_no",
        "fine_tuned_unnecessary_escalation_yes_no",
    ),
}


def require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]

    if missing:
        raise ValueError(
            "The following required columns are missing:\n"
            + "\n".join(f"- {column}" for column in missing)
        )


def yes_count(series: pd.Series) -> int:
    return int(
        series.astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .sum()
    )


def save_metric_plot(summary: pd.DataFrame) -> None:
    plot_data = summary.set_index("metric")[["base_llama", "fine_tuned_llama"]]

    axis = plot_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    axis.set_title("Base Llama vs Fine-Tuned Llama")
    axis.set_xlabel("Evaluation metric")
    axis.set_ylabel("Average score (0–2)")
    axis.set_ylim(0, 2)
    axis.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "metric_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_preference_plot(preference_counts: pd.Series) -> None:
    order = ["Base", "Fine-tuned", "Tie"]
    plot_data = preference_counts.reindex(order, fill_value=0)

    axis = plot_data.plot(
        kind="bar",
        figsize=(8, 5),
    )

    axis.set_title("Preferred Response Counts")
    axis.set_xlabel("Preferred model")
    axis.set_ylabel("Number of examples")
    axis.tick_params(axis="x", rotation=0)

    for container in axis.containers:
        axis.bar_label(container)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "preference_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_safety_plot(safety_summary: pd.DataFrame) -> None:
    plot_data = safety_summary.set_index("safety_flag")[
        ["base_llama", "fine_tuned_llama"]
    ]

    axis = plot_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    axis.set_title("Safety Error Counts")
    axis.set_xlabel("Safety error")
    axis.set_ylabel("Number of examples")
    axis.tick_params(axis="x", rotation=0)

    for container in axis.containers:
        axis.bar_label(container)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "safety_error_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation workbook not found:\n{INPUT_FILE.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        header=1,
    )

    dataframe = dataframe.dropna(subset=["evaluation_id"]).copy()

    required_columns = ["preferred_model_base_fine_tuned_tie"]

    for base_column, fine_column in METRICS.values():
        required_columns.extend([base_column, fine_column])

    for base_column, fine_column in SAFETY_FLAGS.values():
        required_columns.extend([base_column, fine_column])

    require_columns(dataframe, required_columns)

    metric_rows = []

    for metric_name, (base_column, fine_column) in METRICS.items():
        dataframe[base_column] = pd.to_numeric(
            dataframe[base_column],
            errors="coerce",
        )
        dataframe[fine_column] = pd.to_numeric(
            dataframe[fine_column],
            errors="coerce",
        )

        metric_rows.append(
            {
                "metric": metric_name,
                "base_llama": dataframe[base_column].mean(),
                "fine_tuned_llama": dataframe[fine_column].mean(),
            }
        )

    metric_summary = pd.DataFrame(metric_rows)
    metric_summary["fine_tuned_minus_base"] = (
        metric_summary["fine_tuned_llama"]
        - metric_summary["base_llama"]
    )

    base_score_columns = [
        columns[0] for columns in METRICS.values()
    ]
    fine_score_columns = [
        columns[1] for columns in METRICS.values()
    ]

    dataframe["base_total_0_10"] = dataframe[base_score_columns].sum(axis=1)
    dataframe["fine_tuned_total_0_10"] = dataframe[
        fine_score_columns
    ].sum(axis=1)

    overall_summary = pd.DataFrame(
        [
            {
                "metric": "Overall score",
                "base_llama": dataframe["base_total_0_10"].mean(),
                "fine_tuned_llama": dataframe[
                    "fine_tuned_total_0_10"
                ].mean(),
            }
        ]
    )

    overall_summary["fine_tuned_minus_base"] = (
        overall_summary["fine_tuned_llama"]
        - overall_summary["base_llama"]
    )

    safety_rows = []

    for flag_name, (base_column, fine_column) in SAFETY_FLAGS.items():
        safety_rows.append(
            {
                "safety_flag": flag_name,
                "base_llama": yes_count(dataframe[base_column]),
                "fine_tuned_llama": yes_count(dataframe[fine_column]),
            }
        )

    safety_summary = pd.DataFrame(safety_rows)
    safety_summary["fine_tuned_minus_base"] = (
        safety_summary["fine_tuned_llama"]
        - safety_summary["base_llama"]
    )

    preference_counts = (
        dataframe["preferred_model_base_fine_tuned_tie"]
        .astype(str)
        .str.strip()
        .value_counts()
    )

    metric_summary.to_csv(
        OUTPUT_DIR / "metric_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        OUTPUT_DIR / "overall_summary.csv",
        index=False,
    )
    safety_summary.to_csv(
        OUTPUT_DIR / "safety_summary.csv",
        index=False,
    )
    preference_counts.rename_axis("preferred_model").reset_index(
        name="count"
    ).to_csv(
        OUTPUT_DIR / "preference_summary.csv",
        index=False,
    )

    save_metric_plot(metric_summary)
    save_preference_plot(preference_counts)
    save_safety_plot(safety_summary)

    print("\nAUTOMATED PRELIMINARY RESPONSE EVALUATION")
    print("=" * 47)
    print(f"Examples evaluated: {len(dataframe)}")

    print("\nAverage metric scores (0–2):")
    print(metric_summary.round(3).to_string(index=False))

    print("\nOverall average scores (0–10):")
    print(overall_summary.round(3).to_string(index=False))

    print("\nPreferred response counts:")
    print(preference_counts.to_string())

    print("\nSafety error counts:")
    print(safety_summary.to_string(index=False))

    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()