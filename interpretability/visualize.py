import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_mean_std(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a 'mean ± std' string column into two numeric Series."""
    split = series.str.split(r"\s*±\s*", expand=True)
    return split[0].astype(float), split[1].astype(float)


def generate_plots(
    raw_csv_path="outputs/intervention_table_raw.csv",
    summary_csv_path="outputs/intervention_table_summary.csv",
    primary_csv_path="outputs/intervention_table.csv",
    spectrum_csv_path="outputs/svd_variance_spectrum.csv",
    output_dir="outputs/plots",
):
    os.makedirs(output_dir, exist_ok=True)

    raw = pd.read_csv(raw_csv_path)
    raw = raw.rename(columns={"seed": "Seed", "svd_rank": "Rank"})

    summary = pd.read_csv(summary_csv_path)
    summary = summary.rename(columns={"SVD Rank": "Rank"})

    for col in [
        "Explained Var (%)",
        "Base Loss",
        "Feature Loss",
        "Feature Effect Size",
        "Control Loss",
        "Control Effect Size",
        "Causal Delta",
    ]:
        if col in summary.columns and pd.api.types.is_string_dtype(summary[col]):
            mean_s, std_s = _parse_mean_std(summary[col])
            summary[col] = mean_s
            summary[col + " Std"] = std_s

    primary = pd.read_csv(primary_csv_path)
    spectrum = pd.read_csv(spectrum_csv_path)

    logger.info("Loaded raw table: %s", raw.shape)
    logger.info("Loaded summary table: %s", summary.shape)
    logger.info("Loaded primary table: %s", primary.shape)
    logger.info("Loaded SVD spectrum: %s", spectrum.shape)

    ablation = summary[
        summary["Intervention"].str.contains(
            "Ablation",
            case=False,
            na=False,
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        ablation["Explained Var (%)"],
        ablation["Causal Delta"],
        yerr=ablation["Causal Delta Std"],
        fmt="o",
        capsize=4,
        markersize=7,
    )

    for _, row in ablation.iterrows():
        ax.annotate(
            f"Rank {int(row['Rank'])}",
            (
                row["Explained Var (%)"],
                row["Causal Delta"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Explained variance (%)")
    ax.set_ylabel("Causal Delta")
    ax.set_title(
        "Representation Variance vs Causal Importance"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "1_variance_vs_causal_delta.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    intervention_order = [
        "Ablation",
        "Clamping",
        "Amplification",
        "Swapping",
    ]

    for intervention in intervention_order:

        data = summary[
            summary["Intervention"].str.contains(
                intervention,
                case=False,
                na=False,
            )
        ].copy()

        if data.empty:
            continue

        data = data.sort_values("Rank")

        ax.errorbar(
            data["Rank"],
            data["Causal Delta"],
            yerr=data["Causal Delta Std"],
            marker="o",
            capsize=3,
            label=intervention,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("SVD rank")
    ax.set_ylabel("Causal Delta")
    ax.set_title(
        "Causal Effect Across SVD Ranks"
    )

    ax.set_xticks(
        sorted(summary["Rank"].unique())
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "2_causal_delta_across_ranks.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    rank1_ablation = raw[
        (raw["Rank"] == 1)
        & raw["Intervention"].str.contains(
            "Ablation",
            case=False,
            na=False,
        )
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    if "Seed" in rank1_ablation.columns:

        seeds = sorted(
            rank1_ablation["Seed"].unique()
        )

        seed_values = []

        for seed in seeds:

            data = rank1_ablation[
                rank1_ablation["Seed"] == seed
            ]

            value = data["Causal Delta"].mean()

            seed_values.append(value)

            ax.scatter(
                seed,
                value,
                s=80,
            )

        mean_value = np.mean(seed_values)
        sd_value = np.std(
            seed_values,
            ddof=1,
        )

        ax.axhline(
            mean_value,
            linestyle="--",
            linewidth=1,
            label=(
                f"Mean = {mean_value:.3f}"
            ),
        )

        ax.fill_between(
            seeds,
            mean_value - sd_value,
            mean_value + sd_value,
            alpha=0.15,
            label=(
                f"±1 Std = {sd_value:.3f}"
            ),
        )

        ax.set_xticks(seeds)

    ax.axhline(
        0,
        linestyle=":",
        linewidth=1,
    )

    ax.set_xlabel("Evaluation seed")
    ax.set_ylabel("Causal Delta")
    ax.set_title(
        "Rank-1 Ablation Stability Across Seeds"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "3_seed_stability.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    rank1 = primary.copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    y_positions = np.arange(
        len(rank1)
    )

    for y, (_, row) in zip(
        y_positions,
        rank1.iterrows(),
    ):

        control = row["Control Effect Size"]
        feature = row["Feature Effect Size"]

        ax.plot(
            [control, feature],
            [y, y],
            linewidth=3,
        )

        ax.scatter(
            control,
            y,
            s=90,
            label=(
                "Random control"
                if y == 0
                else None
            ),
        )

        ax.scatter(
            feature,
            y,
            s=90,
            label=(
                "SVD direction"
                if y == 0
                else None
            ),
        )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(y_positions)

    ax.set_yticklabels(
        rank1["Intervention"]
    )

    ax.set_xlabel("Effect size")
    ax.set_ylabel("Intervention")
    ax.set_title(
        "SVD Direction vs Matched Random Control"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "4_svd_vs_random_control.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    rank1_raw = raw[
        raw["Rank"] == 1
    ].copy()

    intervention_order = [
        "Ablation",
        "Clamping",
        "Amplification",
        "Swapping",
    ]

    distributions = []
    labels = []

    for intervention in intervention_order:

        data = rank1_raw[
            rank1_raw["Intervention"].str.contains(
                intervention,
                case=False,
                na=False,
            )
        ]["Causal Delta"].dropna()

        if len(data) == 0:
            continue

        distributions.append(
            data.to_numpy()
        )

        labels.append(intervention)

    fig, ax = plt.subplots(figsize=(9, 5))

    parts = ax.violinplot(
        distributions,
        positions=np.arange(
            1,
            len(distributions) + 1,
        ),
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    for i, values in enumerate(
        distributions,
        start=1,
    ):

        x = np.random.default_rng(
            42
        ).normal(
            i,
            0.035,
            size=len(values),
        )

        ax.scatter(
            x,
            values,
            s=45,
            alpha=0.8,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xticks(
        range(
            1,
            len(labels) + 1,
        )
    )

    ax.set_xticklabels(labels)

    ax.set_xlabel("Intervention")
    ax.set_ylabel("Causal Delta")
    ax.set_title(
        "Seed-Level Distribution of Causal Effects"
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            output_dir,
            "5_causal_effect_distribution.png",
        ),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    logger.info(
        "Generated five plots in %s",
        output_dir,
    )


if __name__ == "__main__":
    generate_plots()