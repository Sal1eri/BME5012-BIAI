import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"


def plot_age_distribution_blue(csv_path):
    order = ["CN", "EMCI", "LMCI", "MCI", "AD"]
    palette = {
        "CN": "#8FBC8F",
        "EMCI": "#789CCE",
        "MCI": "#EED776",
        "LMCI": "#F0A472",
        "AD": "#CD5C5C",
    }
    linestyle = {
        "CN": "solid",
        "EMCI": "dashed",
        "LMCI": "dashdot",
        "MCI": "dotted",
        "AD": (0, (5, 1)),
    }
    linewidth = {"CN": 2.8, "EMCI": 3.2, "LMCI": 3.6, "MCI": 3.2, "AD": 4.2}

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = [column.strip().replace('"', "") for column in df.columns]
    df = df[df["Group"].isin(order)].copy()

    age_min, age_max = df["Age"].min(), df["Age"].max()
    sns.set_theme(style="white", context="talk", font_scale=1.1)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    sns.histplot(
        data=df,
        x="Age",
        hue="Group",
        hue_order=order,
        multiple="stack",
        bins=20,
        stat="count",
        palette=palette,
        alpha=0.9,
        edgecolor=None,
        linewidth=0,
        ax=ax1,
        zorder=1,
    )
    ax1.set_xlim(age_min, age_max)
    ax1.set_xlabel("Age (Years)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=14, fontweight="bold")

    ax2 = ax1.twinx()
    for group in order:
        sns.kdeplot(
            data=df[df["Group"] == group],
            x="Age",
            ax=ax2,
            color=palette[group],
            linewidth=linewidth[group],
            linestyle=linestyle[group],
            alpha=0.95,
            bw_adjust=1.3,
            clip=(age_min, age_max),
            common_norm=False,
            legend=False,
            zorder=10,
        )
    ax2.set_ylabel("Density", fontsize=14, fontweight="bold")

    ax1.set_title("Age Distribution by Diagnosis Group", fontsize=18, fontweight="bold", pad=20)
    sns.despine(ax=ax1, left=True)
    sns.despine(ax=ax2, right=True)
    ax1.grid(False)
    ax2.grid(False)
    sns.move_legend(ax1, "upper right", title="Diagnosis Group")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "age_distribution_blue_dual_axis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_age_distribution_blue(REPO_ROOT / "dataset" / "5_class_10_12_2025.csv")
