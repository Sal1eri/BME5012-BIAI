import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

def plot_age_distribution_publication(csv_path):

    # Config
    STACK_ORDER = ['CN', 'MCI', 'EMCI', 'LMCI', 'AD']

    PALETTE = {
        'CN':   '#2E7D32',  # green
        'MCI':  '#FDD835',  # yellow
        'EMCI': '#1E88E5',  # blue
        'LMCI': '#FF6F00',  # orange
        'AD':   '#8B0000'   # dark red
    }

    BINS = 18            # Smoother bins
    KDE_BW = 1.6         # KDE smoothing
    KDE_ALPHA = 0.8
    KDE_LINEWIDTH = 2.2

    # Load data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().replace('"', '') for c in df.columns]

    if not {'Group', 'Age'}.issubset(df.columns):
        raise ValueError("CSV must contain Group and Age columns")

    df = df[df['Group'].isin(STACK_ORDER)].copy()

    age_min, age_max = df['Age'].min(), df['Age'].max()

    # Canvas
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=1.05
    )

    fig, ax_count = plt.subplots(figsize=(12, 8))

    # Left axis: count
    sns.histplot(
        data=df,
        x="Age",
        hue="Group",
        hue_order=STACK_ORDER,
        multiple="stack",
        bins=BINS,
        palette=PALETTE,
        edgecolor=None,
        alpha=0.88,
        ax=ax_count
    )

    ax_count.set_xlim(age_min, age_max)
    ax_count.set_xlabel("Age (Years)", fontsize=14, fontweight="bold")
    ax_count.set_ylabel("Count", fontsize=14, fontweight="bold")

    # Right axis: density
    ax_density = ax_count.twinx()
    ax_density.set_xlim(age_min, age_max)

    for g in STACK_ORDER:
        sns.kdeplot(
            data=df[df["Group"] == g],
            x="Age",
            ax=ax_density,
            color=PALETTE[g],
            bw_adjust=KDE_BW,
            linewidth=KDE_LINEWIDTH,
            alpha=KDE_ALPHA,
            fill=False,
            common_norm=False,
            clip=(age_min, age_max)
        )

    ax_density.set_ylabel("Density", fontsize=14, fontweight="bold")

    # Styling
    ax_count.set_title(
        "Age Distribution Across Diagnostic Groups",
        fontsize=18,
        fontweight="bold",
        pad=18
    )

    ax_count.tick_params(axis='both', colors='#37474F')
    ax_density.tick_params(axis='y', colors='#37474F')

    sns.despine(ax=ax_count, right=True)
    sns.despine(ax=ax_density, left=True)

    # Single legend
    handles, labels = ax_count.get_legend_handles_labels()
    ax_count.legend(
        handles,
        labels,
        title="Diagnosis",
        loc="upper right",
        frameon=True
    )

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "age_distribution_adni_publication.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    CSV_PATH = REPO_ROOT / "dataset" / "5_class_10_12_2025.csv"
    plot_age_distribution_publication(CSV_PATH)
