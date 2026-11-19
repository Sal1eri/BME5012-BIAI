#!/usr/bin/env python3
"""Collect test metrics from experiment logs and write a summary."""
import argparse
import csv
import re
import statistics
from pathlib import Path

import pandas as pd

METRICS = [
    "Accuracy",
    "Macro-F1",
    "Macro-AUC",
    "Balanced Accuracy",
    "Cohen's Kappa",
    "MCC",
]
SHORT = {
    "Accuracy": "Acc",
    "Macro-F1": "F1",
    "Macro-AUC": "AUC",
    "Balanced Accuracy": "BAcc",
    "Cohen's Kappa": "Kappa",
    "MCC": "MCC",
}
DISPLAY_NAMES = {
    "metadata_svm": "SVM (Age/Sex)",
    "metadata_logreg": "LogReg (Age/Sex)",
    "metadata_random_forest": "RF (Age/Sex)",
    "image_pca256_svm": "SVM (3-view, PCA256)",
    "image_pca256_logreg": "LogReg (3-view, PCA256)",
    "image_pca256_random_forest": "RF (3-view, PCA256)",
    "image_single_slice_pca256_svm": "SVM (single-slice, PCA256)",
    "image_single_slice_pca256_logreg": "LogReg (single-slice, PCA256)",
    "image_single_slice_pca256_random_forest": "RF (single-slice, PCA256)",
    "vgg16": "VGG16 (2D-TL)",
    "resnet": "3D-ResNet",
    "hcct_vit": "HCCT-ViT",
    "vit_3d": "3D-ViT",
    "swin_3d_cnn": "Swin-CNN",
    "mvst": "MVST",
}

METRIC_RE = re.compile(
    r"(Accuracy|Macro-F1|Macro-AUC|Balanced Accuracy|Cohen's Kappa|MCC):\s*(-?[0-9.]+)"
)


def parse_metrics(text: str) -> dict:
    return {name: float(value) for name, value in METRIC_RE.findall(text)}


def collect_deep(runs_dir: Path) -> dict:
    groups = {}
    for path in sorted(runs_dir.glob("*_test.log")):
        match = re.match(r"(.+)_seed(\d+)_test\.log$", path.name)
        if not match:
            continue
        model, seed = match.group(1), int(match.group(2))
        values = parse_metrics(path.read_text())
        if values:
            groups.setdefault(model, []).append((seed, values))
    return groups


def collect_deep_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path)
    groups = {}
    for method, rows in frame.groupby("method"):
        entries = []
        for _, row in rows.iterrows():
            entries.append((int(row["seed"]), {key: float(row[key]) for key in METRICS}))
        groups[method] = entries
    return groups


def collect_ml(results_dir: Path) -> dict:
    groups = {}
    for path in sorted(results_dir.glob("*.txt")):
        text = path.read_text()
        if "[TEST] metrics:" not in text:
            continue
        part = text.split("[TEST] metrics:")[-1]
        part = re.split(r"\[(TRAIN|VAL)\] metrics:", part)[0]
        values = parse_metrics(part)
        if values:
            groups[path.stem] = [(0, values)]
    return groups


MEAN_ONLY = False


def format_cell(values: list, key: str) -> str:
    vals = [item[key] * 100 for item in values]
    mean = statistics.mean(vals)
    if MEAN_ONLY or len(vals) == 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {statistics.stdev(vals):.2f}"


def write_csv(path: Path, groups: dict, label: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "method", "seed"] + METRICS)
        for model, entries in sorted(groups.items()):
            for seed, values in entries:
                writer.writerow([label, model, seed] + [values.get(key, "") for key in METRICS])


def score(values: dict) -> float:
    return statistics.mean(values[key] for key in METRICS)


def best_group(groups: dict):
    if not groups:
        return None
    return max(groups.items(), key=lambda item: score(item[1][0][1]))


def best_ml_families(groups: dict) -> dict:
    """Keep the best variant of each ML family (RF / LogReg / SVM)."""
    selected = {}
    for suffix in ["random_forest", "logreg", "svm"]:
        candidates = [(key, value) for key, value in groups.items() if key.endswith(suffix)]
        if candidates:
            key, value = max(candidates, key=lambda item: score(item[1][0][1]))
            selected[key] = value
    return selected


def render_table(table: dict, best: bool, title: str) -> str:
    lines = [f"# {title}", ""]
    header = "| Method | n | " + " | ".join(SHORT[m] for m in METRICS) + " | Avg |"
    lines.extend([header, "| :--- | :---: | " + " | ".join(":---:" for _ in METRICS) + " | :---: |"])

    ordered = sorted(table, key=lambda name: -score(
        max((item for _, item in table[name]), key=score)
        if best
        else {
            key: statistics.mean(item[key] for _, item in table[name])
            for key in METRICS
        }
    ))

    for model in ordered:
        entries = table[model]
        if best:
            chosen = max(entries, key=lambda entry: score(entry[1]))
            values = [chosen[1]]
            n = f"{len(entries)} (best)"
        else:
            values = [item for _, item in entries]
            n = str(len(entries))

        name = DISPLAY_NAMES.get(model, model)
        avg = statistics.mean(statistics.mean(item[key] * 100 for item in values) for key in METRICS)
        row = f"| {name} | {n} | " + " | ".join(
            format_cell(values, key) for key in METRICS
        ) + f" | {avg:.2f} |"
        lines.append(row)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment metrics.")
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--ml", type=Path, default=Path("results/traditional_ml"))
    parser.add_argument("--output", type=Path, default=Path("results/final_metrics.md"))
    parser.add_argument("--best-output", type=Path, default=Path("results/best_metrics.md"))
    args = parser.parse_args()

    deep = collect_deep(args.runs)
    if not deep:
        deep = collect_deep_csv(args.runs.parent / "deep_metrics.csv")
    ml = collect_ml(args.ml)

    table = dict(deep)
    table.update(best_ml_families(ml))

    mean_text = render_table(table, best=False, title="Mean ± std over seeds")
    best_text = render_table(table, best=True, title="Best seed per model")

    args.output.write_text(mean_text, encoding="utf-8")
    args.best_output.write_text(best_text, encoding="utf-8")
    print(mean_text)
    print(best_text)

    if ml:
        write_csv(args.output.with_name("traditional_ml_metrics.csv"), ml, "traditional_ml")
    if deep:
        write_csv(args.output.with_name("deep_metrics.csv"), deep, "deep")


if __name__ == "__main__":
    main()
