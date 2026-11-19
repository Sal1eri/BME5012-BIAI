#!/usr/bin/env python3
"""Traditional machine learning baseline for AD classification metadata."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from utils.dataset import LABEL_MAP
except ImportError as exc:
    raise ImportError("Failed to import LABEL_MAP from utils.dataset; check PYTHONPATH" ) from exc

try:
    from utils.metrics import get_classification_metrics
except ImportError as exc:
    raise ImportError("Failed to import get_classification_metrics from utils.metrics; check repository structure" ) from exc

NUMERIC_FEATURES: List[str] = ["Age"]
CATEGORICAL_FEATURES: List[str] = [
    "Sex",
]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_MAP.items()}


def load_split(csv_path: Path) -> pd.DataFrame:
    """Load a CSV split and attach label ids."""
    df = pd.read_csv(csv_path)
    df = df[df["Group"].isin(LABEL_MAP)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Label"] = df["Group"].map(LABEL_MAP)
    return df


def make_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_model(name: str) -> Pipeline:
    """Build the requested estimator wrapped in a preprocessing pipeline."""
    preprocessor = make_preprocessor()
    if name == "svm":
        estimator = SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    elif name == "logreg":
        estimator = LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            class_weight="balanced",
            solver="lbfgs",
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model '{name}'.")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def evaluate_split(
    model: Pipeline,
    features: pd.DataFrame,
    labels: Iterable[int],
) -> Tuple[Dict[str, float], str]:
    """Compute metrics and a classification report for a split."""
    predictions = model.predict(features)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Selected model does not support probability estimates required for metrics.")
    probabilities = model.predict_proba(features)
    metrics = get_classification_metrics(labels, predictions, probabilities)
    target_names = [ID_TO_LABEL[idx] for idx in sorted(ID_TO_LABEL)]
    ordered_labels = list(sorted(ID_TO_LABEL))
    report = classification_report(
        labels,
        predictions,
        labels=ordered_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    return metrics, report


def fit_and_evaluate(
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
) -> Dict[str, Tuple[Dict[str, float], str]]:
    """Train the model and produce metrics for requested splits."""
    model.fit(train_df[FEATURE_COLUMNS], train_df["Label"])
    results: Dict[str, Tuple[Dict[str, float], str]] = {}
    for split_name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if df is None:
            continue
        metrics, report = evaluate_split(model, df[FEATURE_COLUMNS], df["Label"])
        results[split_name] = (metrics, report)
    return results


def format_metrics(results: Dict[str, Tuple[Dict[str, float], str]]) -> str:
    """Create a readable summary of evaluation outputs."""
    lines: List[str] = []
    for split_name in sorted(results):
        metrics, report = results[split_name]
        lines.append(f"[{split_name.upper()}] metrics:")
        for key, value in metrics.items():
            lines.append(f"  - {key}: {value:.4f}")
        lines.append(report)
    return "\n".join(lines)


def default_split_paths(repo_root: Path) -> Tuple[Path, Path, Path]:
    """Return default CSV split locations."""
    base = repo_root / "configs" / "split_results"
    return (
        base / "train_split_70.csv",
        base / "val_split_10.csv",
        base / "test_split_20.csv",
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_train, default_val, default_test = default_split_paths(repo_root)
    parser = argparse.ArgumentParser(
        description="Train a traditional ML classifier on ADNI metadata splits.",
    )
    parser.add_argument("--train-csv", type=Path, default=default_train, help="Training split CSV path.")
    parser.add_argument("--val-csv", type=Path, default=default_val, help="Validation split CSV path.")
    parser.add_argument("--test-csv", type=Path, default=default_test, help="Test split CSV path.")
    parser.add_argument(
        "--model",
        choices=["svm", "logreg", "random_forest"],
        default="svm",
        help="Estimator to use.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional file to store full classification reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = load_split(args.train_csv)
    val_df = load_split(args.val_csv) if args.val_csv and args.val_csv.exists() else None
    test_df = load_split(args.test_csv) if args.test_csv and args.test_csv.exists() else None

    model = make_model(args.model)
    results = fit_and_evaluate(model, train_df, val_df, test_df)
    summary = format_metrics(results)
    print(summary)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(summary)


if __name__ == "__main__":
    main()
