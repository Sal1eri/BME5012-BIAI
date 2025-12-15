#!/usr/bin/env python3
"""Traditional machine learning baselines for AD classification using metadata or MRI slices."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

try:
    from utils.dataset import LABEL_MAP, ADNI2DDataset
except ImportError as exc:
    raise ImportError("Failed to import LABEL_MAP from utils.dataset; check PYTHONPATH" ) from exc

try:
    from utils.metric import get_classification_metrics
except ImportError as exc:
    raise ImportError("Failed to import get_classification_metrics from utils.metric; check repository structure" ) from exc

NUMERIC_FEATURES: List[str] = ["Age"]
CATEGORICAL_FEATURES: List[str] = [
    "Sex",
    "Visit",
    "Modality",
    "Description",
    "Type",
    "Format",
    "Downloaded",
]
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_MAP.items()}


def load_metadata_split(csv_path: Path) -> pd.DataFrame:
    """Load a CSV split and attach label ids."""
    df = pd.read_csv(csv_path)
    df = df[df["Group"].isin(LABEL_MAP)].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Label"] = df["Group"].map(LABEL_MAP)
    return df


def load_image_split(csv_path: Path, data_root: Path, single_slice: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load MRI slice tensors and flatten them for traditional ML."""
    dataset = ADNI2DDataset(
        csv_file=str(csv_path),
        root_dir=str(data_root),
        transform=None,
        single_slice_mode=single_slice,
    )
    features: List[np.ndarray] = []
    labels: List[int] = []
    for tensor, label in dataset:
        arr = tensor.detach().cpu().numpy()
        features.append(arr.reshape(-1))
        labels.append(int(label.item()))
    if not features:
        raise ValueError(f"No samples found for split at {csv_path} with root {data_root}.")
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def make_metadata_preprocessor() -> ColumnTransformer:
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


def make_image_preprocessor(n_components: Optional[int]) -> Pipeline:
    """Create preprocessing for flattened image arrays."""
    steps: List[Tuple[str, object]] = [
        ("scaler", StandardScaler()),
    ]
    if n_components and n_components > 0:
        steps.append(("pca", PCA(n_components=n_components, whiten=True, random_state=42)))
    return Pipeline(steps=steps)


def make_model(name: str, source: str, pca_components: Optional[int]) -> Pipeline:
    """Build the requested estimator wrapped in a preprocessing pipeline."""
    if source == "metadata":
        preprocessor = make_metadata_preprocessor()
    elif source == "image":
        preprocessor = make_image_preprocessor(pca_components)
    else:
        raise ValueError(f"Unsupported data source '{source}'.")
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
    features: Union[pd.DataFrame, np.ndarray],
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
    train_features: Union[pd.DataFrame, np.ndarray],
    train_labels: Iterable[int],
    val_data: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Iterable[int]]],
    test_data: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Iterable[int]]],
) -> Dict[str, Tuple[Dict[str, float], str]]:
    """Train the model and produce metrics for requested splits."""
    model.fit(train_features, train_labels)
    results: Dict[str, Tuple[Dict[str, float], str]] = {}
    splits = [
        ("train", (train_features, train_labels)),
        ("val", val_data),
        ("test", test_data),
    ]
    for split_name, data in splits:
        if data is None:
            continue
        features, labels = data
        metrics, report = evaluate_split(model, features, labels)
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
    repo_root = Path(__file__).resolve().parent
    default_train, default_val, default_test = default_split_paths(repo_root)
    parser = argparse.ArgumentParser(
        description="Train a traditional ML classifier on ADNI metadata splits.",
    )
    parser.add_argument("--train-csv", type=Path, default=default_train, help="Training split CSV path.")
    parser.add_argument("--val-csv", type=Path, default=default_val, help="Validation split CSV path.")
    parser.add_argument("--test-csv", type=Path, default=default_test, help="Test split CSV path.")
    parser.add_argument(
        "--source",
        choices=["metadata", "image"],
        default="image",
        help="Choose between metadata features or MRI image slices.",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "logreg", "random_forest"],
        default="svm",
        help="Estimator to use.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("dataset/ADNI/ADNI"),
        help="Dataset root directory when using image source.",
    )
    parser.add_argument(
        "--single-slice",
        action="store_true",
        help="Use single axial slice instead of three-view stack.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=256,
        help="Number of PCA components for image source (0 disables PCA).",
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
    if args.source == "metadata":
        train_df = load_metadata_split(args.train_csv)
        val_df = load_metadata_split(args.val_csv) if args.val_csv and args.val_csv.exists() else None
        test_df = load_metadata_split(args.test_csv) if args.test_csv and args.test_csv.exists() else None

        train_features = train_df[FEATURE_COLUMNS]
        train_labels = train_df["Label"].to_numpy()
        val_data = (
            val_df[FEATURE_COLUMNS],
            val_df["Label"].to_numpy(),
        ) if val_df is not None else None
        test_data = (
            test_df[FEATURE_COLUMNS],
            test_df["Label"].to_numpy(),
        ) if test_df is not None else None
        pca_components: Optional[int] = None
    else:
        data_root = args.image_root
        train_features, train_labels = load_image_split(args.train_csv, data_root, args.single_slice)
        val_data = None
        if args.val_csv and args.val_csv.exists():
            val_data = load_image_split(args.val_csv, data_root, args.single_slice)
        test_data = None
        if args.test_csv and args.test_csv.exists():
            test_data = load_image_split(args.test_csv, data_root, args.single_slice)

        if args.pca_components > 0:
            max_components = min(train_features.shape[0], train_features.shape[1])
            pca_components = min(args.pca_components, max_components)
            if pca_components < 1:
                pca_components = None
        else:
            pca_components = None

    model = make_model(args.model, args.source, pca_components)
    results = fit_and_evaluate(model, train_features, train_labels, val_data, test_data)
    summary = format_metrics(results)
    print(summary)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(summary)


if __name__ == "__main__":
    main()
