import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "5_class_10_12_2025.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "configs", "split_results")

TARGET_COLUMN = "Group"
SUBJECT_COLUMN = "Subject"
SEED = 42

# Target proportions; adjusted per class so every split has every class.
TEST_RATIO = 0.20
VAL_RATIO = 0.10


def allocate_subjects(subjects: pd.DataFrame) -> pd.DataFrame:
    """Assign each subject to train/val/test, stratified by class."""
    rng = np.random.default_rng(SEED)
    assignments = []

    for group, frame in subjects.groupby(TARGET_COLUMN):
        ids = frame[SUBJECT_COLUMN].tolist()
        rng.shuffle(ids)
        n = len(ids)

        test_n = max(1, int(round(n * TEST_RATIO)))
        val_n = max(1, int(round(n * VAL_RATIO)))
        while n - test_n - val_n < 1:
            if test_n >= val_n and test_n > 1:
                test_n -= 1
            elif val_n > 1:
                val_n -= 1
            else:
                break

        for index, subject in enumerate(ids):
            if index < test_n:
                split = "test"
            elif index < test_n + val_n:
                split = "val"
            else:
                split = "train"
            assignments.append({"Subject": subject, "Group": group, "split": split})

    return pd.DataFrame(assignments)


def perform_subject_split(csv_path, target_column, output_dir):
    """Split by subject so scans from one subject never cross splits."""
    df = pd.read_csv(csv_path)
    if target_column not in df.columns or SUBJECT_COLUMN not in df.columns:
        raise ValueError(f"CSV must contain '{SUBJECT_COLUMN}' and '{target_column}'")

    subjects = (
        df.groupby(SUBJECT_COLUMN)[target_column]
        .agg(lambda values: values.mode().iloc[0])
        .reset_index()
    )
    assignments = allocate_subjects(subjects)

    splits = {}
    for split in ["train", "val", "test"]:
        ids = assignments.loc[assignments["split"] == split, SUBJECT_COLUMN]
        splits[split] = df[df[SUBJECT_COLUMN].isin(ids)]
        part = splits[split]
        print(
            f"{split}: {len(part)} scans, {part[SUBJECT_COLUMN].nunique()} subjects, "
            f"{part[target_column].value_counts().sort_index().to_dict()}"
        )

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(splits[left][SUBJECT_COLUMN]) & set(splits[right][SUBJECT_COLUMN])

    os.makedirs(output_dir, exist_ok=True)
    splits["train"].to_csv(os.path.join(output_dir, "train_split_70.csv"), index=False)
    splits["val"].to_csv(os.path.join(output_dir, "val_split_10.csv"), index=False)
    splits["test"].to_csv(os.path.join(output_dir, "test_split_20.csv"), index=False)
    print(f"Saved subject-level splits to: {output_dir}")


if __name__ == "__main__":
    perform_subject_split(
        csv_path=CSV_PATH,
        target_column=TARGET_COLUMN,
        output_dir=OUTPUT_DIR,
    )
