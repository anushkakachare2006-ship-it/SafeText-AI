"""Train and evaluate the SafeText AI binary text classifier."""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "hate_speech_pipeline.joblib"
LABELS = ("hate_speech", "not_hate_speech")


def normalize_label(value: object) -> str:
    """Normalize numeric and string labels into comparable CLI values."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def clean_text(value: object) -> str:
    """Normalize tweet markup while retaining words such as negations."""
    text = html.unescape(str(value)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_labels(value: str) -> set[str]:
    return {normalize_label(item) for item in value.split(",") if item.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "labeled_data.csv",
        help="CSV dataset path (default: labeled_data.csv in the project root).",
    )
    parser.add_argument("--text-column", default="tweet", help="Column containing text.")
    parser.add_argument("--target-column", default="class", help="Column containing labels.")
    parser.add_argument(
        "--hate-labels",
        default="0",
        help="Comma-separated original labels meaning hate speech (verified default: 0).",
    )
    parser.add_argument(
        "--not-hate-labels",
        default="1,2",
        help="Comma-separated original labels grouped as not hate (verified default: 1,2).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the fitted pipeline.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hate_labels = parse_labels(args.hate_labels)
    not_hate_labels = parse_labels(args.not_hate_labels)

    if not hate_labels or not not_hate_labels or hate_labels & not_hate_labels:
        raise SystemExit("Hate and not-hate label sets must both be non-empty and disjoint.")
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be greater than 0 and less than 1.")
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}. Supply the real CSV with --data.")

    dataset = pd.read_csv(args.data)
    missing_columns = {args.text_column, args.target_column} - set(dataset.columns)
    if missing_columns:
        columns = ", ".join(sorted(map(str, missing_columns)))
        raise SystemExit(f"Dataset is missing required column(s): {columns}.")

    print(f"Dataset: {args.data.resolve()}")
    print(f"Columns: text={args.text_column!r}, target={args.target_column!r}")
    print(f"Loaded records: {len(dataset)}")
    print("Original target distribution:")
    print(dataset[args.target_column].value_counts(dropna=False).to_string())

    data = dataset[[args.text_column, args.target_column]].copy()
    data[args.text_column] = data[args.text_column].fillna("").astype(str).str.strip()
    data["_label"] = data[args.target_column].map(normalize_label)
    valid_labels = hate_labels | not_hate_labels
    valid = data[args.text_column].ne("") & data["_label"].isin(valid_labels)
    invalid_count = int((~valid).sum())
    data = data.loc[valid, [args.text_column, "_label"]]
    before_deduplication = len(data)
    data = data.drop_duplicates(subset=[args.text_column, "_label"])
    duplicate_count = before_deduplication - len(data)
    data["_binary_label"] = data["_label"].map(
        lambda label: "hate_speech" if label in hate_labels else "not_hate_speech"
    )
    data[args.text_column] = data[args.text_column].map(clean_text)
    data = data.loc[data[args.text_column].ne("")]

    print(f"Removed missing/empty text or unmapped labels: {invalid_count}")
    print(f"Removed duplicate text/label records: {duplicate_count}")
    print(f"Valid records after cleaning: {len(data)}")
    print("Binary target distribution:")
    print(data["_binary_label"].value_counts().to_string())

    if set(data["_binary_label"].unique()) != set(LABELS):
        raise SystemExit("Training requires at least one example from both binary classes.")

    features = data[args.text_column]
    targets = data["_binary_label"]
    class_counts = targets.value_counts()
    test_count = math.ceil(len(data) * args.test_size)
    train_count = len(data) - test_count
    use_stratification = (
        class_counts.min() >= 2
        and test_count >= len(LABELS)
        and train_count >= len(LABELS)
    )
    if not use_stratification:
        print("Stratified split is not supported by these counts; using a regular split.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=args.test_size,
        random_state=42,
        stratify=targets if use_stratification else None,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print(f"Training records: {len(x_train)}")
    print(f"Test records: {len(x_test)}")
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=LABELS, zero_division=0))
    print(f"Macro F1-score: {f1_score(y_test, predictions, labels=LABELS, average='macro'):.4f}")
    print("Confusion matrix (hate_speech, not_hate_speech):")
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)
    print(f"Saved trained pipeline: {args.model_output.resolve()}")


if __name__ == "__main__":
    main()