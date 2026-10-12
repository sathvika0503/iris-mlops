
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "iris.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "iris_model.joblib"))

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
TARGET_COLUMN = "species"
EXPECTED_LABELS = {"setosa", "versicolor", "virginica"}


def load_and_validate_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    df = df[required].copy()

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype("string").str.strip().str.lower()
    )

    df = df.dropna(subset=required)
    df = df[df[TARGET_COLUMN].isin(EXPECTED_LABELS)]

    if len(df) < 15:
        raise ValueError(f"Need at least 15 valid rows; found {len(df)}.")

    counts = df[TARGET_COLUMN].value_counts()
    missing_labels = EXPECTED_LABELS - set(counts.index)

    if missing_labels:
        raise ValueError(
            f"Dataset must contain all 3 Iris labels. "
            f"Missing: {sorted(missing_labels)}"
        )

    if counts.min() < 2:
        raise ValueError("Need at least 2 rows for each label.")

    return df, counts.to_dict()


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=42,
            )),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),
    }


def train_model():
    df, class_counts = load_and_validate_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Keep a separate holdout test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Cross-validation folds must not exceed the smallest training class.
    smallest_class = int(y_train.value_counts().min())
    n_splits = min(5, smallest_class)

    if n_splits < 2:
        raise ValueError(
            "Not enough training samples per class for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://mlflow:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-mlops-starter")

    results = []
    models = build_models()

    # Compare candidates using the same training folds.
    for name, candidate in models.items():
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        results.append({
            "name": name,
            "mean_cv_accuracy": float(scores.mean()),
            "std_cv_accuracy": float(scores.std()),
        })

    # Choose the highest mean cross-validation accuracy.
    results.sort(
        key=lambda item: item["mean_cv_accuracy"],
        reverse=True,
    )

    best_name = results[0]["name"]
    best_cv_accuracy = results[0]["mean_cv_accuracy"]
    best_cv_std = results[0]["std_cv_accuracy"]

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # Evaluate once on the held-out test set.
    predictions = best_model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, predictions))

    labels = sorted(EXPECTED_LABELS)

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    # Save the model in the format expected by the API.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = MODEL_PATH.with_suffix(".tmp")
    joblib.dump(
        {
            "model": best_model,
            "features": FEATURE_COLUMNS,
            "labels": sorted(EXPECTED_LABELS),
        },
        temp_path,
    )
    temp_path.replace(MODEL_PATH)

    # Log the selected model and comparison results in MLflow.
    with mlflow.start_run(run_name=f"best-{best_name}"):
        mlflow.log_param("selected_algorithm", best_name)
        mlflow.log_param("rows_used", len(df))
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("cv_folds", n_splits)

        mlflow.log_metric("best_mean_cv_accuracy", best_cv_accuracy)
        mlflow.log_metric("best_cv_std", best_cv_std)
        mlflow.log_metric("test_accuracy", test_accuracy)

        for result in results:
            mlflow.log_metric(
                f"{result['name']}_cv_mean",
                result["mean_cv_accuracy"],
            )
            mlflow.log_metric(
                f"{result['name']}_cv_std",
                result["std_cv_accuracy"],
            )

        mlflow.log_dict(
            {
                "class_counts": class_counts,
                "features": FEATURE_COLUMNS,
                "cross_validation_results": results,
                "classification_report": report,
            },
            "training_summary.json",
        )

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
        )

    return {
    "status": "trained",
    "selected_algorithm": best_name,
    "rows_used": int(len(df)),
    "class_counts": class_counts,
    "cross_validation_accuracy": round(best_cv_accuracy, 4),
    "cross_validation_std": round(best_cv_std, 4),
    "test_accuracy": round(test_accuracy, 4),
    "model_comparison": results,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_labels": labels,
    "classification_report": report,
    "model_path": str(MODEL_PATH),
}


if __name__ == "__main__":
    print(train_model())
