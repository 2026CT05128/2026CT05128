"""
Adult Census Income Classification
Run: python model/train_models.py

Downloads the UCI Adult dataset, preprocesses it without data leakage,
trains five classifiers, evaluates them, and saves models/results/plots.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "saved_models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def load_dataset():
    """Fetch Adult dataset (UCI dataset ID 2)."""
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # Normalize column names and missing-value markers.
    X.columns = [str(c).strip().replace("-", "_") for c in X.columns]
    y.columns = [str(c).strip().lower() for c in y.columns]
    X = X.replace(r"^\s*\?\s*$", np.nan, regex=True)
    X = X.replace(r"^\s*$", np.nan, regex=True)

    target_col = y.columns[0]
    y = y.iloc[:, 0].astype(str).str.strip()
    y = y.str.replace(r"\.$", "", regex=True)
    y.name = "income"

    # Remove duplicate feature rows while keeping X and y aligned.
    combined = X.copy()
    combined["income"] = y.values
    before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    removed = before - len(combined)
    X = combined.drop(columns="income")
    y = combined["income"]

    # Normalize target labels in the standard UCI representation.
    y = y.str.replace(r"\.$", "", regex=True).str.strip()
    return X, y, removed

def main():
    X, y, duplicates_removed = load_dataset()

    print("\nDATASET OVERVIEW")
    print("----------------")
    print("Rows:", len(X))
    print("Input features:", X.shape[1])
    print("Duplicates removed:", duplicates_removed)
    print("\nFeature types:\n", X.dtypes)
    print("\nMissing values:\n", X.isna().sum().sort_values(ascending=False))
    print("\nClass distribution:\n", y.value_counts())
    print("\nClass percentages:\n", (y.value_counts(normalize=True) * 100).round(2))

    # The original target values are <=50K and >50K.
    positive_label = ">50K"
    if positive_label not in set(y.unique()):
        raise ValueError(f"Expected target label '>50K'. Found: {sorted(y.unique())}")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Save held-out test data for repository reproducibility.
    test_data = X_test.copy()
    test_data["income"] = y_test.values
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_split=10, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=5,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    rows = []
    fitted_models = {}

    y_test_binary = (y_test == positive_label).astype(int)

    for name, estimator in models.items():
        print(f"\nTraining {name} ...")
        # A fresh preprocessing pipeline is fitted only on X_train.
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator)
        ])
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        classes = list(pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(positive_label)
        positive_probabilities = probabilities[:, positive_index]

        metrics = {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test_binary, positive_probabilities),
            "Precision": precision_score(
                y_test, predictions, pos_label=positive_label, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, pos_label=positive_label, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, pos_label=positive_label, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, predictions)
        }
        rows.append(metrics)
        fitted_models[name] = pipeline

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.joblib")
        with open(OUTPUT_DIR / f"{safe_name}_classification_report.txt", "w") as f:
            f.write(classification_report(y_test, predictions, zero_division=0))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, ax=ax, cmap="Blues", xticks_rotation=30
        )
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)

    results = pd.DataFrame(rows).set_index("ML Model Name")
    results.to_csv(OUTPUT_DIR / "evaluation_metrics.csv")
    print("\nEVALUATION METRICS")
    print(results.round(4).to_string())

    # One graph showing all six required metrics for all five models.
    plot_data = results.reset_index().melt(
        id_vars="ML Model Name", var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=plot_data, x="Metric", y="Score",
                hue="ML Model Name", ax=ax)
    ax.set_ylim(-1, 1.05)
    ax.set_title("Classification Model Comparison")
    ax.set_ylabel("Metric score")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "all_metrics_comparison.png", dpi=180)
    plt.close(fig)

    # A separate heatmap makes the numeric comparison easier to read.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(results, annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Evaluation Metrics Heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_heatmap.png", dpi=180)
    plt.close(fig)

    winner = results["F1"].idxmax()
    print(f"\nHighest F1 score: {winner}")
    print("Saved models in:", MODEL_DIR)
    print("Saved metrics and graphs in:", OUTPUT_DIR)
    print("Saved held-out test data:", ROOT / "test_data.csv")
    print("\nInterpretation reminder: compare AUC, F1 and MCC as well as accuracy.")
    print("Use the generated values, not assumed values, in your final report.")

if __name__ == "__main__":
    main()
