"""Customer churn prediction - Phase 2 machine-learning workflow.

The script downloads the public Kaggle dataset, prepares the data, explores
PCA, compares multiple classifiers, tunes Random Forest, evaluates the models,
and saves report-ready plots and tables in the outputs directory.
"""

# =============================================================================
# 1. CONFIGURATION AND IMPORTS
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path

import joblib
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 23
TEST_SIZE = 0.20
TUNING_SAMPLE_SIZE = 80_000
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


# =============================================================================
# 2. DATASET DOWNLOAD AND INTEGRATION
# =============================================================================

print("Downloading the public Kaggle dataset...")
dataset_directory = Path(
    kagglehub.dataset_download(
        "muhammadshahidazeem/customer-churn-dataset"
    )
)

training_files = list(dataset_directory.rglob("*training*.csv"))
testing_files = list(dataset_directory.rglob("*testing*.csv"))

if not training_files or not testing_files:
    raise FileNotFoundError(
        "The expected training and testing CSV files were not found."
    )

training_df = pd.read_csv(training_files[0])
testing_df = pd.read_csv(testing_files[0])

# The supplied files contain the same variables. They are combined before a
# reproducible 80:20 split matching the analysis-notebook methodology.
df = pd.concat([training_df, testing_df], ignore_index=True)
print(f"Original combined shape: {df.shape}")


# =============================================================================
# 3. DATA CLEANING AND INITIAL ASSESSMENT
# =============================================================================

df.columns = [column.strip().lower().replace(" ", "_") for column in df.columns]

missing_rows = int(df.isna().any(axis=1).sum())
duplicate_rows = int(df.duplicated().sum())

# CustomerID is only an identifier and is not a meaningful churn predictor.
df = df.drop(columns="customerid")
df = df.dropna().drop_duplicates().copy()

integer_columns = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "last_interaction",
    "churn",
]
df[integer_columns] = df[integer_columns].astype(int)

print(f"Rows containing missing values: {missing_rows}")
print(f"Duplicate rows: {duplicate_rows}")
print(f"Cleaned shape: {df.shape}")
print("\nTarget distribution:")
print(df["churn"].value_counts().sort_index())


# =============================================================================
# 4. TARGET-DISTRIBUTION VISUALISATION
# =============================================================================

churn_counts = df["churn"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.barplot(
    x=["Not Churned", "Churned"],
    y=churn_counts.values,
    color="#36688D",
    ax=ax,
)

for index, value in enumerate(churn_counts.values):
    ax.text(
        index,
        value + 4_000,
        f"{value:,}\n({value / len(df) * 100:.2f}%)",
        ha="center",
        fontweight="bold",
    )

ax.set_title("Distribution of the Target Variable")
ax.set_xlabel("Customer Status")
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, max(churn_counts.values) * 1.15)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 5. EXPLORATORY AND MULTIVARIATE ANALYSIS
# =============================================================================

correlation_columns = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
    "churn",
]
correlation_matrix = df[correlation_columns].corr()

print("\nNumerical feature correlations with churn:")
print(correlation_matrix["churn"].drop("churn").sort_values(ascending=False).round(3))

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Between Numerical Variables")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "numerical_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

categorical_analysis_columns = ["gender", "subscription_type", "contract_length"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for axis, column in zip(axes, categorical_analysis_columns):
    churn_rate = df.groupby(column)["churn"].mean().sort_values()
    axis.bar(churn_rate.index, churn_rate.values, color="#36688D")

    for index, value in enumerate(churn_rate.values):
        axis.text(
            index,
            value + 0.01,
            f"{value:.1%}",
            ha="center",
            fontweight="bold",
        )

    axis.set_title(column.replace("_", " ").title())
    axis.set_xlabel("")
    axis.set_ylabel("Churn Rate")
    axis.set_ylim(0, min(1, max(churn_rate.values) + 0.12))
    axis.tick_params(axis="x", rotation=20)

fig.suptitle(
    "Churn Rate Across Categorical Customer Groups",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "categorical_churn_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# 6. TRAINING AND TESTING SPLIT
# =============================================================================

X = df.drop(columns="churn")
y = df["churn"]

# This split reproduces the analysis notebook, including its fixed random state,
# so the baseline Random Forest result remains comparable.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Training observations: {len(X_train):,}")
print(f"Testing observations: {len(X_test):,}")

categorical_features = ["gender", "subscription_type", "contract_length"]
numerical_features = [
    "age",
    "tenure",
    "usage_frequency",
    "support_calls",
    "payment_delay",
    "total_spend",
    "last_interaction",
]


# =============================================================================
# 7. PREPROCESSING PIPELINES
# =============================================================================

# Logistic Regression benefits from standardised numerical variables.
logistic_preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# Scaling is unnecessary for tree-based Random Forest models.
forest_preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", "passthrough", numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)


# =============================================================================
# 8. PCA EXPLORATION
# =============================================================================

# PCA is used as an exploratory visualisation rather than as the final model
# input. The dataset has a manageable number of interpretable variables, so
# retaining the original features is preferable for business interpretation.
pca_preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

pca_sample = X_train.sample(
    n=min(10_000, len(X_train)),
    random_state=RANDOM_STATE,
)
pca_target = y_train.loc[pca_sample.index]
pca_features = pca_preprocessor.fit_transform(pca_sample)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_coordinates = pca.fit_transform(pca_features)

fig, ax = plt.subplots(figsize=(7, 5.5))
scatter = ax.scatter(
    pca_coordinates[:, 0],
    pca_coordinates[:, 1],
    c=pca_target,
    cmap="coolwarm",
    alpha=0.35,
    s=12,
)
ax.set_title("PCA Projection of Customer Records")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
legend = ax.legend(*scatter.legend_elements(), title="Churn")
ax.add_artist(legend)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "pca_projection.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(
    "PCA explained variance ratio:",
    np.round(pca.explained_variance_ratio_, 4),
)


# =============================================================================
# 9. BASELINE MODEL TRAINING
# =============================================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", logistic_preprocessor),
        (
            "model",
            LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
        ),
    ]
)

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", forest_preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

decision_tree_model = Pipeline(
    steps=[
        ("preprocessor", forest_preprocessor),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]
)

xgboost_model = Pipeline(
    steps=[
        ("preprocessor", forest_preprocessor),
        (
            "model",
            XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            ),
        ),
    ]
)

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)

print("Training baseline Random Forest...")
random_forest_model.fit(X_train, y_train)

print("Training Decision Tree...")
decision_tree_model.fit(X_train, y_train)

print("Training XGBoost...")
xgboost_model.fit(X_train, y_train)


# =============================================================================
# 10. RANDOM FOREST HYPERPARAMETER TUNING
# =============================================================================

# Tuning is performed on a representative stratified sample to keep the search
# practical on free Colab hardware. The best configuration is then refitted on
# the complete training dataset.
if len(X_train) > TUNING_SAMPLE_SIZE:
    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=TUNING_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
else:
    X_tune, y_tune = X_train, y_train

tuning_pipeline = Pipeline(
    steps=[
        ("preprocessor", forest_preprocessor),
        (
            "model",
            RandomForestClassifier(random_state=42, n_jobs=-1),
        ),
    ]
)

parameter_distributions = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 12, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": [None, "balanced"],
}

cross_validation = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=RANDOM_STATE,
)

search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=parameter_distributions,
    n_iter=8,
    scoring="recall",
    cv=cross_validation,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

print(f"\nTuning Random Forest on {len(X_tune):,} observations...")
search.fit(X_tune, y_tune)
print("Best parameters:")
print(search.best_params_)
print(f"Best mean cross-validation recall: {search.best_score_:.4f}")

tuning_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
tuning_columns = ["rank_test_score", "mean_test_score", "std_test_score", "params"]
tuning_results[tuning_columns].head(10).to_csv(
    OUTPUT_DIR / "tuning_search_results.csv",
    index=False,
)

print("\nTop hyperparameter combinations:")
print(tuning_results[tuning_columns].head(5).to_string(index=False))

# Refit the selected pipeline using the complete training dataset.
tuned_random_forest = search.best_estimator_
tuned_random_forest.fit(X_train, y_train)


# =============================================================================
# 11. MODEL EVALUATION AND COMPARISON
# =============================================================================

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model,
    "Tuned Random Forest": tuned_random_forest,
}


def safe_filename(model_name: str) -> str:
    """Convert a model name into a stable output filename."""
    return model_name.lower().replace(" ", "_")


def evaluate_model(model_name: str, model: Pipeline) -> tuple[dict, np.ndarray]:
    """Calculate metrics and save a report-ready confusion matrix."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        predictions,
        target_names=["Not Churned", "Churned"],
    )
    print(f"\nClassification Report - {model_name}")
    print(report)

    with open(
        OUTPUT_DIR / f"classification_report_{safe_filename(model_name)}.txt",
        "w",
        encoding="utf-8",
    ) as report_file:
        report_file.write(report)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Not Churned", "Churned"],
        cmap="Blues",
        values_format=",d",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"confusion_matrix_{safe_filename(model_name)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return metrics, probabilities


evaluation_rows = []
model_probabilities = {}

for name, model in models.items():
    row, probabilities = evaluate_model(name, model)
    evaluation_rows.append(row)
    model_probabilities[name] = probabilities

comparison_df = pd.DataFrame(evaluation_rows)
metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

print("\nModel comparison:")
print(comparison_df.to_string(index=False, formatters={
    column: "{:.4f}".format for column in metric_columns
}))


# =============================================================================
# 12. ROC AND PRECISION-RECALL CURVES
# =============================================================================

fig, ax = plt.subplots(figsize=(7, 5.5))
for name, probabilities in model_probabilities.items():
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    model_auc = roc_auc_score(y_test, probabilities)
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"{name} (AUC = {model_auc:.3f})",
    )

ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random classifier")
ax.set_title("ROC Curve Comparison")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5.5))
for name, probabilities in model_probabilities.items():
    precision_values, recall_values, _ = precision_recall_curve(
        y_test,
        probabilities,
    )
    pr_auc = auc(recall_values, precision_values)
    ax.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"{name} (AUC = {pr_auc:.3f})",
    )

ax.set_title("Precision-Recall Curve Comparison")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 13. RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================

feature_names = tuned_random_forest.named_steps[
    "preprocessor"
].get_feature_names_out()
feature_importance = tuned_random_forest.named_steps["model"].feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": feature_importance,
    }
).sort_values("Importance", ascending=False)

importance_df["Feature"] = (
    importance_df["Feature"]
    .str.replace("numerical__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
    .str.replace("_", " ")
    .str.title()
)

top_features = importance_df.head(12).sort_values("Importance")
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_features["Feature"], top_features["Importance"], color="#36688D")
ax.set_title("Tuned Random Forest - Feature Importance")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "random_forest_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# =============================================================================
# 14. SAVE TABLES, PARAMETERS, AND FINAL MODEL
# =============================================================================

comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

serialisable_parameters = {
    key: value for key, value in search.best_params_.items()
}
with open(OUTPUT_DIR / "best_parameters.json", "w", encoding="utf-8") as file:
    json.dump(serialisable_parameters, file, indent=2)

joblib.dump(
    tuned_random_forest,
    OUTPUT_DIR / "tuned_random_forest.joblib",
)

print(f"\nAnalysis completed. Report-ready outputs were saved to: {OUTPUT_DIR.resolve()}")
