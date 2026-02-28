import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CV_FOLDS = 5
N_REPEATS = 3  # repeated stratified k-fold for stability

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": "roc_auc",
}

COLUMN_RENAME_MAP = {
    "cp": "chest_pain_type",
    "trestbps": "resting_blood_pressure",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "oldpeak": "st_depression",
    "ca": "num_major_vessels",
    "thal": "thalassemia",
}

MODEL_NAMES = [
    "K-NN (tuned k)",
    "K-NN (GridSearchCV)",
    "Logistic Regression",
    "LR (feature select)",
    "SVM (RBF)",
    "Random Forest",
]

# Columns to drop for the feature-selected LR (Nicholas's approach).
# Listed under both raw and renamed variants so it works on every dataset.
NICHOLAS_DROP_COLS = {
    "fbs", "fasting_blood_sugar", "Fasting blood sugar",
    "restecg", "resting_ecg", "Resting ECG",
}

# ---------------------------------------------------------------------------
# Custom transformer for feature-selected LR (Nicholas's approach)
# ---------------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop columns by name inside a Pipeline. Ignores names not present."""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            to_drop = [c for c in X.columns if c in self.columns_to_drop]
            return X.drop(columns=to_drop)
        return X


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_datasets():
    """Return an ordered dict of {label: (X, y)} for every dataset."""
    datasets = {}

    # 1. Raw
    raw = pd.read_excel("data/heart.xlsx").drop_duplicates()
    datasets["Raw"] = (raw.drop(columns=["target"]), raw["target"])

    # 2. Team Cleaned
    team = pd.read_csv("data/cleaned_heart_data (1).csv")
    if "Unnamed: 0" in team.columns:
        team = team.drop(columns=["Unnamed: 0"])
    datasets["Cleaned Data 1"] = (team.drop(columns=["target"]), team["target"])

    # 3. Enes Cleaned
    enes = pd.read_csv("data/enes_final_cleaned_data.csv")
    datasets["Cleaned Data 2"] = (enes.drop(columns=["target"]), enes["target"])

    # 4. Script Cleaned (generated from heart.xlsx)
    script = pd.read_excel("data/heart.xlsx").drop_duplicates()
    script = script.rename(columns=COLUMN_RENAME_MAP)
    datasets["Cleaned Data 3"] = (script.drop(columns=["target"]),  script["target"])

    return datasets


# ---------------------------------------------------------------------------
# Model builders (as pipelines so scaling is inside each CV fold)
# ---------------------------------------------------------------------------
def _find_best_k(X, y, cv):
    """Find optimal k for KNN via CV over k=1..20."""
    best_k, best_score = 1, 0
    for k in range(1, 21):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_validate(pipe, X, y, cv=cv, scoring="accuracy")
        mean = scores["test_score"].mean()
        if mean > best_score:
            best_k, best_score = k, mean
    return best_k


def _find_best_knn_grid(X, y, cv):
    """Find best KNN params via GridSearchCV."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    grid_params = {
        "knn__n_neighbors": list(range(1, 21)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan", "minkowski"],
    }
    grid = GridSearchCV(pipe, grid_params, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X, y)
    return grid.best_params_


def build_pipelines(X, y, cv):
    """Return {model_name: Pipeline} with scaling built in."""
    print("  Tuning K-NN (best k)...")
    best_k = _find_best_k(X, y, cv)
    print(f"    -> best k = {best_k}")

    print("  Tuning K-NN (GridSearchCV)...")
    best_grid = _find_best_knn_grid(X, y, cv)
    print(f"    -> best params = {best_grid}")

    pipelines = {}

    pipelines["K-NN (tuned k)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=best_k)),
    ])

    pipelines["K-NN (GridSearchCV)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=best_grid["knn__n_neighbors"],
            weights=best_grid["knn__weights"],
            metric=best_grid["knn__metric"],
        )),
    ])

    pipelines["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # Nicholas's approach: LR with fasting_blood_sugar and resting_ecg dropped
    pipelines["LR (feature select)"] = Pipeline([
        ("drop", ColumnDropper(NICHOLAS_DROP_COLS)),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    pipelines["SVM (RBF)"] = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE,
        )),
    ])

    pipelines["Random Forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
        )),
    ])

    return pipelines


# ---------------------------------------------------------------------------
# Evaluation via Repeated Stratified K-Fold
# ---------------------------------------------------------------------------
def evaluate_all(pipelines, X, y, dataset_label):
    """Run repeated stratified k-fold and return results + per-fold data."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    rskf = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )

    rows = []
    fold_data = {}  # {model_name: cv_results dict}

    for model_name in MODEL_NAMES:
        pipe = pipelines[model_name]
        cv_results = cross_validate(
            pipe, X, y, cv=rskf, scoring=SCORING,
            return_train_score=False, n_jobs=-1,
        )
        fold_data[model_name] = cv_results

        rows.append({
            "Model": model_name,
            "Dataset": dataset_label,
            "Accuracy": cv_results["test_accuracy"].mean(),
            "Accuracy Std": cv_results["test_accuracy"].std(),
            "Precision": cv_results["test_precision"].mean(),
            "Recall": cv_results["test_recall"].mean(),
            "F1": cv_results["test_f1"].mean(),
            "F1 Std": cv_results["test_f1"].std(),
            "ROC-AUC": cv_results["test_roc_auc"].mean(),
            "ROC-AUC Std": cv_results["test_roc_auc"].std(),
        })

    return rows, fold_data


# ---------------------------------------------------------------------------
# Visualisation -- Grouped bar chart
# ---------------------------------------------------------------------------
def plot_grouped_bar(df, save_path="results/comparison_by_model.png"):
    """Grouped bar chart -- accuracy by model, grouped by dataset."""
    model_names = df["Model"].unique()
    dataset_names = df["Dataset"].unique()
    n_models = len(model_names)
    n_datasets = len(dataset_names)
    x = np.arange(n_models)
    width = 0.8 / n_datasets

    # Find the overall best model (highest mean accuracy across all datasets)
    best_model = df.groupby("Model")["Accuracy"].mean().idxmax()

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, ds in enumerate(dataset_names):
        subset = df[df["Dataset"] == ds]
        accs = [subset[subset["Model"] == m]["Accuracy"].values[0] * 100
                for m in model_names]
        stds = [subset[subset["Model"] == m]["Accuracy Std"].values[0] * 100
                for m in model_names]
        bars = ax.bar(x + i * width, accs, width, label=ds, yerr=stds,
                      capsize=3, error_kw={"linewidth": 0.8})
        for bar, acc, m in zip(bars, accs, model_names):
            fontw = "bold" if m == best_model else "normal"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.1f}", ha="center", va="bottom", fontsize=7,
                    fontweight=fontw)
            # Star marker on the best model's bars
            if m == best_model:
                ax.plot(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5, "*",
                        color="gold", markersize=8, markeredgecolor="black",
                        markeredgewidth=0.5)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%) -- mean +/- std over 15 folds")
    ax.set_title("Accuracy by Model (Repeated Stratified 5-Fold CV x3)")
    ax.set_xticks(x + width * (n_datasets - 1) / 2)
    xlabels = [f"** {m} **" if m == best_model else m for m in model_names]
    ax.set_xticklabels(xlabels, rotation=15, ha="right")
    for tick_label, m in zip(ax.get_xticklabels(), model_names):
        if m == best_model:
            tick_label.set_fontweight("bold")
    ax.legend(title="Dataset")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Visualisation -- Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(df, save_path="results/comparison_heatmap.png"):
    """Heatmap of accuracy -- rows=models, columns=datasets."""
    pivot = df.pivot(index="Model", columns="Dataset", values="Accuracy") * 100
    # Reorder rows to match MODEL_NAMES
    pivot = pivot.reindex(MODEL_NAMES)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot, annot=False, cmap="RdYlGn",
        linewidths=0.5, ax=ax, vmin=60, vmax=100,
    )

    # Annotate cells, bolding the best value per column (dataset)
    for col_idx, ds in enumerate(pivot.columns):
        best_val = pivot[ds].max()
        for row_idx, model in enumerate(pivot.index):
            val = pivot.loc[model, ds]
            is_best = abs(val - best_val) < 0.01
            ax.text(col_idx + 0.5, row_idx + 0.5, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=11 if is_best else 10,
                    fontweight="bold" if is_best else "normal",
                    color="black")
            if is_best:
                ax.add_patch(plt.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=False, edgecolor="gold", linewidth=3,
                ))

    ax.set_title("Mean Accuracy (%) -- Model x Dataset  [gold border = best per dataset]")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Visualisation -- ROC Curves (one subplot per dataset)
# ---------------------------------------------------------------------------
def plot_roc_curves(all_pipelines, datasets, save_path="results/comparison_roc.png"):
    """Plot ROC curves for each model, one subplot per dataset."""
    from sklearn.model_selection import StratifiedKFold

    dataset_labels = list(datasets.keys())
    n_ds = len(dataset_labels)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for idx, ds_label in enumerate(dataset_labels):
        ax = axes[idx]
        X, y = datasets[ds_label]
        pipelines = all_pipelines[ds_label]

        # Collect all model AUCs first to find the best
        roc_data = {}
        for model_name in MODEL_NAMES:
            pipe = pipelines[model_name]
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)

            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                pipe.fit(X_train, y_train)
                viz = RocCurveDisplay.from_estimator(
                    pipe, X_test, y_test, ax=ax, alpha=0, name="_nolegend_",
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)
                viz.line_.remove()

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            roc_data[model_name] = (mean_fpr, mean_tpr, mean_auc)

        best_model = max(roc_data, key=lambda m: roc_data[m][2])

        for model_name, (fpr, tpr, auc) in roc_data.items():
            is_best = model_name == best_model
            lw = 3.0 if is_best else 1.2
            line_label = f"{model_name} (AUC={auc:.3f})"
            if is_best:
                line_label += " *BEST*"
            ax.plot(fpr, tpr, label=line_label, linewidth=lw,
                    alpha=1.0 if is_best else 0.7)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
        ax.set_title(f"ROC -- {ds_label}")
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Visualisation -- Confusion Matrix Grid
# ---------------------------------------------------------------------------
def plot_confusion_matrices(all_pipelines, datasets, results_df,
                            save_path="results/comparison_confusion.png"):
    """Grid of confusion matrices: rows=datasets, columns=models."""
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    dataset_labels = list(datasets.keys())
    n_ds = len(dataset_labels)
    n_models = len(MODEL_NAMES)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Find the best model per dataset by accuracy
    best_per_ds = {}
    for ds in dataset_labels:
        ds_rows = results_df[results_df["Dataset"] == ds]
        best_per_ds[ds] = ds_rows.loc[ds_rows["Accuracy"].idxmax(), "Model"]

    fig, axes = plt.subplots(n_ds, n_models, figsize=(3.5 * n_models, 3.5 * n_ds))

    for row, label in enumerate(dataset_labels):
        X, y = datasets[label]
        pipelines = all_pipelines[label]

        for col, model_name in enumerate(MODEL_NAMES):
            ax = axes[row][col] if n_ds > 1 else axes[col]
            pipe = pipelines[model_name]
            y_pred = cross_val_predict(pipe, X, y, cv=cv)
            cm = confusion_matrix(y, y_pred)
            is_best = best_per_ds[label] == model_name
            cmap = "YlOrRd" if is_best else "Blues"
            ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"]).plot(
                ax=ax, cmap=cmap, colorbar=False,
            )
            title_text = model_name
            if row == 0:
                if is_best:
                    title_text += " *BEST*"
                ax.set_title(title_text, fontsize=9,
                             fontweight="bold" if is_best else "normal")
            else:
                ax.set_title("")
            if col == 0:
                ax.set_ylabel(f"{label}\nTrue label", fontsize=9)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("" if row < n_ds - 1 else "Predicted", fontsize=8)
            # Gold border for best
            if is_best:
                for spine in ax.spines.values():
                    spine.set_edgecolor("gold")
                    spine.set_linewidth(3)

    plt.suptitle("Confusion Matrices (Stratified 5-Fold CV Predictions)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    datasets = load_datasets()
    all_results = []
    all_pipelines = {}  # {dataset_label: {model_name: Pipeline}}

    for label, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {label}  (samples={len(y)}, features={X.shape[1]})")
        print(f"{'='*60}")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
        pipelines = build_pipelines(X, y, cv)
        all_pipelines[label] = pipelines

        results, _ = evaluate_all(pipelines, X, y, label)
        all_results.extend(results)

        for r in results:
            print(f"  {r['Model']:25s}  "
                  f"Acc={r['Accuracy']:.4f}+/-{r['Accuracy Std']:.4f}  "
                  f"F1={r['F1']:.4f}  "
                  f"AUC={r['ROC-AUC']:.4f}+/-{r['ROC-AUC Std']:.4f}")

    results_df = pd.DataFrame(all_results)

    # Format display columns
    display_cols = ["Model", "Dataset", "Accuracy", "Accuracy Std",
                    "Precision", "Recall", "F1", "F1 Std",
                    "ROC-AUC", "ROC-AUC Std"]
    print(f"\n{'='*60}")
    print("FULL RESULTS TABLE  (Repeated Stratified 5-Fold CV x3)")
    print(f"{'='*60}")
    print(results_df[display_cols].to_string(index=False, float_format="%.4f"))

    # Generate all plots
    plot_grouped_bar(results_df)
    plot_heatmap(results_df)
    plot_roc_curves(all_pipelines, datasets)
    plot_confusion_matrices(all_pipelines, datasets, results_df)
    plt.show()


if __name__ == "__main__":
    main()
