"""
SignSpeak - STEP 2: train and compare the classifiers.

Trains the three models proposed in the synopsis on the landmark CSV:
    1. Random Forest
    2. Support Vector Machine (RBF kernel)
    3. Feedforward Neural Network (MLP)   <- optional comparison model

For each model it reports 5-fold cross-validation, hold-out accuracy, macro F1
and prediction latency, then saves:
    models/signspeak_model.joblib     best model (re-fitted on all data)
    reports/model_comparison.png      bar chart of the three models
    reports/confusion_matrix.png      confusion matrix of the best model
    reports/classification_report.txt per-sign precision / recall / F1
    reports/metrics.json              all numbers, handy for your report

Usage
    python train_model.py
"""
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config

warnings.filterwarnings("ignore", category=FutureWarning)


def build_models():
    rs = config.RANDOM_STATE
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=rs),
        "SVM (RBF)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=rs)),
        "Neural Network (MLP)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                          early_stopping=True, random_state=rs)),
    }


def single_thread(estimator):
    """Real-time inference is one sample at a time -> avoid multi-process overhead."""
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "n_jobs"):
        final.n_jobs = 1
    return estimator


def plot_comparison(results, path):
    names = list(results)
    acc = [results[n]["test_accuracy"] * 100 for n in names]
    f1 = [results[n]["macro_f1"] * 100 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color="#00c8d6")
    b2 = ax.bar(x + 0.18, f1, 0.36, label="Macro F1", color="#ff5ab4")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylim(max(0, min(acc + f1) - 10), 103)
    ax.set_ylabel("Score (%)")
    ax.set_title("SignSpeak - model comparison (hold-out test set)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(cm, labels, title, path):
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(norm, cmap="PuBuGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted sign")
    ax.set_ylabel("True sign")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Fraction of samples")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def top_confusions(cm, labels, n=5):
    pairs = [(cm[i, j], labels[i], labels[j])
             for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)[:n]


def main():
    if not config.DATA_FILE.exists():
        raise SystemExit("[SignSpeak] No dataset found. Run:  python collect_data.py")

    df = pd.read_csv(config.DATA_FILE)
    counts = df["label"].value_counts()
    print(f"\nDataset: {len(df)} samples, {df['label'].nunique()} signs, "
          f"{df.shape[1] - 1} features per sample")
    if df["label"].nunique() < 2 or counts.min() < 10:
        raise SystemExit("[SignSpeak] Need at least 2 signs with >= 10 samples each.\n"
                         f"Samples per sign:\n{counts.to_string()}")
    weak = counts[counts < 100]
    if len(weak):
        print("  Warning - fewer than 100 samples for:", ", ".join(map(str, weak.index)))

    X = df.drop(columns="label").values.astype(np.float32)
    y = df["label"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    folds = StratifiedKFold(n_splits=min(5, int(pd.Series(y_tr).value_counts().min())),
                            shuffle=True, random_state=config.RANDOM_STATE)

    results, fitted = {}, {}
    print(f"\n{'Model':<22}{'CV acc':>9}{'Test acc':>10}{'Macro F1':>10}{'ms/pred':>9}")
    print("-" * 60)
    for name, model in build_models().items():
        cv = cross_val_score(clone(model), X_tr, y_tr, cv=folds, scoring="accuracy")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_s = time.time() - t0

        t0 = time.time()
        for row in X_te[:100]:                       # simulate real-time, 1 sample at a time
            single_thread(model).predict_proba(row.reshape(1, -1))
        latency_ms = (time.time() - t0) / min(100, len(X_te)) * 1000

        pred = model.predict(X_te)
        results[name] = {
            "cv_mean": float(cv.mean()), "cv_std": float(cv.std()),
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro")),
            "train_seconds": round(train_s, 2),
            "latency_ms": round(latency_ms, 2),
        }
        fitted[name] = model
        r = results[name]
        print(f"{name:<22}{r['cv_mean']*100:>8.2f}%{r['test_accuracy']*100:>9.2f}%"
              f"{r['macro_f1']*100:>9.2f}%{r['latency_ms']:>9.1f}")

    best_name = max(results, key=lambda n: (results[n]["macro_f1"], -results[n]["latency_ms"]))
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # ---- reports
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(set(y))
    pred = best.predict(X_te)
    cm = confusion_matrix(y_te, pred, labels=labels)
    plot_comparison(results, config.REPORT_DIR / "model_comparison.png")
    plot_confusion(cm, labels, f"Confusion matrix - {best_name}", config.REPORT_DIR / "confusion_matrix.png")
    report = classification_report(y_te, pred, labels=labels, zero_division=0)
    (config.REPORT_DIR / "classification_report.txt").write_text(f"Model: {best_name}\n\n{report}")
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(
        {"best_model": best_name, "samples": int(len(df)), "signs": labels, "models": results}, indent=2))

    confusions = top_confusions(cm, labels)
    if confusions:
        print("Most confused pairs (true -> predicted):")
        for n, a, b in confusions:
            print(f"   {a} -> {b}   x{n}")

    # ---- final model: re-fit on ALL data for deployment
    final = single_thread(clone(build_models()[best_name]).fit(X, y))
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "name": best_name,
                 "accuracy": results[best_name]["test_accuracy"]}, config.MODEL_FILE)

    print(f"\nSaved model   -> {config.MODEL_FILE}")
    print(f"Saved reports -> {config.REPORT_DIR}")
    print("Next step: python app.py\n")


if __name__ == "__main__":
    main()
