import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_loader import load_nifty
from .features import build_features
from .labeler import label_regimes

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def make_dataset(horizon=5):
    df = load_nifty(start="2007-01-01")
    X = build_features(df)
    y = label_regimes(df, horizon=horizon)
    data = pd.concat([X, y.rename("label")], axis=1).dropna()
    X = data.drop(columns="label")
    y = data["label"].astype(int)
    return X, y


def cv_score(model, X, y, n_splits=4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        scores.append(f1_score(y.iloc[te], pred, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    print("[train] building dataset ...")
    X, y = make_dataset(horizon=5)
    print(f"[train] X={X.shape}  classes={y.value_counts().sort_index().to_dict()}")

    scaler = StandardScaler().fit(X)
    Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    candidates = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.06, num_leaves=31,
            class_weight="balanced", n_jobs=2, random_state=42, verbose=-1),
    }

    results = {}
    for name, mdl in candidates.items():
        m, s = cv_score(mdl, Xs, y)
        results[name] = (m, s)
        print(f"[train] {name:10s} macro-F1 = {m:.4f} +/- {s:.4f}")

    best_name = max(results, key=lambda k: results[k][0])
    print(f"[train] best = {best_name}")

    best = candidates[best_name]
    best.fit(Xs, y)

    joblib.dump(
        {"model": best, "scaler": scaler, "features": list(X.columns), "best_name": best_name},
        MODEL_DIR / "regime_model.joblib",
    )
    print(f"[train] saved model -> {MODEL_DIR/'regime_model.joblib'}")

    pred = best.predict(Xs)
    report = classification_report(y, pred, digits=3)
    print("\n" + report)
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Best model: {best_name}\nmacro-F1 CV: {results[best_name][0]:.4f}\n\n" + report
    )

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot(kind="barh", ax=ax)
        ax.set_title(f"Top 15 features — {best_name}")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "feature_importance.png", dpi=120)
        print(f"[train] feature importance saved")


if __name__ == "__main__":
    main()
