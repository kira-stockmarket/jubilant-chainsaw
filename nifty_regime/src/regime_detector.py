import joblib, pandas as pd, numpy as np
from pathlib import Path
from .data_loader import load_nifty
from .features import build_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "regime_model.joblib"

REGIME_NAMES = {
    0: "STRONG_TREND_UP",
    1: "STRONG_TREND_DOWN",
    2: "RANGE",
    3: "HIGH_VOL_BREAKOUT",
    4: "LOW_VOL_COMPRESSION",
}


class RegimeDetector:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle["features"]

    def _prep(self, df):
        return build_features(df)[self.feature_names]

    def detect_latest(self, df=None):
        df = df if df is not None else load_nifty()
        X = self._prep(df)
        row = X.iloc[[-1]]
        if row.isna().any().any():
            raise ValueError(f"NaN in latest features: {row.isna().sum().to_dict()}")
        Xs = self.scaler.transform(row)
        proba = self.model.predict_proba(Xs)[0]
        cls = int(np.argmax(proba))
        return {
            "date": df.index[-1],
            "regime_id": cls,
            "regime": REGIME_NAMES[cls],
            "confidence": float(proba[cls]),
            "probabilities": {REGIME_NAMES[i]: float(p) for i, p in enumerate(proba)},
            "features": row.iloc[0].to_dict(),
        }

    def detect_history(self, df=None):
        df = df if df is not None else load_nifty()
        X = self._prep(df).dropna()
        preds = self.model.predict(self.scaler.transform(X))
        return pd.DataFrame(
            {"regime_id": preds,
             "regime": [REGIME_NAMES[i] for i in preds]},
            index=X.index,
        )


if __name__ == "__main__":
    import json
    det = RegimeDetector()
    r = det.detect_latest()
    r["date"] = str(r["date"].date())
    r["features"] = {k: round(v, 6) for k, v in r["features"].items()}
    print(json.dumps(r, indent=2))
