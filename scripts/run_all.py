"""
Single entry point for GitHub Actions.
Usage:  python scripts/run_all.py train
        python scripts/run_all.py predict
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nifty_regime.src.data_loader import load_nifty
from nifty_regime.src.regime_detector import RegimeDetector
from nifty_regime.src.breakout_adapter import get_breakout_params
from nifty_regime.src.train import main as train_main


def run_train():
    train_main()


def run_predict():
    df = load_nifty()
    det = RegimeDetector()
    out = det.detect_latest(df)
    atr = out["features"]["atr_14"]
    params = get_breakout_params(out["regime"], out["confidence"], atr=atr)

    summary = {
        "date": str(out["date"].date()),
        "regime": out["regime"],
        "confidence": round(out["confidence"], 4),
        "probabilities": {k: round(v, 4) for k, v in out["probabilities"].items()},
        "atr_14": round(atr, 2),
        "breakout_params": params,
    }

    print(json.dumps(summary, indent=2))

    out_path = Path(__file__).resolve().parents[1] / "nifty_regime" / "reports" / "latest_regime.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[predict] written -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        print("usage: python scripts/run_all.py [train|predict]")
        sys.exit(1)
    {"train": run_train, "predict": run_predict}[sys.argv[1]]()
