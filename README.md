# Nifty Regime Detector

Machine-learning pipeline that classifies Nifty 50 into 5 market regimes
and auto-tunes breakout strategy parameters per regime.

## Regimes
| ID | Name | Meaning |
|----|------|---------|
| 0 | STRONG_TREND_UP | Sustained up move |
| 1 | STRONG_TREND_DOWN | Sustained down move |
| 2 | RANGE | Sideways |
| 3 | HIGH_VOL_BREAKOUT | Volatile expansion |
| 4 | LOW_VOL_COMPRESSION | Quiet squeeze |

## Why "no lag"
- Uses **only causal features** (no future returns, no centered windows).
- Relies on **Efficiency Ratio, Choppiness Index, ATR %, DI spread, streaks** —
  all of which respond the SAME DAY as price moves.
- No slow SMAs → no 100-day lag.

## Workflows
| Workflow | When | What |
|---|---|---|
| `train.yml` | Sunday 03:00 UTC / manual | Downloads data, trains, uploads model artifact |
| `daily_regime.yml` | Mon-Fri 12:00 UTC / manual | Uses trained model to predict today's regime |

## Run on GitHub Actions (no local setup)

1. Open the **Actions** tab
2. Run **Train Regime Model** first (`Run workflow`)
3. Wait for green ✅ (~5 min on free runner)
4. Then run **Daily Nifty Regime**
5. Download the `latest-regime` artifact → `latest_regime.json`

## Local run (2-core OK)
```bash
pip install -r requirements.txt
python scripts/run_all.py train
python scripts/run_all.py predict
