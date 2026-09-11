REGIME_PARAMS = {
    "STRONG_TREND_UP": {
        "allow_long_breakout": True, "allow_short_breakout": False,
        "breakout_buffer_atr": 0.15, "stop_atr": 1.2, "target_atr": 4.0,
        "position_size_pct": 1.0, "trail_atr": 1.5,
    },
    "STRONG_TREND_DOWN": {
        "allow_long_breakout": False, "allow_short_breakout": True,
        "breakout_buffer_atr": 0.15, "stop_atr": 1.2, "target_atr": 4.0,
        "position_size_pct": 1.0, "trail_atr": 1.5,
    },
    "RANGE": {
        "allow_long_breakout": True, "allow_short_breakout": True,
        "breakout_buffer_atr": 0.5, "stop_atr": 1.0, "target_atr": 1.5,
        "position_size_pct": 0.6, "trail_atr": None,
    },
    "HIGH_VOL_BREAKOUT": {
        "allow_long_breakout": True, "allow_short_breakout": True,
        "breakout_buffer_atr": 0.3, "stop_atr": 2.5, "target_atr": 3.5,
        "position_size_pct": 0.5, "trail_atr": 2.0,
    },
    "LOW_VOL_COMPRESSION": {
        "allow_long_breakout": True, "allow_short_breakout": True,
        "breakout_buffer_atr": 0.1, "stop_atr": 0.8, "target_atr": 3.0,
        "position_size_pct": 1.2, "trail_atr": 1.5,
    },
}


def get_breakout_params(regime_name: str, confidence: float = 1.0, atr: float = None):
    p = REGIME_PARAMS[regime_name].copy()
    if confidence < 0.5:
        base = REGIME_PARAMS["RANGE"]
        p = {k: (0.5 * p[k] + 0.5 * base[k]) if isinstance(p[k], (int, float)) else p[k]
             for k in p}
    if atr is not None:
        p["stop_points"] = round(p["stop_atr"] * atr, 2)
        p["target_points"] = round(p["target_atr"] * atr, 2)
        p["buffer_points"] = round(p["breakout_buffer_atr"] * atr, 2)
    return p
