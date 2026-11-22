"""
walkforward_backtest.py — STRICT out-of-sample walk-forward backtest (v2).

Evaluates the upgraded Time-Decayed (xi=0.003) + xG-Blended (0.7*xG + 0.3*Goals)
Dixon-Coles model strictly out-of-sample across:
  * La Liga: Over 2.5, Under 2.5, 1X2 Home Win
  * Ligue 1: Over 2.5, Favorite AH -0.75 (raw vs out-of-sample Platt calibrated)

Fixes from v1:
  * Uses the EXACT TeamStrengthModel(xi=0.003, xg_weight=0.70) causal logic (strict date < as_of).
  * Eliminates same-day league-baseline leakage.
  * Fixes the inverted synthetic-margin bug (bookmaker margin DIVIDES fair odds, never multiplies!).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from autoloop_master import (
    calculate_market_probabilities,
    compute_dixon_coles_probs,
    fit_platt_calibrator,
    _logit,
    _sigmoid,
)
from team_strength import DEFAULT_XI, TeamStrengthModel, XG_WEIGHT


def _metrics(preds: np.ndarray, actuals: np.ndarray, label: str) -> Dict[str, Any]:
    eps = 1e-9
    p = np.clip(np.asarray(preds, dtype=float), eps, 1.0 - eps)
    y = np.asarray(actuals, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean(y))
    brier_base = float(np.mean((base - y) ** 2))
    brier_skill = 1.0 - brier / brier_base if brier_base > 0 else float("nan")
    ll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ll_base = float(-(base * math.log(base + eps) + (1.0 - base) * math.log(1.0 - base + eps)))
    ll_skill = 1.0 - ll / ll_base if ll_base > 0 else float("nan")

    m58 = p >= 0.58
    n58 = int(m58.sum())
    pred58 = float(p[m58].mean() * 100.0) if n58 else 0.0
    real58 = float(y[m58].mean() * 100.0) if n58 else 0.0

    # Calibration max |dev| across bins with >= 25 samples
    edges = np.linspace(0.30, 0.75, 10)
    max_dev = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mb = (p >= lo) & (p < hi)
        if mb.sum() >= 25:
            max_dev = max(max_dev, abs(float(p[mb].mean() - y[mb].mean())))

    return {
        "label": label,
        "matches": int(len(y)),
        "base_rate_pct": round(base * 100.0, 2),
        "mean_pred_pct": round(float(p.mean()) * 100.0, 2),
        "brier": round(brier, 5),
        "brier_baseline": round(brier_base, 5),
        "brier_skill_pct": round(brier_skill * 100.0, 2),
        "logloss": round(ll, 5),
        "logloss_skill_pct": round(ll_skill * 100.0, 2),
        "cal_max_abs_dev": round(max_dev, 4),
        "n_p58": n58,
        "pred_p58_pct": round(pred58, 2),
        "real_p58_pct": round(real58, 2),
        "gap_p58_pct": round(real58 - pred58, 2),
    }


def evaluate_league(
    league: str,
    model: TeamStrengthModel,
    rho: float = -0.10,
    warmup: int = 100,
) -> Dict[str, Any]:
    df = model.get_league_match_dataframe(league)
    if df.empty or len(df) <= warmup:
        return {}

    p_o25, y_o25 = [], []
    p_hw, y_hw = [], []
    p_ah, y_ah = [], []

    # Cache daily causal league baselines so walk-forward runs in ~2 seconds
    day_baseline_cache: Dict[pd.Timestamp, tuple] = {}

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        d = row["date"]
        d_day = d.normalize()
        if d_day not in day_baseline_cache:
            # Strict < d_day prevents any same-day leakage
            mu_h, mu_a, _ = model.league_baselines(league, as_of=d_day)
            day_baseline_cache[d_day] = (mu_h, mu_a)
        mu_h, mu_a = day_baseline_cache[d_day]

        att_h, def_h = model.strength(league, row["home"], as_of=d_day)
        att_a, def_a = model.strength(league, row["away"], as_of=d_day)
        lh = min(5.0, max(0.3, mu_h * att_h * def_a))
        la = min(5.0, max(0.3, mu_a * att_a * def_h))
        mp = calculate_market_probabilities(compute_dixon_coles_probs(lh, la, max_goals=8, rho=rho))

        gh, ga = float(row["goals_home"]), float(row["goals_away"])
        p_o25.append(mp["over_2.5"])
        y_o25.append(1.0 if (gh + ga) > 2.5 else 0.0)
        p_hw.append(mp["home_win"])
        y_hw.append(1.0 if gh > ga else 0.0)

        # Unified numeric favorite side for AH -0.75 (lh >= la -> HOME else AWAY)
        if lh >= la:
            p_ah.append(mp["ah_-0.75_home"])
            diff = gh - ga
        else:
            p_ah.append(mp["ah_-0.75_away"])
            diff = ga - gh
        y_ah.append(1.0 if diff >= 2 else (0.5 if diff == 1 else 0.0))

    res = {
        "over_2.5": _metrics(np.array(p_o25), np.array(y_o25), f"{league} Over 2.5"),
        "home_win": _metrics(np.array(p_hw), np.array(y_hw), f"{league} 1X2 Home Win"),
        "ah_raw": _metrics(np.array(p_ah), np.array(y_ah), f"{league} AH -0.75 (raw)"),
    }

    # Strict expanding out-of-sample Platt calibration for AH -0.75 (fit on first 50%, test on second 50%)
    p_ah_arr = np.array(p_ah)
    y_ah_arr = np.array(y_ah)
    split = len(p_ah_arr) // 2
    if split >= 200:
        a, b = fit_platt_calibrator(p_ah_arr[:split], y_ah_arr[:split])
        p_ah_cal = np.array([_sigmoid(a * _logit(x) + b) for x in p_ah_arr[split:]])
        res["ah_platt_oos"] = _metrics(
            p_ah_cal, y_ah_arr[split:], f"{league} AH -0.75 (Platt OOS a={a}, b={b})"
        )
        res["ah_raw_oos_half"] = _metrics(
            p_ah_arr[split:], y_ah_arr[split:], f"{league} AH -0.75 (raw 2nd-half)"
        )
        res["platt_params"] = (a, b)
    return res


def run(save_csv: bool = True) -> Dict[str, Any]:
    model = TeamStrengthModel(xi=DEFAULT_XI, xg_weight=XG_WEIGHT)
    all_rows = []
    out: Dict[str, Any] = {}
    for league in ("La Liga", "Ligue 1"):
        res = evaluate_league(league, model)
        out[league] = res
        print(f"\n=== {league} (xi={DEFAULT_XI}, xg_weight={XG_WEIGHT}) ===")
        for k, v in res.items():
            if isinstance(v, dict) and "brier" in v:
                all_rows.append(v)
                print(
                    f"  {v['label']:42s} | Brier={v['brier']:.5f} ({v['brier_skill_pct']:+5.2f}%) "
                    f"| LL_skill={v['logloss_skill_pct']:+5.2f}% "
                    f"| p>=0.58: n={v['n_p58']:3d} pred={v['pred_p58_pct']:5.1f}% real={v['real_p58_pct']:5.1f}% (gap={v['gap_p58_pct']:+5.1f}%)"
                )
    if save_csv and all_rows:
        pd.DataFrame(all_rows).to_csv("walkforward_backtest_result.csv", index=False)
    return out


if __name__ == "__main__":
    run()
