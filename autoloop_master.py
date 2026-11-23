"""
autoloop_master.py — Quantitative Automated Football Betting Engine & Pipeline (v2).

Key v2 Upgrades:
  1. Live Fixtures + Live Scores integration via live_data.py (ESPN + football-data.co.uk
     + optional football-data.org / The Odds API), replacing the frozen September ledger.
  2. Leak-free, Time-Decayed (xi=0.003), xG-Blended (0.7*xG + 0.3*Goals) TeamStrengthModel
     connected directly to per-league MLE (mu_h, mu_a, rho) for BOTH La Liga and Ligue 1.
  3. Unified explicit `selection_side` (HOME / AWAY / DRAW) determined numerically from
     expected goals (lh vs la) — completely removing the broken `'Paris' in away` heuristic.
  4. Expanded high-signal markets: 1X2 (Home Win / Away Win), Over/Under 2.5, and
     Platt-calibrated Ligue 1 AH -0.75 (removing historical -5% overconfidence).
  5. Broker Mode safety margin: min_odds = round(1.06 / p_calibrated, 2) (+6% buffer)
     so following min_odds targets genuine +EV instead of zero-margin breakeven.
  6. Honest settlement reporting: displays real-result hit-rate and verified-odds ROI
     only (never fabricated ROI from 1.90 placeholder prices).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize as _scipy_minimize
    _SCIPY_AVAILABLE = True
except Exception:
    _scipy_minimize = None
    _SCIPY_AVAILABLE = False

from live_data import (
    BIG5_LEAGUES,
    REPO_DIR,
    canonical_team_name,
    fetch_all_live_matches,
    fetch_upcoming_fixtures,
    sync_live_history,
)
from settle_ledger_auto import (
    ARCHIVE_WEEKLY_FILE,
    LEDGER,
    PUBLIC_WEEKLY_FILE,
    SUMMARY_FILE,
    build_closing_odds_index,
    build_weekly_transparency_and_archive,
    determine_selection_side,
    get_matchweek_window,
    load_ledger_normalized,
    save_ledger_df,
    settle_ledger,
)
from team_strength import DEFAULT_XI, TeamStrengthModel, get_model, time_decay_weight
from telegram_alerts import (
    format_telegram_report as _tg_format_telegram_report,
    format_weekly_transparency_header,
    send_telegram_alert,
)

PUBLIC_DATA_DIR = REPO_DIR / "data" / "public"
BACKTEST_STATS_FILE = PUBLIC_DATA_DIR / "backtest_verified_stats.json"

# Global baseline priors (used ONLY when a league has no usable history).
_PRIOR_MU_HOME = 1.48
_PRIOR_MU_AWAY = 1.18
_PRIOR_RHO = -0.10

# Broker mode safety margin: require >= +6% buffer over fair calibrated price.
BROKER_SAFETY_MARGIN = 1.06
DEFAULT_MODEL_TIP_MIN = 0.60
DEFAULT_DNB_MIN = 0.70

# Per-league fitted params: {normalized_league: {mu_home, mu_away, rho, status, n}}.
_FITTED_BY_LEAGUE: Dict[str, Dict[str, Any]] = {}

# Verified profitable (league, market) pairs from strict 2019-2026 historical odds & R&D backtest
# (ROI > 0% against real bookmaker odds + positive Brier skill + positive CLV / seasonal stability).
_DEFAULT_APPROVED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("La Liga", "1X2 Home"),
    ("La Liga", "DNB Home"),
    ("La Liga", "Under 2.5"),
    ("Serie A", "1X2 Away"),
    ("Serie A", "Over 2.5"),
)

_DEFAULT_PLATT_PARAMS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("ligue 1", "ah_-0.75"): (0.82, -0.10),
}


def normalize_market_key(market: Any) -> str:
    mk = str(market or "").strip().lower()
    if "dnb" in mk or "draw no bet" in mk or "ah 0" in mk:
        return "dnb away" if "away" in mk else "dnb home"
    if "ah" in mk or "handicap" in mk:
        return "ah_-0.75"
    if "over" in mk and "2.5" in mk:
        return "over 2.5"
    if "under" in mk and "2.5" in mk:
        return "under 2.5"
    if "1x2" in mk or mk in ("home win", "away win", "draw", "home", "away"):
        if "home" in mk:
            return "1x2 home"
        if "away" in mk:
            return "1x2 away"
        if "draw" in mk:
            return "1x2 draw"
    return mk


def load_verified_backtest_config(
    stats_path: Path = BACKTEST_STATS_FILE,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], Tuple[float, float]]]:
    """
    Load backtest-approved profitable (league, market) pairs and out-of-sample
    Platt calibration parameters from `data/public/backtest_verified_stats.json`.
    """
    approved = list(_DEFAULT_APPROVED_PAIRS)
    platt = dict(_DEFAULT_PLATT_PARAMS)
    if stats_path.exists():
        try:
            raw = json.loads(stats_path.read_text(encoding="utf-8"))
            pairs = raw.get("approved_production_pairs")
            if isinstance(pairs, list) and pairs:
                approved = [(str(lg), str(mk)) for lg, mk in pairs]
            fplatt = raw.get("fitted_platt_params")
            if isinstance(fplatt, dict):
                for key_str, ab in fplatt.items():
                    if "|" in key_str and isinstance(ab, (list, tuple)) and len(ab) == 2:
                        lg_k, mk_k = key_str.split("|", 1)
                        platt[(lg_k.strip().lower(), normalize_market_key(mk_k))] = (
                            float(ab[0]),
                            float(ab[1]),
                        )
        except Exception:
            pass
    return approved, platt


APPROVED_PRODUCTION_PAIRS, _PLATT_CALIBRATORS = load_verified_backtest_config()
APPROVED_PAIR_KEYS = {
    (lg.strip().lower(), normalize_market_key(mk)) for lg, mk in APPROVED_PRODUCTION_PAIRS
}
ACTIVE_LEAGUES: Tuple[str, ...] = tuple(sorted({lg for lg, _ in APPROVED_PRODUCTION_PAIRS}))


def _norm_league(league: Any) -> str:
    return str(league or "").strip().lower()


def _prior_result(status: str, n: int = 0) -> Dict[str, Any]:
    return {
        "mu_home": _PRIOR_MU_HOME,
        "mu_away": _PRIOR_MU_AWAY,
        "rho": _PRIOR_RHO,
        "status": status,
        "n": int(n),
    }


def _store_league_result(league_key: str, result: Dict[str, Any]) -> Dict[str, Any]:
    _FITTED_BY_LEAGUE[league_key] = dict(result)
    return result


def get_league_params(league: Any) -> Tuple[float, float, float]:
    """Return (mu_home, mu_away, rho) STRICTLY for the requested league."""
    entry = _FITTED_BY_LEAGUE.get(_norm_league(league))
    if entry:
        return (float(entry["mu_home"]), float(entry["mu_away"]), float(entry["rho"]))
    return (_PRIOR_MU_HOME, _PRIOR_MU_AWAY, _PRIOR_RHO)


def get_league_mle_status(league: Any) -> str:
    entry = _FITTED_BY_LEAGUE.get(_norm_league(league))
    return str(entry.get("status", "prior")) if entry else "prior"


# ==============================================================================
# 1. DYNAMIC PER-LEAGUE MLE CALIBRATION (La Liga + Ligue 1)
# ==============================================================================

def _mle_cache_path() -> Path:
    return REPO_DIR / "league_mle_params.json"


def _load_mle_cache(max_age_days: float = 3.0) -> Dict[str, Dict[str, Any]]:
    try:
        p = _mle_cache_path()
        if not p.exists():
            return {}
        if (time.time() - p.stat().st_mtime) > max_age_days * 86400:
            return {}
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and all(kk in v for kk in ("mu_home", "mu_away", "rho")):
                out[_norm_league(k)] = {
                    "mu_home": float(v["mu_home"]),
                    "mu_away": float(v["mu_away"]),
                    "rho": float(v["rho"]),
                    "status": "cached",
                    "n": int(v.get("n", 0)),
                }
        return out
    except Exception:
        return {}


def _save_mle_cache(league_key: str, entry: Dict[str, Any]) -> None:
    try:
        p = _mle_cache_path()
        merged: Dict[str, Any] = {}
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    merged = {k: v for k, v in raw.items() if isinstance(v, dict)}
            except Exception:
                merged = {}
        merged[league_key] = {
            "mu_home": float(entry["mu_home"]),
            "mu_away": float(entry["mu_away"]),
            "rho": float(entry["rho"]),
            "n": int(entry.get("n", 0)),
        }
        p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _dixon_coles_nll(
    params: np.ndarray, gh: np.ndarray, ga: np.ndarray, w: np.ndarray
) -> float:
    log_mu_h, log_mu_a, rho = float(params[0]), float(params[1]), float(params[2])
    mu_h = float(np.clip(np.exp(log_mu_h), 0.5, 3.0))
    mu_a = float(np.clip(np.exp(log_mu_a), 0.5, 3.0))
    rho = float(np.clip(rho, -0.25, 0.05))
    lg = math.lgamma
    nll = 0.0
    tot_w = 0.0
    log_mh = math.log(mu_h)
    log_ma = math.log(mu_a)
    for i in range(gh.shape[0]):
        x, y = int(gh[i]), int(ga[i])
        if x > 12 or y > 12:
            continue
        lp = x * log_mh - mu_h - lg(x + 1) + y * log_ma - mu_a - lg(y + 1)
        tau = 1.0
        if x == 0 and y == 0:
            tau = max(1e-6, 1.0 - mu_h * mu_a * rho)
        elif x == 1 and y == 0:
            tau = max(1e-6, 1.0 + mu_a * rho)
        elif x == 0 and y == 1:
            tau = max(1e-6, 1.0 + mu_h * rho)
        elif x == 1 and y == 1:
            tau = max(1e-6, 1.0 - rho)
        lp += math.log(tau)
        wi = float(w[i])
        nll -= wi * lp
        tot_w += wi
    if tot_w > 0:
        nll /= tot_w
    # Mild regularization toward Dixon-Coles negative-dependence prior (-0.08)
    return float(nll) + 0.03 * ((rho + 0.08) ** 2)


def fit_league_mle_dynamic(
    league: str = "La Liga",
    team_model: Optional[TeamStrengthModel] = None,
    as_of: Any = None,
    max_matches: int = 760,
    time_budget_s: float = 4.0,
    xi: float = DEFAULT_XI,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Fit scoring baselines (mu_home, mu_away) and Dixon-Coles rho by time-weighted
    Maximum Likelihood on THIS league's matches strictly before `as_of`.
    Works for BOTH La Liga and Ligue 1 via TeamStrengthModel.
    """
    league_key = _norm_league(league)
    t0 = time.time()

    if use_cache and as_of is None:
        cached = _load_mle_cache().get(league_key)
        if cached:
            entry = {
                "mu_home": float(np.clip(cached["mu_home"], 0.5, 3.0)),
                "mu_away": float(np.clip(cached["mu_away"], 0.5, 3.0)),
                "rho": float(np.clip(cached["rho"], -0.25, 0.05)),
                "status": "cached",
                "n": int(cached.get("n", 0)),
            }
            return _store_league_result(league_key, entry)

    try:
        model = team_model or get_model()
        df = model.get_league_match_dataframe(league, as_of=as_of)
        if len(df) < 100:
            return _store_league_result(league_key, _prior_result("prior(thin-history)", int(len(df))))
        df = df.sort_values("date").tail(max_matches)
        gh = df["goals_home"].to_numpy(dtype=float)
        ga = df["goals_away"].to_numpy(dtype=float)
        ref_dt = pd.to_datetime(as_of) if as_of is not None else df["date"].max()
        days_ago = (ref_dt - df["date"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        w = np.exp(-float(xi) * np.clip(days_ago, 0.0, 2000.0))
    except Exception:
        return _store_league_result(league_key, _prior_result("prior(load-fail)"))

    try:
        m0_h = float(np.average(gh, weights=w))
        m0_a = float(np.average(ga, weights=w))
        if _SCIPY_AVAILABLE and _scipy_minimize is not None:
            x0 = np.array(
                [math.log(max(0.5, min(3.0, m0_h))), math.log(max(0.5, min(3.0, m0_a))), -0.08],
                dtype=float,
            )
            bounds = [(math.log(0.5), math.log(3.0)), (math.log(0.5), math.log(3.0)), (-0.25, 0.05)]
            res = _scipy_minimize(
                _dixon_coles_nll,
                x0,
                args=(gh, ga, w),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 25, "ftol": 1e-6, "gtol": 1e-4},
            )
            if time.time() - t0 > time_budget_s:
                raise TimeoutError("mle-budget")
            if res is not None and np.all(np.isfinite(res.x)):
                best_mh = float(np.clip(math.exp(float(res.x[0])), 0.5, 3.0))
                best_ma = float(np.clip(math.exp(float(res.x[1])), 0.5, 3.0))
                best_rho = float(np.clip(float(res.x[2]), -0.25, 0.05))
            else:
                raise RuntimeError("scipy-invalid")
        else:
            # Pure-NumPy fallback MLE (moment-matched baselines + 1D grid search over rho)
            best_mh = float(np.clip(m0_h, 0.5, 3.0))
            best_ma = float(np.clip(m0_a, 0.5, 3.0))
            lmh, lma = math.log(best_mh), math.log(best_ma)
            rho_grid = np.linspace(-0.25, 0.05, 31)
            best_rho = min(rho_grid, key=lambda r: _dixon_coles_nll(np.array([lmh, lma, r]), gh, ga, w))
            best_rho = float(best_rho)

        entry = {
            "mu_home": best_mh,
            "mu_away": best_ma,
            "rho": best_rho,
            "status": "fitted",
            "n": int(len(df)),
        }
        if as_of is None:
            _save_mle_cache(league_key, entry)
        return _store_league_result(league_key, entry)
    except Exception:
        pass
    return _store_league_result(league_key, _prior_result("prior(fit-fallback)", int(len(df))))


# ==============================================================================
# 2. DIXON-COLES PRICING & PLATT PROBABILITY CALIBRATION
# ==============================================================================

def dixon_coles_tau(x: int, y: int, lambda_h: float, lambda_a: float, rho: float = -0.10) -> float:
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lambda_h * lambda_a * rho)
    elif x == 1 and y == 0:
        return max(0.0, 1.0 + lambda_a * rho)
    elif x == 0 and y == 1:
        return max(0.0, 1.0 + lambda_h * rho)
    elif x == 1 and y == 1:
        return max(0.0, 1.0 - rho)
    return 1.0


def compute_dixon_coles_probs(
    lambda_h: float,
    lambda_a: float,
    max_goals: int = 8,
    rho: Optional[float] = None,
    league: Any = None,
) -> np.ndarray:
    if rho is None:
        rho = get_league_params(league)[2] if league is not None else _PRIOR_RHO
    lambda_h = max(0.2, min(5.0, float(lambda_h)))
    lambda_a = max(0.2, min(5.0, float(lambda_a)))

    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    p_h = np.array([(lambda_h ** x * math.exp(-lambda_h)) / math.factorial(x) for x in range(max_goals + 1)])
    p_a = np.array([(lambda_a ** y * math.exp(-lambda_a)) / math.factorial(y) for y in range(max_goals + 1)])

    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            matrix[x, y] = dixon_coles_tau(x, y, lambda_h, lambda_a, rho) * p_h[x] * p_a[y]

    total_p = matrix.sum()
    if total_p > 0:
        matrix /= total_p
    return matrix


def calculate_market_probabilities(prob_matrix: np.ndarray) -> Dict[str, float]:
    max_goals = prob_matrix.shape[0] - 1
    under_25 = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            if x + y <= 2:
                under_25 += prob_matrix[x, y]
    over_25 = 1.0 - under_25

    p_0_any = prob_matrix[0, :].sum()
    p_any_0 = prob_matrix[:, 0].sum()
    p_0_0 = prob_matrix[0, 0]
    btts_yes = 1.0 - (p_0_any + p_any_0 - p_0_0)

    home_win = float(np.tril(prob_matrix, -1).sum())
    draw = float(np.trace(prob_matrix))
    away_win = float(np.triu(prob_matrix, 1).sum())

    p_diff_ge_2_h = p_diff_1_h = p_diff_ge_2_a = p_diff_1_a = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            diff = x - y
            if diff >= 2:
                p_diff_ge_2_h += prob_matrix[x, y]
            elif diff == 1:
                p_diff_1_h += prob_matrix[x, y]
            elif diff <= -2:
                p_diff_ge_2_a += prob_matrix[x, y]
            elif diff == -1:
                p_diff_1_a += prob_matrix[x, y]

    ah_minus_075_home = p_diff_ge_2_h + 0.5 * p_diff_1_h
    ah_minus_075_away = p_diff_ge_2_a + 0.5 * p_diff_1_a
    non_draw = home_win + away_win
    dnb_home = home_win / non_draw if non_draw > 0 else 0.5
    dnb_away = away_win / non_draw if non_draw > 0 else 0.5

    return {
        "over_2.5": round(float(over_25), 4),
        "under_2.5": round(float(under_25), 4),
        "btts_yes": round(float(btts_yes), 4),
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "dnb_home": round(float(dnb_home), 4),
        "dnb_away": round(float(dnb_away), 4),
        "ah_-0.75_home": round(float(ah_minus_075_home), 4),
        "ah_-0.75_away": round(float(ah_minus_075_away), 4),
    }


def _logit(p: float) -> float:
    pc = min(0.999, max(0.001, float(p)))
    return math.log(pc / (1.0 - pc))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_platt_calibrator(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Fit Platt scaling parameters (a, b) minimizing log-loss of
    sigmoid(a * logit(p) + b) against realized outcomes (supports 0, 0.5, 1).
    Works with scipy (L-BFGS-B) or pure-NumPy gradient descent fallback.
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-4, 1.0 - 1e-4)
    y = np.clip(np.asarray(targets, dtype=float), 0.0, 1.0)
    if len(p) < 50:
        return (1.0, 0.0)
    logits = np.log(p / (1.0 - p))

    def _loss(ab: np.ndarray) -> float:
        a, b = float(ab[0]), float(ab[1])
        z = np.clip(a * logits + b, -20.0, 20.0)
        q = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-9
        ll = -np.mean(y * np.log(np.clip(q, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - q, eps, 1.0)))
        return float(ll) + 0.005 * ((a - 1.0) ** 2 + b ** 2)

    if _SCIPY_AVAILABLE and _scipy_minimize is not None:
        res = _scipy_minimize(
            _loss,
            np.array([1.0, 0.0]),
            method="L-BFGS-B",
            bounds=[(0.4, 1.5), (-0.6, 0.6)],
        )
        if res is not None and np.all(np.isfinite(res.x)):
            return (round(float(res.x[0]), 4), round(float(res.x[1]), 4))
        return (1.0, 0.0)

    # Pure-NumPy gradient descent fallback
    a, b = 1.0, 0.0
    lr = 0.2
    for _ in range(300):
        z = np.clip(a * logits + b, -20.0, 20.0)
        q = 1.0 / (1.0 + np.exp(-z))
        diff = q - y
        ga = float(np.mean(diff * logits)) + 0.01 * (a - 1.0)
        gb = float(np.mean(diff)) + 0.01 * b
        a = float(np.clip(a - lr * ga, 0.4, 1.5))
        b = float(np.clip(b - lr * gb, -0.6, 0.6))
    return (round(a, 4), round(b, 4))


def calibrate_probability(
    p_raw: float,
    league: str,
    market: str,
    calibrators: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
) -> float:
    """Apply Platt calibration for markets with empirically fitted parameters."""
    cal_map = calibrators if calibrators is not None else _PLATT_CALIBRATORS
    mkey = normalize_market_key(market)
    params = cal_map.get((_norm_league(league), mkey))
    if not params:
        return round(float(np.clip(p_raw, 0.01, 0.99)), 4)
    a, b = params
    p_cal = _sigmoid(a * _logit(p_raw) + b)
    return round(float(np.clip(p_cal, 0.01, 0.99)), 4)


def compute_broker_min_odds(
    p_calibrated: float, safety_margin: float = BROKER_SAFETY_MARGIN
) -> float:
    """
    Compute Broker Mode minimum odds with explicit +6% safety margin:
    min_odds = round(1.06 / p_calibrated, 2).
    """
    p = max(0.01, min(0.99, float(p_calibrated)))
    return round(float(safety_margin) / p, 2)


# ==============================================================================
# 3. MARKET VIG REMOVAL, +EV & KELLY SIZING
# ==============================================================================

def remove_market_vig(
    odds_target: float, odds_counter: Optional[float] = None, default_margin: float = 1.055
) -> float:
    if odds_target <= 1.0:
        return 0.5
    if odds_counter is not None and odds_counter > 1.0:
        overround = (1.0 / odds_target) + (1.0 / odds_counter)
        p_fair = (1.0 / odds_target) / overround
    else:
        p_fair = (1.0 / odds_target) / default_margin
    return max(0.01, min(0.99, float(p_fair)))


def compute_ev(p_model: float, bookie_odds: float) -> float:
    return float((p_model * bookie_odds) - 1.0)


def calculate_fractional_kelly(
    p_model: float, bookie_odds: float, fraction: float = 0.25, max_stake: float = 0.05
) -> float:
    if bookie_odds <= 1.0:
        return 0.0
    ev = compute_ev(p_model, bookie_odds)
    if ev <= 0.0:
        return 0.0
    full_kelly = ev / (bookie_odds - 1.0)
    return round(float(min(max_stake, max(0.0, fraction * full_kelly))), 4)


# ==============================================================================
# 4. EXPECTED GOALS (LAMBDA) ESTIMATION — CONNECTED TO MLE & CAUSAL TEAM STRENGTH
# ==============================================================================

def estimate_match_lambdas(
    home_team: str,
    away_team: str,
    league: str,
    league_avg_home: Optional[float] = None,
    league_avg_away: Optional[float] = None,
    team_model: Optional[TeamStrengthModel] = None,
    fixture_date: Any = None,
    use_mle_baselines: bool = True,
) -> Tuple[float, float]:
    """
    Compute (lambda_home, lambda_away) using:
      * Per-league MLE baselines (mu_h, mu_a) from fit_league_mle_dynamic when
        calibrated for live fixtures (or strictly causal as_of league baselines
        for historical/walk-forward evaluation).
      * Leak-free, time-decayed (xi=0.003), xG-blended club strengths (att, def).
    """
    model = team_model or get_model()
    as_of = pd.to_datetime(fixture_date, errors="coerce") if fixture_date is not None else None
    if pd.isna(as_of):
        as_of = None

    # Resolve league baselines (strictly per league; MLE connected when fitted!)
    if league_avg_home is not None and league_avg_away is not None:
        mu_h, mu_a = float(league_avg_home), float(league_avg_away)
    else:
        mle_entry = _FITTED_BY_LEAGUE.get(_norm_league(league))
        is_future_or_live = as_of is None or as_of >= pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        if use_mle_baselines and mle_entry and mle_entry.get("status") in ("fitted", "cached") and is_future_or_live:
            mu_h = float(mle_entry["mu_home"])
            mu_a = float(mle_entry["mu_away"])
        else:
            mu_h, mu_a, _ = model.league_baselines(league, as_of=as_of)

        if league_avg_home is not None:
            mu_h = float(league_avg_home)
        if league_avg_away is not None:
            mu_a = float(league_avg_away)

    att_h, def_h = model.strength(league, home_team, as_of=as_of)
    att_a, def_a = model.strength(league, away_team, as_of=as_of)

    lambda_h = mu_h * att_h * def_a
    lambda_a = mu_a * att_a * def_h
    return (
        round(float(min(5.0, max(0.3, lambda_h))), 3),
        round(float(min(5.0, max(0.3, lambda_a))), 3),
    )


# ==============================================================================
# 5. AGENTS: DATA INGESTION (LIVE FIXTURES), QUANT, RISK, CRITIC, PORTFOLIO
# ==============================================================================

class DataIngestionAgent:
    """
    Ingests upcoming fixtures from LIVE football feeds (ESPN / football-data.org)
    and merges with any pending verified-odds rows from paper_trading_ledger_2026.csv.
    Falls back cleanly to ledger PENDING rows when offline.
    """

    def __init__(
        self,
        ledger_path: str = LEDGER,
        window_days: int = 8,
        leagues: Tuple[str, ...] = ACTIVE_LEAGUES,
        use_live_fixtures: bool = True,
    ) -> None:
        p = Path(ledger_path)
        self.ledger_path = p if p.exists() else (REPO_DIR / ledger_path)
        self.window_days = int(window_days)
        self.active_leagues = tuple(leagues)
        self.use_live_fixtures = bool(use_live_fixtures)
        self.window_start = pd.Timestamp.now().normalize()
        self.window_end = self.window_start + pd.Timedelta(days=self.window_days)

    def leagues(self) -> List[str]:
        found = set(self.active_leagues)
        if self.ledger_path.exists():
            try:
                ldf, _ = load_ledger_normalized(self.ledger_path)
                found.update(x for x in ldf["league"].dropna().astype(str).str.strip() if x)
            except Exception:
                pass
        return sorted(found)

    def _load_ledger_upcoming(self) -> pd.DataFrame:
        if not self.ledger_path.exists():
            return pd.DataFrame()
        df, _ = load_ledger_normalized(self.ledger_path)
        if df.empty:
            return df
        mask = (
            (pd.to_datetime(df["date"]) >= self.window_start)
            & (pd.to_datetime(df["date"]) <= self.window_end)
            & (df["status"] == "PENDING")
        )
        out = df[mask].copy()
        if not out.empty:
            out["home"] = out.apply(lambda r: canonical_team_name(r["home"], str(r["league"])), axis=1)
            out["away"] = out.apply(lambda r: canonical_team_name(r["away"], str(r["league"])), axis=1)
        return out

    def active_bets(self, live_records: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Return candidate upcoming bets for the next `window_days`:
          * Generates candidates for ALL live upcoming La Liga & Ligue 1 fixtures across
            high-signal markets (1X2 Home Win, 1X2 Away Win, Over 2.5, Under 2.5,
            and calibrated Ligue 1 AH -0.75).
          * Merges existing PENDING ledger rows (preserving any verified the-odds-api prices).
        """
        ledger_df = self._load_ledger_upcoming()
        # Index existing ledger rows to preserve verified odds if already fetched
        ledger_lookup: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
        for _, r in ledger_df.iterrows():
            key = (
                str(r["date"])[:10],
                str(r["league"]),
                str(r["home"]),
                str(r["away"]),
                str(r["market"]).strip(),
                str(r.get("selection_side", "") or "").strip().upper(),
            )
            ledger_lookup[key] = r.to_dict()

        rows: List[Dict[str, Any]] = []
        seen_keys = set()

        if self.use_live_fixtures:
            live_df = fetch_upcoming_fixtures(
                leagues=self.active_leagues,
                window_days=self.window_days,
                now=self.window_start,
                live_records=live_records,
            )
            print(f"[STEP 2] Live fixtures discovered for next {self.window_days}d: {len(live_df)}")
            for _, fx in live_df.iterrows():
                lg = str(fx["league"])
                h = str(fx["home"])
                a = str(fx["away"])
                dt = fx["date"]
                dk = str(dt)[:10]

                # Candidate markets: ONLY backtest-approved profitable (league, market) pairs
                all_candidates = [
                    ("1X2 Home Win", "HOME"),
                    ("DNB Home", "HOME"),
                    ("1X2 Away Win", "AWAY"),
                    ("Over 2.5", ""),
                    ("Under 2.5", ""),
                ]
                candidates = [
                    (mk, sd)
                    for mk, sd in all_candidates
                    if (_norm_league(lg), normalize_market_key(mk)) in APPROVED_PAIR_KEYS
                ]

                for market, side in candidates:
                    key = (dk, lg, h, a, market, side)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    existing = ledger_lookup.get(key)
                    if existing:
                        rows.append(existing)
                    else:
                        rows.append({
                            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                            "date": dt,
                            "home": h,
                            "away": a,
                            "league": lg,
                            "market": market,
                            "selection_side": side,
                            "odds_source": "",
                            "market_odds": np.nan,
                            "stake": 0.0,
                            "status": "PENDING",
                            "result": "",
                            "profit": np.nan,
                        })

        # Also include any upcoming ledger rows not covered by live feed
        for key, rec in ledger_lookup.items():
            if key not in seen_keys:
                seen_keys.add(key)
                rows.append(rec)

        out = pd.DataFrame(rows) if rows else pd.DataFrame()
        print(f"[STEP 2] Total candidate market rows to evaluate: {len(out)}")
        return out


class QuantCalibrationAgent:
    """Per-league MLE calibration + Dixon-Coles pricing + Platt probability calibration."""

    def __init__(self, team_model: Optional[TeamStrengthModel] = None) -> None:
        self.team_model = team_model

    def calibrate_leagues(self, leagues: List[str]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        model = self.team_model or get_model()
        for lg in leagues:
            mle = fit_league_mle_dynamic(league=lg, team_model=model)
            results[lg] = mle
            print(
                f"[MLE] {lg}: μH={mle['mu_home']:.3f} μA={mle['mu_away']:.3f} "
                f"ρ={mle['rho']:.3f} status={mle['status']} n={mle['n']}"
            )
        return results

    def price_fixture(
        self, home: str, away: str, league: str, fixture_date: Any = None
    ) -> Dict[str, Any]:
        model = self.team_model or get_model()
        lh, la = estimate_match_lambdas(
            home, away, league, team_model=model, fixture_date=fixture_date
        )
        rho = get_league_params(league)[2]
        matrix = compute_dixon_coles_probs(lh, la, league=league)
        return {
            "lh": lh,
            "la": la,
            "rho": float(rho),
            "market_probs": calculate_market_probabilities(matrix),
            "mle_status": get_league_mle_status(league),
        }


class RiskManagementAgent:
    def __init__(self, kelly_fraction: float = 0.25, max_stake: float = 0.05) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_stake = max_stake

    def evaluate(self, p_calibrated: float, odds: float) -> Dict[str, float]:
        p_fair = remove_market_vig(odds)
        ev = compute_ev(p_calibrated, odds)
        return {
            "p_fair": round(p_fair, 4),
            "ev": float(ev),
            "ev_percent": round(ev * 100.0, 2),
            "edge": round(float(p_calibrated) - p_fair, 4),
            "stake_kelly": calculate_fractional_kelly(
                p_calibrated, odds, fraction=self.kelly_fraction, max_stake=self.max_stake
            ),
        }


class ModelValidatorAgent:
    """The Critic: enforces profitable-market whitelist, probability bounds, +EV gate, and confidence thresholds."""

    def __init__(
        self,
        min_ev_pct: float = 2.0,
        model_tip_min: float = DEFAULT_MODEL_TIP_MIN,
        dnb_min: float = DEFAULT_DNB_MIN,
        ah_ligue1_min: float = 0.62,
        enforce_approved_markets: bool = True,
    ) -> None:
        self.min_ev_pct = float(min_ev_pct)
        self.model_tip_min = float(model_tip_min)
        self.dnb_min = float(dnb_min)
        self.ah_ligue1_min = float(ah_ligue1_min)
        self.enforce_approved_markets = bool(enforce_approved_markets)

    def _audit_row(self, row: Dict[str, Any]) -> Tuple[str, str]:
        try:
            p = float(row.get("p_calibrated", row.get("p_model", float("nan"))))
            odds = float(row.get("market_odds", 0.0))
            ev_pct = float(row.get("ev_percent", float("nan")))
            lh = float(row.get("lh", 1.5))
            la = float(row.get("la", 1.2))
        except (TypeError, ValueError):
            return ("REJECTED", "unparseable numeric fields")

        if self.enforce_approved_markets:
            pair_key = (_norm_league(row.get("league", "")), normalize_market_key(row.get("market", "")))
            if pair_key not in APPROVED_PAIR_KEYS:
                return ("REJECTED", f"market {pair_key} excluded by historical odds ROI filter")

        if not (np.isfinite(p) and np.isfinite(odds)):
            return ("REJECTED", "non-finite model/odds fields")
        if not (0.01 <= p <= 0.99):
            return ("REJECTED", f"calibrated prob out of bounds ({p:.3f})")
        if not (odds > 1.0):
            return ("REJECTED", f"invalid odds ({odds})")
        if not (0.2 <= lh <= 5.0 and 0.2 <= la <= 5.0):
            return ("REJECTED", f"lambda anomaly ({lh}/{la})")

        price_type = str(row.get("price_type", "model"))
        if price_type == "verified":
            if not np.isfinite(ev_pct) or ev_pct < self.min_ev_pct:
                return ("REJECTED", f"EV {ev_pct:+.1f}% below {self.min_ev_pct:.1f}% gate")
            return ("APPROVED", f"verified price, EV {ev_pct:+.1f}%")

        # Broker Mode: dedicated threshold per market class (DNB >= 0.70, core 1X2/OU >= 0.60)
        mk_low = str(row.get("market", "")).lower()
        lg_low = _norm_league(row.get("league", ""))
        if "dnb" in mk_low or "draw no bet" in mk_low or "ah 0" in mk_low:
            min_conf = self.dnb_min
        elif lg_low == "ligue 1" and "ah" in mk_low:
            min_conf = self.ah_ligue1_min
        else:
            min_conf = self.model_tip_min
        if p < min_conf:
            return ("REJECTED", f"calibrated confidence p={p:.2f} below {min_conf:.0%} gate")
        m = float(row.get("min_odds") or 0.0)
        return ("APPROVED", f"model tip (p_cal={p:.1%}, find odds >= {m:.2f} [+6% margin])")

    def audit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df.empty:
            return df, df
        verdicts, reasons = [], []
        for _, r in df.iterrows():
            v, reason = self._audit_row(r.to_dict())
            verdicts.append(v)
            reasons.append(reason)
        df = df.copy()
        df["verdict"] = verdicts
        df["reject_reason"] = reasons
        consensus = df[df["verdict"] == "APPROVED"].copy()
        rejected = df[df["verdict"] != "APPROVED"].copy()
        print(f"[CRITIC] Consensus: {len(consensus)} approved / {len(rejected)} rejected.")
        return consensus, rejected


class PortfolioExposureController:
    def __init__(
        self, matchday_cap: float = 0.12, global_cap: float = 0.30, kelly_fraction: float = 0.25
    ) -> None:
        self.matchday_cap = float(matchday_cap)
        self.global_cap = float(global_cap)
        self.kelly_fraction = kelly_fraction
        self.scaled_matchdays = 0
        self.scaled_global = False

    @staticmethod
    def _group(day: Any) -> str:
        try:
            return pd.to_datetime(day).strftime("%Y-%m-%d")
        except Exception:
            return str(day)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        raw_col = out.get("stake_kelly", out.get("stake"))
        if raw_col is None:
            return out
        out["stake_kelly"] = pd.to_numeric(raw_col, errors="coerce").fillna(0.0)
        out["stake_raw"] = out["stake_kelly"]

        self.scaled_matchdays = 0
        for day, grp in out.groupby(out["date"].map(self._group)):
            grp_sum = float(grp["stake_kelly"].sum())
            if grp_sum > self.matchday_cap and grp_sum > 0:
                factor = self.matchday_cap / grp_sum
                mask = out["date"].map(self._group) == day
                out.loc[mask, "stake_kelly"] = out.loc[mask, "stake_kelly"] * factor
                self.scaled_matchdays += 1

        total = float(out["stake_kelly"].sum())
        if total > self.global_cap and total > 0:
            factor = self.global_cap / total
            out["stake_kelly"] = out["stake_kelly"] * factor
            self.scaled_global = True

        out["stake_kelly"] = out["stake_kelly"].round(4)
        return out

    def audit_note(self) -> str:
        parts = []
        if self.scaled_matchdays > 0:
            parts.append(f"matchday-cap applied x{self.scaled_matchdays}")
        if self.scaled_global:
            parts.append("global-cap applied")
        return "، ".join(parts) if parts else "within caps"


# ==============================================================================
# 6. PIPELINE ORCHESTRATION & LEDGER PERSISTENCE
# ==============================================================================

def run_settlement(
    live_records: Optional[List[Dict[str, Any]]] = None,
    use_live: bool = True,
) -> Dict[str, Any]:
    print("[STEP 1] Running Ledger Settlement & History Sync Engine...")
    ledger_path = Path(LEDGER)
    if not ledger_path.exists():
        ledger_path = REPO_DIR / LEDGER
    summary_path = ledger_path.parent / SUMMARY_FILE
    try:
        return settle_ledger(
            ledger_path=ledger_path,
            summary_path=summary_path,
            live_records=live_records,
            use_live=use_live,
            sync_history=use_live,
        )
    except Exception as e:
        print(f"[SETTLE EXCEPTION] {e}")
        return {
            "settled_now": 0,
            "wins": 0,
            "half_wins": 0,
            "losses": 0,
            "total_settled_all_time": 0,
            "hit_rate_percent": 0.0,
            "verified_settled_count": 0,
            "verified_profit_units": 0.0,
            "verified_roi_percent": None,
            "pending_after": 0,
        }


def fetch_and_evaluate_opportunities(
    data_agent: Optional[DataIngestionAgent] = None,
    quant_agent: Optional[QuantCalibrationAgent] = None,
    risk_agent: Optional[RiskManagementAgent] = None,
    live_records: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    data_agent = data_agent or DataIngestionAgent()
    quant_agent = quant_agent or QuantCalibrationAgent()
    risk_agent = risk_agent or RiskManagementAgent()
    active_bets = data_agent.active_bets(live_records=live_records)
    if active_bets.empty:
        return pd.DataFrame()

    upgraded_rows: List[Dict[str, Any]] = []
    for _, row in active_bets.iterrows():
        league = str(row.get("league", "La Liga"))
        home = canonical_team_name(row["home"], league)
        away = canonical_team_name(row["away"], league)
        market = str(row.get("market", "Over 2.5")).strip()
        mk_low = market.lower()
        odds = pd.to_numeric(row.get("market_odds"), errors="coerce")

        # 1. Dixon-Coles pricing using league MLE + causal time-decayed xG strengths
        pricing = quant_agent.price_fixture(home, away, league, fixture_date=row["date"])
        mp = pricing["market_probs"]
        lh, la = pricing["lh"], pricing["la"]

        # 2. Extract model probability & unified explicit selection_side (NO string hacks!)
        explicit_side = row.get("selection_side")
        selection_side = ""

        if "over" in mk_low and "2.5" in mk_low:
            p_raw = mp["over_2.5"]
        elif "under" in mk_low and "2.5" in mk_low:
            p_raw = mp["under_2.5"]
        elif "1x2" in mk_low or mk_low in ("home win", "away win", "draw"):
            if "home" in mk_low or str(explicit_side).upper() == "HOME":
                selection_side = "HOME"
                p_raw = mp["home_win"]
            elif "away" in mk_low or str(explicit_side).upper() == "AWAY":
                selection_side = "AWAY"
                p_raw = mp["away_win"]
            elif "draw" in mk_low or str(explicit_side).upper() == "DRAW":
                selection_side = "DRAW"
                p_raw = mp["draw"]
            else:
                selection_side = determine_selection_side(
                    home, away, league, fixture_date=row["date"], lh=lh, la=la
                )
                p_raw = mp["home_win"] if selection_side == "HOME" else mp["away_win"]
        elif "dnb" in mk_low or "draw no bet" in mk_low or "ah 0" in mk_low:
            if "home" in mk_low or str(explicit_side).upper() == "HOME":
                selection_side = "HOME"
                p_raw = mp["dnb_home"]
            elif "away" in mk_low or str(explicit_side).upper() == "AWAY":
                selection_side = "AWAY"
                p_raw = mp["dnb_away"]
            else:
                selection_side = determine_selection_side(
                    home, away, league, fixture_date=row["date"], lh=lh, la=la
                )
                p_raw = mp["dnb_home"] if selection_side == "HOME" else mp["dnb_away"]
        elif "ah" in mk_low or "handicap" in mk_low:
            selection_side = determine_selection_side(
                home, away, league, fixture_date=row["date"],
                explicit_side=explicit_side, lh=lh, la=la,
            )
            p_raw = mp["ah_-0.75_home"] if selection_side == "HOME" else mp["ah_-0.75_away"]
        elif "btts" in mk_low:
            p_raw = mp["btts_yes"] if "no" not in mk_low else (1.0 - mp["btts_yes"])
        else:
            continue

        # 3. Apply Platt probability calibration (removes Ligue 1 AH -0.75 overconfidence)
        p_cal = calibrate_probability(p_raw, league, market)

        # 4. Pricing path: verified bookmaker odds vs Broker Mode (+6% safety margin)
        odds_source = str(row.get("odds_source", "") or "").strip()
        has_real = odds_source == "the-odds-api" and pd.notna(odds) and float(odds) > 1.0
        min_odds_req = compute_broker_min_odds(p_cal, BROKER_SAFETY_MARGIN)

        if has_real:
            risk = risk_agent.evaluate(p_cal, float(odds))
            price_type = "verified"
            min_odds = None
            use_odds = round(float(odds), 2)
            p_fair = risk["p_fair"]
            ev_percent = risk["ev_percent"]
            edge = risk["edge"]
            stake_kelly = risk["stake_kelly"]
        else:
            price_type = "model"
            min_odds = min_odds_req
            use_odds = min_odds_req
            p_fair = float(p_cal)
            # Expected value IF bettor achieves >= min_odds (+6% target margin)
            ev_percent = round(compute_ev(p_cal, min_odds_req) * 100.0, 2)
            edge = 0.0
            conf_req = DEFAULT_DNB_MIN if ("dnb" in mk_low or "draw no bet" in mk_low) else DEFAULT_MODEL_TIP_MIN
            stake_kelly = 0.02 if p_cal >= conf_req else 0.0

        row_dict = row.to_dict()
        row_dict["home"] = home
        row_dict["away"] = away
        row_dict["league"] = league
        row_dict["market"] = market
        row_dict["selection_side"] = selection_side
        row_dict["odds_source"] = odds_source
        row_dict["price_type"] = price_type
        row_dict["min_odds"] = min_odds
        row_dict["p_model"] = round(float(p_raw), 4)
        row_dict["p_calibrated"] = round(float(p_cal), 4)
        row_dict["p_fair"] = float(p_fair)
        row_dict["ev_percent"] = float(ev_percent)
        row_dict["edge"] = float(edge)
        row_dict["stake_kelly"] = float(stake_kelly)
        row_dict["market_odds"] = float(use_odds)
        row_dict["lh"] = lh
        row_dict["la"] = la
        row_dict["rho"] = round(float(pricing["rho"]), 4)
        row_dict["mle_status"] = pricing["mle_status"]
        upgraded_rows.append(row_dict)

    return pd.DataFrame(upgraded_rows)


def record_approved_bets_to_ledger(
    df_opps: pd.DataFrame, ledger_path: Path = Path(LEDGER)
) -> int:
    """Persist newly approved live picks into paper_trading_ledger_2026.csv for future settlement."""
    if df_opps is None or df_opps.empty:
        return 0
    if not ledger_path.exists():
        ledger_path = REPO_DIR / LEDGER
    if not ledger_path.exists():
        return 0

    df, is_v4 = load_ledger_normalized(ledger_path)

    existing_keys = set()
    for _, r in df.iterrows():
        existing_keys.add((
            str(r.get("date", ""))[:10],
            str(r.get("league", "")),
            canonical_team_name(r.get("home", ""), str(r.get("league", ""))),
            canonical_team_name(r.get("away", ""), str(r.get("league", ""))),
            str(r.get("market", "")).strip(),
            str(r.get("selection_side", "") or "").strip().upper(),
        ))

    new_rows = []
    for _, r in df_opps.iterrows():
        key = (
            str(r["date"])[:10],
            str(r["league"]),
            str(r["home"]),
            str(r["away"]),
            str(r["market"]).strip(),
            str(r.get("selection_side", "") or "").strip().upper(),
        )
        if key in existing_keys:
            continue
        existing_keys.add(key)
        is_verified = str(r.get("odds_source", "")).strip() == "the-odds-api"
        new_rows.append({
            "date": pd.to_datetime(str(r["date"])[:19]),
            "league": r["league"],
            "home": r["home"],
            "away": r["away"],
            "market": r["market"],
            "selection_side": r.get("selection_side", ""),
            "p_calibrated": r["p_calibrated"],
            "min_odds": r.get("min_odds", np.nan),
            "market_odds": r["market_odds"] if is_verified else np.nan,
            "stake": r.get("stake_kelly", 0.02),
            "status": "PENDING",
            "result": "",
            "profit": np.nan,
            "odds_source": r.get("odds_source", "") if is_verified else "",
        })

    if new_rows:
        out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_ledger_df(out, ledger_path, is_v4=is_v4)
    return len(new_rows)


# ==============================================================================
# 7. TELEGRAM FORMATTING & DISPATCH (DELEGATED TO telegram_alerts.py)
# ==============================================================================

def format_telegram_report(
    summary: Dict[str, Any],
    df_opps: pd.DataFrame,
    critic_note: str = "",
    transparency_banner: Optional[Dict[str, Any]] = None,
) -> str:
    return _tg_format_telegram_report(
        summary=summary,
        df_opps=df_opps,
        critic_note=critic_note,
        transparency_banner=transparency_banner,
        mle_by_league=_FITTED_BY_LEAGUE,
        get_league_params_fn=get_league_params,
        get_league_status_fn=get_league_mle_status,
    )


# ==============================================================================
# 8. DEDUP & MAIN ENTRYPOINT
# ==============================================================================

def _fixture_key(r: Any) -> str:
    d = str(r["date"])[:10]
    side = str(r.get("selection_side", "") or "").strip().upper()
    return f"{d}|{r['home']}|{r['away']}|{r['market']}|{side}"


def _load_sent_keys() -> set:
    try:
        p = Path("autoloop_last_run.json")
        if not p.exists():
            p = REPO_DIR / "autoloop_last_run.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return set(data.get("sent_fixture_keys", []))
    except Exception:
        pass
    return set()


def dedup_opportunities(df_opps: pd.DataFrame, sent_keys: set) -> Tuple[pd.DataFrame, set, int]:
    if df_opps is None or df_opps.empty:
        return df_opps, set(), 0
    new_rows, new_keys, skipped = [], set(), 0
    for _, r in df_opps.iterrows():
        k = _fixture_key(r)
        if k in sent_keys:
            skipped += 1
        else:
            new_rows.append(r.to_dict())
            new_keys.add(k)
    new_df = pd.DataFrame(new_rows, columns=df_opps.columns) if new_rows else df_opps.iloc[0:0]
    return new_df, new_keys, skipped


def export_public_json_bundle(
    df_opps: pd.DataFrame,
    summary: Dict[str, Any],
    ledger_path: Path = Path(LEDGER),
    public_dir: Path = PUBLIC_DATA_DIR,
    archive_path: Optional[Path] = None,
    closing_index: Optional[Dict[Any, Any]] = None,
) -> Dict[str, Path]:
    """
    Export clean, structured public JSON artifacts to `data/public/` for
    consumption by the global web platform (Next.js):
      * predictions_latest.json (includes top-level `transparency_banner`)
      * performance_summary.json (includes weekly Conservative Floor + Kelly Weighted + Flat 1u + Closing Odds metrics)
      * weekly_archive.json (mirrored to `data/archive/weekly_archive.json` for permanent closed-week records)
      (backtest_verified_stats.json is maintained by historical_odds_backtest.py)
    """
    public_dir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    is_default_public = public_dir.resolve() == PUBLIC_DATA_DIR.resolve()
    eff_archive = archive_path if archive_path is not None else (
        ARCHIVE_WEEKLY_FILE if is_default_public else (public_dir.parent / "archive" / "weekly_archive.json")
    )
    eff_public_archive = PUBLIC_WEEKLY_FILE if is_default_public else (public_dir / "weekly_archive.json")

    # 0. Load latest ledger (including newly recorded picks) and build Matchweek Window archive + transparency_banner
    lpath = ledger_path if ledger_path.exists() else (REPO_DIR / LEDGER)
    ldf = pd.DataFrame()
    by_league: Dict[str, Dict[str, Any]] = {}
    by_market: Dict[str, Dict[str, Any]] = {}
    if lpath.exists():
        try:
            ldf, _ = load_ledger_normalized(lpath)
            done = ldf[ldf["status"].isin(["WIN", "HALF WIN", "LOSS"])]
            for lg, g in done.groupby("league"):
                w = int((g["status"] == "WIN").sum())
                hw = int((g["status"] == "HALF WIN").sum())
                l = int((g["status"] == "LOSS").sum())
                tot = len(g)
                by_league[str(lg)] = {
                    "settled": tot,
                    "wins": w,
                    "half_wins": hw,
                    "losses": l,
                    "hit_rate_percent": round((w + 0.5 * hw) / tot * 100.0, 2) if tot else 0.0,
                }
            for mk, g in done.groupby("market"):
                w = int((g["status"] == "WIN").sum())
                hw = int((g["status"] == "HALF WIN").sum())
                l = int((g["status"] == "LOSS").sum())
                tot = len(g)
                by_market[str(mk)] = {
                    "settled": tot,
                    "wins": w,
                    "half_wins": hw,
                    "losses": l,
                    "hit_rate_percent": round((w + 0.5 * hw) / tot * 100.0, 2) if tot else 0.0,
                }
        except Exception:
            ldf = pd.DataFrame()

    weekly_archive, transparency_banner = build_weekly_transparency_and_archive(
        df=ldf,
        archive_path=eff_archive,
        public_archive_path=eff_public_archive,
        closing_index=closing_index,
        persist=True,
    )
    summary["transparency_banner"] = transparency_banner
    summary["weekly_summary_count"] = int(weekly_archive["total_weeks"])
    summary["closed_weeks_count"] = int(weekly_archive["closed_weeks_count"])

    # 1. predictions_latest.json (with `transparency_banner` at top!)
    preds_list: List[Dict[str, Any]] = []
    if df_opps is not None and not df_opps.empty:
        for _, r in df_opps.iterrows():
            mw = get_matchweek_window(r["date"])
            preds_list.append({
                "date": str(r["date"])[:19],
                "matchweek_id": mw["matchweek_id"],
                "league": str(r["league"]),
                "home": str(r["home"]),
                "away": str(r["away"]),
                "market": str(r["market"]),
                "selection_side": str(r.get("selection_side", "") or ""),
                "p_model": round(float(r.get("p_model", 0.0)), 4),
                "p_calibrated": round(float(r.get("p_calibrated", r.get("p_model", 0.0))), 4),
                "price_type": str(r.get("price_type", "model")),
                "min_odds": None if pd.isna(r.get("min_odds")) else round(float(r["min_odds"]), 2),
                "market_odds": round(float(r.get("market_odds", 0.0)), 2),
                "ev_percent": round(float(r.get("ev_percent", 0.0)), 2),
                "stake_kelly": round(float(r.get("stake_kelly", 0.0)), 4),
                "expected_goals": {
                    "home": round(float(r.get("lh", 0.0)), 3),
                    "away": round(float(r.get("la", 0.0)), 3),
                },
            })

    mle_snapshot = {
        lg: {
            "mu_home": round(float(v["mu_home"]), 3),
            "mu_away": round(float(v["mu_away"]), 3),
            "rho": round(float(v["rho"]), 3),
            "status": str(v.get("status", "prior")),
            "n": int(v.get("n", 0)),
        }
        for lg, v in sorted(_FITTED_BY_LEAGUE.items())
    }

    predictions_payload = {
        "generated_at": now_iso,
        "engine_version": "v4-alpha-promoted",
        "transparency_banner": transparency_banner,
        "min_confidence_pct": round(DEFAULT_MODEL_TIP_MIN * 100.0, 1),
        "dnb_min_confidence_pct": round(DEFAULT_DNB_MIN * 100.0, 1),
        "safety_margin_pct": round((BROKER_SAFETY_MARGIN - 1.0) * 100.0, 1),
        "approved_leagues": list(ACTIVE_LEAGUES),
        "approved_pairs": [[lg, mk] for lg, mk in APPROVED_PRODUCTION_PAIRS],
        "mle_calibration": mle_snapshot,
        "count": len(preds_list),
        "predictions": preds_list,
    }
    pred_path = public_dir / "predictions_latest.json"
    pred_path.write_text(json.dumps(predictions_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. performance_summary.json (with Matchweek Window weekly metrics)
    compact_weekly = [
        {k: v for k, v in wk.items() if k != "picks"}
        for wk in weekly_archive.get("weeks", [])
    ]
    perf_payload = {
        "generated_at": now_iso,
        "engine_version": "v4-alpha-promoted",
        "matchweek_window_rule": weekly_archive["matchweek_window_rule"],
        "valuation_basis": weekly_archive["valuation_basis"],
        "approved_production_pairs": [[lg, mk] for lg, mk in APPROVED_PRODUCTION_PAIRS],
        "transparency_banner": transparency_banner,
        "weekly_performance": compact_weekly,
        "settlement": summary,
        "breakdown_by_league": by_league,
        "breakdown_by_market": by_market,
    }
    perf_path = public_dir / "performance_summary.json"
    perf_path.write_text(json.dumps(perf_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "predictions_latest": pred_path,
        "performance_summary": perf_path,
        "weekly_archive": eff_public_archive,
    }


def main() -> None:
    start_time = time.time()
    print(f"[AUTOLOOP v3] Start: {datetime.now(timezone.utc).isoformat()} | Approved leagues: {ACTIVE_LEAGUES}")

    # Fetch live matches for Big-5 (keeps full Big-5 history & settlement up to date)
    live_records = fetch_all_live_matches(leagues=BIG5_LEAGUES, days_back=45, days_forward=10)
    print(f"[LIVE FEED] Loaded {len(live_records)} unified Big-5 match records.")

    # 1. Settle completed bets + sync newly finished matches into historical datasets
    summary = run_settlement(live_records=live_records, use_live=True)

    # Refresh TeamStrengthModel after live history sync
    team_model = get_model(refresh=True)
    data_agent = DataIngestionAgent(leagues=ACTIVE_LEAGUES, use_live_fixtures=True)
    quant_agent = QuantCalibrationAgent(team_model=team_model)
    risk_agent = RiskManagementAgent(kelly_fraction=0.25, max_stake=0.05)
    critic_agent = ModelValidatorAgent(
        min_ev_pct=2.0,
        model_tip_min=DEFAULT_MODEL_TIP_MIN,
        dnb_min=DEFAULT_DNB_MIN,
        ah_ligue1_min=0.62,
        enforce_approved_markets=True,
    )

    # 2. Per-league Dynamic MLE calibration for active profitable leagues
    try:
        quant_agent.calibrate_leagues(sorted(set(ACTIVE_LEAGUES)))
    except Exception as e:
        print(f"[MLE] skipped ({e}); using priors.")

    # 3. Evaluate live upcoming gameweek opportunities
    df_evaluated = fetch_and_evaluate_opportunities(
        data_agent, quant_agent, risk_agent, live_records=live_records
    )

    # 4. Critic audit + portfolio exposure caps
    df_opps, df_rejected = critic_agent.audit(df_evaluated)
    portfolio_controller = PortfolioExposureController(matchday_cap=0.12, global_cap=0.30, kelly_fraction=0.25)
    df_opps = portfolio_controller.apply(df_opps)
    port_note = portfolio_controller.audit_note()

    # Keep at most the highest-confidence pick per fixture so we don't stack correlated markets
    if not df_opps.empty:
        df_opps = (
            df_opps.sort_values(["p_calibrated", "ev_percent"], ascending=False)
            .drop_duplicates(subset=["league", "home", "away"], keep="first")
            .sort_values("date")
            .reset_index(drop=True)
        )

    # Record approved picks to ledger for automatic future settlement
    recorded = record_approved_bets_to_ledger(df_opps)
    if recorded:
        summary["pending_after"] = int(summary.get("pending_after", 0)) + int(recorded)
        print(f"[LEDGER] Recorded {recorded} new upcoming pick(s) for future settlement.")

    # Export full approved slate & verified performance to data/public/ (Next.js ready)
    exported = export_public_json_bundle(df_opps, summary)
    print(f"[PUBLIC EXPORT] Updated: {[str(p.name) for p in exported.values()]}")

    # 5. Dedup Telegram notifications (so chat never gets duplicate alerts)
    sent_keys = _load_sent_keys()
    df_telegram, newly_sent_keys, skipped_sent = dedup_opportunities(df_opps, sent_keys)
    if skipped_sent:
        print(f"[DEDUP] skipped {skipped_sent} already-dispatched recommendation(s).")

    critic_note = (
        f"{len(df_telegram)} معتمدة / {len(df_rejected)} مرفوضة"
        + (f" / {skipped_sent} مكررة" if skipped_sent else "")
        + f" (هامش أمان +6% | المحفظة: {port_note})"
    )
    report_text = format_telegram_report(summary, df_telegram, critic_note=critic_note)
    print("\n--- [TELEGRAM PREVIEW] ---")
    print(report_text)
    print("--------------------------\n")

    sent_ok = send_telegram_alert(report_text)
    persisted_keys = sorted(sent_keys | newly_sent_keys) if sent_ok else sorted(sent_keys)

    Path("telegram_last_report.txt").write_text(report_text, encoding="utf-8")
    out_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settlement": summary,
        "opportunities_count": int(len(df_telegram)),
        "opportunities": df_telegram.astype(str).to_dict(orient="records") if not df_telegram.empty else [],
        "rejected_count": int(len(df_rejected)),
        "rejected": df_rejected.astype(str).to_dict(orient="records") if not df_rejected.empty else [],
        "report_markdown": report_text,
        "telegram_dispatched": bool(sent_ok),
        "sent_fixture_keys": persisted_keys,
    }
    Path("autoloop_last_run.json").write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("[SAVED] telegram_last_report.txt & autoloop_last_run.json updated.")
    print(f"[AUTOLOOP v3] Complete in {time.time() - start_time:.2f}s.")


def _send_telegram_test() -> None:
    msg = (
        "🧪 <b>اختبار الإرسال — Telegram Delivery Test (v2)</b>\n"
        "إذا وصلت هذه الرسالة فهذا دليل أن البوت يعمل ويصل إلى دردشتك.\n"
        f"• الزمن: <code>{datetime.now(timezone.utc).isoformat()}</code>\n"
        "• يليها تقرير تسوية الجولة القادمة."
    )
    print("--- [TELEGRAM TEST] ---")
    print(msg)
    print("------------------------")
    send_telegram_alert(msg)


if __name__ == "__main__":
    if "--telegram-test" in sys.argv:
        _send_telegram_test()
    else:
        main()
