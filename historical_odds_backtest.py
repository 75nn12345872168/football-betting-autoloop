"""
historical_odds_backtest.py — Strict Out-of-Sample Walk-Forward ROI & CLV Backtest (Big-5).

Evaluates the upgraded quantitative engine across all 5 major European leagues
(La Liga, Ligue 1, EPL, Bundesliga, Serie A) against REAL historical bookmaker
odds & Pinnacle closing lines (football-data.co.uk, 2018-2026) with ZERO leakage:

  1. Strictly causal (< match_day) TeamStrengthModel: xi=0.003, 0.7*xG + 0.3*Goals.
  2. Strictly causal (< season_start) per-league Dixon-Coles MLE (mu_h, mu_a, rho).
  3. Strictly causal (< season) expanding-window Platt Calibration (fit ONLY on
     strictly prior out-of-sample seasons).
  4. Entry criteria:
       (a) p_calibrated >= 0.58
       (b) Market_Odds >= 1.06 / p_calibrated  (+6% EV safety margin)
  5. Reports Flat 1u & Fractional Kelly (0.25x, max 5% cap) ROI%, Net Profit,
     Max Drawdown%, Win Rate%, Avg Entry Odds, Brier Skill%, and Pinnacle CLV%.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autoloop_master import (
    BROKER_SAFETY_MARGIN,
    _logit,
    _sigmoid,
    calculate_fractional_kelly,
    calculate_market_probabilities,
    compute_dixon_coles_probs,
    fit_league_mle_dynamic,
    fit_platt_calibrator,
)
from live_data import BIG5_HISTORY_CSV, BIG5_LEAGUES, REPO_DIR
from team_strength import DEFAULT_XI, TeamStrengthModel, XG_WEIGHT

PUBLIC_DIR = REPO_DIR / "data" / "public"
BACKTEST_JSON = PUBLIC_DIR / "backtest_verified_stats.json"

MIN_CONF = 0.58
SAFETY_MARGIN = BROKER_SAFETY_MARGIN  # 1.06
KELLY_FRACTION = 0.25
KELLY_MAX_STAKE = 0.05
START_BANKROLL = 100.0

MARKETS = ("1X2 Home", "1X2 Away", "Over 2.5", "Under 2.5")


def _vig_free_prob_3way(o_target: float, o1: float, o2: float, o3: float) -> Optional[float]:
    if not all(x is not None and np.isfinite(x) and x > 1.0 for x in (o_target, o1, o2, o3)):
        return None
    s = (1.0 / o1) + (1.0 / o2) + (1.0 / o3)
    if s <= 0:
        return None
    return (1.0 / o_target) / s


def _vig_free_prob_2way(o_target: float, o_other: float) -> Optional[float]:
    if not all(x is not None and np.isfinite(x) and x > 1.0 for x in (o_target, o_other)):
        return None
    s = (1.0 / o_target) + (1.0 / o_other)
    if s <= 0:
        return None
    return (1.0 / o_target) / s


def _simulate_strategy(
    bets: List[Dict[str, Any]], odds_key: str
) -> Dict[str, Any]:
    """
    Simulate Flat 1u and 0.25x Fractional Kelly staking on bets that satisfy:
      p_cal >= 0.58 AND odds >= 1.06 / p_cal.
    """
    qualified = []
    for b in bets:
        odds = b.get(odds_key)
        p_cal = b["p_cal"]
        if odds is None or not np.isfinite(odds) or odds <= 1.0:
            continue
        if p_cal >= MIN_CONF and odds >= (SAFETY_MARGIN / p_cal):
            qualified.append((b, float(odds)))

    n = len(qualified)
    if n == 0:
        return {
            "total_bets": 0,
            "wins": 0,
            "win_rate_pct": 0.0,
            "avg_pred_pct": 0.0,
            "avg_odds": 0.0,
            "flat_profit_units": 0.0,
            "flat_roi_pct": 0.0,
            "flat_max_dd_pct": 0.0,
            "kelly_profit_units": 0.0,
            "kelly_roi_pct": 0.0,
            "kelly_bankroll_end": START_BANKROLL,
            "kelly_max_dd_pct": 0.0,
            "clv_fair_pct": None,
            "clv_raw_pct": None,
            "clv_sample_n": 0,
        }

    # Flat 1-unit bankroll (starts at 100u so drawdown % is directly comparable)
    flat_eq = START_BANKROLL
    flat_peak = START_BANKROLL
    flat_max_dd = 0.0
    flat_pnl = 0.0

    # Fractional Kelly bankroll (starts at 100u)
    kelly_eq = START_BANKROLL
    kelly_peak = START_BANKROLL
    kelly_max_dd = 0.0
    kelly_staked_total = 0.0
    kelly_pnl_total = 0.0

    wins = 0
    odds_list = []
    pred_list = []
    clv_fair_list = []
    clv_raw_list = []

    for b, odds in qualified:
        y = int(b["y"])
        p_cal = float(b["p_cal"])
        wins += y
        odds_list.append(odds)
        pred_list.append(p_cal)

        # Flat 1u
        pnl_u = (odds - 1.0) if y == 1 else -1.0
        flat_pnl += pnl_u
        flat_eq += pnl_u
        if flat_eq > flat_peak:
            flat_peak = flat_eq
        dd_f = (flat_peak - flat_eq) / flat_peak * 100.0 if flat_peak > 0 else 0.0
        if dd_f > flat_max_dd:
            flat_max_dd = dd_f

        # Fractional Kelly
        f_stake = calculate_fractional_kelly(p_cal, odds, fraction=KELLY_FRACTION, max_stake=KELLY_MAX_STAKE)
        stake_u = kelly_eq * f_stake
        k_pnl = stake_u * (odds - 1.0) if y == 1 else -stake_u
        kelly_staked_total += stake_u
        kelly_pnl_total += k_pnl
        kelly_eq = max(0.01, kelly_eq + k_pnl)
        if kelly_eq > kelly_peak:
            kelly_peak = kelly_eq
        dd_k = (kelly_peak - kelly_eq) / kelly_peak * 100.0 if kelly_peak > 0 else 0.0
        if dd_k > kelly_max_dd:
            kelly_max_dd = dd_k

        # CLV vs Pinnacle Closing Line
        p_close_fair = b.get("p_close_fair")
        o_close = b.get("close_odds")
        if p_close_fair is not None and np.isfinite(p_close_fair):
            clv_fair_list.append((odds * p_close_fair - 1.0) * 100.0)
        if o_close is not None and np.isfinite(o_close) and o_close > 1.0:
            clv_raw_list.append((odds / o_close - 1.0) * 100.0)

    return {
        "total_bets": int(n),
        "wins": int(wins),
        "win_rate_pct": round(wins / n * 100.0, 2),
        "avg_pred_pct": round(float(np.mean(pred_list)) * 100.0, 2),
        "avg_odds": round(float(np.mean(odds_list)), 3),
        "flat_profit_units": round(flat_pnl, 2),
        "flat_roi_pct": round(flat_pnl / n * 100.0, 2),
        "flat_max_dd_pct": round(flat_max_dd, 2),
        "kelly_profit_units": round(kelly_pnl_total, 2),
        "kelly_roi_pct": round((kelly_pnl_total / kelly_staked_total * 100.0) if kelly_staked_total > 0 else 0.0, 2),
        "kelly_bankroll_end": round(kelly_eq, 2),
        "kelly_max_dd_pct": round(kelly_max_dd, 2),
        "clv_fair_pct": round(float(np.mean(clv_fair_list)), 2) if clv_fair_list else None,
        "clv_raw_pct": round(float(np.mean(clv_raw_list)), 2) if clv_raw_list else None,
        "clv_sample_n": int(len(clv_fair_list)),
    }


def _brier_and_ll(probs: np.ndarray, actuals: np.ndarray) -> Tuple[float, float, float]:
    eps = 1e-9
    p = np.clip(probs, eps, 1.0 - eps)
    y = actuals
    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean(y))
    b_base = float(np.mean((base - y) ** 2))
    b_skill = (1.0 - brier / b_base) * 100.0 if b_base > 0 else 0.0
    ll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    ll_base = float(-(base * math.log(base + eps) + (1.0 - base) * math.log(1.0 - base + eps)))
    ll_skill = (1.0 - ll / ll_base) * 100.0 if ll_base > 0 else 0.0
    return round(brier, 5), round(b_skill, 2), round(ll_skill, 2)


def run_backtest(
    csv_path: Path = BIG5_HISTORY_CSV,
    save_json: bool = True,
) -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Historical Big-5 dataset not found: {csv_path}. Run build_big5_dataset.py first.")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["league", "date", "home", "away", "goals_home", "goals_away"])
    df = df.sort_values(["date", "league"]).reset_index(drop=True)

    # Causal strength model loaded cleanly from the unified dataset
    model = TeamStrengthModel(
        la_csv="", ligue_json="", ligue_live_csv="",
        big5_csv=str(csv_path), xi=DEFAULT_XI, xg_weight=XG_WEIGHT,
    )

    # Stage 1: Generate strictly causal out-of-sample RAW probabilities for every match
    print(f"[BACKTEST] Generating strictly causal predictions across {len(df)} Big-5 matches...")
    raw_records: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        (lg, mk): [] for lg in BIG5_LEAGUES for mk in MARKETS
    }

    # Fit per-league seasonal MLE strictly BEFORE each season starts (zero look-ahead)
    mle_season_cache: Dict[Tuple[str, int], Tuple[float, float, float]] = {}

    for lg in BIG5_LEAGUES:
        lg_df = df[df["league"] == lg].sort_values("date").reset_index(drop=True)
        seasons = sorted(lg_df["season"].dropna().astype(int).unique().tolist())

        for _, row in lg_df.iterrows():
            d_day = row["date"].normalize()
            season = int(row["season"])
            if (lg, season) not in mle_season_cache:
                # Fit MLE strictly before this season's first match date
                first_day = lg_df[lg_df["season"] == season]["date"].min().normalize()
                mle = fit_league_mle_dynamic(
                    league=lg, team_model=model, as_of=first_day, use_cache=False
                )
                mle_season_cache[(lg, season)] = (mle["mu_home"], mle["mu_away"], mle["rho"])

            _, _, rho = mle_season_cache[(lg, season)]
            # Strictly causal < d_day baselines + team strengths
            mu_h, mu_a, _ = model.league_baselines(lg, as_of=d_day)
            att_h, def_h = model.strength(lg, row["home"], as_of=d_day)
            att_a, def_a = model.strength(lg, row["away"], as_of=d_day)

            lh = min(5.0, max(0.3, mu_h * att_h * def_a))
            la = min(5.0, max(0.3, mu_a * att_a * def_h))
            mp = calculate_market_probabilities(compute_dixon_coles_probs(lh, la, max_goals=8, rho=rho))

            gh, ga = int(row["goals_home"]), int(row["goals_away"])
            psh, psd, psa = row.get("PSH"), row.get("PSD"), row.get("PSA")
            psch, pscd, psca = row.get("PSCH"), row.get("PSCD"), row.get("PSCA")
            maxh, maxa = row.get("MaxH"), row.get("MaxA")
            po, pu = row.get("P_over25"), row.get("P_under25")
            pco, pcu = row.get("PC_over25"), row.get("PC_under25")
            maxo, maxu = row.get("Max_over25"), row.get("Max_under25")

            # Use Max market odds, falling back to Pinnacle if Max missing; and Pinnacle odds separately
            market_entries = [
                (
                    "1X2 Home",
                    mp["home_win"],
                    1 if gh > ga else 0,
                    maxh if pd.notna(maxh) else psh,
                    psh,
                    psch,
                    _vig_free_prob_3way(psch, psch, pscd, psca),
                ),
                (
                    "1X2 Away",
                    mp["away_win"],
                    1 if ga > gh else 0,
                    maxa if pd.notna(maxa) else psa,
                    psa,
                    psca,
                    _vig_free_prob_3way(psca, psch, pscd, psca),
                ),
                (
                    "Over 2.5",
                    mp["over_2.5"],
                    1 if (gh + ga) > 2.5 else 0,
                    maxo if pd.notna(maxo) else po,
                    po,
                    pco,
                    _vig_free_prob_2way(pco, pcu),
                ),
                (
                    "Under 2.5",
                    mp["under_2.5"],
                    1 if (gh + ga) < 2.5 else 0,
                    maxu if pd.notna(maxu) else pu,
                    pu,
                    pcu,
                    _vig_free_prob_2way(pcu, pco),
                ),
            ]
            for mk, p_raw, y, best_odds, pin_odds, close_odds, p_close_fair in market_entries:
                raw_records[(lg, mk)].append({
                    "season": season,
                    "date": str(row["date"])[:10],
                    "p_raw": float(p_raw),
                    "y": int(y),
                    "max_odds": float(best_odds) if pd.notna(best_odds) else None,
                    "pin_odds": float(pin_odds) if pd.notna(pin_odds) else None,
                    "close_odds": float(close_odds) if pd.notna(close_odds) else None,
                    "p_close_fair": p_close_fair,
                })

    # Stage 2: Expanding-window out-of-sample Platt Calibration (strictly season < current_season)
    # Test seasons: 2019..2026 (2018 is warmup & initial Platt fit only — NEVER bet on!)
    print("[BACKTEST] Applying strictly expanding out-of-sample Platt calibration & evaluating ROI/CLV...")
    results_by_cell: List[Dict[str, Any]] = []
    all_oos_bets_max: List[Dict[str, Any]] = []
    fitted_platt_latest: Dict[str, Tuple[float, float]] = {}

    for lg in BIG5_LEAGUES:
        for mk in MARKETS:
            recs = raw_records[(lg, mk)]
            seasons = sorted({r["season"] for r in recs})
            oos_bets: List[Dict[str, Any]] = []

            for s in seasons:
                if s <= 2018:
                    continue  # 2018 is strictly warmup
                prior = [r for r in recs if r["season"] < s]
                cur = [r for r in recs if r["season"] == s]
                if len(prior) >= 100:
                    a, b = fit_platt_calibrator(
                        np.array([r["p_raw"] for r in prior]),
                        np.array([r["y"] for r in prior]),
                    )
                else:
                    a, b = 1.0, 0.0
                for r in cur:
                    p_cal = float(np.clip(_sigmoid(a * _logit(r["p_raw"]) + b), 0.01, 0.99))
                    item = dict(r)
                    item["p_cal"] = round(p_cal, 4)
                    oos_bets.append(item)

            # Fit final deployable Platt params on all completed history for production use
            if len(recs) >= 200:
                af, bf = fit_platt_calibrator(
                    np.array([r["p_raw"] for r in recs]),
                    np.array([r["y"] for r in recs]),
                )
                fitted_platt_latest[f"{lg.lower()}|{mk.lower()}"] = (af, bf)

            p_arr = np.array([r["p_cal"] for r in oos_bets], dtype=float)
            y_arr = np.array([r["y"] for r in oos_bets], dtype=float)
            brier, b_skill, ll_skill = _brier_and_ll(p_arr, y_arr)

            max_stats = _simulate_strategy(oos_bets, "max_odds")
            pin_stats = _simulate_strategy(oos_bets, "pin_odds")
            all_oos_bets_max.extend(oos_bets)

            # Qualification for production: profitable against real market odds (ROI > 0%),
            # meaningful sample size (>= 15 bets), and positive Brier skill.
            profitable_max = (
                max_stats["total_bets"] >= 15
                and max_stats["flat_roi_pct"] > 0.0
                and b_skill > 0.0
            )
            profitable_pin = (
                pin_stats["total_bets"] >= 10
                and pin_stats["flat_roi_pct"] > 0.0
            )

            cell = {
                "league": lg,
                "market": mk,
                "oos_matches": int(len(oos_bets)),
                "brier": brier,
                "brier_skill_pct": b_skill,
                "logloss_skill_pct": ll_skill,
                "best_market_odds": max_stats,
                "pinnacle_odds": pin_stats,
                "profitable_market": bool(profitable_max),
                "profitable_pinnacle": bool(profitable_pin),
            }
            results_by_cell.append(cell)

    # Aggregate portfolio summary across ALL cells vs PROFITABLE-ONLY cells
    overall_all_max = _simulate_strategy(all_oos_bets_max, "max_odds")
    overall_all_pin = _simulate_strategy(all_oos_bets_max, "pin_odds")

    profitable_pairs = {(c["league"], c["market"]) for c in results_by_cell if c["profitable_market"]}
    prof_bets = []
    for lg, mk in profitable_pairs:
        recs = raw_records[(lg, mk)]
        seasons = sorted({r["season"] for r in recs})
        for s in seasons:
            if s <= 2018:
                continue
            prior = [r for r in recs if r["season"] < s]
            cur = [r for r in recs if r["season"] == s]
            a, b = fit_platt_calibrator(
                np.array([r["p_raw"] for r in prior]),
                np.array([r["y"] for r in prior]),
            ) if len(prior) >= 100 else (1.0, 0.0)
            for r in cur:
                item = dict(r)
                item["p_cal"] = round(float(np.clip(_sigmoid(a * _logit(r["p_raw"]) + b), 0.01, 0.99)), 4)
                prof_bets.append(item)
    prof_bets.sort(key=lambda x: x["date"])
    profitable_portfolio_max = _simulate_strategy(prof_bets, "max_odds")
    profitable_portfolio_pin = _simulate_strategy(prof_bets, "pin_odds")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "model": "Dixon-Coles MLE + Causal Time-Decayed xG TeamStrength + Expanding OOS Platt Calibration",
            "xi_decay": DEFAULT_XI,
            "xg_weight": XG_WEIGHT,
            "entry_min_confidence": MIN_CONF,
            "entry_safety_margin": SAFETY_MARGIN,
            "entry_rule": f"p_calibrated >= {MIN_CONF} AND Market_Odds >= {SAFETY_MARGIN} / p_calibrated",
            "kelly_fraction": KELLY_FRACTION,
            "kelly_max_stake": KELLY_MAX_STAKE,
            "oos_seasons": "2019-2026 (2018 warmup excluded from betting)",
            "data_leakage_protection": "strict < match_day for strengths/baselines; strict < season for MLE and Platt",
        },
        "overall_all_big5_best_market": overall_all_max,
        "overall_all_big5_pinnacle": overall_all_pin,
        "profitable_selection_best_market": profitable_portfolio_max,
        "profitable_selection_pinnacle": profitable_portfolio_pin,
        "approved_production_pairs": sorted([[lg, mk] for lg, mk in profitable_pairs]),
        "fitted_platt_params": {k: list(v) for k, v in sorted(fitted_platt_latest.items())},
        "breakdown": results_by_cell,
    }

    if save_json:
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
        BACKTEST_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print clean summary table
    print("\n=== STRICT OUT-OF-SAMPLE HISTORICAL ODDS BACKTEST (2019-2026, BEST MARKET ODDS) ===")
    print(f"{'League':11s} {'Market':10s} {'Bets':>5s} {'Win%':>6s} {'AvgOdd':>7s} {'FlatP&L':>8s} {'FlatROI':>8s} {'FlatDD':>7s} {'KelROI':>8s} {'CLV%':>7s} {'BrierSk':>8s}")
    for c in results_by_cell:
        m = c["best_market_odds"]
        clv = f"{m['clv_fair_pct']:+6.2f}%" if m["clv_fair_pct"] is not None else "    N/A"
        tag = "*" if c["profitable_market"] else " "
        print(
            f"{tag}{c['league']:10s} {c['market']:10s} {m['total_bets']:5d} {m['win_rate_pct']:5.1f}% "
            f"{m['avg_odds']:7.3f} {m['flat_profit_units']:+8.2f} {m['flat_roi_pct']:+7.2f}% "
            f"{m['flat_max_dd_pct']:6.2f}% {m['kelly_roi_pct']:+7.2f}% {clv} {c['brier_skill_pct']:+7.2f}%"
        )

    print("\n=== SAME FILTER AGAINST PINNACLE OPENING ODDS ===")
    for c in results_by_cell:
        m = c["pinnacle_odds"]
        if m["total_bets"] == 0:
            continue
        clv = f"{m['clv_fair_pct']:+6.2f}%" if m["clv_fair_pct"] is not None else "    N/A"
        print(
            f" {c['league']:10s} {c['market']:10s} {m['total_bets']:5d} {m['win_rate_pct']:5.1f}% "
            f"{m['avg_odds']:7.3f} {m['flat_profit_units']:+8.2f} {m['flat_roi_pct']:+7.2f}% "
            f"{m['flat_max_dd_pct']:6.2f}% {m['kelly_roi_pct']:+7.2f}% {clv}"
        )

    print("\n=== AGGREGATE PORTFOLIOS ===")
    print("ALL Big-5 (Best Market):      ", overall_all_max)
    print("ALL Big-5 (Pinnacle):         ", overall_all_pin)
    print("PROFITABLE ONLY (Best Market):", profitable_portfolio_max)
    print("PROFITABLE ONLY (Pinnacle):   ", profitable_portfolio_pin)
    print("Approved Production Pairs:    ", payload["approved_production_pairs"])
    return payload


if __name__ == "__main__":
    run_backtest()
