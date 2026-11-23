"""
research_alpha_lab.py — Strict Out-of-Sample Quantitative R&D Lab (2019-2026).

Tests three independent alpha hypotheses against the verified v3 Baseline
(+5.45% Best Market ROI, +6.13% Pinnacle ROI, 610 bets, +0.16% CLV, 13.12% Max DD)
with ZERO Data Leakage (strict < match_day features, strict < season MLE & Platt):

  Experiment 1 — Feature Engineering:
    (1A) Home/Away venue-split attack & defense strengths (shrunk toward pooled strength).
    (1B) Finishing Luck Mean-Reversion (time-weighted Goals - xG over last 6 games).
    (1C) Schedule Fatigue / Rest Days penalty (<= 3 days rest offensive discount).
    (1D) Combined combinations (A+B, B+C, A+C, A+B+C).

  Experiment 2 — Threshold & EV Margin Sweep:
    Grid of p_calibrated in [0.58, 0.60, 0.62] x EV_margin in [+6%, +8%, +10%]
    across Best Market & Pinnacle Opening odds, plus season-by-season stability.

  Experiment 3 — Draw No Bet (DNB / AH 0) vs Direct 1X2:
    Exact tradable replication DNB odds O_dnb = O_side * (1 - 1 / O_draw) on
    La Liga Home DNB and Serie A Away DNB vs direct 1X2, measuring ROI, Max DD,
    and Pinnacle Closing DNB CLV%.
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from autoloop_master import (
    _logit,
    _sigmoid,
    calculate_fractional_kelly,
    calculate_market_probabilities,
    compute_dixon_coles_probs,
    fit_league_mle_dynamic,
    fit_platt_calibrator,
)
from live_data import BIG5_HISTORY_CSV, BIG5_LEAGUES, REPO_DIR, canonical_team_name
from team_strength import DEFAULT_XI, K_SHRINK, XG_WEIGHT, TeamStrengthModel, blend_xg_goals, time_decay_weight

APPROVED_4_PAIRS = (
    ("La Liga", "1X2 Home"),
    ("La Liga", "Under 2.5"),
    ("Serie A", "1X2 Away"),
    ("Serie A", "Over 2.5"),
)

MARKETS = ("1X2 Home", "1X2 Away", "Over 2.5", "Under 2.5")


def _vig_free_3way(o_t: float, o1: float, o2: float, o3: float) -> Optional[float]:
    if not all(x is not None and np.isfinite(x) and x > 1.0 for x in (o_t, o1, o2, o3)):
        return None
    s = (1.0 / o1) + (1.0 / o2) + (1.0 / o3)
    return (1.0 / o_t) / s if s > 0 else None


def _vig_free_2way(o_t: float, o_other: float) -> Optional[float]:
    if not all(x is not None and np.isfinite(x) and x > 1.0 for x in (o_t, o_other)):
        return None
    s = (1.0 / o_t) + (1.0 / o_other)
    return (1.0 / o_t) / s if s > 0 else None


def _synth_dnb_odds(o_side: float, o_draw: float) -> Optional[float]:
    """
    Exact tradable replication odds for Draw No Bet (AH 0):
    Put 1/O_draw on Draw (refund on draw) and (1 - 1/O_draw) on side win.
    Net decimal payout on win = O_side * (1 - 1/O_draw).
    """
    if not (o_side is not None and o_draw is not None and np.isfinite(o_side) and np.isfinite(o_draw)):
        return None
    if o_side <= 1.0 or o_draw <= 1.0:
        return None
    odnb = o_side * (1.0 - 1.0 / o_draw)
    return float(odnb) if odnb > 1.0 else None


class ResearchStrengthEngine:
    """
    Fast, strictly causal (< as_of_day) feature & strength engine supporting
    configurable toggles for:
      * venue_split: Home/Away split attack & defense shrunk toward pooled team strength
      * luck_beta: Finishing Luck Mean-Reversion over last 6 games (Goals - xG)
      * fatigue_discount: Offensive multiplier when rest_days <= 3 (e.g. 0.94)
    """

    def __init__(self, df: pd.DataFrame, xi: float = DEFAULT_XI, xg_weight: float = XG_WEIGHT) -> None:
        self.xi = xi
        self.xg_weight = xg_weight
        self.leagues: Dict[str, Dict[str, Any]] = {}
        self._baseline_cache: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}

        for lg in BIG5_LEAGUES:
            sub = df[df["league"] == lg].sort_values("date")
            matches = []
            events_all: Dict[str, List[Tuple[float, float, float, float, float, float, float, int]]] = {}
            events_h: Dict[str, List[Tuple[float, float, float]]] = {}
            events_a: Dict[str, List[Tuple[float, float, float]]] = {}

            for _, r in sub.iterrows():
                ts = float(r["date"].value) / 1e9
                h = canonical_team_name(r["home"], lg)
                a = canonical_team_name(r["away"], lg)
                gh, ga = float(r["goals_home"]), float(r["goals_away"])
                xgh = float(r["xG_home"]) if pd.notna(r["xG_home"]) else gh
                xga = float(r["xG_away"]) if pd.notna(r["xG_away"]) else ga
                eh = blend_xg_goals(gh, xgh, xg_weight)
                ea = blend_xg_goals(ga, xga, xg_weight)
                matches.append((ts, h, a, gh, ga, eh, ea))
                # (ts, eff_for, eff_ag, goals_for, xg_for, goals_ag, xg_ag, is_home)
                events_all.setdefault(h, []).append((ts, eh, ea, gh, xgh, ga, xga, 1))
                events_all.setdefault(a, []).append((ts, ea, eh, ga, xga, gh, xgh, 0))
                events_h.setdefault(h, []).append((ts, eh, ea))
                events_a.setdefault(a, []).append((ts, ea, eh))

            self.leagues[lg] = {
                "matches": matches,
                "match_ts": [m[0] for m in matches],
                "events_all": events_all,
                "events_all_ts": {t: [e[0] for e in v] for t, v in events_all.items()},
                "events_h": events_h,
                "events_h_ts": {t: [e[0] for e in v] for t, v in events_h.items()},
                "events_a": events_a,
                "events_a_ts": {t: [e[0] for e in v] for t, v in events_a.items()},
            }

    def baselines(self, league: str, ref_ts: float) -> Tuple[float, float, float, float]:
        day_int = int(ref_ts // 86400)
        ck = (league, day_int)
        if ck in self._baseline_cache:
            return self._baseline_cache[ck]
        s = self.leagues[league]
        idx = bisect.bisect_left(s["match_ts"], ref_ts)
        sub = s["matches"][:idx]
        if not sub:
            return 1.50, 1.20, 1.50, 1.20
        w_sum = gh_sum = ga_sum = eh_sum = ea_sum = 0.0
        for ts, _, _, gh, ga, eh, ea in sub:
            w = math.exp(-self.xi * max(0.0, (ref_ts - ts) / 86400.0))
            w_sum += w
            gh_sum += w * gh
            ga_sum += w * ga
            eh_sum += w * eh
            ea_sum += w * ea
        res = (gh_sum / w_sum, ga_sum / w_sum, eh_sum / w_sum, ea_sum / w_sum)
        self._baseline_cache[ck] = res
        return res

    def _pooled_strength(self, league: str, team: str, ref_ts: float, lpg_eff: float) -> Tuple[float, float]:
        s = self.leagues[league]
        evs = s["events_all"].get(team)
        if not evs:
            return 1.0, 1.0
        idx = bisect.bisect_left(s["events_all_ts"][team], ref_ts)
        played = evs[:idx]
        if len(played) < 5:
            return 1.0, 1.0
        w_sum = ef_sum = ea_sum = 0.0
        for ts, ef, ea, _, _, _, _, _ in played:
            w = math.exp(-self.xi * max(0.0, (ref_ts - ts) / 86400.0))
            w_sum += w
            ef_sum += w * ef
            ea_sum += w * ea
        sh = w_sum / (w_sum + K_SHRINK)
        att = 1.0 + sh * ((ef_sum / w_sum) / max(0.05, lpg_eff) - 1.0)
        deff = 1.0 + sh * ((ea_sum / w_sum) / max(0.05, lpg_eff) - 1.0)
        return max(0.3, min(2.2, att)), max(0.3, min(2.2, deff))

    def _venue_strength(
        self, league: str, team: str, is_home: bool, ref_ts: float,
        mu_h_eff: float, mu_a_eff: float, k_venue: float = 12.0,
    ) -> Tuple[float, float]:
        """Hierarchical Home/Away split shrunk toward the team's pooled strength."""
        lpg = (mu_h_eff + mu_a_eff) / 2.0
        p_att, p_def = self._pooled_strength(league, team, ref_ts, lpg)
        s = self.leagues[league]
        evs = (s["events_h"] if is_home else s["events_a"]).get(team)
        if not evs:
            return p_att, p_def
        ts_arr = (s["events_h_ts"] if is_home else s["events_a_ts"])[team]
        idx = bisect.bisect_left(ts_arr, ref_ts)
        played = evs[:idx]
        if len(played) < 4:
            return p_att, p_def
        w_sum = ef_sum = ea_sum = 0.0
        for ts, ef, ea in played:
            w = math.exp(-self.xi * max(0.0, (ref_ts - ts) / 86400.0))
            w_sum += w
            ef_sum += w * ef
            ea_sum += w * ea
        denom_att = mu_h_eff if is_home else mu_a_eff
        denom_def = mu_a_eff if is_home else mu_h_eff
        raw_att = (ef_sum / w_sum) / max(0.05, denom_att)
        raw_def = (ea_sum / w_sum) / max(0.05, denom_def)
        sh_v = w_sum / (w_sum + k_venue)
        att = p_att + sh_v * (raw_att - p_att)
        deff = p_def + sh_v * (raw_def - p_def)
        return max(0.3, min(2.2, att)), max(0.3, min(2.2, deff))

    def _finishing_luck_last6(self, league: str, team: str, ref_ts: float) -> Tuple[float, float]:
        """
        Time-weighted average (Goals - xG) over the team's last 6 strictly prior matches.
        Returns (attack_luck, defense_luck) where:
          * attack_luck > 0 means team scored MORE than xG (lucky finishing -> expected reversion down)
          * defense_luck > 0 means team conceded MORE than xGA (unlucky defending -> expected reversion)
        """
        s = self.leagues[league]
        evs = s["events_all"].get(team)
        if not evs:
            return 0.0, 0.0
        idx = bisect.bisect_left(s["events_all_ts"][team], ref_ts)
        recent = evs[max(0, idx - 6):idx]
        if len(recent) < 3:
            return 0.0, 0.0
        w_sum = att_luck = def_luck = 0.0
        for k, (ts, _, _, gf, xgf, ga, xga, _) in enumerate(recent):
            # Exponential recency weighting within the last 6 games
            w = math.exp(-0.015 * max(0.0, (ref_ts - ts) / 86400.0))
            w_sum += w
            att_luck += w * (gf - xgf)
            def_luck += w * (ga - xga)
        return (
            float(np.clip(att_luck / w_sum, -1.2, 1.2)),
            float(np.clip(def_luck / w_sum, -1.2, 1.2)),
        )

    def _rest_days(self, league: str, team: str, ref_ts: float) -> Optional[float]:
        """Days since team's previous league match strictly before ref_ts."""
        s = self.leagues[league]
        ts_arr = s["events_all_ts"].get(team)
        if not ts_arr:
            return None
        idx = bisect.bisect_left(ts_arr, ref_ts)
        if idx == 0:
            return None
        return max(0.0, (ref_ts - ts_arr[idx - 1]) / 86400.0)

    def predict_lambdas(
        self,
        league: str,
        home: str,
        away: str,
        ref_ts: float,
        venue_split: bool = False,
        luck_beta: float = 0.0,
        fatigue_discount: float = 1.0,
    ) -> Tuple[float, float]:
        mu_h, mu_a, mu_h_eff, mu_a_eff = self.baselines(league, ref_ts)
        lpg = (mu_h_eff + mu_a_eff) / 2.0

        if venue_split:
            att_h, def_h = self._venue_strength(league, home, True, ref_ts, mu_h_eff, mu_a_eff)
            att_a, def_a = self._venue_strength(league, away, False, ref_ts, mu_h_eff, mu_a_eff)
        else:
            att_h, def_h = self._pooled_strength(league, home, ref_ts, lpg)
            att_a, def_a = self._pooled_strength(league, away, ref_ts, lpg)

        lh = mu_h * att_h * def_a
        la = mu_a * att_a * def_h

        # (B) Finishing Luck Mean-Reversion:
        # Overperforming finishing (Goals > xG recently) reverts downward;
        # unlucky conceding (GA > xGA recently) reverts downward on opponent scoring.
        if luck_beta != 0.0:
            hl_att, hl_def = self._finishing_luck_last6(league, home, ref_ts)
            al_att, al_def = self._finishing_luck_last6(league, away, ref_ts)
            lh *= math.exp(-luck_beta * (hl_att + 0.5 * al_def))
            la *= math.exp(-luck_beta * (al_att + 0.5 * hl_def))

        # (C) Schedule Fatigue (Rest Days <= 3.5 days -> offensive discount)
        if fatigue_discount < 1.0:
            rh = self._rest_days(league, home, ref_ts)
            ra = self._rest_days(league, away, ref_ts)
            if rh is not None and rh <= 3.5:
                lh *= fatigue_discount
            if ra is not None and ra <= 3.5:
                la *= fatigue_discount

        return min(5.0, max(0.3, lh)), min(5.0, max(0.3, la))


def simulate_bets(
    bets: List[Dict[str, Any]],
    odds_key: str,
    min_conf: float = 0.58,
    ev_margin: float = 1.06,
    is_dnb: bool = False,
) -> Dict[str, Any]:
    qual = []
    for b in bets:
        odds = b.get(odds_key)
        p_cal = b["p_cal"]
        if odds is None or not np.isfinite(odds) or odds <= 1.0:
            continue
        if p_cal >= min_conf and odds >= (ev_margin / p_cal):
            qual.append((b, float(odds)))

    n = len(qual)
    if n == 0:
        return {
            "bets": 0, "win_rate": 0.0, "avg_odds": 0.0,
            "flat_pnl": 0.0, "flat_roi": 0.0, "flat_dd": 0.0,
            "kelly_roi": 0.0, "kelly_end": 100.0, "kelly_dd": 0.0,
            "clv_fair": None, "pos_seasons": "0/0", "season_rois": {},
        }

    flat_eq = flat_peak = 100.0
    flat_dd = flat_pnl = 0.0
    kelly_eq = kelly_peak = 100.0
    kelly_dd = kelly_staked = kelly_pnl = 0.0
    wins = pushes = 0
    odds_arr = []
    clv_arr = []
    by_season: Dict[int, List[float]] = {}

    for b, odds in qual:
        y = b["y"]  # 1=win, 0=loss, -1=push (DNB draw)
        p_cal = b["p_cal"]
        odds_arr.append(odds)
        if y == 1:
            wins += 1
            pnl = odds - 1.0
        elif y == -1:
            pushes += 1
            pnl = 0.0
        else:
            pnl = -1.0

        flat_pnl += pnl
        flat_eq += pnl
        flat_peak = max(flat_peak, flat_eq)
        flat_dd = max(flat_dd, (flat_peak - flat_eq) / flat_peak * 100.0)
        by_season.setdefault(int(b["season"]), []).append(pnl)

        f_stake = calculate_fractional_kelly(p_cal, odds, fraction=0.25, max_stake=0.05)
        stake_u = kelly_eq * f_stake
        kpnl = stake_u * pnl
        kelly_staked += stake_u
        kelly_pnl += kpnl
        kelly_eq = max(0.01, kelly_eq + kpnl)
        kelly_peak = max(kelly_peak, kelly_eq)
        kelly_dd = max(kelly_dd, (kelly_peak - kelly_eq) / kelly_peak * 100.0)

        pcf = b.get("p_close_fair")
        if pcf is not None and np.isfinite(pcf):
            clv_arr.append((odds * pcf - 1.0) * 100.0)

    decided = max(1, n - pushes)
    season_rois = {
        s: round(sum(vals) / len(vals) * 100.0, 1)
        for s, vals in sorted(by_season.items())
    }
    pos_s = sum(1 for r in season_rois.values() if r > 0)

    return {
        "bets": int(n),
        "pushes": int(pushes),
        "win_rate": round(wins / decided * 100.0, 2),
        "avg_odds": round(float(np.mean(odds_arr)), 3),
        "flat_pnl": round(flat_pnl, 2),
        "flat_roi": round(flat_pnl / n * 100.0, 2),
        "flat_dd": round(flat_dd, 2),
        "kelly_roi": round((kelly_pnl / kelly_staked * 100.0) if kelly_staked > 0 else 0.0, 2),
        "kelly_end": round(kelly_eq, 2),
        "kelly_dd": round(kelly_dd, 2),
        "clv_fair": round(float(np.mean(clv_arr)), 2) if clv_arr else None,
        "pos_seasons": f"{pos_s}/{len(season_rois)}",
        "season_rois": season_rois,
    }


def evaluate_config(
    df: pd.DataFrame,
    engine: ResearchStrengthEngine,
    mle_cache: Dict[Tuple[str, int], float],
    venue_split: bool = False,
    luck_beta: float = 0.0,
    fatigue_discount: float = 1.0,
    target_pairs: Tuple[Tuple[str, str], ...] = APPROVED_4_PAIRS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate strictly causal OOS calibrated bets (2019-2026) for `target_pairs`."""
    raw_by_pair: Dict[Tuple[str, str], List[Dict[str, Any]]] = {p: [] for p in target_pairs}
    active_leagues = sorted({lg for lg, _ in target_pairs})

    for lg in active_leagues:
        lg_df = df[df["league"] == lg].sort_values("date")
        lg_markets = [mk for l, mk in target_pairs if l == lg]
        for _, row in lg_df.iterrows():
            d_day = row["date"].normalize()
            ref_ts = float(d_day.value) / 1e9
            season = int(row["season"])
            rho = mle_cache[(lg, season)]

            lh, la = engine.predict_lambdas(
                lg, row["home"], row["away"], ref_ts,
                venue_split=venue_split, luck_beta=luck_beta, fatigue_discount=fatigue_discount,
            )
            mp = calculate_market_probabilities(compute_dixon_coles_probs(lh, la, max_goals=8, rho=rho))
            gh, ga = int(row["goals_home"]), int(row["goals_away"])
            psh, psd, psa = row.get("PSH"), row.get("PSD"), row.get("PSA")
            psch, pscd, psca = row.get("PSCH"), row.get("PSCD"), row.get("PSCA")
            maxh, maxd, maxa = row.get("MaxH"), row.get("MaxD"), row.get("MaxA")
            po, pu = row.get("P_over25"), row.get("P_under25")
            pco, pcu = row.get("PC_over25"), row.get("PC_under25")
            maxo, maxu = row.get("Max_over25"), row.get("Max_under25")

            for mk in lg_markets:
                if mk == "1X2 Home":
                    p_raw = mp["home_win"]
                    y = 1 if gh > ga else 0
                    mo = maxh if pd.notna(maxh) else psh
                    po_v = psh
                    pcf = _vig_free_3way(psch, psch, pscd, psca)
                elif mk == "1X2 Away":
                    p_raw = mp["away_win"]
                    y = 1 if ga > gh else 0
                    mo = maxa if pd.notna(maxa) else psa
                    po_v = psa
                    pcf = _vig_free_3way(psca, psch, pscd, psca)
                elif mk == "Over 2.5":
                    p_raw = mp["over_2.5"]
                    y = 1 if (gh + ga) > 2.5 else 0
                    mo = maxo if pd.notna(maxo) else po
                    po_v = po
                    pcf = _vig_free_2way(pco, pcu)
                elif mk == "Under 2.5":
                    p_raw = mp["under_2.5"]
                    y = 1 if (gh + ga) < 2.5 else 0
                    mo = maxu if pd.notna(maxu) else pu
                    po_v = pu
                    pcf = _vig_free_2way(pcu, pco)
                elif mk == "DNB Home":
                    denom = mp["home_win"] + mp["away_win"]
                    p_raw = mp["home_win"] / denom if denom > 0 else 0.5
                    y = 1 if gh > ga else (-1 if gh == ga else 0)
                    mo = _synth_dnb_odds(maxh if pd.notna(maxh) else psh, maxd if pd.notna(maxd) else psd)
                    po_v = _synth_dnb_odds(psh, psd)
                    # Pinnacle closing conditional fair DNB prob = P_close(H) / (P_close(H) + P_close(A))
                    ph_c = _vig_free_3way(psch, psch, pscd, psca)
                    pa_c = _vig_free_3way(psca, psch, pscd, psca)
                    pcf = (ph_c / (ph_c + pa_c)) if (ph_c and pa_c and (ph_c + pa_c) > 0) else None
                elif mk == "DNB Away":
                    denom = mp["home_win"] + mp["away_win"]
                    p_raw = mp["away_win"] / denom if denom > 0 else 0.5
                    y = 1 if ga > gh else (-1 if gh == ga else 0)
                    mo = _synth_dnb_odds(maxa if pd.notna(maxa) else psa, maxd if pd.notna(maxd) else psd)
                    po_v = _synth_dnb_odds(psa, psd)
                    ph_c = _vig_free_3way(psch, psch, pscd, psca)
                    pa_c = _vig_free_3way(psca, psch, pscd, psca)
                    pcf = (pa_c / (ph_c + pa_c)) if (ph_c and pa_c and (ph_c + pa_c) > 0) else None
                else:
                    continue

                raw_by_pair[(lg, mk)].append({
                    "league": lg,
                    "market": mk,
                    "season": season,
                    "date": str(row["date"])[:10],
                    "p_raw": float(p_raw),
                    "y": int(y),
                    "max_odds": float(mo) if mo is not None and pd.notna(mo) else None,
                    "pin_odds": float(po_v) if po_v is not None and pd.notna(po_v) else None,
                    "p_close_fair": pcf,
                })

    # Expanding OOS Platt calibration strictly on prior seasons < s
    oos_all: List[Dict[str, Any]] = []
    brier_by_pair: Dict[str, float] = {}
    for pair, recs in raw_by_pair.items():
        seasons = sorted({r["season"] for r in recs})
        pair_oos = []
        for s in seasons:
            if s <= 2018:
                continue
            # For DNB, Platt calibration trains on decided (non-draw) prior matches only
            prior = [r for r in recs if r["season"] < s and r["y"] != -1]
            cur = [r for r in recs if r["season"] == s]
            if len(prior) >= 80:
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
                pair_oos.append(item)
                oos_all.append(item)

        dec = [r for r in pair_oos if r["y"] != -1]
        if dec:
            p_a = np.array([r["p_cal"] for r in dec])
            y_a = np.array([r["y"] for r in dec])
            brier_by_pair[f"{pair[0]} {pair[1]}"] = round(float(np.mean((p_a - y_a) ** 2)), 5)

    oos_all.sort(key=lambda x: x["date"])
    return oos_all, brier_by_pair


def main() -> None:
    df = pd.read_csv(BIG5_HISTORY_CSV, parse_dates=["date"])
    df = df.dropna(subset=["league", "date", "home", "away", "goals_home", "goals_away"])
    df = df.sort_values(["date", "league"]).reset_index(drop=True)

    print(f"[ALPHA LAB] Loaded {len(df)} Big-5 matches (2018-2026). Precomputing causal seasonal MLE rho...")
    base_model = TeamStrengthModel(
        la_csv="", ligue_json="", ligue_live_csv="", big5_csv=str(BIG5_HISTORY_CSV),
        xi=DEFAULT_XI, xg_weight=XG_WEIGHT,
    )
    mle_cache: Dict[Tuple[str, int], float] = {}
    for lg in BIG5_LEAGUES:
        lg_df = df[df["league"] == lg]
        for s in sorted(lg_df["season"].dropna().astype(int).unique()):
            first_day = lg_df[lg_df["season"] == s]["date"].min().normalize()
            mle = fit_league_mle_dynamic(league=lg, team_model=base_model, as_of=first_day, use_cache=False)
            mle_cache[(lg, s)] = float(mle["rho"])

    engine = ResearchStrengthEngine(df, xi=DEFAULT_XI, xg_weight=XG_WEIGHT)

    # =========================================================================
    # EXPERIMENT 1: FEATURE ENGINEERING (vs BASELINE on Approved 4 Pairs)
    # =========================================================================
    print("\n" + "=" * 92)
    print("EXPERIMENT 1 — FEATURE ENGINEERING (Strict OOS 2019-2026, Approved 4 Pairs, p>=0.58, +6% EV)")
    print("=" * 92)
    exp1_configs = [
        ("0. Baseline (v3)", False, 0.0, 1.0),
        ("1A. Home/Away Split", True, 0.0, 1.0),
        ("1B. Luck Reversion (b=0.06)", False, 0.06, 1.0),
        ("1B2. Luck Reversion (b=0.10)", False, 0.10, 1.0),
        ("1C. Rest Days <=3d (-5%)", False, 0.0, 0.95),
        ("1C2. Rest Days <=3d (-8%)", False, 0.0, 0.92),
        ("1D. Split + Luck(0.06)", True, 0.06, 1.0),
        ("1E. Luck(0.06) + Rest(-5%)", False, 0.06, 0.95),
        ("1F. ALL (Split+Luck+Rest)", True, 0.06, 0.95),
    ]
    print(f"{'Experiment':28s} | {'Bets':>4s} {'Win%':>6s} {'BestROI':>8s} {'BestDD':>7s} {'CLV%':>7s} {'PosSzn':>6s} | {'PinBets':>7s} {'PinROI':>8s} {'PinDD':>7s}")
    exp1_results = {}
    baseline_oos = None
    for name, vs, lb, fd in exp1_configs:
        oos, briers = evaluate_config(df, engine, mle_cache, venue_split=vs, luck_beta=lb, fatigue_discount=fd)
        if name.startswith("0."):
            baseline_oos = oos
        m_best = simulate_bets(oos, "max_odds", min_conf=0.58, ev_margin=1.06)
        m_pin = simulate_bets(oos, "pin_odds", min_conf=0.58, ev_margin=1.06)
        exp1_results[name] = {"best": m_best, "pin": m_pin, "briers": briers}
        clv_s = f"{m_best['clv_fair']:+6.2f}%" if m_best["clv_fair"] is not None else "    N/A"
        print(
            f"{name:28s} | {m_best['bets']:4d} {m_best['win_rate']:5.1f}% {m_best['flat_roi']:+7.2f}% "
            f"{m_best['flat_dd']:6.2f}% {clv_s} {m_best['pos_seasons']:>6s} | "
            f"{m_pin['bets']:7d} {m_pin['flat_roi']:+7.2f}% {m_pin['flat_dd']:6.2f}%"
        )

    print("\n  Season-by-season Flat ROI% (Best Market) for Experiment 1:")
    for name, res in exp1_results.items():
        print(f"    {name:28s}: {res['best']['season_rois']} | avg_brier={np.mean(list(res['briers'].values())):.5f}")

    # =========================================================================
    # EXPERIMENT 2: THRESHOLD & EV MARGIN SWEEP (on Baseline OOS predictions)
    # =========================================================================
    print("\n" + "=" * 92)
    print("EXPERIMENT 2 — THRESHOLD & EV MARGIN SWEEP (p_cal in [0.58, 0.60, 0.62] x EV in [+6%, +8%, +10%])")
    print("=" * 92)
    print(f"{'p_min':>6s} {'EV_mgn':>7s} | {'Bets':>4s} {'Win%':>6s} {'AvgOdd':>7s} {'BestROI':>8s} {'BestPnL':>8s} {'BestDD':>7s} {'CLV%':>7s} {'PosSzn':>6s} | {'PinBets':>7s} {'PinROI':>8s} {'PinDD':>7s}")
    sweep_results = []
    for p_thr in (0.58, 0.60, 0.62):
        for ev_m in (1.06, 1.08, 1.10):
            mb = simulate_bets(baseline_oos, "max_odds", min_conf=p_thr, ev_margin=ev_m)
            mp = simulate_bets(baseline_oos, "pin_odds", min_conf=p_thr, ev_margin=ev_m)
            clv_s = f"{mb['clv_fair']:+6.2f}%" if mb["clv_fair"] is not None else "    N/A"
            print(
                f"{p_thr:6.2f} +{(ev_m-1)*100:4.0f}%  | {mb['bets']:4d} {mb['win_rate']:5.1f}% {mb['avg_odds']:7.3f} "
                f"{mb['flat_roi']:+7.2f}% {mb['flat_pnl']:+8.2f} {mb['flat_dd']:6.2f}% {clv_s} {mb['pos_seasons']:>6s} | "
                f"{mp['bets']:7d} {mp['flat_roi']:+7.2f}% {mp['flat_dd']:6.2f}%"
            )
            sweep_results.append({"p_min": p_thr, "ev_margin": ev_m, "best": mb, "pin": mp})

    # =========================================================================
    # EXPERIMENT 3: DRAW NO BET (DNB / AH 0) vs DIRECT 1X2
    # =========================================================================
    print("\n" + "=" * 92)
    print("EXPERIMENT 3 — DRAW NO BET (DNB / AH 0) vs DIRECT 1X2 (La Liga Home & Serie A Away)")
    print("=" * 92)
    dnb_pairs = (
        ("La Liga", "1X2 Home"),
        ("La Liga", "DNB Home"),
        ("Serie A", "1X2 Away"),
        ("Serie A", "DNB Away"),
    )
    dnb_oos, _ = evaluate_config(df, engine, mle_cache, target_pairs=dnb_pairs)
    print(f"{'League + Market':22s} | {'Bets':>4s} {'Push':>4s} {'Win%':>6s} {'AvgOdd':>7s} {'BestROI':>8s} {'BestPnL':>8s} {'BestDD':>7s} {'CLV%':>7s} {'PosSzn':>6s} | {'PinBets':>7s} {'PinROI':>8s} {'PinDD':>7s}")
    dnb_summary = {}
    for lg, mk in dnb_pairs:
        sub_b = [r for r in dnb_oos if r["league"] == lg and r["market"] == mk]
        mb = simulate_bets(sub_b, "max_odds", min_conf=0.58, ev_margin=1.06, is_dnb=("DNB" in mk))
        mp = simulate_bets(sub_b, "pin_odds", min_conf=0.58, ev_margin=1.06, is_dnb=("DNB" in mk))
        clv_s = f"{mb['clv_fair']:+6.2f}%" if mb["clv_fair"] is not None else "    N/A"
        label = f"{lg} {mk}"
        print(
            f"{label:22s} | {mb['bets']:4d} {mb['pushes']:4d} {mb['win_rate']:5.1f}% {mb['avg_odds']:7.3f} "
            f"{mb['flat_roi']:+7.2f}% {mb['flat_pnl']:+8.2f} {mb['flat_dd']:6.2f}% {clv_s} {mb['pos_seasons']:>6s} | "
            f"{mp['bets']:7d} {mp['flat_roi']:+7.2f}% {mp['flat_dd']:6.2f}%"
        )
        dnb_summary[label] = {"best": mb, "pin": mp}

    # Also check DNB with slightly higher DNB confidence (since DNB probs naturally sit higher than 1X2)
    print("\n  DNB Sensitivity at p_cal >= 0.70 (comparable odds band to 1X2 >= 0.58):")
    for lg, mk in (("La Liga", "DNB Home"), ("Serie A", "DNB Away")):
        sub_b = [r for r in dnb_oos if r["league"] == lg and r["market"] == mk]
        for p_thr in (0.65, 0.70, 0.75):
            mb = simulate_bets(sub_b, "max_odds", min_conf=p_thr, ev_margin=1.06, is_dnb=True)
            mp = simulate_bets(sub_b, "pin_odds", min_conf=p_thr, ev_margin=1.06, is_dnb=True)
            clv_s = f"{mb['clv_fair']:+6.2f}%" if mb["clv_fair"] is not None else "    N/A"
            print(
                f"    {lg} {mk} p>={p_thr:.2f} | bets={mb['bets']:3d} push={mb['pushes']:2d} win={mb['win_rate']:5.1f}% "
                f"avgOdd={mb['avg_odds']:.3f} BestROI={mb['flat_roi']:+6.2f}% DD={mb['flat_dd']:5.2f}% CLV={clv_s} "
                f"Pos={mb['pos_seasons']} | PinROI={mp['flat_roi']:+6.2f}%"
            )

    # Save full research log to scratch/json for auditing
    out_path = REPO_DIR / "research_alpha_results.json"
    out_path.write_text(
        json.dumps(
            {"exp1": exp1_results, "exp2": sweep_results, "exp3": dnb_summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[ALPHA LAB] Full JSON results saved to {out_path}")


if __name__ == "__main__":
    main()
