"""
team_strength.py — Leak-Free, Time-Decayed, xG-Blended Club Attack/Defense Strengths.

Upgrades over v1:
  1. Exponential Time Decay (default xi = 0.003, half-life ~231 days) so recent
     matches carry far more weight than matches from years ago.
  2. Strict causal `as_of` cutoff for BOTH team strengths AND league scoring
     baselines (mu_h, mu_a) — zero look-ahead data leakage.
  3. xG + Goals blending (`0.7 * xG + 0.3 * Goals`) whenever xG is available
     (both La Liga CSV and Ligue 1 JSON / live football-data.co.uk CSV), reducing
     finishing noise while gracefully falling back to Goals when xG is absent.
  4. Canonical team-name normalization across all data sources.
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from live_data import canonical_team_name

REPO_DIR = Path(__file__).parent
LA_CSV = "la_liga_matches_2018_2026.csv"
LIGUE_JSON = "raw_Ligue1_8seasons.json"
LIGUE_LIVE_CSV = "live_matches_ligue1.csv"

# Shrinkage: a team with effective weight W gets weight W/(W + K_SHRINK) toward its own rate.
K_SHRINK = 15.0
DEFAULT_XI = 0.003
XG_WEIGHT = 0.70


def time_decay_weight(days_ago: float, xi: float = DEFAULT_XI) -> float:
    """Exponential time-decay weight: w(dt) = exp(-xi * max(0, days_ago))."""
    return math.exp(-float(xi) * max(0.0, float(days_ago)))


def blend_xg_goals(goals: float, xg: Optional[float], xg_weight: float = XG_WEIGHT) -> float:
    """Blend expected goals and actual goals (0.7*xG + 0.3*Goals) when xG is finite."""
    if xg is not None:
        try:
            xg_f = float(xg)
            if math.isfinite(xg_f) and xg_f >= 0.0:
                return float(xg_weight) * xg_f + (1.0 - float(xg_weight)) * float(goals)
        except Exception:
            pass
    return float(goals)


def _to_ts_seconds(dt: Any) -> float:
    ts = pd.to_datetime(dt, errors="coerce")
    if pd.isna(ts):
        return float("nan")
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return float(ts.value) / 1e9


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    cand = REPO_DIR / path_str
    return cand


class TeamStrengthModel:
    def __init__(
        self,
        la_csv: str = LA_CSV,
        ligue_json: str = LIGUE_JSON,
        ligue_live_csv: str = LIGUE_LIVE_CSV,
        min_matches: int = 5,
        xi: float = DEFAULT_XI,
        xg_weight: float = XG_WEIGHT,
    ) -> None:
        self.min_matches = int(min_matches)
        self.xi = float(xi)
        self.xg_weight = float(xg_weight)
        # league -> {
        #   'default_mu_h': float, 'default_mu_a': float,
        #   'matches': sorted list of (ts_sec, home, away, gh, ga, eff_h, eff_a),
        #   'match_ts': sorted list of ts_sec (parallel to matches for bisect),
        #   'events': {team: sorted list of (ts_sec, eff_for, eff_ag, raw_for, raw_ag)},
        #   'event_ts': {team: sorted list of ts_sec}
        # }
        self.leagues: Dict[str, Dict[str, Any]] = {}
        self._baseline_cache: Dict[Tuple[str, Optional[int], float], Tuple[float, float, float, float]] = {}
        self._build_la(la_csv)
        self._build_ligue(ligue_json, ligue_live_csv)

    def _finalize_league(
        self,
        league: str,
        raw_matches: List[Tuple[float, str, str, float, float, float, float]],
        default_mu_h: float,
        default_mu_a: float,
    ) -> None:
        # Deduplicate by (date_day_int, home, away) keeping the last/richest entry
        dedup: Dict[Tuple[int, str, str], Tuple[float, str, str, float, float, float, float]] = {}
        for rec in raw_matches:
            ts_sec, home, away, gh, ga, eff_h, eff_a = rec
            if not math.isfinite(ts_sec) or not home or not away:
                continue
            day_key = int(ts_sec // 86400)
            dedup[(day_key, home, away)] = rec

        matches = sorted(dedup.values(), key=lambda x: x[0])
        match_ts = [m[0] for m in matches]

        events: Dict[str, List[Tuple[float, float, float, float, float]]] = {}
        for ts_sec, home, away, gh, ga, eff_h, eff_a in matches:
            events.setdefault(home, []).append((ts_sec, eff_h, eff_a, gh, ga))
            events.setdefault(away, []).append((ts_sec, eff_a, eff_h, ga, gh))

        event_ts: Dict[str, List[float]] = {}
        for team in events:
            events[team].sort(key=lambda x: x[0])
            event_ts[team] = [e[0] for e in events[team]]

        self.leagues[league] = {
            "default_mu_h": float(default_mu_h),
            "default_mu_a": float(default_mu_a),
            "matches": matches,
            "match_ts": match_ts,
            "events": events,
            "event_ts": event_ts,
        }

    # ------------------------------------------------------------------ la liga
    def _build_la(self, csv_path: str) -> None:
        path = _resolve_path(csv_path)
        if not path.exists():
            self._finalize_league("La Liga", [], 1.48, 1.18)
            return
        df = pd.read_csv(path, parse_dates=["date"])
        if "isResult" in df.columns:
            df = df[df["isResult"].astype(str).str.lower() == "true"]
        df = df.dropna(subset=["date", "goals_home", "goals_away"])

        raw_matches: List[Tuple[float, str, str, float, float, float, float]] = []
        has_xg = "xG_home" in df.columns and "xG_away" in df.columns
        for _, r in df.iterrows():
            ts_sec = _to_ts_seconds(r["date"])
            home = canonical_team_name(r["home"], "La Liga")
            away = canonical_team_name(r["away"], "La Liga")
            gh = float(r["goals_home"])
            ga = float(r["goals_away"])
            xg_h = r["xG_home"] if has_xg else None
            xg_a = r["xG_away"] if has_xg else None
            eff_h = blend_xg_goals(gh, xg_h, self.xg_weight)
            eff_a = blend_xg_goals(ga, xg_a, self.xg_weight)
            raw_matches.append((ts_sec, home, away, gh, ga, eff_h, eff_a))

        self._finalize_league("La Liga", raw_matches, 1.48, 1.18)

    # ---------------------------------------------------------------- ligue 1
    def _build_ligue(self, json_path: str, live_csv_path: str) -> None:
        raw_matches: List[Tuple[float, str, str, float, float, float, float]] = []
        path = _resolve_path(json_path)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for _, sdata in raw.items():
                    if not isinstance(sdata, dict):
                        continue
                    for d in sdata.get("dates", []):
                        if not d.get("isResult") or not d.get("goals"):
                            continue
                        ts_sec = _to_ts_seconds(d.get("datetime"))
                        home = canonical_team_name((d.get("h") or {}).get("title", ""), "Ligue 1")
                        away = canonical_team_name((d.get("a") or {}).get("title", ""), "Ligue 1")
                        gh = float(d["goals"]["h"])
                        ga = float(d["goals"]["a"])
                        xg_obj = d.get("xG") or {}
                        eff_h = blend_xg_goals(gh, xg_obj.get("h"), self.xg_weight)
                        eff_a = blend_xg_goals(ga, xg_obj.get("a"), self.xg_weight)
                        raw_matches.append((ts_sec, home, away, gh, ga, eff_h, eff_a))
            except Exception:
                pass

        # Also merge live synced Ligue 1 matches if available
        lpath = _resolve_path(live_csv_path)
        if lpath.exists():
            try:
                ldf = pd.read_csv(lpath, parse_dates=["date"])
                ldf = ldf.dropna(subset=["date", "goals_home", "goals_away"])
                has_xg = "xG_home" in ldf.columns and "xG_away" in ldf.columns
                for _, r in ldf.iterrows():
                    ts_sec = _to_ts_seconds(r["date"])
                    home = canonical_team_name(r["home"], "Ligue 1")
                    away = canonical_team_name(r["away"], "Ligue 1")
                    gh = float(r["goals_home"])
                    ga = float(r["goals_away"])
                    xg_h = r["xG_home"] if has_xg else None
                    xg_a = r["xG_away"] if has_xg else None
                    eff_h = blend_xg_goals(gh, xg_h, self.xg_weight)
                    eff_a = blend_xg_goals(ga, xg_a, self.xg_weight)
                    raw_matches.append((ts_sec, home, away, gh, ga, eff_h, eff_a))
            except Exception:
                pass

        self._finalize_league("Ligue 1", raw_matches, 1.55, 1.20)

    # ---------------------------------------------------------------- helpers
    def get_league_match_dataframe(
        self, league: str, as_of: Any = None
    ) -> pd.DataFrame:
        """Return strictly-causal match DataFrame [date, home, away, goals_home, goals_away, eff_h, eff_a]."""
        s = self.leagues.get(str(league))
        if not s or not s["matches"]:
            return pd.DataFrame(columns=["date", "home", "away", "goals_home", "goals_away", "eff_h", "eff_a"])
        matches = s["matches"]
        if as_of is not None:
            cutoff_ts = _to_ts_seconds(as_of)
            idx = bisect.bisect_left(s["match_ts"], cutoff_ts)
            matches = matches[:idx]
        if not matches:
            return pd.DataFrame(columns=["date", "home", "away", "goals_home", "goals_away", "eff_h", "eff_a"])
        return pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(m[0], unit="s"),
                    "home": m[1],
                    "away": m[2],
                    "goals_home": m[3],
                    "goals_away": m[4],
                    "eff_h": m[5],
                    "eff_a": m[6],
                }
                for m in matches
            ]
        )

    def _causal_baselines(
        self,
        league: str,
        as_of: Any = None,
        xi: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Compute strictly causal (< as_of) time-decayed league averages:
        (mu_h_raw_goals, mu_a_raw_goals, mu_h_eff_xg, mu_a_eff_xg).
        """
        s = self.leagues.get(str(league))
        if not s:
            return 1.48, 1.18, 1.48, 1.18
        def_h = s["default_mu_h"]
        def_a = s["default_mu_a"]
        matches = s["matches"]
        if not matches:
            return def_h, def_a, def_h, def_a

        if as_of is not None:
            ref_ts = _to_ts_seconds(as_of)
            idx = bisect.bisect_left(s["match_ts"], ref_ts)
            sub = matches[:idx]
            cache_key = (str(league), int(ref_ts // 86400), round(self.xi if xi is None else float(xi), 6))
        else:
            sub = matches
            ref_ts = matches[-1][0]
            cache_key = (str(league), None, round(self.xi if xi is None else float(xi), 6))

        cached = self._baseline_cache.get(cache_key)
        if cached is not None:
            return cached

        if not sub:
            return def_h, def_a, def_h, def_a

        decay = self.xi if xi is None else float(xi)
        w_sum = 0.0
        gh_sum = ga_sum = eh_sum = ea_sum = 0.0
        for ts_sec, _, _, gh, ga, eff_h, eff_a in sub:
            days_ago = max(0.0, (ref_ts - ts_sec) / 86400.0)
            w = time_decay_weight(days_ago, decay)
            w_sum += w
            gh_sum += w * gh
            ga_sum += w * ga
            eh_sum += w * eff_h
            ea_sum += w * eff_a

        if w_sum <= 1e-9:
            return def_h, def_a, def_h, def_a
        res = (
            gh_sum / w_sum,
            ga_sum / w_sum,
            eh_sum / w_sum,
            ea_sum / w_sum,
        )
        self._baseline_cache[cache_key] = res
        return res

    def league_baselines(
        self,
        league: str,
        as_of: Any = None,
        xi: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Return strictly causal (mu_h, mu_a, events) for `league` using only
        matches played strictly before `as_of`, weighted by time decay.
        """
        s = self.leagues.get(str(league), {})
        mu_h, mu_a, _, _ = self._causal_baselines(league, as_of=as_of, xi=xi)
        return float(mu_h), float(mu_a), s.get("events", {})

    def strength(
        self,
        league: str,
        team: str,
        as_of: Any = None,
        xi: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Return time-decayed, xG-blended (att, def) for a club relative to its
        strictly causal league baseline, using ONLY matches played strictly
        before `as_of` (zero look-ahead leakage).
        """
        s = self.leagues.get(str(league))
        if not s:
            return 1.0, 1.0
        cname = canonical_team_name(team, league)
        ev = s["events"].get(cname)
        if not ev:
            return 1.0, 1.0

        if as_of is not None:
            ref_ts = _to_ts_seconds(as_of)
            idx = bisect.bisect_left(s["event_ts"][cname], ref_ts)
            played = ev[:idx]
        else:
            played = ev
            ref_ts = s["matches"][-1][0] if s["matches"] else ev[-1][0]

        if len(played) < self.min_matches:
            return 1.0, 1.0

        decay = self.xi if xi is None else float(xi)
        w_sum = 0.0
        eff_for_sum = 0.0
        eff_ag_sum = 0.0
        for ts_sec, eff_for, eff_ag, _, _ in played:
            days_ago = max(0.0, (ref_ts - ts_sec) / 86400.0)
            w = time_decay_weight(days_ago, decay)
            w_sum += w
            eff_for_sum += w * eff_for
            eff_ag_sum += w * eff_ag

        if w_sum <= 1e-9:
            return 1.0, 1.0

        _, _, mu_h_eff, mu_a_eff = self._causal_baselines(league, as_of=as_of, xi=decay)
        league_eff_per_game = max(0.05, (mu_h_eff + mu_a_eff) / 2.0)

        raw_att = (eff_for_sum / w_sum) / league_eff_per_game
        raw_def = (eff_ag_sum / w_sum) / league_eff_per_game
        # Effective sample-size shrinkage toward 1.0
        sh = w_sum / (w_sum + K_SHRINK)
        att = 1.0 + sh * (raw_att - 1.0)
        deff = 1.0 + sh * (raw_def - 1.0)
        return round(max(0.3, min(2.2, att)), 3), round(max(0.3, min(2.2, deff)), 3)


# Reusable singleton, built once per process.
_MODEL: Optional[TeamStrengthModel] = None


def get_model(refresh: bool = False) -> TeamStrengthModel:
    global _MODEL
    if _MODEL is None or refresh:
        _MODEL = TeamStrengthModel()
    return _MODEL
