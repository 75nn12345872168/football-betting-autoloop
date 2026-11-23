"""
settle_ledger_auto.py — Automated Real-Result Ledger Settlement Engine.

Upgrades over v1:
  1. Multi-source live result ingestion via live_data.py (ESPN scoreboard API,
     football-data.co.uk season CSVs, optional football-data.org & The Odds API
     /scores) + automatic sync into historical datasets.
  2. Fixes the critical `_LOCAL_FILES` string-iteration bug: uses proper lists
     of snapshot file paths as an offline fallback.
  3. Unified selection side (`selection_side` = HOME / AWAY / DRAW): reads the
     explicit column first, and falls back to the SAME numeric `lh` vs `la`
     comparison used by autoloop_master.py (eliminating BIG_CLUBS / string
     heuristics mismatches like `Le Mans vs Lyon` and `Lyon vs Paris FC`).
  4. Settles Over/Under 2.5, 1X2 (Home Win / Draw / Away Win), AH -0.75, and BTTS.
  5. Honest reporting: separates real-result hit-rate from verified-odds ROI,
     NEVER computing fake ROI from fabricated 1.90 placeholder prices.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from live_data import (
    BIG5_LEAGUES,
    REPO_DIR,
    canonical_team_name,
    fetch_all_live_matches,
    sync_live_history,
)
from team_strength import get_model

LEDGER = "paper_trading_ledger_2026.csv"
SUMMARY_FILE = "ledger_settlement_summary.json"
ARCHIVE_WEEKLY_FILE = REPO_DIR / "data" / "archive" / "weekly_archive.json"
PUBLIC_WEEKLY_FILE = REPO_DIR / "data" / "public" / "weekly_archive.json"
HISTORICAL_ODDS_FILE = REPO_DIR / "data" / "historical_big5_odds.csv"
CONSERVATIVE_SAFETY_MARGIN = 1.06

V4_LEDGER_COLUMNS: List[str] = [
    "Date",
    "League",
    "HomeTeam",
    "AwayTeam",
    "Market",
    "Selection",
    "Calibrated_Prob",
    "Min_Odds",
    "Market_Odds",
    "Stake_Pct",
    "Result",
    "PnL_Units",
]


def _parse_v4_result_cell(val: Any) -> Tuple[str, str]:
    """Parse v4 `Result` cell ('PENDING', 'WIN', 'WIN (2-0)', 'PUSH (1-1)', etc.) -> (status, score)."""
    raw = str(val or "").strip()
    if not raw or raw.lower() == "nan" or raw.upper() == "PENDING":
        return "PENDING", ""
    up = raw.upper()
    score = ""
    if "(" in raw and ")" in raw:
        score = raw[raw.find("(") + 1 : raw.rfind(")")].strip()
    for st in ("HALF WIN", "WIN", "LOSS", "PUSH", "UNKNOWN_MARKET", "UNKNOWN_AH"):
        if up.startswith(st):
            return st, score
    return "PENDING", ""


def _format_v4_result_cell(status: Any, score: Any) -> str:
    st = str(status or "PENDING").strip().upper()
    if not st or st == "NAN":
        return "PENDING"
    sc = str(score or "").strip()
    if st == "PENDING" or not sc or sc.lower() == "nan":
        return st
    return f"{st} ({sc})"


def load_ledger_normalized(ledger_path: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Load either the clean v4 12-column ledger CSV or a legacy lowercase ledger CSV
    into a unified internal DataFrame, returning `(df, is_v4_schema)`.
    """
    raw = pd.read_csv(ledger_path)
    is_v4 = "HomeTeam" in raw.columns or "Date" in raw.columns

    if is_v4:
        rows = []
        for _, r in raw.iterrows():
            st, sc = _parse_v4_result_cell(r.get("Result", "PENDING"))
            mo = pd.to_numeric(r.get("Market_Odds"), errors="coerce")
            has_real = pd.notna(mo) and float(mo) > 1.0
            rows.append({
                "date": pd.to_datetime(r.get("Date"), errors="coerce"),
                "league": str(r.get("League", "La Liga")),
                "home": str(r.get("HomeTeam", "")),
                "away": str(r.get("AwayTeam", "")),
                "market": str(r.get("Market", "")),
                "selection_side": "" if pd.isna(r.get("Selection")) else str(r.get("Selection")).strip(),
                "p_calibrated": pd.to_numeric(r.get("Calibrated_Prob"), errors="coerce"),
                "min_odds": pd.to_numeric(r.get("Min_Odds"), errors="coerce"),
                "market_odds": mo,
                "stake": float(pd.to_numeric(r.get("Stake_Pct"), errors="coerce") or 0.0),
                "status": st,
                "result": sc,
                "profit": pd.to_numeric(r.get("PnL_Units"), errors="coerce"),
                "odds_source": "the-odds-api" if has_real else "",
            })
        cols = [
            "date", "league", "home", "away", "market", "selection_side",
            "p_calibrated", "min_odds", "market_odds", "stake",
            "status", "result", "profit", "odds_source",
        ]
        df = pd.DataFrame(rows, columns=cols)
    else:
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col, default in [
            ("league", "La Liga"),
            ("market", "Over 2.5"),
            ("selection_side", ""),
            ("odds_source", ""),
            ("stake", 0.0),
            ("market_odds", np.nan),
            ("profit", np.nan),
            ("result", ""),
            ("status", "PENDING"),
            ("p_calibrated", np.nan),
            ("min_odds", np.nan),
        ]:
            if col not in df.columns:
                df[col] = default

    df["result"] = df["result"].fillna("").astype(object)
    df["selection_side"] = df["selection_side"].fillna("").astype(object)
    df["odds_source"] = df["odds_source"].fillna("").astype(object)
    df["status"] = df["status"].fillna("PENDING").astype(object)
    return df, is_v4


def save_ledger_df(df: pd.DataFrame, ledger_path: Path, is_v4: bool = True) -> None:
    """Persist normalized ledger DataFrame cleanly in v4 12-column schema (or legacy if requested)."""
    if not is_v4:
        df.to_csv(ledger_path, index=False)
        return

    out_rows = []
    for _, r in df.iterrows():
        dt_val = r.get("date")
        dt_str = pd.to_datetime(dt_val).strftime("%Y-%m-%d %H:%M:%S") if pd.notna(dt_val) else ""
        mo = pd.to_numeric(r.get("market_odds"), errors="coerce")
        has_real = str(r.get("odds_source", "")).strip() == "the-odds-api" and pd.notna(mo) and float(mo) > 1.0
        out_rows.append({
            "Date": dt_str,
            "League": str(r.get("league", "")),
            "HomeTeam": str(r.get("home", "")),
            "AwayTeam": str(r.get("away", "")),
            "Market": str(r.get("market", "")),
            "Selection": str(r.get("selection_side", "") or ""),
            "Calibrated_Prob": "" if pd.isna(r.get("p_calibrated")) else round(float(r["p_calibrated"]), 4),
            "Min_Odds": "" if pd.isna(r.get("min_odds")) else round(float(r["min_odds"]), 2),
            "Market_Odds": round(float(mo), 3) if has_real else "",
            "Stake_Pct": round(float(pd.to_numeric(r.get("stake"), errors="coerce") or 0.0), 4),
            "Result": _format_v4_result_cell(r.get("status", "PENDING"), r.get("result", "")),
            "PnL_Units": "" if pd.isna(r.get("profit")) else round(float(r["profit"]), 4),
        })
    pd.DataFrame(out_rows, columns=V4_LEDGER_COLUMNS).to_csv(ledger_path, index=False)

# Fixed: each league maps to a LIST of fallback snapshot paths (never a bare str!)
_LOCAL_FILES: Dict[str, List[str]] = {
    "La Liga": [
        "data/historical_big5_odds.csv",
        "understat_raw_2026.json",
        "la_liga_matches_2018_2026.csv",
    ],
    "Ligue 1": [
        "data/historical_big5_odds.csv",
        "live_matches_ligue1.csv",
        "raw_Ligue1_8seasons.json",
    ],
    "EPL": ["data/historical_big5_odds.csv"],
    "Bundesliga": ["data/historical_big5_odds.csv"],
    "Serie A": ["data/historical_big5_odds.csv"],
}


def determine_selection_side(
    home: str,
    away: str,
    league: str,
    fixture_date: Any = None,
    explicit_side: Any = None,
    lh: Optional[float] = None,
    la: Optional[float] = None,
    team_model=None,
) -> str:
    """
    Unified selection-side resolver shared across autoloop_master.py,
    fetch_odds.py, and settle_ledger_auto.py.

    Priority:
      1. Explicit `selection_side` column ('HOME', 'AWAY', 'DRAW') if present.
      2. Numeric comparison of expected goals `lh` vs `la` (if provided).
      3. Numeric comparison of causal `TeamStrengthModel` expected goals `lh` vs `la`.
    """
    if explicit_side is not None and not pd.isna(explicit_side):
        s = str(explicit_side).strip().upper()
        if s in ("HOME", "AWAY", "DRAW"):
            return s

    if lh is not None and la is not None:
        return "HOME" if float(lh) >= float(la) else "AWAY"

    model = team_model or get_model()
    as_of = pd.to_datetime(fixture_date, errors="coerce") if fixture_date is not None else None
    if pd.isna(as_of):
        as_of = None
    mu_h, mu_a, _ = model.league_baselines(league, as_of=as_of)
    att_h, def_h = model.strength(league, home, as_of=as_of)
    att_a, def_a = model.strength(league, away, as_of=as_of)
    exp_h = mu_h * att_h * def_a
    exp_a = mu_a * att_a * def_h
    return "HOME" if exp_h >= exp_a else "AWAY"


def _load_local_results(league: str, base_dir: Path = REPO_DIR) -> List[Dict[str, Any]]:
    """Load completed match results from local JSON/CSV fallback snapshots."""
    results: List[Dict[str, Any]] = []
    for cand in _LOCAL_FILES.get(league, []):
        p = Path(cand)
        if not p.exists():
            p = base_dir / cand
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".json":
                raw = json.loads(p.read_text(encoding="utf-8"))
                date_items: List[Dict[str, Any]] = []
                if isinstance(raw, dict) and "dates" in raw:
                    date_items = raw["dates"]
                elif isinstance(raw, dict):
                    for v in raw.values():
                        if isinstance(v, dict):
                            date_items.extend(v.get("dates", []))
                for d in date_items:
                    if d.get("isResult") and d.get("goals"):
                        results.append({
                            "league": league,
                            "date_key": str(d.get("datetime", ""))[:10],
                            "home": canonical_team_name((d.get("h") or {}).get("title", ""), league),
                            "away": canonical_team_name((d.get("a") or {}).get("title", ""), league),
                            "goals_home": int(d["goals"]["h"]),
                            "goals_away": int(d["goals"]["a"]),
                            "completed": True,
                        })
            elif p.suffix.lower() == ".csv":
                df = pd.read_csv(p)
                if "league" in df.columns:
                    df = df[df["league"].astype(str).str.strip() == league]
                if "isResult" in df.columns:
                    df = df[df["isResult"].astype(str).str.lower() == "true"]
                df = df.dropna(subset=["date", "goals_home", "goals_away"])
                for _, r in df.iterrows():
                    results.append({
                        "league": league,
                        "date_key": str(r["date"])[:10],
                        "home": canonical_team_name(r["home"], league),
                        "away": canonical_team_name(r["away"], league),
                        "goals_home": int(float(r["goals_home"])),
                        "goals_away": int(float(r["goals_away"])),
                        "completed": True,
                    })
        except Exception:
            continue
    return results


def build_result_index(
    leagues: Tuple[str, ...] = ("La Liga", "Ligue 1"),
    live_records: Optional[List[Dict[str, Any]]] = None,
    use_live: bool = True,
    base_dir: Path = REPO_DIR,
) -> Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, int, int]]]:
    """
    Build lookup index `(league, canonical_home, canonical_away) -> [(date, gh, ga)]`
    combining live APIs (ESPN, football-data.co.uk, football-data.org, Odds API)
    with local snapshot fallbacks.
    """
    records: List[Dict[str, Any]] = []
    # Local snapshots first, then live records override/extend
    for lg in leagues:
        records.extend(_load_local_results(lg, base_dir=base_dir))

    if live_records is not None:
        records.extend(live_records)
    elif use_live:
        records.extend(fetch_all_live_matches(leagues=leagues, days_back=45, days_forward=2))

    index: Dict[Tuple[str, str, str], Dict[str, Tuple[pd.Timestamp, int, int]]] = {}
    for m in records:
        if not m.get("completed"):
            continue
        gh, ga = m.get("goals_home"), m.get("goals_away")
        if gh is None or ga is None:
            continue
        lg = str(m.get("league", ""))
        h = canonical_team_name(m.get("home", ""), lg)
        a = canonical_team_name(m.get("away", ""), lg)
        dk = str(m.get("date_key") or m.get("date", ""))[:10]
        dt = pd.to_datetime(dk, errors="coerce")
        if not lg or not h or not a or pd.isna(dt):
            continue
        index.setdefault((lg, h, a), {})[dk] = (dt, int(gh), int(ga))

    return {k: list(v.values()) for k, v in index.items()}


def fetch_result(
    league: str,
    home: str,
    away: str,
    date: Any,
    result_index: Optional[Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, int, int]]]] = None,
    max_date_diff_days: int = 3,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Look up real completed score (gh, ga) for a fixture.
    Uses canonical team names and tolerates up to `max_date_diff_days` for
    rescheduled/timezone-shifted fixtures (choosing the closest date).
    """
    idx = result_index if result_index is not None else build_result_index(leagues=(league,))
    h = canonical_team_name(home, league)
    a = canonical_team_name(away, league)
    target_dt = pd.to_datetime(date, errors="coerce")
    if pd.isna(target_dt):
        return None, None
    target_day = target_dt.normalize()

    candidates = idx.get((league, h, a), [])
    best_match = None
    best_diff = max_date_diff_days + 1
    for dt, gh, ga in candidates:
        diff = abs((dt.normalize() - target_day).days)
        if diff < best_diff:
            best_diff = diff
            best_match = (gh, ga)
    if best_match is not None and best_diff <= max_date_diff_days:
        return best_match
    return None, None


def settle_row(
    row: Dict[str, Any],
    cutoff: Optional[pd.Timestamp] = None,
    result_index: Optional[Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, int, int]]]] = None,
    team_model=None,
) -> Dict[str, Any]:
    """Resolve a PENDING row against a REAL completed result."""
    cut = (cutoff if cutoff is not None else pd.Timestamp.now()).normalize()
    league = str(row.get("league", "La Liga"))
    market = str(row.get("market", "Over 2.5")).strip()
    home = canonical_team_name(row["home"], league)
    away = canonical_team_name(row["away"], league)
    date = row["date"]

    dt = pd.to_datetime(date, errors="coerce")
    if pd.isna(dt) or dt.normalize() > cut:
        return row  # still upcoming -> remain PENDING

    gh, ga = fetch_result(league, home, away, date, result_index=result_index)
    if gh is None or ga is None:
        row["status"] = "PENDING"
        return row

    stake = float(row.get("stake", row.get("stake_kelly", 0.0)) or 0.0)
    odds_source = str(row.get("odds_source", "") or "").strip()
    has_verified_odds = odds_source == "the-odds-api"
    odds = float(row.get("market_odds", 0.0) or 0.0) if has_verified_odds else 0.0

    mk_low = market.lower()
    row["result"] = f"{gh}-{ga}"

    if "over" in mk_low and "2.5" in mk_low:
        won = (gh + ga) > 2.5
        status = "WIN" if won else "LOSS"
        profit = (stake * (odds - 1.0) if won else -stake) if has_verified_odds else 0.0
    elif "under" in mk_low and "2.5" in mk_low:
        won = (gh + ga) < 2.5
        status = "WIN" if won else "LOSS"
        profit = (stake * (odds - 1.0) if won else -stake) if has_verified_odds else 0.0
    elif "1x2" in mk_low or mk_low in ("home win", "away win", "draw"):
        if "home" in mk_low:
            side = "HOME"
        elif "away" in mk_low:
            side = "AWAY"
        elif "draw" in mk_low:
            side = "DRAW"
        else:
            side = determine_selection_side(
                home, away, league, fixture_date=date,
                explicit_side=row.get("selection_side"), team_model=team_model,
            )
        row["selection_side"] = side
        if side == "HOME":
            won = gh > ga
        elif side == "AWAY":
            won = ga > gh
        else:
            won = gh == ga
        status = "WIN" if won else "LOSS"
        profit = (stake * (odds - 1.0) if won else -stake) if has_verified_odds else 0.0
    elif "dnb" in mk_low or "draw no bet" in mk_low or "ah 0" in mk_low:
        if "home" in mk_low:
            side = "HOME"
        elif "away" in mk_low:
            side = "AWAY"
        else:
            side = determine_selection_side(
                home, away, league, fixture_date=date,
                explicit_side=row.get("selection_side"), team_model=team_model,
            )
        row["selection_side"] = side
        if gh == ga:
            status = "PUSH"
            profit = 0.0
        else:
            won = (gh > ga) if side == "HOME" else (ga > gh)
            status = "WIN" if won else "LOSS"
            profit = (stake * (odds - 1.0) if won else -stake) if has_verified_odds else 0.0
    elif "ah" in mk_low or "handicap" in mk_low:
        side = determine_selection_side(
            home, away, league, fixture_date=date,
            explicit_side=row.get("selection_side"), team_model=team_model,
        )
        row["selection_side"] = side
        diff = (gh - ga) if side == "HOME" else (ga - gh)
        if diff >= 2:
            status = "WIN"
            profit = stake * (odds - 1.0) if has_verified_odds else 0.0
        elif diff == 1:
            status = "HALF WIN"
            profit = stake * 0.5 * (odds - 1.0) if has_verified_odds else 0.0
        else:
            status = "LOSS"
            profit = -stake if has_verified_odds else 0.0
    elif "btts" in mk_low:
        both = (gh > 0) and (ga > 0)
        want_yes = "no" not in mk_low
        won = both if want_yes else (not both)
        status = "WIN" if won else "LOSS"
        profit = (stake * (odds - 1.0) if won else -stake) if has_verified_odds else 0.0
    else:
        row["status"] = "UNKNOWN_MARKET"
        row["profit"] = 0.0
        return row

    row["status"] = status
    row["profit"] = round(float(profit), 4) if has_verified_odds else np.nan
    return row


# ==============================================================================
# MATCHWEEK WINDOW (TUE 00:00 UTC -> MON 23:59 UTC), WEEKLY P&L & ARCHIVING
# ==============================================================================

def get_matchweek_window(dt: Any) -> Dict[str, str]:
    """
    Map any kickoff timestamp to its unified European Football Matchweek Window:
      Tuesday 00:00:00 UTC -> Monday 23:59:59 UTC.
    This guarantees Monday night fixtures stay grouped with their weekend slate.
    """
    ts = pd.to_datetime(dt, errors="coerce")
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
    elif getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    # Python weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    days_since_tuesday = (int(ts.weekday()) - 1) % 7
    win_start = ts.normalize() - pd.Timedelta(days=days_since_tuesday)
    win_end = win_start + pd.Timedelta(days=6)
    start_str = win_start.strftime("%Y-%m-%d")
    end_str = win_end.strftime("%Y-%m-%d")
    return {
        "matchweek_id": f"MW-{start_str}_to_{end_str}",
        "window_start": start_str,
        "window_end": end_str,
        "window_start_utc": f"{start_str}T00:00:00Z",
        "window_end_utc": f"{end_str}T23:59:59Z",
        "label": f"{start_str} → {end_str}",
    }


def _bet_pnl_multiplier(status: str, odds: Optional[float]) -> Optional[float]:
    """Return net P&L per 1.0 unit staked for a settled bet at decimal `odds`."""
    st = str(status or "").strip().upper()
    if st == "PUSH":
        return 0.0
    if odds is None or pd.isna(odds) or float(odds) <= 1.0:
        return None
    dec = float(odds)
    if st == "WIN":
        return dec - 1.0
    if st == "HALF WIN":
        return 0.5 * (dec - 1.0)
    if st == "LOSS":
        return -1.0
    return None


def _extract_closing_odds_for_market(
    odds_rec: Dict[str, Any], market: str, selection_side: str = ""
) -> Optional[float]:
    """
    Extract real closing decimal odds from football-data.co.uk record.
    Uses Pinnacle Closing (PSCH/PSCD/PSCA/PC_over25/PC_under25) with clean
    fallback to Pinnacle Opening (PSH/PSD/PSA) or Best Market (MaxH/MaxD/MaxA).
    For DNB markets, applies exact replication: O_DNB = O_side * (1 - 1 / O_draw).
    """
    def _num(*keys: str) -> Optional[float]:
        for k in keys:
            v = pd.to_numeric(odds_rec.get(k), errors="coerce")
            if pd.notna(v) and float(v) > 1.0:
                return float(v)
        return None

    mk = str(market or "").strip().lower()
    side = str(selection_side or "").strip().upper()

    if "dnb" in mk or "draw no bet" in mk or "ah 0" in mk:
        is_away = ("away" in mk) or (side == "AWAY")
        o_side = _num("PSCA", "PSA", "MaxA") if is_away else _num("PSCH", "PSH", "MaxH")
        o_draw = _num("PSCD", "PSD", "MaxD")
        if o_side and o_draw and o_draw > 1.0:
            rep = o_side * (1.0 - (1.0 / o_draw))
            return round(rep, 3) if rep > 1.0 else None
        return None
    if "over" in mk and "2.5" in mk:
        o = _num("PC_over25", "PC>2.5", "P_over25", "Max_over25", "Max>2.5")
        return round(o, 3) if o else None
    if "under" in mk and "2.5" in mk:
        o = _num("PC_under25", "PC<2.5", "P_under25", "Max_under25", "Max<2.5")
        return round(o, 3) if o else None
    if "1x2" in mk or mk in ("home win", "away win", "draw"):
        if "away" in mk or side == "AWAY":
            o = _num("PSCA", "PSA", "MaxA")
        elif "draw" in mk or side == "DRAW":
            o = _num("PSCD", "PSD", "MaxD")
        else:
            o = _num("PSCH", "PSH", "MaxH")
        return round(o, 3) if o else None
    return None


def build_closing_odds_index(
    historical_path: Path = HISTORICAL_ODDS_FILE,
    extra_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, Dict[str, Any]]]]:
    """
    Build lookup index `(league, canonical_home, canonical_away) -> [(date, odds_dict)]`
    from `data/historical_big5_odds.csv` (sourced from football-data.co.uk) and
    optional extra records.
    """
    index: Dict[Tuple[str, str, str], Dict[str, Tuple[pd.Timestamp, Dict[str, Any]]]] = {}
    odds_cols = (
        "PSCH", "PSCD", "PSCA", "PC_over25", "PC_under25",
        "PSH", "PSD", "PSA", "P_over25", "P_under25",
        "MaxH", "MaxD", "MaxA", "Max_over25", "Max_under25",
        "PC>2.5", "PC<2.5", "Max>2.5", "Max<2.5",
    )

    if historical_path and historical_path.exists():
        try:
            df = pd.read_csv(historical_path)
            present_cols = [c for c in odds_cols if c in df.columns]
            if present_cols and all(c in df.columns for c in ("league", "date", "home", "away")):
                # Focus on recent seasons (2024+) for fast lookup during settlement
                df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
                recent = df[df["_dt"] >= pd.Timestamp("2024-07-01")]
                for _, r in recent.iterrows():
                    lg = str(r["league"]).strip()
                    h = canonical_team_name(r["home"], lg)
                    a = canonical_team_name(r["away"], lg)
                    dt = r["_dt"]
                    if pd.isna(dt) or not lg or not h or not a:
                        continue
                    dk = dt.strftime("%Y-%m-%d")
                    rec = {c: r[c] for c in present_cols if pd.notna(r[c])}
                    if rec:
                        index.setdefault((lg, h, a), {})[dk] = (dt, rec)
        except Exception:
            pass

    if extra_records:
        for m in extra_records:
            lg = str(m.get("league", "")).strip()
            h = canonical_team_name(m.get("home", ""), lg)
            a = canonical_team_name(m.get("away", ""), lg)
            dt = pd.to_datetime(m.get("date_key") or m.get("date"), errors="coerce")
            if pd.isna(dt) or not lg or not h or not a:
                continue
            rec = {c: m[c] for c in odds_cols if c in m and pd.notna(m[c])}
            if rec:
                dk = dt.strftime("%Y-%m-%d")
                index.setdefault((lg, h, a), {})[dk] = (dt, rec)

    return {k: list(v.values()) for k, v in index.items()}


def lookup_closing_odds(
    league: str,
    home: str,
    away: str,
    fixture_date: Any,
    market: str,
    selection_side: str,
    closing_index: Optional[Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, Dict[str, Any]]]]],
    max_date_diff_days: int = 3,
) -> Optional[float]:
    if not closing_index:
        return None
    h = canonical_team_name(home, league)
    a = canonical_team_name(away, league)
    target = pd.to_datetime(fixture_date, errors="coerce")
    if pd.isna(target):
        return None
    target_day = target.normalize()
    best_rec = None
    best_diff = max_date_diff_days + 1
    for dt, rec in closing_index.get((league, h, a), []):
        diff = abs((dt.normalize() - target_day).days)
        if diff < best_diff:
            best_diff = diff
            best_rec = rec
    if best_rec is None or best_diff > max_date_diff_days:
        return None
    return _extract_closing_odds_for_market(best_rec, market, selection_side)


def _compute_week_metrics_from_picks(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute weekly transparency metrics from normalized pick dicts:
      1. conservative_min_odds_roi_pct & kelly_bankroll_change_pct (Kelly-weighted on Min_Odds floor)
      2. flat_net_units & flat_roi_pct (flat 1u on Min_Odds floor for backtest parity)
      3. closing_odds_roi_pct (auto-populated when football-data.co.uk closing odds exist)
    """
    settled_statuses = {"WIN", "HALF WIN", "LOSS", "PUSH"}
    total_picks = len(picks)
    wins = sum(1 for p in picks if p["status"] == "WIN")
    half_wins = sum(1 for p in picks if p["status"] == "HALF WIN")
    losses = sum(1 for p in picks if p["status"] == "LOSS")
    pushes = sum(1 for p in picks if p["status"] == "PUSH")
    settled_picks = sum(1 for p in picks if p["status"] in settled_statuses)
    pending_picks = sum(1 for p in picks if p["status"] == "PENDING")

    decisive = wins + half_wins + losses
    hit_rate_pct = round(((wins + 0.5 * half_wins) / decisive) * 100.0, 2) if decisive > 0 else 0.0

    kelly_pnl_frac = 0.0
    kelly_staked_frac = 0.0
    flat_net_units = 0.0
    flat_counted = 0

    closing_kelly_pnl = 0.0
    closing_kelly_staked = 0.0
    closing_sample_n = 0

    for p in picks:
        st = p["status"]
        if st not in settled_statuses:
            p["conservative_kelly_pnl_pct"] = None
            p["flat_pnl_units"] = None
            continue

        stake_pct = float(p.get("stake_pct") or 0.0)
        if stake_pct <= 0.0:
            stake_pct = 0.02

        # Conservative floor odds: verified market_odds if recorded, otherwise Min_Odds floor
        floor_odds = p.get("market_odds") if (p.get("market_odds") and float(p["market_odds"]) > 1.0) else p.get("min_odds")
        if (floor_odds is None or float(floor_odds) <= 1.0) and p.get("calibrated_prob"):
            prob = float(p["calibrated_prob"])
            if 0.01 <= prob <= 0.99:
                floor_odds = round(CONSERVATIVE_SAFETY_MARGIN / prob, 2)

        mult = _bet_pnl_multiplier(st, floor_odds)
        if mult is not None:
            bet_kelly_pnl = stake_pct * mult
            kelly_pnl_frac += bet_kelly_pnl
            kelly_staked_frac += stake_pct
            flat_net_units += mult
            flat_counted += 1
            p["conservative_kelly_pnl_pct"] = round(bet_kelly_pnl * 100.0, 3)
            p["flat_pnl_units"] = round(mult, 4)
        else:
            p["conservative_kelly_pnl_pct"] = None
            p["flat_pnl_units"] = None

        c_odds = p.get("closing_odds")
        if c_odds is not None and float(c_odds) > 1.0:
            c_mult = _bet_pnl_multiplier(st, float(c_odds))
            if c_mult is not None:
                closing_kelly_pnl += stake_pct * c_mult
                closing_kelly_staked += stake_pct
                closing_sample_n += 1

    kelly_bankroll_change_pct = round(kelly_pnl_frac * 100.0, 2)
    conservative_roi_pct = (
        round((kelly_pnl_frac / kelly_staked_frac) * 100.0, 2)
        if kelly_staked_frac > 0
        else None
    )
    flat_roi_pct = (
        round((flat_net_units / flat_counted) * 100.0, 2)
        if flat_counted > 0
        else None
    )
    closing_roi_pct = (
        round((closing_kelly_pnl / closing_kelly_staked) * 100.0, 2)
        if closing_kelly_staked > 0
        else None
    )

    if settled_picks == 0:
        pnl_status = "PENDING"
    elif kelly_bankroll_change_pct > 0:
        pnl_status = "PROFIT"
    elif kelly_bankroll_change_pct < 0:
        pnl_status = "LOSS"
    else:
        pnl_status = "BREAKEVEN"

    archive_status = "CLOSED_ARCHIVED" if (total_picks > 0 and pending_picks == 0) else "ACTIVE"

    return {
        "archive_status": archive_status,
        "status": pnl_status,
        "total_picks": total_picks,
        "settled_picks": settled_picks,
        "pending_picks": pending_picks,
        "wins": wins,
        "half_wins": half_wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate_percent": hit_rate_pct,
        "total_staked_pct": round(kelly_staked_frac * 100.0, 2),
        "conservative_min_odds_roi_pct": conservative_roi_pct,
        "kelly_bankroll_change_pct": kelly_bankroll_change_pct,
        "flat_net_units": round(flat_net_units, 4),
        "flat_roi_pct": flat_roi_pct,
        "closing_odds_roi_pct": closing_roi_pct,
        "closing_odds_sample_n": closing_sample_n,
    }


def build_weekly_transparency_and_archive(
    df: pd.DataFrame,
    archive_path: Path = ARCHIVE_WEEKLY_FILE,
    public_archive_path: Optional[Path] = PUBLIC_WEEKLY_FILE,
    closing_index: Optional[Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, Dict[str, Any]]]]] = None,
    reference_now: Optional[pd.Timestamp] = None,
    persist: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Group ledger rows by Matchweek Window (Tue 00:00 UTC -> Mon 23:59 UTC),
    compute Conservative Floor + Kelly Weighted + Flat 1u + Closing Odds ROI,
    permanently archive any completed matchweek as `"CLOSED_ARCHIVED"`, and
    build both `weekly_archive` payload and `transparency_banner` summary.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    ref_ts = reference_now if reference_now is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    current_mw_meta = get_matchweek_window(ref_ts)

    if closing_index is None:
        closing_index = build_closing_odds_index()

    # 1. Load previously permanently archived CLOSED_ARCHIVED weeks (never lost!)
    persisted_closed: Dict[str, Dict[str, Any]] = {}
    if archive_path and archive_path.exists():
        try:
            raw_arch = json.loads(archive_path.read_text(encoding="utf-8"))
            for wk in raw_arch.get("weeks", []):
                if isinstance(wk, dict) and wk.get("archive_status") == "CLOSED_ARCHIVED" and wk.get("matchweek_id"):
                    persisted_closed[str(wk["matchweek_id"])] = wk
        except Exception:
            pass

    # 2. Group ledger rows by Matchweek Window
    grouped_picks: Dict[str, Tuple[Dict[str, str], List[Dict[str, Any]]]] = {}
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            dt_val = pd.to_datetime(r.get("date"), errors="coerce")
            if pd.isna(dt_val):
                continue
            mw = get_matchweek_window(dt_val)
            mw_id = mw["matchweek_id"]
            league = str(r.get("league", "")).strip()
            home = canonical_team_name(r.get("home", ""), league)
            away = canonical_team_name(r.get("away", ""), league)
            market = str(r.get("market", "")).strip()
            side = str(r.get("selection_side", "") or "").strip().upper()
            p_cal = pd.to_numeric(r.get("p_calibrated"), errors="coerce")
            min_o = pd.to_numeric(r.get("min_odds"), errors="coerce")
            mkt_o = pd.to_numeric(r.get("market_odds"), errors="coerce")
            has_real = str(r.get("odds_source", "")).strip() == "the-odds-api" and pd.notna(mkt_o) and float(mkt_o) > 1.0
            stake_pct = float(pd.to_numeric(r.get("stake"), errors="coerce") or 0.0)
            status = str(r.get("status", "PENDING") or "PENDING").strip().upper()
            score = str(r.get("result", "") or "").strip()

            c_odds = lookup_closing_odds(
                league, home, away, dt_val, market, side, closing_index=closing_index
            )

            pick_rec = {
                "date": dt_val.strftime("%Y-%m-%d %H:%M:%S"),
                "league": league,
                "home": home,
                "away": away,
                "market": market,
                "selection_side": side,
                "calibrated_prob": None if pd.isna(p_cal) else round(float(p_cal), 4),
                "min_odds": None if pd.isna(min_o) else round(float(min_o), 2),
                "market_odds": round(float(mkt_o), 3) if has_real else None,
                "closing_odds": c_odds,
                "stake_pct": round(stake_pct, 4),
                "status": status,
                "score": score,
            }
            if mw_id not in grouped_picks:
                grouped_picks[mw_id] = (mw, [])
            grouped_picks[mw_id][1].append(pick_rec)

    # 3. Merge live grouped weeks with permanently archived closed weeks
    merged_weeks: Dict[str, Dict[str, Any]] = {}

    # Re-enrich previously archived weeks with closing odds if newly available
    for mw_id, old_wk in persisted_closed.items():
        picks = [dict(p) for p in old_wk.get("picks", [])]
        for p in picks:
            if p.get("closing_odds") is None:
                c_odds = lookup_closing_odds(
                    p.get("league", ""),
                    p.get("home", ""),
                    p.get("away", ""),
                    p.get("date"),
                    p.get("market", ""),
                    p.get("selection_side", ""),
                    closing_index=closing_index,
                )
                if c_odds is not None:
                    p["closing_odds"] = c_odds
        metrics = _compute_week_metrics_from_picks(picks)
        metrics["archive_status"] = "CLOSED_ARCHIVED"
        merged_weeks[mw_id] = {
            "matchweek_id": mw_id,
            "window_start": old_wk.get("window_start", ""),
            "window_end": old_wk.get("window_end", ""),
            "window_start_utc": old_wk.get("window_start_utc", ""),
            "window_end_utc": old_wk.get("window_end_utc", ""),
            "label": old_wk.get("label", mw_id),
            "closed_at": old_wk.get("closed_at") or now_iso,
            **metrics,
            "picks": picks,
        }

    # Apply ledger weeks (overrides/creates active weeks, closes completed weeks)
    for mw_id, (mw, picks) in grouped_picks.items():
        picks_sorted = sorted(picks, key=lambda x: (x["date"], x["league"], x["home"]))
        metrics = _compute_week_metrics_from_picks(picks_sorted)
        prior_closed_at = persisted_closed.get(mw_id, {}).get("closed_at")
        closed_at = (
            prior_closed_at or now_iso
            if metrics["archive_status"] == "CLOSED_ARCHIVED"
            else None
        )
        merged_weeks[mw_id] = {
            **mw,
            "closed_at": closed_at,
            **metrics,
            "picks": picks_sorted,
        }

    # Sort weeks newest-first by window_start
    sorted_weeks = sorted(
        merged_weeks.values(),
        key=lambda w: str(w.get("window_start", "")),
        reverse=True,
    )
    closed_weeks = [w for w in sorted_weeks if w.get("archive_status") == "CLOSED_ARCHIVED"]
    active_weeks = [w for w in sorted_weeks if w.get("archive_status") != "CLOSED_ARCHIVED"]

    # Determine current week for the transparency banner:
    # Prefer the active matchweek (nearest live slate), else current calendar matchweek, else empty stub
    if active_weeks:
        # Pick the active week containing `ref_ts`, or the earliest upcoming active week, or latest active week
        cur_match = next(
            (w for w in active_weeks if w["matchweek_id"] == current_mw_meta["matchweek_id"]),
            None,
        )
        current_week_full = cur_match or active_weeks[0]
    elif current_mw_meta["matchweek_id"] in merged_weeks:
        current_week_full = merged_weeks[current_mw_meta["matchweek_id"]]
    else:
        current_week_full = {
            **current_mw_meta,
            "closed_at": None,
            **_compute_week_metrics_from_picks([]),
            "picks": [],
        }

    last_closed_full = closed_weeks[0] if closed_weeks else None

    def _compact_week(w: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not w:
            return None
        return {k: v for k, v in w.items() if k != "picks"}

    transparency_banner = {
        "valuation_basis": "Conservative Min Odds Floor (+6% EV Margin) · Kelly Weighted",
        "matchweek_window_rule": "Tue 00:00 UTC – Mon 23:59 UTC",
        "current_week": _compact_week(current_week_full),
        "last_closed_week": _compact_week(last_closed_full),
        "closed_weeks_count": len(closed_weeks),
    }

    weekly_archive_payload = {
        "generated_at": now_iso,
        "engine_version": "v4-alpha-promoted",
        "matchweek_window_rule": "Tue 00:00 UTC – Mon 23:59 UTC",
        "valuation_basis": "Conservative Min Odds Floor (+6% EV Margin) · Kelly Weighted",
        "total_weeks": len(sorted_weeks),
        "closed_weeks_count": len(closed_weeks),
        "active_weeks_count": len(active_weeks),
        "transparency_banner": transparency_banner,
        "weeks": sorted_weeks,
    }

    if persist:
        for out_p in (archive_path, public_archive_path):
            if out_p is not None:
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(
                    json.dumps(weekly_archive_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    return weekly_archive_payload, transparency_banner


def settle_ledger(
    ledger_path: Path = Path(LEDGER),
    summary_path: Path = Path(SUMMARY_FILE),
    cutoff: Optional[pd.Timestamp] = None,
    live_records: Optional[List[Dict[str, Any]]] = None,
    use_live: bool = True,
    sync_history: bool = True,
    archive_path: Optional[Path] = None,
    public_archive_path: Optional[Path] = None,
    persist_archive: bool = True,
) -> Dict[str, Any]:
    """Run settlement over `ledger_path`, save updated CSV + summary JSON + weekly archive, and return summary."""
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")

    cut = (cutoff if cutoff is not None else pd.Timestamp.now()).normalize()
    df, is_v4 = load_ledger_normalized(ledger_path)

    leagues = tuple(sorted({str(x).strip() for x in df["league"].dropna().unique()} | set(BIG5_LEAGUES)))
    if live_records is None and use_live:
        live_records = fetch_all_live_matches(leagues=leagues, days_back=45, days_forward=10)
    if sync_history and live_records:
        try:
            added = sync_live_history(live_records=live_records)
            if any(added.values()):
                print(f"[HISTORY SYNC] Added new completed matches: {added}")
        except Exception as e:
            print(f"[HISTORY SYNC WARN] {e}")

    result_index = build_result_index(
        leagues=leagues, live_records=live_records, use_live=use_live, base_dir=ledger_path.parent
    )
    team_model = get_model(refresh=bool(sync_history and live_records))

    # Populate explicit selection_side on any legacy AH/1X2 row that lacked it
    for idx, row in df.iterrows():
        mk_low = str(row.get("market", "")).lower()
        if ("ah" in mk_low or "handicap" in mk_low) and not str(row.get("selection_side", "")).strip():
            side = determine_selection_side(
                row["home"], row["away"], str(row["league"]),
                fixture_date=row["date"], team_model=team_model,
            )
            df.at[idx, "selection_side"] = side

    pending_mask = df["status"] == "PENDING"
    pending = df[pending_mask].copy()
    played = pending[pd.to_datetime(pending["date"]).dt.normalize() <= cut]

    settled_count = 0
    not_found = 0
    ambiguous = 0
    for idx, row in played.iterrows():
        updated = settle_row(row.to_dict(), cutoff=cut, result_index=result_index, team_model=team_model)
        df.at[idx, "status"] = updated["status"]
        df.at[idx, "result"] = updated.get("result", "")
        df.at[idx, "profit"] = updated.get("profit", np.nan)
        if updated.get("selection_side"):
            df.at[idx, "selection_side"] = updated["selection_side"]
        if updated["status"] != "PENDING":
            settled_count += 1
            if updated["status"] in ("UNKNOWN_AH", "UNKNOWN_MARKET"):
                ambiguous += 1
        else:
            not_found += 1

    # Real-result hit-rate across ALL settled rows (PUSH draws excluded from hit-rate denominator)
    done = df[df["status"].isin(["WIN", "LOSS", "HALF WIN"])]
    pushes = df[df["status"] == "PUSH"]
    wins = done[done["status"] == "WIN"]
    half = done[done["status"] == "HALF WIN"]
    losses = done[done["status"] == "LOSS"]
    settled_units = len(done)
    weighted_wins = len(wins) + 0.5 * len(half)
    hit_rate = (weighted_wins / settled_units * 100.0) if settled_units else 0.0

    # Verified-odds ROI ONLY (strictly excludes placeholder 1.90 rows!)
    verified_done = done[done["odds_source"].astype(str).str.strip() == "the-odds-api"]
    verified_profit = float(pd.to_numeric(verified_done["profit"], errors="coerce").fillna(0.0).sum()) if not verified_done.empty else 0.0
    verified_staked = float(pd.to_numeric(verified_done["stake"], errors="coerce").fillna(0.0).sum()) if not verified_done.empty else 0.0
    verified_roi = round(verified_profit / verified_staked * 100.0, 2) if verified_staked > 0 else None

    save_ledger_df(df, ledger_path, is_v4=is_v4)

    # Matchweek Window (Tue-Mon) P&L + Permanent Weekly Archive
    is_default_repo_ledger = ledger_path.resolve().parent == REPO_DIR.resolve()
    eff_archive = archive_path if archive_path is not None else (
        ARCHIVE_WEEKLY_FILE if is_default_repo_ledger else (ledger_path.parent / "weekly_archive.json")
    )
    eff_public_archive = public_archive_path if public_archive_path is not None else (
        PUBLIC_WEEKLY_FILE if is_default_repo_ledger else None
    )
    closing_index = build_closing_odds_index(extra_records=live_records)
    weekly_archive, transparency_banner = build_weekly_transparency_and_archive(
        df=df,
        archive_path=eff_archive,
        public_archive_path=eff_public_archive,
        closing_index=closing_index,
        reference_now=cut,
        persist=persist_archive,
    )

    summary = {
        "cutoff": str(cut),
        "pending_before": int(len(pending)),
        "eligible_played": int(len(played)),
        "settled_now": int(settled_count),
        "result_not_found": int(not_found),
        "ambiguous_average": int(ambiguous),
        "pending_after": int((df["status"] == "PENDING").sum()),
        "total_settled_all_time": int(settled_units),
        "wins": int(len(wins)),
        "half_wins": int(len(half)),
        "losses": int(len(losses)),
        "pushes": int(len(pushes)),
        "settled_units": int(settled_units),
        "hit_rate_percent": round(float(hit_rate), 2),
        # Verified-odds-only financial metrics (no fake 1.90 ROI!)
        "verified_settled_count": int(len(verified_done)),
        "verified_profit_units": round(verified_profit, 4),
        "verified_staked": round(verified_staked, 4),
        "verified_roi_percent": verified_roi,
        "leagues": sorted(df["league"].dropna().astype(str).unique().tolist()),
        "transparency_banner": transparency_banner,
        "weekly_summary_count": int(weekly_archive["total_weeks"]),
        "closed_weeks_count": int(weekly_archive["closed_weeks_count"]),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> Dict[str, Any]:
    ledger = Path(LEDGER)
    if not ledger.exists():
        ledger = REPO_DIR / LEDGER
    if not ledger.exists():
        print(json.dumps({"error": "ledger not found"}))
        sys.exit(1)
    summary = settle_ledger(ledger_path=ledger, summary_path=ledger.parent / SUMMARY_FILE)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
