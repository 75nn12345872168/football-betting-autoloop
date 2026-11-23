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


def settle_ledger(
    ledger_path: Path = Path(LEDGER),
    summary_path: Path = Path(SUMMARY_FILE),
    cutoff: Optional[pd.Timestamp] = None,
    live_records: Optional[List[Dict[str, Any]]] = None,
    use_live: bool = True,
    sync_history: bool = True,
) -> Dict[str, Any]:
    """Run settlement over `ledger_path`, save updated CSV + summary JSON, and return summary."""
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")

    cut = (cutoff if cutoff is not None else pd.Timestamp.now()).normalize()
    df = pd.read_csv(ledger_path, parse_dates=["date"])

    for col, default in [
        ("league", "La Liga"),
        ("market", "Over 2.5"),
        ("selection_side", ""),
        ("odds_source", ""),
        ("stake", 0.0),
        ("market_odds", np.nan),
        ("profit", np.nan),
        ("result", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df["result"] = df["result"].astype(object)
    df["selection_side"] = df["selection_side"].fillna("").astype(object)
    df["odds_source"] = df["odds_source"].fillna("").astype(object)

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

    # Real-result hit-rate across ALL settled rows (independent of fabricated odds)
    done = df[df["status"].isin(["WIN", "LOSS", "HALF WIN"])]
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

    df.to_csv(ledger_path, index=False)
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
        "settled_units": int(settled_units),
        "hit_rate_percent": round(float(hit_rate), 2),
        # Verified-odds-only financial metrics (no fake 1.90 ROI!)
        "verified_settled_count": int(len(verified_done)),
        "verified_profit_units": round(verified_profit, 4),
        "verified_staked": round(verified_staked, 4),
        "verified_roi_percent": verified_roi,
        "leagues": sorted(df["league"].dropna().astype(str).unique().tolist()),
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
