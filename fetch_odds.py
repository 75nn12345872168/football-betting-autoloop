"""
fetch_odds.py — Attach REAL bookmaker odds to the ledger from The Odds API.

Upgrades over v1:
  1. Uses canonical_team_name() from live_data.py so team names always match
     between Understat / ESPN / football-data and The Odds API.
  2. Selects the BEST available decimal price (max across bookmakers) instead
     of the worst (min).
  3. Supports 1X2 (Home Win / Draw / Away Win), Over/Under 2.5, and AH -0.75
     using the unified explicit `selection_side` column (`HOME` / `AWAY`).
  4. Never fabricates prices: bets without a verified bookmaker quote remain
     odds_source='' and route to Broker Mode (min_odds with +6% safety margin).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

from live_data import ODDS_API_SPORTS, canonical_team_name

LEDGER = "paper_trading_ledger_2026.csv"
CONFIG = "odds_config.json"
BASE = "https://api.the-odds-api.com/v4"

SPORT_MAP: Dict[str, str] = dict(ODDS_API_SPORTS)


def get_api_key() -> str:
    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if key:
        return key
    cfg = Path(CONFIG)
    if cfg.exists():
        try:
            key = str(json.loads(cfg.read_text(encoding="utf-8")).get("api_key", "")).strip()
        except Exception:
            pass
    if not key:
        sys.exit(
            "ERR: no THE_ODDS_API_KEY in env or odds_config.json — "
            "get a key at https://the-odds-api.com (free tier) and set it."
        )
    return key


def _best(current: Optional[float], candidate: float) -> float:
    """Return the best (highest) valid decimal price across bookmakers."""
    if candidate <= 1.0:
        return current if current is not None else 0.0
    if current is None or candidate > current:
        return float(candidate)
    return float(current)


def parse_odds_events(
    events: list, league: str
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Parse The Odds API JSON events into a canonical-team-keyed map taking the
    BEST (maximum) available decimal odds per market across bookmakers.
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for m in events:
        raw_home = str(m.get("home_team", ""))
        raw_away = str(m.get("away_team", ""))
        home = canonical_team_name(raw_home, league)
        away = canonical_team_name(raw_away, league)
        if not home or not away:
            continue

        rec: Dict[str, Any] = {
            "over25": None,
            "under25": None,
            "home_win": None,
            "draw": None,
            "away_win": None,
            "spreads": {"HOME": {}, "AWAY": {}},  # line -> best price
        }

        for bm in m.get("bookmakers", []):
            for mk in bm.get("markets", []):
                mktype = mk.get("key")
                for o in mk.get("outcomes", []):
                    try:
                        price = float(o["price"])
                    except Exception:
                        continue
                    if price <= 1.0:
                        continue
                    oname = str(o.get("name", "")).strip()
                    oname_canon = canonical_team_name(oname, league)

                    if mktype == "totals":
                        pt = o.get("point")
                        if pt is not None and abs(float(pt) - 2.5) < 1e-6:
                            if oname.lower() == "over":
                                rec["over25"] = _best(rec["over25"], price)
                            elif oname.lower() == "under":
                                rec["under25"] = _best(rec["under25"], price)

                    elif mktype == "h2h":
                        if oname.lower() == "draw":
                            rec["draw"] = _best(rec["draw"], price)
                        elif oname == raw_home or oname_canon == home:
                            rec["home_win"] = _best(rec["home_win"], price)
                        elif oname == raw_away or oname_canon == away:
                            rec["away_win"] = _best(rec["away_win"], price)

                    elif mktype == "spreads":
                        pt = o.get("point")
                        if pt is None:
                            continue
                        line = round(float(pt), 2)
                        if oname == raw_home or oname_canon == home:
                            prev = rec["spreads"]["HOME"].get(line)
                            rec["spreads"]["HOME"][line] = _best(prev, price)
                        elif oname == raw_away or oname_canon == away:
                            prev = rec["spreads"]["AWAY"].get(line)
                            rec["spreads"]["AWAY"][line] = _best(prev, price)

        out[(home, away)] = rec
    return out


def fetch_odds(
    session: requests.Session, key: str, league: str, sport_key: str, regions: str = "eu"
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    url = f"{BASE}/sports/{sport_key}/odds/"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": "h2h,totals,spreads",
        "oddsFormat": "decimal",
    }
    r = session.get(url, params=params, timeout=20)
    if r.status_code == 401:
        raise PermissionError("The Odds API rejected the key (401/INVALID_KEY).")
    r.raise_for_status()
    return parse_odds_events(r.json(), league)


def extract_market_price(
    rec: Dict[str, Any], market: str, selection_side: str = ""
) -> Optional[float]:
    """Extract the best verified bookmaker price for a given market & selection_side."""
    mk = str(market or "").strip().lower()
    side = str(selection_side or "").strip().upper()

    if "over" in mk and "2.5" in mk:
        return rec.get("over25")
    if "under" in mk and "2.5" in mk:
        return rec.get("under25")
    if "1x2" in mk or mk in ("home win", "away win", "draw", "home", "away"):
        if side == "HOME" or "home" in mk:
            return rec.get("home_win")
        if side == "AWAY" or "away" in mk:
            return rec.get("away_win")
        if side == "DRAW" or "draw" in mk:
            return rec.get("draw")
        return None
    if "ah" in mk or "handicap" in mk:
        if side not in ("HOME", "AWAY"):
            return None
        # Exact -0.75 spread line only — never fabricate from unrelated lines
        return (rec.get("spreads") or {}).get(side, {}).get(-0.75)
    return None


def main() -> None:
    key = get_api_key()
    ledger = Path(LEDGER)
    if not ledger.exists():
        sys.exit(f"ERR: {LEDGER} not found")

    df = pd.read_csv(ledger, parse_dates=["date"])
    if "odds_source" not in df.columns:
        df["odds_source"] = ""
    if "market_odds" not in df.columns:
        df["market_odds"] = float("nan")
    if "selection_side" not in df.columns:
        df["selection_side"] = ""

    session = requests.Session()
    session.headers.update({"User-Agent": "autoloop/2.0"})

    attached = 0
    pending_mask = df["status"] == "PENDING"
    for league, sport_key in SPORT_MAP.items():
        subset = df[pending_mask & (df["league"] == league)]
        if subset.empty:
            continue
        try:
            odds_map = fetch_odds(session, key, league, sport_key)
        except PermissionError as e:
            sys.exit(f"ERR: {e}")
        except requests.HTTPError as e:
            print(f"[ODDS] {league}: HTTP error {e.response.status_code}; skipping")
            continue

        for idx in subset.index:
            home = canonical_team_name(df.at[idx, "home"], league)
            away = canonical_team_name(df.at[idx, "away"], league)
            market = str(df.at[idx, "market"])
            side = str(df.at[idx, "selection_side"] or "").strip().upper()
            rec = odds_map.get((home, away))
            if not rec:
                continue
            price = extract_market_price(rec, market, side)
            if price is not None and price > 1.0:
                df.at[idx, "market_odds"] = round(float(price), 3)
                df.at[idx, "odds_source"] = "the-odds-api"
                attached += 1
        time.sleep(1.1)

    df.to_csv(ledger, index=False)
    verified = int(((df["status"] == "PENDING") & (df["odds_source"] == "the-odds-api")).sum())
    print(
        json.dumps(
            {
                "attached_real_odds": attached,
                "pending_with_real_odds": verified,
                "pending_still_unverified": int(
                    ((df["status"] == "PENDING") & (df["odds_source"] != "the-odds-api")).sum()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
