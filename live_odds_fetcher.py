"""
live_odds_fetcher.py — Live Bookmaker Odds Ingestion via The Odds API (v4).

Features:
  * Secure: reads THE_ODDS_API_KEY exclusively from environment (never hardcoded).
  * Fail-Safe: returns empty/None gracefully on missing key, network timeout,
    quota exhaustion (401/402/429), or malformed API payload without breaking autoloop.
  * Quota-Conscious: queries ONLY approved production leagues
    (soccer_epl, soccer_spain_la_liga, soccer_italy_serie_a) with h2h,totals,
    decimal odds, regions=eu,uk,us.
  * Exact DNB Home replication from h2h market:
        O_DNB_Home = O_home * (1 - 1 / O_draw)
  * Extracts live_best_odds, live_best_bookmaker, and live_pinnacle_odds per market.
  * Smart diacritic-aware canonical + fuzzy team name matching (e.g. Alavés <-> Alaves,
    Inter Milan <-> Inter, Parma <-> Parma Calcio 1913).
"""

from __future__ import annotations

import difflib
import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from live_data import canonical_team_name

THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

# Query ONLY approved production leagues to conserve API quota.
ODDS_API_SPORT_KEYS: Dict[str, str] = {
    "EPL": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
}

_STOP_TOKENS = {
    "fc", "cf", "cd", "ud", "rcd", "ca", "ac", "as", "ss", "ssc", "sc", "afc",
    "calcio", "club", "de", "del", "sv", "1913", "1907", "1909", "bc",
}


def compute_dnb_replication_odds(o_side: Optional[float], o_draw: Optional[float]) -> Optional[float]:
    """
    Exact tradable Draw No Bet replication odds from decimal 1X2 prices:
        O_DNB = O_side * (1 - 1 / O_draw)
    """
    if o_side is None or o_draw is None:
        return None
    try:
        s = float(o_side)
        d = float(o_draw)
    except (TypeError, ValueError):
        return None
    if s <= 1.0 or d <= 1.0:
        return None
    rep = s * (1.0 - (1.0 / d))
    return round(rep, 3) if rep > 1.0 else None


def normalize_team_text(name: Any) -> str:
    """Strip Latin diacritics (Alavés -> alaves), punctuation, and extra whitespace."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    decomposed = unicodedata.normalize("NFKD", raw)
    ascii_str = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", ascii_str).lower()
    return " ".join(cleaned.split())


def _core_tokens(name: str) -> List[str]:
    norm = normalize_team_text(name)
    tokens = [t for t in norm.split() if t and t not in _STOP_TOKENS]
    return tokens or norm.split()


def team_similarity_score(name_a: str, name_b: str, league: str = "") -> float:
    """
    Compute robust similarity score in [0, 1] between two club names using:
      1. Canonical team name equality (1.0)
      2. Diacritic-stripped normalized equality (0.99)
      3. Core token containment / Jaccard + SequenceMatcher ratio
    """
    ca = canonical_team_name(name_a, league)
    cb = canonical_team_name(name_b, league)
    if ca and cb and ca.lower() == cb.lower():
        return 1.0

    na = normalize_team_text(ca or name_a)
    nb = normalize_team_text(cb or name_b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 0.99

    ta = set(_core_tokens(na))
    tb = set(_core_tokens(nb))
    generic = {"real", "manchester", "athletic", "atletico", "city", "united", "sporting", "racing"}
    if ta and tb:
        if ta == tb:
            return 0.97
        inter = ta & tb
        union = ta | tb
        jaccard = len(inter) / len(union)
        # If the two names share ONLY generic tokens (e.g., "Manchester City" vs "Manchester United",
        # "Real Madrid" vs "Real Betis") and each has distinct non-matching tokens, reject fuzzy match.
        distinct_a = ta - inter
        distinct_b = tb - inter
        if inter and inter.issubset(generic) and distinct_a and distinct_b:
            rem_seq = difflib.SequenceMatcher(
                None, " ".join(sorted(distinct_a)), " ".join(sorted(distinct_b))
            ).ratio()
            return round(min(0.45, rem_seq * 0.5), 4)

        meaningful_inter = inter - generic
        if meaningful_inter and (ta.issubset(tb) or tb.issubset(ta)):
            return 0.92
    else:
        jaccard = 0.0

    seq = difflib.SequenceMatcher(None, " ".join(_core_tokens(na)), " ".join(_core_tokens(nb))).ratio()
    return round(max(seq, jaccard), 4)


def _normalize_market_slot(market: str, selection_side: str = "") -> str:
    mk = str(market or "").strip().lower()
    sd = str(selection_side or "").strip().upper()
    if "dnb" in mk or "draw no bet" in mk or "ah 0" in mk:
        return "dnb_away" if ("away" in mk or sd == "AWAY") else "dnb_home"
    if "over" in mk and "2.5" in mk:
        return "over_2.5"
    if "under" in mk and "2.5" in mk:
        return "under_2.5"
    if "1x2" in mk or mk in ("home win", "away win", "draw", "home", "away"):
        if "away" in mk or sd == "AWAY":
            return "1x2_away"
        if "draw" in mk or sd == "DRAW":
            return "1x2_draw"
        return "1x2_home"
    return ""


def parse_odds_api_events(events: List[Dict[str, Any]], league: str) -> List[Dict[str, Any]]:
    """
    Parse raw The Odds API v4 events list for `league` into structured fixture
    market snapshots containing best odds, best bookmaker, and Pinnacle odds
    for 1X2, Totals 2.5, and replicated DNB Home/Away.
    """
    parsed: List[Dict[str, Any]] = []
    if not isinstance(events, list):
        return parsed

    for ev in events:
        if not isinstance(ev, dict):
            continue
        raw_home = str(ev.get("home_team") or "").strip()
        raw_away = str(ev.get("away_team") or "").strip()
        if not raw_home or not raw_away:
            continue

        c_home = canonical_team_name(raw_home, league)
        c_away = canonical_team_name(raw_away, league)
        commence_time = str(ev.get("commence_time") or "")

        # Per-market trackers: slot -> {best_odds, best_bookmaker, pinnacle_odds, count}
        slots: Dict[str, Dict[str, Any]] = {
            k: {"best_odds": None, "best_bookmaker": None, "pinnacle_odds": None, "count": 0}
            for k in ("1x2_home", "1x2_draw", "1x2_away", "dnb_home", "dnb_away", "over_2.5", "under_2.5")
        }

        def _record(slot_name: str, price: Optional[float], bk_title: str, is_pinnacle: bool) -> None:
            if price is None or price <= 1.0:
                return
            p = round(float(price), 3)
            st = slots[slot_name]
            st["count"] += 1
            if st["best_odds"] is None or p > st["best_odds"]:
                st["best_odds"] = p
                st["best_bookmaker"] = bk_title
            if is_pinnacle:
                st["pinnacle_odds"] = p

        for bm in ev.get("bookmakers") or []:
            if not isinstance(bm, dict):
                continue
            bk_key = str(bm.get("key") or "").strip().lower()
            bk_title = str(bm.get("title") or bk_key or "Unknown").strip()
            is_pin = bk_key == "pinnacle" or "pinnacle" in bk_title.lower()

            o_h: Optional[float] = None
            o_d: Optional[float] = None
            o_a: Optional[float] = None

            for mkt in bm.get("markets") or []:
                mkey = str(mkt.get("key") or "").strip().lower()
                outcomes = mkt.get("outcomes") or []

                if mkey == "h2h":
                    for oc in outcomes:
                        oname = str(oc.get("name") or "").strip()
                        try:
                            price = float(oc.get("price"))
                        except (TypeError, ValueError):
                            continue
                        if price <= 1.0:
                            continue
                        if normalize_team_text(oname) == "draw":
                            o_d = price
                        elif team_similarity_score(oname, raw_home, league) >= team_similarity_score(oname, raw_away, league):
                            o_h = price
                        else:
                            o_a = price

                    _record("1x2_home", o_h, bk_title, is_pin)
                    _record("1x2_draw", o_d, bk_title, is_pin)
                    _record("1x2_away", o_a, bk_title, is_pin)

                    # Exact DNB replication on this bookmaker's h2h prices
                    dnb_h = compute_dnb_replication_odds(o_h, o_d)
                    dnb_a = compute_dnb_replication_odds(o_a, o_d)
                    _record("dnb_home", dnb_h, bk_title, is_pin)
                    _record("dnb_away", dnb_a, bk_title, is_pin)

                elif mkey == "totals":
                    for oc in outcomes:
                        try:
                            pt = float(oc.get("point"))
                            price = float(oc.get("price"))
                        except (TypeError, ValueError):
                            continue
                        if abs(pt - 2.5) > 1e-6 or price <= 1.0:
                            continue
                        oname = str(oc.get("name") or "").strip().lower()
                        if oname == "over":
                            _record("over_2.5", price, bk_title, is_pin)
                        elif oname == "under":
                            _record("under_2.5", price, bk_title, is_pin)

        parsed.append({
            "league": league,
            "raw_home": raw_home,
            "raw_away": raw_away,
            "home": c_home,
            "away": c_away,
            "commence_time": commence_time,
            "markets": slots,
        })

    return parsed


def fetch_live_odds_for_leagues(
    leagues: Iterable[str] = tuple(ODDS_API_SPORT_KEYS.keys()),
    api_key: Optional[str] = None,
    regions: str = "eu,uk,us",
    markets: str = "h2h,totals",
    timeout: int = 25,
    retries: int = 1,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch live bookmaker odds from The Odds API (v4) for approved leagues ONLY.
    Fail-Safe: returns {} gracefully if API key is missing, quota is exhausted,
    or any network/HTTP error occurs.
    """
    key = (api_key if api_key is not None else os.getenv("THE_ODDS_API_KEY", "")).strip()
    if not key:
        return {}

    results: Dict[str, List[Dict[str, Any]]] = {}
    remaining_quota: Optional[str] = None

    for lg in leagues:
        sport_key = ODDS_API_SPORT_KEYS.get(str(lg).strip())
        if not sport_key:
            continue
        url = f"{THE_ODDS_API_BASE}/{sport_key}/odds"
        last_err: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                resp = requests.get(
                    url,
                    params={
                        "apiKey": key,
                        "regions": regions,
                        "markets": markets,
                        "oddsFormat": "decimal",
                    },
                    timeout=timeout,
                )
                if resp.headers.get("x-requests-remaining") is not None:
                    remaining_quota = str(resp.headers.get("x-requests-remaining"))

                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code}"
                    break

                payload = resp.json()
                if not isinstance(payload, list):
                    last_err = "non-list payload"
                    break

                parsed_events = parse_odds_api_events(payload, lg)
                results[lg] = parsed_events
                print(f"[LIVE ODDS] {lg}: fetched {len(parsed_events)} live event(s).")
                last_err = None
                break
            except Exception as exc:
                last_err = str(exc).replace(key, "***")[:120]
        if last_err:
            print(f"[LIVE ODDS WARN] {lg} ({sport_key}): {last_err} — continuing fail-safe.")

    if remaining_quota is not None:
        print(f"[LIVE ODDS] The Odds API requests remaining: {remaining_quota}")
    return results


def match_live_odds_for_fixture(
    league: str,
    home: str,
    away: str,
    market: str,
    selection_side: str = "",
    live_odds_by_league: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    min_team_similarity: float = 0.72,
) -> Dict[str, Any]:
    """
    Smart fuzzy lookup for a fixture + market inside fetched live odds snapshots.
    Returns:
      {
        "live_best_odds": float | None,
        "live_best_bookmaker": str | None,
        "live_pinnacle_odds": float | None,
      }
    """
    empty = {
        "live_best_odds": None,
        "live_best_bookmaker": None,
        "live_pinnacle_odds": None,
    }
    if not live_odds_by_league:
        return empty
    events = live_odds_by_league.get(league)
    if not events:
        return empty

    slot = _normalize_market_slot(market, selection_side)
    if not slot:
        return empty

    best_event = None
    best_score = 0.0
    for ev in events:
        sh = max(
            team_similarity_score(home, ev["home"], league),
            team_similarity_score(home, ev.get("raw_home", ""), league),
        )
        sa = max(
            team_similarity_score(away, ev["away"], league),
            team_similarity_score(away, ev.get("raw_away", ""), league),
        )
        if sh >= min_team_similarity and sa >= min_team_similarity:
            combined = 0.5 * (sh + sa)
            if combined > best_score:
                best_score = combined
                best_event = ev

    if not best_event:
        return empty

    quote = (best_event.get("markets") or {}).get(slot)
    if not quote or quote.get("best_odds") is None:
        return empty

    return {
        "live_best_odds": round(float(quote["best_odds"]), 2),
        "live_best_bookmaker": quote.get("best_bookmaker"),
        "live_pinnacle_odds": (
            round(float(quote["pinnacle_odds"]), 2)
            if quote.get("pinnacle_odds") is not None
            else None
        ),
    }


def evaluate_live_actionability(
    calibrated_prob: float,
    min_acceptable_odds: Optional[float],
    live_best_odds: Optional[float],
) -> Dict[str, Any]:
    """
    Compute live EV % and actionable status:
      * live_ev_pct = round((calibrated_prob * live_best_odds - 1) * 100, 2)
      * is_live_actionable = True iff live_best_odds >= min_acceptable_odds
    """
    if live_best_odds is None or float(live_best_odds) <= 1.0:
        return {"live_ev_pct": None, "is_live_actionable": False}
    p = float(calibrated_prob)
    o = float(live_best_odds)
    live_ev_pct = round((p * o - 1.0) * 100.0, 2)
    actionable = (
        bool(o >= float(min_acceptable_odds))
        if (min_acceptable_odds is not None and float(min_acceptable_odds) > 1.0)
        else bool(live_ev_pct > 0.0)
    )
    return {
        "live_ev_pct": live_ev_pct,
        "is_live_actionable": actionable,
    }
