"""
live_data.py — Live Fixtures, Completed Scores, History Sync & Canonical Team Mapping.

Replaces the broken Understat JSON endpoint and frozen September CSV with a
clean, resilient multi-source data layer:
  1. ESPN public soccer scoreboard API (no API key required; day-by-day queries
     for both upcoming fixtures and recent completed scores).
  2. football-data.co.uk current-season CSVs (no API key required; full season
     completed scores including HxG / AxG when available).
  3. Optional football-data.org v4 API (when FOOTBALL_DATA_API_KEY is set).
  4. Optional The Odds API /scores endpoint (when THE_ODDS_API_KEY is set).
  5. Clean local JSON/CSV snapshots fallback when offline.

Also provides a single canonical team-name normalizer shared across
autoloop_master.py, team_strength.py, fetch_odds.py, and settle_ledger_auto.py.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

REPO_DIR = Path(__file__).parent

ESPN_LEAGUE_CODES: Dict[str, str] = {
    "La Liga": "esp.1",
    "Ligue 1": "fra.1",
}

FD_ORG_COMPETITIONS: Dict[str, str] = {
    "La Liga": "PD",
    "Ligue 1": "FL1",
}

FD_UK_LEAGUE_FILES: Dict[str, str] = {
    "La Liga": "SP1.csv",
    "Ligue 1": "F1.csv",
}

ODDS_API_SPORTS: Dict[str, str] = {
    "La Liga": "soccer_spain_la_liga",
    "Ligue 1": "soccer_france_ligue_one",
}

LIVE_CACHE_FILE = REPO_DIR / "live_results_cache.json"
LA_LIGA_HISTORY_CSV = REPO_DIR / "la_liga_matches_2018_2026.csv"
LIGUE1_LIVE_CSV = REPO_DIR / "live_matches_ligue1.csv"


# ==============================================================================
# 1. CANONICAL TEAM NAME MAPPING (Understat / ESPN / Odds API / football-data)
# ==============================================================================

_CANONICAL_ALIASES: Dict[str, str] = {
    # --- La Liga ---
    "alaves": "Alaves",
    "deportivo alaves": "Alaves",
    "alaves sad": "Alaves",
    "athletic bilbao": "Athletic Club",
    "athletic club": "Athletic Club",
    "athletic club bilbao": "Athletic Club",
    "ath bilbao": "Athletic Club",
    "atletico madrid": "Atletico Madrid",
    "atletico de madrid": "Atletico Madrid",
    "club atletico de madrid": "Atletico Madrid",
    "ath madrid": "Atletico Madrid",
    "barcelona": "Barcelona",
    "fc barcelona": "Barcelona",
    "cadiz": "Cadiz",
    "cadiz cf": "Cadiz",
    "celta": "Celta Vigo",
    "celta vigo": "Celta Vigo",
    "celta de vigo": "Celta Vigo",
    "rc celta": "Celta Vigo",
    "rc celta de vigo": "Celta Vigo",
    "eibar": "Eibar",
    "sd eibar": "Eibar",
    "elche": "Elche",
    "elche cf": "Elche",
    "espanyol": "Espanyol",
    "rcd espanyol": "Espanyol",
    "rcd espanyol de barcelona": "Espanyol",
    "espanol": "Espanyol",
    "getafe": "Getafe",
    "getafe cf": "Getafe",
    "girona": "Girona",
    "girona fc": "Girona",
    "granada": "Granada",
    "granada cf": "Granada",
    "huesca": "SD Huesca",
    "sd huesca": "SD Huesca",
    "las palmas": "Las Palmas",
    "ud las palmas": "Las Palmas",
    "leganes": "Leganes",
    "cd leganes": "Leganes",
    "levante": "Levante",
    "levante ud": "Levante",
    "malaga": "Malaga",
    "malaga cf": "Malaga",
    "mallorca": "Mallorca",
    "rcd mallorca": "Mallorca",
    "real mallorca": "Mallorca",
    "osasuna": "Osasuna",
    "ca osasuna": "Osasuna",
    "oviedo": "Real Oviedo",
    "real oviedo": "Real Oviedo",
    "racing santander": "Racing Santander",
    "real racing club": "Racing Santander",
    "santander": "Racing Santander",
    "rayo vallecano": "Rayo Vallecano",
    "vallecano": "Rayo Vallecano",
    "rayo vallecano de madrid": "Rayo Vallecano",
    "real betis": "Real Betis",
    "betis": "Real Betis",
    "real betis balompie": "Real Betis",
    "real madrid": "Real Madrid",
    "real madrid cf": "Real Madrid",
    "real sociedad": "Real Sociedad",
    "sociedad": "Real Sociedad",
    "real sociedad de futbol": "Real Sociedad",
    "real valladolid": "Real Valladolid",
    "valladolid": "Real Valladolid",
    "sevilla": "Sevilla",
    "sevilla fc": "Sevilla",
    "valencia": "Valencia",
    "valencia cf": "Valencia",
    "villarreal": "Villarreal",
    "villarreal cf": "Villarreal",
    "almeria": "Almeria",
    "ud almeria": "Almeria",
    # --- Ligue 1 ---
    "ajaccio": "Ajaccio",
    "ac ajaccio": "Ajaccio",
    "amiens": "Amiens",
    "amiens sc": "Amiens",
    "angers": "Angers",
    "angers sco": "Angers",
    "auxerre": "Auxerre",
    "aj auxerre": "Auxerre",
    "bordeaux": "Bordeaux",
    "girondins de bordeaux": "Bordeaux",
    "brest": "Brest",
    "stade brestois": "Brest",
    "stade brestois 29": "Brest",
    "clermont": "Clermont Foot",
    "clermont foot": "Clermont Foot",
    "clermont foot 63": "Clermont Foot",
    "dijon": "Dijon",
    "dijon fco": "Dijon",
    "guingamp": "Guingamp",
    "ea guingamp": "Guingamp",
    "le havre": "Le Havre",
    "le havre ac": "Le Havre",
    "havre": "Le Havre",
    "le mans": "Le Mans",
    "le mans fc": "Le Mans",
    "lens": "Lens",
    "rc lens": "Lens",
    "lille": "Lille",
    "lille osc": "Lille",
    "losc": "Lille",
    "losc lille": "Lille",
    "lorient": "Lorient",
    "fc lorient": "Lorient",
    "lyon": "Lyon",
    "olympique lyonnais": "Lyon",
    "olympique lyon": "Lyon",
    "marseille": "Marseille",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",
    "metz": "Metz",
    "fc metz": "Metz",
    "monaco": "Monaco",
    "as monaco": "Monaco",
    "as monaco fc": "Monaco",
    "montpellier": "Montpellier",
    "montpellier hsc": "Montpellier",
    "nantes": "Nantes",
    "fc nantes": "Nantes",
    "nice": "Nice",
    "ogc nice": "Nice",
    "nimes": "Nimes",
    "nimes olympique": "Nimes",
    "paris fc": "Paris FC",
    "paris saint germain": "Paris Saint Germain",
    "paris saint-germain": "Paris Saint Germain",
    "paris saint-germain fc": "Paris Saint Germain",
    "paris sg": "Paris Saint Germain",
    "psg": "Paris Saint Germain",
    "reims": "Reims",
    "stade de reims": "Reims",
    "rennes": "Rennes",
    "stade rennais": "Rennes",
    "stade rennais fc": "Rennes",
    "stade rennais fc 1901": "Rennes",
    "saint-etienne": "Saint-Etienne",
    "saint etienne": "Saint-Etienne",
    "as saint-etienne": "Saint-Etienne",
    "st etienne": "Saint-Etienne",
    "strasbourg": "Strasbourg",
    "rc strasbourg": "Strasbourg",
    "rc strasbourg alsace": "Strasbourg",
    "toulouse": "Toulouse",
    "toulouse fc": "Toulouse",
    "troyes": "Troyes",
    "estac troyes": "Troyes",
}


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def canonical_team_name(name: Any, league: Optional[str] = None) -> str:
    """Normalize a club name across Understat, ESPN, football-data, and The Odds API."""
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""
    norm = _strip_accents(raw).lower()
    norm = re.sub(r"[-_./]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[norm]
    # Also try hyphenated form for clubs like Saint-Etienne
    hyph = norm.replace(" ", "-")
    if hyph in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[hyph]
    # Try stripping common leading/trailing club tokens
    stripped = re.sub(
        r"^(fc|cf|rc|rcd|cd|ud|sd|ca|as|ac|aj|ogc|losc|olympique de|olympique|stade de|stade|club|deportivo)\s+",
        "",
        norm,
    )
    stripped = re.sub(
        r"\s+(fc|cf|ud|sd|cd|sc|sco|hsc|osc|ac|sad|1901|29|63)$",
        "",
        stripped,
    ).strip()
    if stripped in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[stripped]
    return raw


def _current_season_code(ref_date: Optional[datetime] = None) -> str:
    """Return football-data.co.uk 4-digit season code, e.g. '2627' for Nov 2026."""
    dt = ref_date or datetime.now(timezone.utc)
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or val == "" or pd.isna(val):
            return None
        f = float(val)
        return f if f == f else None
    except Exception:
        return None


# ==============================================================================
# 2. SOURCE FETCHERS (ESPN, football-data.co.uk, football-data.org, Odds API)
# ==============================================================================

def fetch_espn_matches(
    league: str,
    days_back: int = 10,
    days_forward: int = 10,
    ref_date: Optional[datetime] = None,
    timeout: int = 12,
) -> List[Dict[str, Any]]:
    """
    Fetch recent completed scores + upcoming fixtures from ESPN's public
    scoreboard API (no API key required). Uses single-day queries in parallel.
    """
    code = ESPN_LEAGUE_CODES.get(league)
    if not code:
        return []
    now = ref_date or datetime.now(timezone.utc)
    day_strs = [
        (now + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(-days_back, days_forward + 1)
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; autoloop/2.0)"})

    def _get_day(day_str: str) -> List[Dict[str, Any]]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={day_str}"
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception:
            return []
        rows: List[Dict[str, Any]] = []
        for ev in data.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors", [])
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home_c or not away_c:
                continue
            home = canonical_team_name(home_c.get("team", {}).get("displayName", ""), league)
            away = canonical_team_name(away_c.get("team", {}).get("displayName", ""), league)
            st_type = ev.get("status", {}).get("type", {})
            completed = bool(st_type.get("completed", False))
            state = str(st_type.get("state", "")).lower()
            dt_raw = str(ev.get("date", ""))
            dt_iso = dt_raw.replace("Z", "+00:00")
            try:
                dt_parsed = pd.to_datetime(dt_iso, utc=True).tz_localize(None)
                dt_str = dt_parsed.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_str = dt_raw[:19].replace("T", " ")

            gh = int(float(home_c["score"])) if completed and _safe_float(home_c.get("score")) is not None else None
            ga = int(float(away_c["score"])) if completed and _safe_float(away_c.get("score")) is not None else None
            rows.append({
                "league": league,
                "date": dt_str,
                "date_key": dt_str[:10],
                "home": home,
                "away": away,
                "completed": completed,
                "state": state,
                "goals_home": gh,
                "goals_away": ga,
                "xG_home": None,
                "xG_away": None,
                "source": "espn",
            })
        return rows

    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(_get_day, d) for d in day_strs]
        for f in as_completed(futs):
            for rec in f.result():
                key = (rec["date_key"], rec["home"], rec["away"])
                results[key] = rec

    return sorted(results.values(), key=lambda x: x["date"])


def fetch_fd_uk_season_results(
    league: str,
    ref_date: Optional[datetime] = None,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetch current-season completed results (including HxG / AxG when present)
    from football-data.co.uk CSVs (no API key required).
    """
    fname = FD_UK_LEAGUE_FILES.get(league)
    if not fname:
        return []
    season_code = _current_season_code(ref_date)
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{fname}"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.strip():
            return []
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        gh = _safe_float(row.get("FTHG"))
        ga = _safe_float(row.get("FTAG"))
        if gh is None or ga is None:
            continue
        raw_date = str(row.get("Date", "")).strip()
        raw_time = str(row.get("Time", "15:00")).strip()
        if not raw_date or raw_date.lower() == "nan":
            continue
        if not raw_time or raw_time.lower() == "nan":
            raw_time = "15:00"
        dt = pd.to_datetime(f"{raw_date} {raw_time}", dayfirst=True, errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            continue
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        home = canonical_team_name(row.get("HomeTeam", ""), league)
        away = canonical_team_name(row.get("AwayTeam", ""), league)
        if not home or not away:
            continue
        rows.append({
            "league": league,
            "date": dt_str,
            "date_key": dt_str[:10],
            "home": home,
            "away": away,
            "completed": True,
            "state": "post",
            "goals_home": int(gh),
            "goals_away": int(ga),
            "xG_home": _safe_float(row.get("HxG")),
            "xG_away": _safe_float(row.get("AxG")),
            "source": "football-data.co.uk",
        })
    return sorted(rows, key=lambda x: x["date"])


def fetch_football_data_org_matches(
    league: str,
    api_key: Optional[str] = None,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Optional fetcher using football-data.org v4 API when FOOTBALL_DATA_API_KEY is set."""
    key = (api_key or os.environ.get("FOOTBALL_DATA_API_KEY", "")).strip()
    comp = FD_ORG_COMPETITIONS.get(league)
    if not key or not comp:
        return []
    url = f"https://api.football-data.org/v4/competitions/{comp}/matches"
    try:
        r = requests.get(url, headers={"X-Auth-Token": key}, timeout=timeout)
        if r.status_code != 200:
            return []
        matches = r.json().get("matches", [])
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for m in matches:
        status = str(m.get("status", "")).upper()
        completed = status == "FINISHED"
        ft = (m.get("score") or {}).get("fullTime") or {}
        gh = int(ft["home"]) if completed and ft.get("home") is not None else None
        ga = int(ft["away"]) if completed and ft.get("away") is not None else None
        dt_raw = str(m.get("utcDate", ""))
        try:
            dt_str = pd.to_datetime(dt_raw, utc=True).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        home = canonical_team_name((m.get("homeTeam") or {}).get("name", ""), league)
        away = canonical_team_name((m.get("awayTeam") or {}).get("name", ""), league)
        rows.append({
            "league": league,
            "date": dt_str,
            "date_key": dt_str[:10],
            "home": home,
            "away": away,
            "completed": completed,
            "state": "post" if completed else "pre",
            "goals_home": gh,
            "goals_away": ga,
            "xG_home": None,
            "xG_away": None,
            "source": "football-data.org",
        })
    return sorted(rows, key=lambda x: x["date"])


def fetch_odds_api_scores(
    league: str,
    api_key: Optional[str] = None,
    days_from: int = 3,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Optional fetcher using The Odds API /scores when THE_ODDS_API_KEY is set."""
    key = (api_key or os.environ.get("THE_ODDS_API_KEY", "")).strip()
    sport = ODDS_API_SPORTS.get(league)
    if not key or not sport:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/scores/"
    try:
        r = requests.get(url, params={"apiKey": key, "daysFrom": days_from}, timeout=timeout)
        if r.status_code != 200:
            return []
        events = r.json()
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for ev in events:
        completed = bool(ev.get("completed", False))
        home_raw = str(ev.get("home_team", ""))
        away_raw = str(ev.get("away_team", ""))
        home = canonical_team_name(home_raw, league)
        away = canonical_team_name(away_raw, league)
        gh = ga = None
        if completed and isinstance(ev.get("scores"), list):
            score_map = {str(s.get("name", "")): s.get("score") for s in ev["scores"]}
            gh_f = _safe_float(score_map.get(home_raw))
            ga_f = _safe_float(score_map.get(away_raw))
            if gh_f is not None and ga_f is not None:
                gh, ga = int(gh_f), int(ga_f)
        dt_raw = str(ev.get("commence_time", ""))
        try:
            dt_str = pd.to_datetime(dt_raw, utc=True).tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        rows.append({
            "league": league,
            "date": dt_str,
            "date_key": dt_str[:10],
            "home": home,
            "away": away,
            "completed": completed,
            "state": "post" if completed else "pre",
            "goals_home": gh,
            "goals_away": ga,
            "xG_home": None,
            "xG_away": None,
            "source": "the-odds-api-scores",
        })
    return sorted(rows, key=lambda x: x["date"])


# ==============================================================================
# 3. UNIFIED LIVE SYNC, UPCOMING FIXTURES & HISTORY AUTO-UPDATER
# ==============================================================================

def _merge_matches(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge match records across sources by (league, home, away, nearby date).
    Prefers completed records with xG over records without xG.
    """
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for rec in records:
        key = (rec["league"], rec["date_key"], rec["home"], rec["away"])
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(rec)
            continue
        # Prefer completed score if newly available
        if rec.get("completed") and not existing.get("completed"):
            existing["completed"] = True
            existing["goals_home"] = rec.get("goals_home")
            existing["goals_away"] = rec.get("goals_away")
            existing["state"] = "post"
        # Preserve xG whenever any source provides it
        if existing.get("xG_home") is None and rec.get("xG_home") is not None:
            existing["xG_home"] = rec["xG_home"]
        if existing.get("xG_away") is None and rec.get("xG_away") is not None:
            existing["xG_away"] = rec["xG_away"]
    return sorted(merged.values(), key=lambda x: x["date"])


def fetch_all_live_matches(
    leagues: Tuple[str, ...] = ("La Liga", "Ligue 1"),
    days_back: int = 14,
    days_forward: int = 10,
    use_cache_hours: float = 3.0,
    cache_path: Path = LIVE_CACHE_FILE,
) -> List[Dict[str, Any]]:
    """
    Fetch unified recent scores + upcoming fixtures across all configured live
    sources with disk caching (default 3h TTL) and graceful offline fallback.
    """
    if use_cache_hours > 0 and cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_h < use_cache_hours:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list) and cached:
                    return [m for m in cached if m.get("league") in leagues]
            except Exception:
                pass

    all_records: List[Dict[str, Any]] = []
    for lg in leagues:
        # 1. Optional football-data.org (full fixtures + scores)
        all_records.extend(fetch_football_data_org_matches(lg))
        # 2. Free full-season scores + HxG/AxG from football-data.co.uk
        all_records.extend(fetch_fd_uk_season_results(lg))
        # 3. Free ESPN scoreboard (upcoming fixtures + latest scores)
        all_records.extend(fetch_espn_matches(lg, days_back=days_back, days_forward=days_forward))
        # 4. Optional The Odds API /scores
        all_records.extend(fetch_odds_api_scores(lg))

    merged = _merge_matches(all_records)
    if merged:
        try:
            cache_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return merged

    # Offline fallback: return stale cache if present
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return [m for m in cached if m.get("league") in leagues]
        except Exception:
            pass
    return []


def fetch_upcoming_fixtures(
    leagues: Tuple[str, ...] = ("La Liga", "Ligue 1"),
    window_days: int = 8,
    now: Optional[pd.Timestamp] = None,
    live_records: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Return upcoming (unplayed) fixtures inside [now, now + window_days] as a
    clean DataFrame with canonical team names: [date, league, home, away, source].
    """
    ref = (now or pd.Timestamp.now()).normalize()
    end = ref + pd.Timedelta(days=window_days)
    records = live_records if live_records is not None else fetch_all_live_matches(
        leagues=leagues, days_back=2, days_forward=max(window_days + 2, 10)
    )

    upcoming: List[Dict[str, Any]] = []
    seen = set()
    for m in records:
        if m.get("completed"):
            continue
        if m.get("league") not in leagues:
            continue
        dt = pd.to_datetime(m.get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        if ref <= dt <= end:
            key = (m["league"], dt.strftime("%Y-%m-%d"), m["home"], m["away"])
            if key in seen:
                continue
            seen.add(key)
            upcoming.append({
                "date": dt,
                "league": m["league"],
                "home": m["home"],
                "away": m["away"],
                "source": m.get("source", "live"),
            })

    if not upcoming:
        return pd.DataFrame(columns=["date", "league", "home", "away", "source"])
    return pd.DataFrame(upcoming).sort_values("date").reset_index(drop=True)


def sync_live_history(
    live_records: Optional[List[Dict[str, Any]]] = None,
    la_csv: Path = LA_LIGA_HISTORY_CSV,
    ligue1_csv: Path = LIGUE1_LIVE_CSV,
) -> Dict[str, int]:
    """
    Append newly completed matches from live feeds into local historical CSVs
    (deduplicated by date_key + home + away), keeping TeamStrengthModel fresh.
    """
    records = live_records if live_records is not None else fetch_all_live_matches()
    completed = [
        m for m in records
        if m.get("completed") and m.get("goals_home") is not None and m.get("goals_away") is not None
    ]
    added_counts = {"La Liga": 0, "Ligue 1": 0}

    # --- 1. Update La Liga CSV ---
    la_done = [m for m in completed if m.get("league") == "La Liga"]
    if la_done and la_csv.exists():
        df = pd.read_csv(la_csv)
        existing_keys = set()
        for _, r in df.iterrows():
            dk = str(r.get("date", ""))[:10]
            h = canonical_team_name(r.get("home", ""), "La Liga")
            a = canonical_team_name(r.get("away", ""), "La Liga")
            existing_keys.add((dk, h, a))
        new_rows = []
        for m in la_done:
            key = (m["date_key"], m["home"], m["away"])
            if key in existing_keys:
                continue
            dt = pd.to_datetime(m["date"])
            season_yr = dt.year if dt.month >= 7 else dt.year - 1
            gh, ga = int(m["goals_home"]), int(m["goals_away"])
            new_rows.append({
                "season": season_yr,
                "season_label": f"{season_yr}/{season_yr + 1}",
                "date": m["date"],
                "isResult": True,
                "home": m["home"],
                "away": m["away"],
                "home_id": "",
                "away_id": "",
                "goals_home": gh,
                "goals_away": ga,
                "xG_home": m.get("xG_home") if m.get("xG_home") is not None else "",
                "xG_away": m.get("xG_away") if m.get("xG_away") is not None else "",
                "total_goals": gh + ga,
                "btts": 1 if (gh > 0 and ga > 0) else 0,
                "over25": 1 if (gh + ga) > 2.5 else 0,
            })
            existing_keys.add(key)
        if new_rows:
            out_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            out_df.to_csv(la_csv, index=False)
            added_counts["La Liga"] = len(new_rows)

    # --- 2. Update Ligue 1 live CSV (read alongside raw_Ligue1_8seasons.json) ---
    l1_done = [m for m in completed if m.get("league") == "Ligue 1"]
    if l1_done:
        cols = ["date", "home", "away", "goals_home", "goals_away", "xG_home", "xG_away", "source"]
        if ligue1_csv.exists():
            df1 = pd.read_csv(ligue1_csv)
        else:
            df1 = pd.DataFrame(columns=cols)
        existing_keys = set()
        for _, r in df1.iterrows():
            existing_keys.add((
                str(r.get("date", ""))[:10],
                canonical_team_name(r.get("home", ""), "Ligue 1"),
                canonical_team_name(r.get("away", ""), "Ligue 1"),
            ))
        new_rows = []
        for m in l1_done:
            key = (m["date_key"], m["home"], m["away"])
            if key in existing_keys:
                continue
            new_rows.append({
                "date": m["date"],
                "home": m["home"],
                "away": m["away"],
                "goals_home": int(m["goals_home"]),
                "goals_away": int(m["goals_away"]),
                "xG_home": m.get("xG_home") if m.get("xG_home") is not None else "",
                "xG_away": m.get("xG_away") if m.get("xG_away") is not None else "",
                "source": m.get("source", "live"),
            })
            existing_keys.add(key)
        if new_rows:
            out_df1 = pd.concat([df1, pd.DataFrame(new_rows)], ignore_index=True)
            out_df1.sort_values("date").to_csv(ligue1_csv, index=False)
            added_counts["Ligue 1"] = len(new_rows)

    return added_counts
