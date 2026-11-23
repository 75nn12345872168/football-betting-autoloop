"""
live_data.py — Big-5 European Leagues Live Fixtures, Scores, History Sync & Canonical Team Mapping.

Supports all 5 major European leagues:
  * La Liga    (ESPN esp.1 | football-data.co.uk SP1.csv | football-data.org PD  | Odds API soccer_spain_la_liga)
  * Ligue 1    (ESPN fra.1 | football-data.co.uk F1.csv  | football-data.org FL1 | Odds API soccer_france_ligue_one)
  * EPL        (ESPN eng.1 | football-data.co.uk E0.csv  | football-data.org PL  | Odds API soccer_epl)
  * Bundesliga (ESPN ger.1 | football-data.co.uk D1.csv  | football-data.org BL1 | Odds API soccer_germany_bundesliga)
  * Serie A    (ESPN ita.1 | football-data.co.uk I1.csv  | football-data.org SA  | Odds API soccer_italy_serie_a)

Provides:
  1. Multi-source live fixture & completed score fetching (zero-key ESPN + football-data.co.uk,
     plus optional football-data.org v4 & The Odds API /scores).
  2. Automatic history sync into local CSV datasets so TeamStrengthModel stays fresh.
  3. Comprehensive canonical team-name mapping across Understat, ESPN, football-data.co.uk,
     football-data.org, and The Odds API for all Big-5 clubs.
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

BIG5_LEAGUES: Tuple[str, ...] = ("La Liga", "Ligue 1", "EPL", "Bundesliga", "Serie A")

ESPN_LEAGUE_CODES: Dict[str, str] = {
    "La Liga": "esp.1",
    "Ligue 1": "fra.1",
    "EPL": "eng.1",
    "Bundesliga": "ger.1",
    "Serie A": "ita.1",
}

FD_ORG_COMPETITIONS: Dict[str, str] = {
    "La Liga": "PD",
    "Ligue 1": "FL1",
    "EPL": "PL",
    "Bundesliga": "BL1",
    "Serie A": "SA",
}

FD_UK_LEAGUE_FILES: Dict[str, str] = {
    "La Liga": "SP1.csv",
    "Ligue 1": "F1.csv",
    "EPL": "E0.csv",
    "Bundesliga": "D1.csv",
    "Serie A": "I1.csv",
}

ODDS_API_SPORTS: Dict[str, str] = {
    "La Liga": "soccer_spain_la_liga",
    "Ligue 1": "soccer_france_ligue_one",
    "EPL": "soccer_epl",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
}

LIVE_CACHE_FILE = REPO_DIR / "live_results_cache.json"
LA_LIGA_HISTORY_CSV = REPO_DIR / "la_liga_matches_2018_2026.csv"
LIGUE1_LIVE_CSV = REPO_DIR / "live_matches_ligue1.csv"
BIG5_HISTORY_CSV = REPO_DIR / "data" / "historical_big5_odds.csv"


# ==============================================================================
# 1. CANONICAL TEAM NAME MAPPING (Big-5 across Understat/ESPN/FD-UK/FD-Org/Odds)
# ==============================================================================

_CANONICAL_ALIASES: Dict[str, str] = {
    # --- La Liga ---
    "alaves": "Alaves",
    "deportivo alaves": "Alaves",
    "alaves sad": "Alaves",
    "almeria": "Almeria",
    "ud almeria": "Almeria",
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
    "deportivo": "Deportivo La Coruna",
    "deportivo la coruna": "Deportivo La Coruna",
    "la coruna": "Deportivo La Coruna",
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
    "caen": "Caen",
    "sm caen": "Caen",
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

    # --- EPL (Premier League) ---
    "arsenal": "Arsenal",
    "arsenal fc": "Arsenal",
    "aston villa": "Aston Villa",
    "aston villa fc": "Aston Villa",
    "bournemouth": "Bournemouth",
    "afc bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brentford fc": "Brentford",
    "brighton": "Brighton",
    "brighton and hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "burnley": "Burnley",
    "burnley fc": "Burnley",
    "cardiff": "Cardiff",
    "cardiff city": "Cardiff",
    "chelsea": "Chelsea",
    "chelsea fc": "Chelsea",
    "coventry": "Coventry",
    "coventry city": "Coventry",
    "crystal palace": "Crystal Palace",
    "crystal palace fc": "Crystal Palace",
    "everton": "Everton",
    "everton fc": "Everton",
    "fulham": "Fulham",
    "fulham fc": "Fulham",
    "huddersfield": "Huddersfield",
    "huddersfield town": "Huddersfield",
    "ipswich": "Ipswich",
    "ipswich town": "Ipswich",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leicester": "Leicester",
    "leicester city": "Leicester",
    "liverpool": "Liverpool",
    "liverpool fc": "Liverpool",
    "luton": "Luton",
    "luton town": "Luton",
    "man city": "Manchester City",
    "manchester city": "Manchester City",
    "manchester city fc": "Manchester City",
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester united": "Manchester United",
    "manchester united fc": "Manchester United",
    "middlesbrough": "Middlesbrough",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "newcastle utd": "Newcastle United",
    "norwich": "Norwich",
    "norwich city": "Norwich",
    "nott'm forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "sheffield united": "Sheffield United",
    "sheffield utd": "Sheffield United",
    "southampton": "Southampton",
    "southampton fc": "Southampton",
    "sunderland": "Sunderland",
    "sunderland afc": "Sunderland",
    "tottenham": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "spurs": "Tottenham",
    "watford": "Watford",
    "watford fc": "Watford",
    "west brom": "West Bromwich Albion",
    "west bromwich albion": "West Bromwich Albion",
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",

    # --- Bundesliga ---
    "arminia bielefeld": "Arminia Bielefeld",
    "bielefeld": "Arminia Bielefeld",
    "augsburg": "Augsburg",
    "fc augsburg": "Augsburg",
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    "bayer 04 leverkusen": "Bayer Leverkusen",
    "bayern munich": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "fc bayern munich": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "bochum": "Bochum",
    "vfl bochum": "Bochum",
    "vfl bochum 1848": "Bochum",
    "borussia dortmund": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "borussia m gladbach": "Borussia M.Gladbach",
    "borussia monchengladbach": "Borussia M.Gladbach",
    "monchengladbach": "Borussia M.Gladbach",
    "m'gladbach": "Borussia M.Gladbach",
    "m gladbach": "Borussia M.Gladbach",
    "gladbach": "Borussia M.Gladbach",
    "darmstadt": "Darmstadt",
    "sv darmstadt 98": "Darmstadt",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "ein frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    "fc cologne": "FC Cologne",
    "fc koln": "FC Cologne",
    "1 fc koln": "FC Cologne",
    "koln": "FC Cologne",
    "cologne": "FC Cologne",
    "fc heidenheim": "FC Heidenheim",
    "heidenheim": "FC Heidenheim",
    "1 fc heidenheim": "FC Heidenheim",
    "1 fc heidenheim 1846": "FC Heidenheim",
    "fortuna dusseldorf": "Fortuna Duesseldorf",
    "fortuna duesseldorf": "Fortuna Duesseldorf",
    "dusseldorf": "Fortuna Duesseldorf",
    "freiburg": "Freiburg",
    "sc freiburg": "Freiburg",
    "greuther furth": "Greuther Fuerth",
    "greuther fuerth": "Greuther Fuerth",
    "spvgg greuther furth": "Greuther Fuerth",
    "hamburg": "Hamburger SV",
    "hamburger sv": "Hamburger SV",
    "hannover": "Hannover 96",
    "hannover 96": "Hannover 96",
    "hertha": "Hertha Berlin",
    "hertha berlin": "Hertha Berlin",
    "hertha bsc": "Hertha Berlin",
    "hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim",
    "tsg 1899 hoffenheim": "Hoffenheim",
    "holstein kiel": "Holstein Kiel",
    "kiel": "Holstein Kiel",
    "mainz": "Mainz 05",
    "mainz 05": "Mainz 05",
    "1 fsv mainz 05": "Mainz 05",
    "fsv mainz 05": "Mainz 05",
    "nurnberg": "Nuernberg",
    "nuernberg": "Nuernberg",
    "1 fc nurnberg": "Nuernberg",
    "paderborn": "Paderborn",
    "sc paderborn 07": "Paderborn",
    "rasenballsport leipzig": "RasenBallsport Leipzig",
    "rb leipzig": "RasenBallsport Leipzig",
    "leipzig": "RasenBallsport Leipzig",
    "schalke": "Schalke 04",
    "schalke 04": "Schalke 04",
    "fc schalke 04": "Schalke 04",
    "st pauli": "St. Pauli",
    "fc st pauli": "St. Pauli",
    "stuttgart": "VfB Stuttgart",
    "vfb stuttgart": "VfB Stuttgart",
    "union berlin": "Union Berlin",
    "1 fc union berlin": "Union Berlin",
    "fc union berlin": "Union Berlin",
    "werder bremen": "Werder Bremen",
    "sv werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",
    "wolfsburg": "Wolfsburg",
    "vfl wolfsburg": "Wolfsburg",

    # --- Serie A ---
    "ac milan": "AC Milan",
    "milan": "AC Milan",
    "atalanta": "Atalanta",
    "atalanta bc": "Atalanta",
    "benevento": "Benevento",
    "bologna": "Bologna",
    "bologna fc": "Bologna",
    "bologna fc 1909": "Bologna",
    "brescia": "Brescia",
    "cagliari": "Cagliari",
    "cagliari calcio": "Cagliari",
    "chievo": "Chievo",
    "chievo verona": "Chievo",
    "como": "Como",
    "como 1907": "Como",
    "cremonese": "Cremonese",
    "us cremonese": "Cremonese",
    "crotone": "Crotone",
    "empoli": "Empoli",
    "empoli fc": "Empoli",
    "fiorentina": "Fiorentina",
    "acf fiorentina": "Fiorentina",
    "frosinone": "Frosinone",
    "frosinone calcio": "Frosinone",
    "genoa": "Genoa",
    "genoa cfc": "Genoa",
    "hellas verona": "Verona",
    "verona": "Verona",
    "inter": "Inter",
    "inter milan": "Inter",
    "internazionale": "Inter",
    "fc internazionale milano": "Inter",
    "juventus": "Juventus",
    "juventus fc": "Juventus",
    "lazio": "Lazio",
    "ss lazio": "Lazio",
    "lecce": "Lecce",
    "us lecce": "Lecce",
    "monza": "Monza",
    "ac monza": "Monza",
    "napoli": "Napoli",
    "ssc napoli": "Napoli",
    "palermo": "Palermo",
    "parma": "Parma Calcio 1913",
    "parma calcio 1913": "Parma Calcio 1913",
    "pisa": "Pisa",
    "pisa sc": "Pisa",
    "roma": "Roma",
    "as roma": "Roma",
    "salernitana": "Salernitana",
    "us salernitana 1919": "Salernitana",
    "sampdoria": "Sampdoria",
    "uc sampdoria": "Sampdoria",
    "sassuolo": "Sassuolo",
    "us sassuolo": "Sassuolo",
    "spal": "SPAL 2013",
    "spal 2013": "SPAL 2013",
    "spezia": "Spezia",
    "spezia calcio": "Spezia",
    "torino": "Torino",
    "torino fc": "Torino",
    "udinese": "Udinese",
    "udinese calcio": "Udinese",
    "venezia": "Venezia",
    "venezia fc": "Venezia",
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
    norm = norm.replace("&", " and ")
    norm = re.sub(r"[-_./]+", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    if norm in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[norm]
    no_apos = norm.replace("'", "")
    if no_apos in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[no_apos]
    hyph = norm.replace(" ", "-")
    if hyph in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[hyph]
    stripped = re.sub(
        r"^(1 |fc|cf|rc|rcd|cd|ud|sd|ca|as|ac|aj|ogc|losc|vfl|vfb|tsg|sc|sv|ss|us|uc|ssc|acf|afc|olympique de|olympique|stade de|stade|club|deportivo)\s+",
        "",
        no_apos,
    )
    stripped = re.sub(
        r"\s+(fc|cf|ud|sd|cd|sc|sco|hsc|osc|ac|afc|bc|cfc|sad|calcio|1901|1907|1909|1913|1919|1899|1848|1846|29|63|04|05|07|96|98)$",
        "",
        stripped,
    ).strip()
    if stripped in _CANONICAL_ALIASES:
        return _CANONICAL_ALIASES[stripped]
    return raw


def _current_season_code(ref_date: Optional[datetime] = None) -> str:
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
    code = ESPN_LEAGUE_CODES.get(league)
    if not code:
        return []
    now = ref_date or datetime.now(timezone.utc)
    day_strs = [
        (now + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(-days_back, days_forward + 1)
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; autoloop/3.0)"})

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
    season_code: Optional[str] = None,
    ref_date: Optional[datetime] = None,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """
    Fetch completed results + real odds (+ HxG / AxG when present) from
    football-data.co.uk for a given season_code (defaults to current season).
    """
    fname = FD_UK_LEAGUE_FILES.get(league)
    if not fname:
        return []
    scode = season_code or _current_season_code(ref_date)
    url = f"https://www.football-data.co.uk/mmz4281/{scode}/{fname}"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text.strip():
            return []
        df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
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
            "season_code": scode,
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
            # Opening / market odds
            "PSH": _safe_float(row.get("PSH")),
            "PSD": _safe_float(row.get("PSD")),
            "PSA": _safe_float(row.get("PSA")),
            "MaxH": _safe_float(row.get("MaxH") if pd.notna(row.get("MaxH")) else row.get("BbMxH")),
            "MaxD": _safe_float(row.get("MaxD") if pd.notna(row.get("MaxD")) else row.get("BbMxD")),
            "MaxA": _safe_float(row.get("MaxA") if pd.notna(row.get("MaxA")) else row.get("BbMxA")),
            "AvgH": _safe_float(row.get("AvgH") if pd.notna(row.get("AvgH")) else row.get("BbAvH")),
            "AvgD": _safe_float(row.get("AvgD") if pd.notna(row.get("AvgD")) else row.get("BbAvD")),
            "AvgA": _safe_float(row.get("AvgA") if pd.notna(row.get("AvgA")) else row.get("BbAvA")),
            "P_over25": _safe_float(row.get("P>2.5")),
            "P_under25": _safe_float(row.get("P<2.5")),
            "Max_over25": _safe_float(row.get("Max>2.5") if pd.notna(row.get("Max>2.5")) else row.get("BbMx>2.5")),
            "Max_under25": _safe_float(row.get("Max<2.5") if pd.notna(row.get("Max<2.5")) else row.get("BbMx<2.5")),
            "Avg_over25": _safe_float(row.get("Avg>2.5") if pd.notna(row.get("Avg>2.5")) else row.get("BbAv>2.5")),
            "Avg_under25": _safe_float(row.get("Avg<2.5") if pd.notna(row.get("Avg<2.5")) else row.get("BbAv<2.5")),
            # Pinnacle closing odds (true CLV benchmark)
            "PSCH": _safe_float(row.get("PSCH")),
            "PSCD": _safe_float(row.get("PSCD")),
            "PSCA": _safe_float(row.get("PSCA")),
            "PC_over25": _safe_float(row.get("PC>2.5")),
            "PC_under25": _safe_float(row.get("PC<2.5")),
            "source": "football-data.co.uk",
        })
    return sorted(rows, key=lambda x: x["date"])


def fetch_football_data_org_matches(
    league: str,
    api_key: Optional[str] = None,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
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
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for rec in records:
        key = (rec["league"], rec["date_key"], rec["home"], rec["away"])
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(rec)
            continue
        if rec.get("completed") and not existing.get("completed"):
            existing["completed"] = True
            existing["goals_home"] = rec.get("goals_home")
            existing["goals_away"] = rec.get("goals_away")
            existing["state"] = "post"
        for field in (
            "xG_home", "xG_away", "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA",
            "AvgH", "AvgD", "AvgA", "P_over25", "P_under25", "Max_over25", "Max_under25",
            "Avg_over25", "Avg_under25", "PSCH", "PSCD", "PSCA", "PC_over25", "PC_under25",
        ):
            if existing.get(field) is None and rec.get(field) is not None:
                existing[field] = rec[field]
    return sorted(merged.values(), key=lambda x: x["date"])


def fetch_all_live_matches(
    leagues: Tuple[str, ...] = BIG5_LEAGUES,
    days_back: int = 14,
    days_forward: int = 10,
    use_cache_hours: float = 3.0,
    cache_path: Path = LIVE_CACHE_FILE,
) -> List[Dict[str, Any]]:
    if use_cache_hours > 0 and cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_h < use_cache_hours:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list) and cached:
                    cached_leagues = {m.get("league") for m in cached}
                    if set(leagues).issubset(cached_leagues):
                        return [m for m in cached if m.get("league") in leagues]
            except Exception:
                pass

    all_records: List[Dict[str, Any]] = []
    for lg in leagues:
        all_records.extend(fetch_football_data_org_matches(lg))
        all_records.extend(fetch_fd_uk_season_results(lg))
        all_records.extend(fetch_espn_matches(lg, days_back=days_back, days_forward=days_forward))
        all_records.extend(fetch_odds_api_scores(lg))

    merged = _merge_matches(all_records)
    if merged:
        try:
            cache_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return merged

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return [m for m in cached if m.get("league") in leagues]
        except Exception:
            pass
    return []


def fetch_upcoming_fixtures(
    leagues: Tuple[str, ...] = BIG5_LEAGUES,
    window_days: int = 8,
    now: Optional[pd.Timestamp] = None,
    live_records: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
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
    big5_csv: Path = BIG5_HISTORY_CSV,
) -> Dict[str, int]:
    """
    Append newly completed Big-5 matches into local historical CSVs (deduplicated).
    Updates `data/historical_big5_odds.csv` (all 5 leagues) as well as legacy La Liga/Ligue 1 CSVs.
    """
    records = live_records if live_records is not None else fetch_all_live_matches()
    completed = [
        m for m in records
        if m.get("completed") and m.get("goals_home") is not None and m.get("goals_away") is not None
    ]
    added_counts: Dict[str, int] = {lg: 0 for lg in BIG5_LEAGUES}

    # --- 1. Sync unified Big-5 historical CSV ---
    if completed and big5_csv.exists():
        bdf = pd.read_csv(big5_csv)
        existing_keys = set()
        for _, r in bdf.iterrows():
            lg = str(r.get("league", ""))
            existing_keys.add((
                lg,
                str(r.get("date", ""))[:10],
                canonical_team_name(r.get("home", ""), lg),
                canonical_team_name(r.get("away", ""), lg),
            ))
        new_rows = []
        for m in completed:
            lg = str(m.get("league", ""))
            if lg not in BIG5_LEAGUES:
                continue
            key = (lg, m["date_key"], m["home"], m["away"])
            if key in existing_keys:
                continue
            existing_keys.add(key)
            dt = pd.to_datetime(m["date"])
            sy = dt.year if dt.month >= 7 else dt.year - 1
            row_d = {
                "league": lg,
                "season": sy,
                "date": m["date"],
                "home": m["home"],
                "away": m["away"],
                "goals_home": int(m["goals_home"]),
                "goals_away": int(m["goals_away"]),
                "xG_home": m.get("xG_home"),
                "xG_away": m.get("xG_away"),
            }
            for col in (
                "PSH", "PSD", "PSA", "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
                "P_over25", "P_under25", "Max_over25", "Max_under25", "Avg_over25", "Avg_under25",
                "PSCH", "PSCD", "PSCA", "PC_over25", "PC_under25",
            ):
                row_d[col] = m.get(col)
            new_rows.append(row_d)
            added_counts[lg] = added_counts.get(lg, 0) + 1
        if new_rows:
            out_bdf = pd.concat([bdf, pd.DataFrame(new_rows)], ignore_index=True)
            out_bdf.sort_values(["date", "league"]).to_csv(big5_csv, index=False)

    # --- 2. Keep legacy La Liga CSV in sync ---
    la_done = [m for m in completed if m.get("league") == "La Liga"]
    if la_done and la_csv.exists():
        df = pd.read_csv(la_csv)
        existing_keys = {
            (str(r.get("date", ""))[:10], canonical_team_name(r.get("home", ""), "La Liga"), canonical_team_name(r.get("away", ""), "La Liga"))
            for _, r in df.iterrows()
        }
        new_rows = []
        for m in la_done:
            key = (m["date_key"], m["home"], m["away"])
            if key in existing_keys:
                continue
            existing_keys.add(key)
            dt = pd.to_datetime(m["date"])
            sy = dt.year if dt.month >= 7 else dt.year - 1
            gh, ga = int(m["goals_home"]), int(m["goals_away"])
            new_rows.append({
                "season": sy,
                "season_label": f"{sy}/{sy + 1}",
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
        if new_rows:
            pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True).to_csv(la_csv, index=False)

    # --- 3. Keep legacy Ligue 1 live CSV in sync ---
    l1_done = [m for m in completed if m.get("league") == "Ligue 1"]
    if l1_done:
        cols = ["date", "home", "away", "goals_home", "goals_away", "xG_home", "xG_away", "source"]
        df1 = pd.read_csv(ligue1_csv) if ligue1_csv.exists() else pd.DataFrame(columns=cols)
        existing_keys = {
            (str(r.get("date", ""))[:10], canonical_team_name(r.get("home", ""), "Ligue 1"), canonical_team_name(r.get("away", ""), "Ligue 1"))
            for _, r in df1.iterrows()
        }
        new_rows = []
        for m in l1_done:
            key = (m["date_key"], m["home"], m["away"])
            if key in existing_keys:
                continue
            existing_keys.add(key)
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
        if new_rows:
            pd.concat([df1, pd.DataFrame(new_rows)], ignore_index=True).sort_values("date").to_csv(ligue1_csv, index=False)

    return added_counts
