"""
build_big5_dataset.py — Builds `data/historical_big5_odds.csv` for all Big-5 European Leagues.

Combines:
  1. Real historical bookmaker & Pinnacle closing odds (2018-2026, seasons 1819..2627)
     from football-data.co.uk for SP1 (La Liga), F1 (Ligue 1), E0 (EPL),
     D1 (Bundesliga), and I1 (Serie A).
  2. Real match-level xG (xG_home, xG_away) from Understat local archives
     (la_liga_matches_2018_2026.csv, raw_Ligue1_8seasons.json,
      raw_EPL_8seasons.json, raw_Bundesliga_8seasons.json, raw_Serie_A_8seasons.json)
     and football-data.co.uk 26/27 HxG/AxG columns.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from live_data import (
    BIG5_HISTORY_CSV,
    BIG5_LEAGUES,
    REPO_DIR,
    canonical_team_name,
    fetch_fd_uk_season_results,
)

SEASONS = ("1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627")

UNDERSTAT_JSON_CANDIDATES: Dict[str, List[Path]] = {
    "Ligue 1": [REPO_DIR / "raw_Ligue1_8seasons.json", Path("C:/Users/User/raw_Ligue1_8seasons.json")],
    "EPL": [REPO_DIR / "raw_EPL_8seasons.json", Path("C:/Users/User/raw_EPL_8seasons.json")],
    "Bundesliga": [REPO_DIR / "raw_Bundesliga_8seasons.json", Path("C:/Users/User/raw_Bundesliga_8seasons.json")],
    "Serie A": [REPO_DIR / "raw_Serie_A_8seasons.json", Path("C:/Users/User/raw_Serie_A_8seasons.json")],
}


def _load_xg_index() -> Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, float, float]]]:
    """Return index (league, home, away) -> [(date, xg_h, xg_a)] from Understat archives."""
    idx: Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, float, float]]] = {}

    # 1. La Liga CSV
    la_csv = REPO_DIR / "la_liga_matches_2018_2026.csv"
    if la_csv.exists():
        df = pd.read_csv(la_csv, parse_dates=["date"])
        df = df.dropna(subset=["date", "xG_home", "xG_away"])
        for _, r in df.iterrows():
            h = canonical_team_name(r["home"], "La Liga")
            a = canonical_team_name(r["away"], "La Liga")
            idx.setdefault(("La Liga", h, a), []).append(
                (r["date"].normalize(), float(r["xG_home"]), float(r["xG_away"]))
            )

    # 2. Other 4 leagues from Understat 8-season JSONs
    for league, paths in UNDERSTAT_JSON_CANDIDATES.items():
        found = next((p for p in paths if p.exists()), None)
        if not found:
            print(f"[WARN] No Understat JSON found for {league}")
            continue
        raw = json.loads(found.read_text(encoding="utf-8"))
        for _, sdata in raw.items():
            if not isinstance(sdata, dict):
                continue
            for d in sdata.get("dates", []):
                if not d.get("isResult") or not d.get("xG"):
                    continue
                xh = d["xG"].get("h")
                xa = d["xG"].get("a")
                if xh is None or xa is None:
                    continue
                dt = pd.to_datetime(d.get("datetime"), errors="coerce")
                if pd.isna(dt):
                    continue
                h = canonical_team_name((d.get("h") or {}).get("title", ""), league)
                a = canonical_team_name((d.get("a") or {}).get("title", ""), league)
                idx.setdefault((league, h, a), []).append((dt.normalize(), float(xh), float(xa)))

    return idx


def _lookup_xg(
    xg_index: Dict[Tuple[str, str, str], List[Tuple[pd.Timestamp, float, float]]],
    league: str,
    home: str,
    away: str,
    dt_str: str,
) -> Tuple[Any, Any]:
    target = pd.to_datetime(dt_str, errors="coerce")
    if pd.isna(target):
        return None, None
    tday = target.normalize()
    best = None
    best_diff = 4
    for d, xh, xa in xg_index.get((league, home, away), []):
        diff = abs((d - tday).days)
        if diff < best_diff:
            best_diff = diff
            best = (round(xh, 4), round(xa, 4))
    return best if best is not None else (None, None)


def build_dataset(out_csv: Path = BIG5_HISTORY_CSV) -> pd.DataFrame:
    print("[BUILD] Loading local Understat xG archives...")
    xg_index = _load_xg_index()
    print(f"[BUILD] Loaded xG match pairs: {len(xg_index)}")

    tasks = [(lg, sc) for lg in BIG5_LEAGUES for sc in SEASONS]
    all_rows: List[Dict[str, Any]] = []

    print(f"[BUILD] Downloading {len(tasks)} season CSVs from football-data.co.uk...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch_fd_uk_season_results, lg, sc): (lg, sc) for lg, sc in tasks}
        for f in as_completed(futs):
            lg, sc = futs[f]
            recs = f.result()
            all_rows.extend(recs)

    print(f"[BUILD] Fetched {len(all_rows)} historical matches with real bookmaker odds.")

    merged_rows: List[Dict[str, Any]] = []
    seen = set()
    for r in sorted(all_rows, key=lambda x: (x["date"], x["league"])):
        lg = r["league"]
        h = r["home"]
        a = r["away"]
        dk = r["date_key"]
        key = (lg, dk, h, a)
        if key in seen:
            continue
        seen.add(key)

        xg_h = r.get("xG_home")
        xg_a = r.get("xG_away")
        if xg_h is None or xg_a is None:
            uxh, uxa = _lookup_xg(xg_index, lg, h, a, r["date"])
            if uxh is not None and uxa is not None:
                xg_h, xg_a = uxh, uxa

        dt = pd.to_datetime(r["date"])
        season_yr = dt.year if dt.month >= 7 else dt.year - 1
        merged_rows.append({
            "league": lg,
            "season": season_yr,
            "date": r["date"],
            "home": h,
            "away": a,
            "goals_home": int(r["goals_home"]),
            "goals_away": int(r["goals_away"]),
            "xG_home": xg_h,
            "xG_away": xg_a,
            "PSH": r.get("PSH"),
            "PSD": r.get("PSD"),
            "PSA": r.get("PSA"),
            "MaxH": r.get("MaxH"),
            "MaxD": r.get("MaxD"),
            "MaxA": r.get("MaxA"),
            "AvgH": r.get("AvgH"),
            "AvgD": r.get("AvgD"),
            "AvgA": r.get("AvgA"),
            "P_over25": r.get("P_over25"),
            "P_under25": r.get("P_under25"),
            "Max_over25": r.get("Max_over25"),
            "Max_under25": r.get("Max_under25"),
            "Avg_over25": r.get("Avg_over25"),
            "Avg_under25": r.get("Avg_under25"),
            "PSCH": r.get("PSCH"),
            "PSCD": r.get("PSCD"),
            "PSCA": r.get("PSCA"),
            "PC_over25": r.get("PC_over25"),
            "PC_under25": r.get("PC_under25"),
        })

    df = pd.DataFrame(merged_rows).sort_values(["date", "league"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n[BUILD] Saved {len(df)} rows to {out_csv}")
    for lg, g in df.groupby("league"):
        xg_pct = g["xG_home"].notna().mean() * 100.0
        ps_pct = g["PSH"].notna().mean() * 100.0
        psc_pct = g["PSCH"].notna().mean() * 100.0
        ou_pct = g["Max_over25"].notna().mean() * 100.0
        print(
            f"  {lg:12s}: n={len(g):4d} | xG={xg_pct:5.1f}% | Pinnacle_1X2={ps_pct:5.1f}% "
            f"| Pinnacle_Close={psc_pct:5.1f}% | O/U2.5={ou_pct:5.1f}% | {str(g['date'].min())[:10]}..{str(g['date'].max())[:10]}"
        )
    return df


if __name__ == "__main__":
    build_dataset()
