#!/usr/bin/env python3
"""
Professional La Liga 2025-2026 Team Statistics Fetcher
Fetches team stats from FBref (free, reliable source) and exports to CSV.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys
from typing import Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
FBREF_BASE_URL = "https://fbref.com"
LALIGA_2025_2026_URL = "https://fbref.com/en/comps/12/2025-2026/2025-2026-La-Liga-Stats"
OUTPUT_CSV = "laliga_teams_stats_2026.csv"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2


def fetch_page(url: str, retries: int = MAX_RETRIES) -> Optional[requests.Response]:
    """Fetch a page with retry logic and proper headers."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"All retries exhausted for {url}")
                return None
    return None


def parse_standard_stats_table(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse the standard team stats table (league table)."""
    # Find the main league table
    table = soup.find('table', {'id': 'results2025-2026121_overall'})
    if not table:
        # Try alternative table IDs
        for table_id in ['results2025-2026121_overall', 'stats_squads_standard_for', 'results2025-2026121']:
            table = soup.find('table', {'id': table_id})
            if table:
                break
    
    if not table:
        raise ValueError("Could not find standard stats table")
    
    # Parse with pandas
    df = pd.read_html(str(table))[0]
    
    # Clean column names (handle multi-level headers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if str(c) != '']).strip() for col in df.columns.values]
    df.columns = [str(c).strip() for c in df.columns]
    
    return df


def parse_advanced_stats_table(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse advanced stats table (xG, possession, passing, shots)."""
    # Look for shooting, possession, or misc tables
    advanced_tables = {}
    
    table_configs = {
        'shooting': 'stats_squads_shooting_for',
        'possession': 'stats_squads_possession_for',
        'passing': 'stats_squads_passing_for',
        'misc': 'stats_squads_misc_for',
    }
    
    for name, table_id in table_configs.items():
        table = soup.find('table', {'id': table_id})
        if table:
            try:
                df = pd.read_html(str(table))[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join([str(c) for c in col if str(c) != '']).strip() for col in df.columns.values]
                df.columns = [str(c).strip() for c in df.columns]
                advanced_tables[name] = df
                logger.info(f"Found {name} table with {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to parse {name} table: {e}")
    
    return advanced_tables


def extract_team_name(row) -> str:
    """Extract clean team name from a row."""
    # Try common column names for team
    for col in ['Squad', 'Team', 'Club', 'team', 'squad']:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
    return "Unknown"


def clean_numeric(val) -> float:
    """Clean and convert to numeric."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    # Remove commas, percentage signs, etc.
    cleaned = str(val).replace(',', '').replace('%', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def merge_team_data(standard_df: pd.DataFrame, advanced_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge standard and advanced stats by team name."""
    results = []
    
    for idx, row in standard_df.iterrows():
        team = extract_team_name(row)
        if team == "Unknown":
            continue
        
        # Basic stats
        team_data = {
            'Rank': clean_numeric(row.get('Rk', row.get('Rank', idx + 1))),
            'Team': team,
            'MP': clean_numeric(row.get('MP', row.get('Matches Played', 0))),
            'W': clean_numeric(row.get('W', row.get('Wins', 0))),
            'D': clean_numeric(row.get('D', row.get('Draws', 0))),
            'L': clean_numeric(row.get('L', row.get('Losses', 0))),
            'GF': clean_numeric(row.get('GF', row.get('Goals For', 0))),
            'GA': clean_numeric(row.get('GA', row.get('Goals Against', 0))),
            'GD': clean_numeric(row.get('GD', row.get('Goal Difference', 0))),
            'Pts': clean_numeric(row.get('Pts', row.get('Points', 0))),
        }
        
        # Advanced stats - find matching team in each advanced table
        # xG and xGA from shooting table
        shooting_df = advanced_tables.get('shooting', pd.DataFrame())
        if not shooting_df.empty:
            shooting_row = shooting_df[shooting_df.apply(lambda r: extract_team_name(r) == team, axis=1)]
            if not shooting_row.empty:
                sr = shooting_row.iloc[0]
                team_data['xG'] = clean_numeric(sr.get('xG', sr.get('Expected Goals', 0)))
                team_data['xGA'] = clean_numeric(sr.get('xGA', sr.get('Expected Goals Against', 0)))
                team_data['Shots_on_Target'] = clean_numeric(sr.get('SoT', sr.get('Shots on Target', 0)))
            else:
                team_data['xG'] = 0.0
                team_data['xGA'] = 0.0
                team_data['Shots_on_Target'] = 0.0
        else:
            team_data['xG'] = 0.0
            team_data['xGA'] = 0.0
            team_data['Shots_on_Target'] = 0.0
        
        # Possession from possession table
        possession_df = advanced_tables.get('possession', pd.DataFrame())
        if not possession_df.empty:
            poss_row = possession_df[possession_df.apply(lambda r: extract_team_name(r) == team, axis=1)]
            if not poss_row.empty:
                pr = poss_row.iloc[0]
                team_data['Possession'] = clean_numeric(pr.get('Poss', pr.get('Possession', 0)))
            else:
                team_data['Possession'] = 0.0
        else:
            team_data['Possession'] = 0.0
        
        # Passing Accuracy from passing table
        passing_df = advanced_tables.get('passing', pd.DataFrame())
        if not passing_df.empty:
            pass_row = passing_df[passing_df.apply(lambda r: extract_team_name(r) == team, axis=1)]
            if not pass_row.empty:
                par = pass_row.iloc[0]
                team_data['Passing_Accuracy'] = clean_numeric(par.get('Cmp%', par.get('Pass Completion %', 0)))
            else:
                team_data['Passing_Accuracy'] = 0.0
        else:
            team_data['Passing_Accuracy'] = 0.0
        
        results.append(team_data)
    
    return pd.DataFrame(results)


def fetch_laliga_stats() -> Optional[pd.DataFrame]:
    """Main function to fetch and process La Liga stats."""
    logger.info(f"Fetching La Liga 2025-2026 stats from {LALIGA_2025_2026_URL}")
    
    # Fetch main page
    response = fetch_page(LALIGA_2025_2026_URL)
    if not response:
        logger.error("Failed to fetch main page")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Parse standard stats
    try:
        standard_df = parse_standard_stats_table(soup)
        logger.info(f"Parsed standard stats: {len(standard_df)} teams")
    except Exception as e:
        logger.error(f"Failed to parse standard stats: {e}")
        return None
    
    # Parse advanced stats
    advanced_tables = parse_advanced_stats_table(soup)
    logger.info(f"Parsed {len(advanced_tables)} advanced stat tables")
    
    # Merge data
    merged_df = merge_team_data(standard_df, advanced_tables)
    logger.info(f"Merged data for {len(merged_df)} teams")
    
    # Ensure all required columns exist
    required_columns = [
        'Rank', 'Team', 'MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts',
        'xG', 'xGA', 'Possession', 'Passing_Accuracy', 'Shots_on_Target'
    ]
    
    for col in required_columns:
        if col not in merged_df.columns:
            merged_df[col] = 0.0
    
    # Reorder columns
    merged_df = merged_df[required_columns]
    
    # Sort by Rank
    merged_df = merged_df.sort_values('Rank').reset_index(drop=True)
    
    return merged_df


def save_to_csv(df: pd.DataFrame, filepath: str) -> bool:
    """Save DataFrame to CSV with UTF-8 BOM encoding."""
    try:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Successfully saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return False


def main():
    """Main execution function."""
    print("=" * 60)
    print("La Liga 2025-2026 Team Statistics Fetcher")
    print("=" * 60)
    
    try:
        # Fetch data
        df = fetch_laliga_stats()
        
        if df is None or df.empty:
            print("❌ ERROR: No data fetched. Check network connection or source availability.")
            sys.exit(1)
        
        # Save to CSV
        if not save_to_csv(df, OUTPUT_CSV):
            print("❌ ERROR: Failed to save CSV file.")
            sys.exit(1)
        
        # Print summary
        print(f"\n✅ SUCCESS: Data fetched and saved to '{OUTPUT_CSV}'")
        print(f"📊 Total teams: {len(df)}")
        print(f"📋 Columns: {', '.join(df.columns.tolist())}")
        
        # Display first 5 rows
        print("\n" + "=" * 60)
        print("FIRST 5 ROWS (Preview):")
        print("=" * 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', 30)
        print(df.head().to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✅ Process completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()