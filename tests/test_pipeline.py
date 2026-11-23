"""
tests/test_pipeline.py — Automated Verification Suite for football-betting-autoloop (v2).

Covers:
  (a) Unified selection_side consistency between pricing and settlement for
      `Le Mans vs Lyon` and `Lyon vs Paris FC` (no string-matching bug).
  (b) Automatic settlement settles ALL completed past matches (0 completed matches left PENDING)
      and `_LOCAL_FILES` is a proper list (no str char-iteration bug).
  (c) Zero look-ahead leakage (`as_of`) for BOTH `mu_h`/`mu_a` league baselines AND
      club strengths, active Time Decay (xi=0.003), and active xG blending (0.7*xG + 0.3*Goals).
  (d) Broker Mode `min_odds` includes the exact +6% safety margin (`round(1.06 / p_cal, 2)`),
      Platt calibration shrinks overconfident Ligue 1 AH -0.75 probabilities, and
      Telegram report suppresses fake 1.90 ROI when verified_settled_count == 0.
  (e) Canonical team-name mapping + best-odds (`max`) selection in fetch_odds.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import autoloop_master as am
import fetch_odds as fo
import live_data as ld
import settle_ledger_auto as sla
from team_strength import DEFAULT_XI, TeamStrengthModel, blend_xg_goals, time_decay_weight


class TestSelectionSideConsistency(unittest.TestCase):
    """(a) Pricing and settlement MUST agree on selection_side for Le Mans vs Lyon & Lyon vs Paris FC."""

    def test_le_mans_vs_lyon_and_lyon_vs_paris_fc(self):
        model = TeamStrengthModel()
        # 1. Pricing side and settlement side MUST match 100% for both fixtures
        for league, home, away, dt in [
            ("Ligue 1", "Le Mans", "Lyon", "2027-01-02"),
            ("Ligue 1", "Lyon", "Paris FC", "2027-05-29"),
        ]:
            lh, la = am.estimate_match_lambdas(home, away, league, team_model=model, fixture_date=dt)
            pricing_side = sla.determine_selection_side(home, away, league, fixture_date=dt, lh=lh, la=la, team_model=model)
            settle_side = sla.determine_selection_side(home, away, league, fixture_date=dt, team_model=model)
            self.assertEqual(
                pricing_side,
                settle_side,
                f"Mismatch between pricing ({pricing_side}) and settlement ({settle_side}) for {home} vs {away}",
            )

        # 2. Verify numeric favorite selection does NOT get tricked by 'Paris' in 'Paris FC'
        #    and selects AWAY whenever away expected goals exceed home expected goals (lh < la).
        self.assertEqual(
            sla.determine_selection_side("Lyon", "Paris FC", "Ligue 1", lh=2.10, la=0.90),
            "HOME",
        )
        self.assertEqual(
            sla.determine_selection_side("Le Mans", "Lyon", "Ligue 1", lh=0.85, la=1.95),
            "AWAY",
        )

        # 3. Verify settle_row honors explicit selection_side and settles accurately
        fake_index = {
            ("Ligue 1", "Le Mans", "Lyon"): [(pd.Timestamp("2027-01-02"), 0, 2)],
            ("Ligue 1", "Lyon", "Paris FC"): [(pd.Timestamp("2027-05-29"), 2, 0)],
        }
        r1 = sla.settle_row(
            {
                "date": "2027-01-02",
                "league": "Ligue 1",
                "home": "Le Mans",
                "away": "Lyon",
                "market": "AH -0.75",
                "selection_side": "AWAY",
                "odds_source": "the-odds-api",
                "market_odds": 1.85,
                "stake": 0.02,
                "status": "PENDING",
            },
            cutoff=pd.Timestamp("2027-06-01"),
            result_index=fake_index,
        )
        self.assertEqual(r1["status"], "WIN")
        self.assertEqual(r1["selection_side"], "AWAY")

        r2 = sla.settle_row(
            {
                "date": "2027-05-29",
                "league": "Ligue 1",
                "home": "Lyon",
                "away": "Paris FC",
                "market": "AH -0.75",
                "selection_side": "HOME",
                "odds_source": "the-odds-api",
                "market_odds": 1.80,
                "stake": 0.02,
                "status": "PENDING",
            },
            cutoff=pd.Timestamp("2027-06-01"),
            result_index=fake_index,
        )
        self.assertEqual(r2["status"], "WIN")
        self.assertEqual(r2["selection_side"], "HOME")


class TestAutomaticSettlement(unittest.TestCase):
    """(b) Settlement engine resolves every completed past match and fixes _LOCAL_FILES."""

    def test_local_files_is_list_not_str(self):
        for league, paths in sla._LOCAL_FILES.items():
            self.assertIsInstance(paths, list, f"_LOCAL_FILES[{league}] must be a list, got {type(paths)}")
            self.assertTrue(all(isinstance(p, str) and len(p) > 3 for p in paths))

    def test_all_completed_matches_settled_none_left_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ledger_path = tmp_dir / "ledger.csv"
            summary_path = tmp_dir / "summary.json"

            df = pd.DataFrame([
                {
                    "date": "2026-11-10 15:00:00",
                    "league": "La Liga",
                    "home": "Barcelona",
                    "away": "Sevilla",
                    "market": "Over 2.5",
                    "selection_side": "",
                    "odds_source": "",
                    "market_odds": np.nan,
                    "stake": 0.02,
                    "status": "PENDING",
                },
                {
                    "date": "2026-11-11 18:00:00",
                    "league": "La Liga",
                    "home": "Real Madrid",
                    "away": "Getafe",
                    "market": "1X2 Home Win",
                    "selection_side": "HOME",
                    "odds_source": "the-odds-api",
                    "market_odds": 1.45,
                    "stake": 0.02,
                    "status": "PENDING",
                },
                {
                    "date": "2026-11-12 20:00:00",
                    "league": "Ligue 1",
                    "home": "Lyon",
                    "away": "Paris FC",
                    "market": "AH -0.75",
                    "selection_side": "HOME",
                    "odds_source": "",
                    "market_odds": np.nan,
                    "stake": 0.02,
                    "status": "PENDING",
                },
                {
                    "date": "2026-12-20 15:00:00",
                    "league": "La Liga",
                    "home": "Barcelona",
                    "away": "Valencia",
                    "market": "Over 2.5",
                    "selection_side": "",
                    "odds_source": "",
                    "market_odds": np.nan,
                    "stake": 0.02,
                    "status": "PENDING",
                },
            ])
            df.to_csv(ledger_path, index=False)

            live_records = [
                {
                    "league": "La Liga",
                    "date": "2026-11-10 15:00:00",
                    "date_key": "2026-11-10",
                    "home": "Barcelona",
                    "away": "Sevilla",
                    "completed": True,
                    "goals_home": 3,
                    "goals_away": 1,
                },
                {
                    "league": "La Liga",
                    "date": "2026-11-11 18:00:00",
                    "date_key": "2026-11-11",
                    "home": "Real Madrid",
                    "away": "Getafe",
                    "completed": True,
                    "goals_home": 2,
                    "goals_away": 0,
                },
                {
                    "league": "Ligue 1",
                    "date": "2026-11-12 20:00:00",
                    "date_key": "2026-11-12",
                    "home": "Lyon",
                    "away": "Paris FC",
                    "completed": True,
                    "goals_home": 1,
                    "goals_away": 0,
                },
            ]

            summary = sla.settle_ledger(
                ledger_path=ledger_path,
                summary_path=summary_path,
                cutoff=pd.Timestamp("2026-11-23"),
                live_records=live_records,
                use_live=False,
                sync_history=False,
            )

            settled_df = pd.read_csv(ledger_path)
            past_rows = settled_df[pd.to_datetime(settled_df["date"]) <= pd.Timestamp("2026-11-23")]
            self.assertEqual((past_rows["status"] == "PENDING").sum(), 0)
            self.assertEqual(summary["settled_now"], 3)
            self.assertEqual(summary["result_not_found"], 0)
            self.assertEqual(summary["pending_after"], 1)  # future 2026-12-20 stays PENDING
            self.assertEqual(summary["wins"], 2)
            self.assertEqual(summary["half_wins"], 1)
            self.assertEqual(summary["verified_settled_count"], 1)
            self.assertAlmostEqual(summary["verified_roi_percent"], 45.0, places=1)


class TestNoLeakageTimeDecayAndXG(unittest.TestCase):
    """(c) Strict causal `as_of` prevents future leakage in mu_h, mu_a & strength; Time Decay + xG active."""

    def test_no_future_leakage_in_baselines_or_strength(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_base = Path(tmp) / "la_base.csv"
            csv_plus_future = Path(tmp) / "la_future.csv"

            base_rows = []
            for i in range(12):
                dt = pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i)
                base_rows.append({
                    "date": dt.strftime("%Y-%m-%d 15:00:00"),
                    "isResult": True,
                    "home": "TeamA" if i % 2 == 0 else "TeamB",
                    "away": "TeamB" if i % 2 == 0 else "TeamA",
                    "goals_home": 2,
                    "goals_away": 1,
                    "xG_home": 1.8,
                    "xG_away": 0.9,
                })
            pd.DataFrame(base_rows).to_csv(csv_base, index=False)

            # Append extreme future matches AFTER as_of = 2024-05-01
            future_rows = list(base_rows)
            for i in range(10):
                dt = pd.Timestamp("2024-06-01") + pd.Timedelta(days=7 * i)
                future_rows.append({
                    "date": dt.strftime("%Y-%m-%d 15:00:00"),
                    "isResult": True,
                    "home": "TeamA",
                    "away": "TeamB",
                    "goals_home": 8,
                    "goals_away": 0,
                    "xG_home": 7.5,
                    "xG_away": 0.1,
                })
            pd.DataFrame(future_rows).to_csv(csv_plus_future, index=False)

            m_base = TeamStrengthModel(la_csv=str(csv_base), ligue_json="", ligue_live_csv="", xi=DEFAULT_XI)
            m_fut = TeamStrengthModel(la_csv=str(csv_plus_future), ligue_json="", ligue_live_csv="", xi=DEFAULT_XI)

            as_of = pd.Timestamp("2024-05-01")
            mu_h1, mu_a1, _ = m_base.league_baselines("La Liga", as_of=as_of)
            mu_h2, mu_a2, _ = m_fut.league_baselines("La Liga", as_of=as_of)
            self.assertAlmostEqual(mu_h1, mu_h2, places=6)
            self.assertAlmostEqual(mu_a1, mu_a2, places=6)

            s1 = m_base.strength("La Liga", "TeamA", as_of=as_of)
            s2 = m_fut.strength("La Liga", "TeamA", as_of=as_of)
            self.assertEqual(s1, s2)

    def test_time_decay_prioritizes_recent_matches(self):
        self.assertAlmostEqual(time_decay_weight(0, xi=0.003), 1.0, places=6)
        self.assertLess(time_decay_weight(365, xi=0.003), 0.35)

        with tempfile.TemporaryDirectory() as tmp:
            csv_p = Path(tmp) / "decay.csv"
            rows = []
            # 10 old low-scoring matches in 2020 (TeamA scored 0)
            for i in range(10):
                rows.append({
                    "date": (pd.Timestamp("2020-01-01") + pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d"),
                    "isResult": True,
                    "home": "TeamA",
                    "away": "TeamB",
                    "goals_home": 0,
                    "goals_away": 2,
                    "xG_home": 0.2,
                    "xG_away": 2.0,
                })
            # 10 recent high-scoring matches in 2026 (TeamA scored 4)
            for i in range(10):
                rows.append({
                    "date": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=7 * i)).strftime("%Y-%m-%d"),
                    "isResult": True,
                    "home": "TeamA",
                    "away": "TeamB",
                    "goals_home": 4,
                    "goals_away": 0,
                    "xG_home": 3.5,
                    "xG_away": 0.3,
                })
            pd.DataFrame(rows).to_csv(csv_p, index=False)

            m_decay = TeamStrengthModel(la_csv=str(csv_p), ligue_json="", ligue_live_csv="", xi=0.003)
            m_flat = TeamStrengthModel(la_csv=str(csv_p), ligue_json="", ligue_live_csv="", xi=0.0)
            as_of = pd.Timestamp("2026-04-01")
            att_decay, _ = m_decay.strength("La Liga", "TeamA", as_of=as_of)
            att_flat, _ = m_flat.strength("La Liga", "TeamA", as_of=as_of)
            # With time decay, TeamA's recent 4-goal surge dominates over 2020 slump
            self.assertGreater(att_decay, att_flat + 0.15)

    def test_xg_blending_active(self):
        # 0.7 * 2.0 + 0.3 * 0.0 = 1.4
        self.assertAlmostEqual(blend_xg_goals(0.0, 2.0, xg_weight=0.7), 1.4, places=6)
        self.assertAlmostEqual(blend_xg_goals(3.0, None, xg_weight=0.7), 3.0, places=6)


class TestBrokerMinOddsCalibrationAndOddsFetch(unittest.TestCase):
    """(d) +6% safety margin min_odds, Platt calibration, canonical mapping & best-odds (max)."""

    def test_broker_min_odds_has_6_percent_safety_margin(self):
        self.assertEqual(am.compute_broker_min_odds(0.60), round(1.06 / 0.60, 2))  # 1.77
        self.assertEqual(am.compute_broker_min_odds(0.70), round(1.06 / 0.70, 2))  # 1.51
        # At min_odds, EV must be >= +5% (accounting for 2-decimal rounding)
        for p in (0.58, 0.65, 0.75):
            mo = am.compute_broker_min_odds(p)
            self.assertGreater(am.compute_ev(p, mo), 0.05)

    def test_platt_calibration_reduces_ligue1_ah_overconfidence(self):
        p_raw = 0.70
        p_cal = am.calibrate_probability(p_raw, "Ligue 1", "AH -0.75")
        self.assertLess(p_cal, p_raw)
        # Uncalibrated market without Platt entry stays unchanged
        self.assertEqual(am.calibrate_probability(0.62, "Unknown League", "Over 2.5"), 0.62)

    def test_telegram_report_hides_fake_roi_when_unverified(self):
        summary = {
            "settled_now": 2,
            "total_settled_all_time": 10,
            "wins": 6,
            "half_wins": 1,
            "losses": 3,
            "hit_rate_percent": 65.0,
            "verified_settled_count": 0,
            "verified_profit_units": 0.0,
            "verified_roi_percent": None,
            "pending_after": 5,
        }
        report = am.format_telegram_report(summary, pd.DataFrame())
        self.assertIn("65.0%", report)
        self.assertIn("بانتظار صفقات مسوّاة بأسعار سوق حقيقية", report)
        self.assertNotIn("20.99%", report)

    def test_canonical_mapping_and_best_max_odds(self):
        self.assertEqual(ld.canonical_team_name("Atlético Madrid", "La Liga"), "Atletico Madrid")
        self.assertEqual(ld.canonical_team_name("Athletic Bilbao", "La Liga"), "Athletic Club")
        self.assertEqual(ld.canonical_team_name("Paris Saint-Germain", "Ligue 1"), "Paris Saint Germain")
        self.assertEqual(ld.canonical_team_name("Olympique de Marseille", "Ligue 1"), "Marseille")

        events = [{
            "home_team": "Atlético Madrid",
            "away_team": "Athletic Bilbao",
            "bookmakers": [
                {
                    "key": "b1",
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Atlético Madrid", "price": 1.80},
                            {"name": "Draw", "price": 3.40},
                            {"name": "Athletic Bilbao", "price": 4.20},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 2.5, "price": 1.95},
                            {"name": "Under", "point": 2.5, "price": 1.85},
                        ]},
                    ],
                },
                {
                    "key": "b2",
                    "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Atlético Madrid", "price": 1.92},
                            {"name": "Draw", "price": 3.60},
                            {"name": "Athletic Bilbao", "price": 4.60},
                        ]},
                        {"key": "totals", "outcomes": [
                            {"name": "Over", "point": 2.5, "price": 2.10},
                            {"name": "Under", "point": 2.5, "price": 1.80},
                        ]},
                    ],
                },
            ],
        }]
        parsed = fo.parse_odds_events(events, "La Liga")
        rec = parsed[("Atletico Madrid", "Athletic Club")]
        # Must select BEST (max) price, not worst (min)
        self.assertEqual(rec["home_win"], 1.92)
        self.assertEqual(rec["over25"], 2.10)
        self.assertEqual(fo.extract_market_price(rec, "1X2 Home Win", "HOME"), 1.92)

    def test_mle_fits_both_leagues_and_connects_to_lambdas(self):
        model = TeamStrengthModel()
        m_la = am.fit_league_mle_dynamic("La Liga", team_model=model, use_cache=False)
        m_l1 = am.fit_league_mle_dynamic("Ligue 1", team_model=model, use_cache=False)
        self.assertEqual(m_la["status"], "fitted")
        self.assertEqual(m_l1["status"], "fitted")
        lh, la = am.estimate_match_lambdas("Barcelona", "Sevilla", "La Liga", team_model=model)
        self.assertTrue(0.3 <= lh <= 5.0 and 0.3 <= la <= 5.0)


class TestBig5ExpansionEVAndPublicExports(unittest.TestCase):
    """(f) Big-5 league coverage, profitable-market EV gating, and Next.js data/public JSON exports."""

    def test_big5_canonical_names_and_strengths_loaded(self):
        self.assertEqual(ld.canonical_team_name("Man City", "EPL"), "Manchester City")
        self.assertEqual(ld.canonical_team_name("Nott'm Forest", "EPL"), "Nottingham Forest")
        self.assertEqual(ld.canonical_team_name("Ein Frankfurt", "Bundesliga"), "Eintracht Frankfurt")
        self.assertEqual(ld.canonical_team_name("M'gladbach", "Bundesliga"), "Borussia M.Gladbach")
        self.assertEqual(ld.canonical_team_name("Hellas Verona", "Serie A"), "Verona")
        self.assertEqual(ld.canonical_team_name("Inter Milan", "Serie A"), "Inter")

        model = TeamStrengthModel()
        for lg in ld.BIG5_LEAGUES:
            df = model.get_league_match_dataframe(lg)
            self.assertGreater(len(df), 2000, f"Expected >2000 matches for {lg}, got {len(df)}")
            mu_h, mu_a, _ = model.league_baselines(lg)
            self.assertTrue(0.8 <= mu_h <= 2.5 and 0.6 <= mu_a <= 2.2)

    def test_ev_kelly_and_profitable_market_whitelist(self):
        # EV = p * odds - 1
        self.assertAlmostEqual(am.compute_ev(0.60, 1.80), 0.08, places=6)
        self.assertGreater(am.calculate_fractional_kelly(0.60, 1.80, fraction=0.25, max_stake=0.05), 0.0)
        self.assertEqual(am.calculate_fractional_kelly(0.50, 1.90), 0.0)

        critic = am.ModelValidatorAgent(min_ev_pct=2.0, model_tip_min=0.58, enforce_approved_markets=True)
        df_test = pd.DataFrame([
            {
                "league": "La Liga", "home": "Real Madrid", "away": "Alaves",
                "market": "1X2 Home Win", "p_calibrated": 0.65, "market_odds": 1.63,
                "min_odds": 1.63, "ev_percent": 6.0, "price_type": "model", "lh": 2.2, "la": 0.7,
            },
            {
                "league": "Serie A", "home": "Inter", "away": "Atalanta",
                "market": "Over 2.5", "p_calibrated": 0.62, "market_odds": 1.71,
                "min_odds": 1.71, "ev_percent": 6.0, "price_type": "model", "lh": 1.9, "la": 1.4,
            },
            {
                "league": "EPL", "home": "Arsenal", "away": "Chelsea",
                "market": "1X2 Away Win", "p_calibrated": 0.65, "market_odds": 1.70,
                "min_odds": 1.63, "ev_percent": 10.5, "price_type": "model", "lh": 1.0, "la": 2.1,
            },
        ])
        approved, rejected = critic.audit(df_test)
        self.assertEqual(len(approved), 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected.iloc[0]["league"], "EPL")

    def test_public_json_bundle_exports_valid_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            pub_dir = Path(tmp) / "public"
            sample_opps = pd.DataFrame([
                {
                    "date": "2026-11-28 18:00:00",
                    "league": "La Liga",
                    "home": "Real Madrid",
                    "away": "Alaves",
                    "market": "1X2 Home Win",
                    "selection_side": "HOME",
                    "p_model": 0.66,
                    "p_calibrated": 0.64,
                    "price_type": "model",
                    "min_odds": 1.66,
                    "market_odds": 1.66,
                    "ev_percent": 6.24,
                    "stake_kelly": 0.02,
                    "lh": 2.15,
                    "la": 0.72,
                }
            ])
            summary = {
                "settled_now": 0,
                "total_settled_all_time": 21,
                "wins": 9,
                "half_wins": 7,
                "losses": 5,
                "hit_rate_percent": 59.52,
                "verified_settled_count": 0,
                "verified_roi_percent": None,
                "pending_after": 67,
            }
            paths = am.export_public_json_bundle(sample_opps, summary, public_dir=pub_dir)
            pred_data = json.loads(paths["predictions_latest"].read_text(encoding="utf-8"))
            perf_data = json.loads(paths["performance_summary"].read_text(encoding="utf-8"))

            self.assertEqual(pred_data["count"], 1)
            self.assertEqual(pred_data["predictions"][0]["selection_side"], "HOME")
            self.assertEqual(pred_data["predictions"][0]["min_odds"], 1.66)
            self.assertEqual(perf_data["settlement"]["hit_rate_percent"], 59.52)

        # Verify repo's backtest_verified_stats.json exists and has required structure
        bt_path = REPO_ROOT / "data" / "public" / "backtest_verified_stats.json"
        self.assertTrue(bt_path.exists(), "data/public/backtest_verified_stats.json must exist")
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
        self.assertIn("approved_production_pairs", bt)
        self.assertEqual(len(bt["breakdown"]), 25)

    def test_dnb_pricing_replication_odds_push_settlement_and_v4_thresholds(self):
        # 1. Exact conditional DNB probability
        mat = am.compute_dixon_coles_probs(2.0, 0.9, rho=-0.10)
        mp = am.calculate_market_probabilities(mat)
        expected_dnb = mp["home_win"] / (mp["home_win"] + mp["away_win"])
        self.assertAlmostEqual(mp["dnb_home"], expected_dnb, places=3)

        # 2. Exact tradable DNB replication odds from real 1X2 odds: O_H * (1 - 1/O_D)
        rec = {"home_win": 2.00, "draw": 4.00, "away_win": 3.80}
        self.assertAlmostEqual(fo.extract_market_price(rec, "DNB Home", "HOME"), 1.50, places=3)

        # 3. Draw settles as PUSH (0.0 profit), Win settles as WIN
        fake_index = {
            ("La Liga", "Real Madrid", "Sevilla"): [(pd.Timestamp("2026-11-15"), 1, 1)],
            ("La Liga", "Barcelona", "Alaves"): [(pd.Timestamp("2026-11-16"), 2, 0)],
        }
        r_push = sla.settle_row(
            {
                "date": "2026-11-15", "league": "La Liga", "home": "Real Madrid", "away": "Sevilla",
                "market": "DNB Home", "selection_side": "HOME", "odds_source": "the-odds-api",
                "market_odds": 1.55, "stake": 0.02, "status": "PENDING",
            },
            cutoff=pd.Timestamp("2026-11-23"), result_index=fake_index,
        )
        self.assertEqual(r_push["status"], "PUSH")
        self.assertEqual(r_push["profit"], 0.0)

        r_win = sla.settle_row(
            {
                "date": "2026-11-16", "league": "La Liga", "home": "Barcelona", "away": "Alaves",
                "market": "DNB Home", "selection_side": "HOME", "odds_source": "the-odds-api",
                "market_odds": 1.50, "stake": 0.02, "status": "PENDING",
            },
            cutoff=pd.Timestamp("2026-11-23"), result_index=fake_index,
        )
        self.assertEqual(r_win["status"], "WIN")
        self.assertAlmostEqual(r_win["profit"], 0.01, places=4)

        # 4. Promoted v4 thresholds: core 1X2/OU requires >= 0.60, DNB requires >= 0.70
        critic = am.ModelValidatorAgent(enforce_approved_markets=True)
        self.assertEqual(critic.model_tip_min, 0.60)
        self.assertEqual(critic.dnb_min, 0.70)
        df_thr = pd.DataFrame([
            {
                "league": "La Liga", "home": "Real Madrid", "away": "Alaves",
                "market": "1X2 Home Win", "p_calibrated": 0.59, "market_odds": 1.80,
                "min_odds": 1.80, "ev_percent": 6.0, "price_type": "model", "lh": 2.0, "la": 0.8,
            },
            {
                "league": "La Liga", "home": "Real Madrid", "away": "Alaves",
                "market": "DNB Home", "p_calibrated": 0.68, "market_odds": 1.56,
                "min_odds": 1.56, "ev_percent": 6.0, "price_type": "model", "lh": 2.0, "la": 0.8,
            },
            {
                "league": "La Liga", "home": "Real Madrid", "away": "Alaves",
                "market": "DNB Home", "p_calibrated": 0.73, "market_odds": 1.45,
                "min_odds": 1.45, "ev_percent": 6.0, "price_type": "model", "lh": 2.2, "la": 0.7,
            },
        ])
        app, rej = critic.audit(df_thr)
        self.assertEqual(len(app), 1)
        self.assertEqual(len(rej), 2)
        self.assertEqual(app.iloc[0]["market"], "DNB Home")


if __name__ == "__main__":
    unittest.main()
