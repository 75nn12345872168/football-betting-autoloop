"""
telegram_alerts.py — Institutional Telegram Alert Formatter & Dispatcher (AlphaPitch).

Includes:
  * Weekly Matchweek Transparency Header (Tue 00:00 UTC -> Mon 23:59 UTC):
      - Active Matchweek status (PROFIT / LOSS / BREAKEVEN / PENDING),
        Conservative Floor Kelly Bankroll Change %, Min-Odds ROI %, Flat 1u Net Units.
      - Last Closed & Archived Matchweek outcome.
  * Honest settlement metrics (never computes fake ROI from placeholder prices).
  * Broker Mode (+6% safety margin) vs Verified Market Odds formatting.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd


def _format_week_line(wk: Dict[str, Any], prefix: str) -> str:
    label = str(wk.get("label") or wk.get("matchweek_id") or "")
    status = str(wk.get("status", "PENDING")).upper()
    settled = int(wk.get("settled_picks", 0))
    total = int(wk.get("total_picks", 0))
    pending = int(wk.get("pending_picks", 0))
    wins = int(wk.get("wins", 0))
    half_wins = int(wk.get("half_wins", 0))
    losses = int(wk.get("losses", 0))
    pushes = int(wk.get("pushes", 0))
    kelly_chg = float(wk.get("kelly_bankroll_change_pct", 0.0) or 0.0)
    flat_u = float(wk.get("flat_net_units", 0.0) or 0.0)
    cons_roi = wk.get("conservative_min_odds_roi_pct")
    close_roi = wk.get("closing_odds_roi_pct")

    if status == "PROFIT":
        icon = "🟢"
        st_ar = "ربح"
    elif status == "LOSS":
        icon = "🔴"
        st_ar = "خسارة"
    elif status == "BREAKEVEN":
        icon = "⚪"
        st_ar = "تعادل"
    else:
        icon = "⏳"
        st_ar = "قيد التنفيذ"

    if settled == 0:
        return (
            f"{icon} <b>{prefix}</b> (<code>{label}</code>): "
            f"<b>{st_ar}</b> — {total} صفقة معلّقة بانتظار التسوية"
        )

    roi_txt = f" | ROI(Min Odds): <b>{float(cons_roi):+.2f}%</b>" if cons_roi is not None else ""
    close_txt = f" | Closing ROI: <b>{float(close_roi):+.2f}%</b>" if close_roi is not None else ""
    push_txt = f" / {pushes}P" if pushes > 0 else ""
    pend_txt = f" ({pending} معلّقة)" if pending > 0 else ""
    return (
        f"{icon} <b>{prefix}</b> (<code>{label}</code>): "
        f"<b>{st_ar} {kelly_chg:+.2f}%</b> رأس مال Kelly | "
        f"<b>{flat_u:+.2f}u</b> ثابت{roi_txt}{close_txt} "
        f"[{wins}W/{half_wins}HW/{losses}L{push_txt}{pend_txt}]"
    )


def format_weekly_transparency_header(transparency_banner: Optional[Dict[str, Any]]) -> str:
    """Build a concise Arabic transparency summary header for Telegram messages."""
    if not transparency_banner or not isinstance(transparency_banner, dict):
        return ""
    cur = transparency_banner.get("current_week")
    last_closed = transparency_banner.get("last_closed_week")
    if not cur and not last_closed:
        return ""

    lines = ["🛡️ <b>الشفافية الأسبوعية (نافذة الجولة: الثلاثاء–الاثنين UTC | Min Odds Floor)</b>"]
    if isinstance(cur, dict):
        lines.append(_format_week_line(cur, "الجولة الجارية"))
    if isinstance(last_closed, dict):
        lines.append(_format_week_line(last_closed, "آخر جولة مؤرشفة"))
    else:
        lines.append("📦 <b>آخر جولة مؤرشفة</b>: <i>لا توجد جولات مكتملة ومؤرشفة بعد (بداية سجل v4)</i>")
    return "\n".join(lines) + "\n\n"


def format_telegram_report(
    summary: Dict[str, Any],
    df_opps: pd.DataFrame,
    critic_note: str = "",
    transparency_banner: Optional[Dict[str, Any]] = None,
    mle_by_league: Optional[Dict[str, Dict[str, Any]]] = None,
    get_league_params_fn: Optional[Callable[[str], Tuple[float, float, float]]] = None,
    get_league_status_fn: Optional[Callable[[str], str]] = None,
) -> str:
    settled_now = int(summary.get("settled_now", 0))
    total_settled = int(summary.get("total_settled_all_time", 0))
    wins = int(summary.get("wins", 0))
    half_wins = int(summary.get("half_wins", 0))
    losses = int(summary.get("losses", 0))
    hit_rate = float(summary.get("hit_rate_percent", 0.0))
    pending = int(summary.get("pending_after", 0))

    v_count = int(summary.get("verified_settled_count", 0))
    v_profit = float(summary.get("verified_profit_units", 0.0))
    v_roi = summary.get("verified_roi_percent")
    if v_count > 0 and v_roi is not None:
        roi_line = f"• العائد المالي المؤكد (ROI | {v_count} صفقة): <b>{v_roi:+.2f}%</b> ({v_profit:+.3f} وحدة)"
    else:
        roi_line = "• العائد المالي المؤكد (ROI): <i>بانتظار صفقات مسوّاة بأسعار سوق حقيقية</i>"

    _mle_bits = []
    if mle_by_league:
        for _lg in sorted(mle_by_league.keys()):
            if get_league_params_fn and get_league_status_fn:
                _p_h, _p_a, _p_r = get_league_params_fn(_lg)
                _st = get_league_status_fn(_lg)
            else:
                v = mle_by_league[_lg]
                _p_h, _p_a, _p_r = float(v["mu_home"]), float(v["mu_away"]), float(v["rho"])
                _st = str(v.get("status", "prior"))
            _mle_bits.append(f"{_lg.title()}: μH={_p_h:.2f} μA={_p_a:.2f} ρ={_p_r:.3f} ({_st})")
    _mle_txt = " | ".join(_mle_bits) if _mle_bits else "no active leagues"
    _critic_block = f"• تدقيق الناقد: <code>{critic_note}</code>\n" if critic_note else ""

    banner = transparency_banner if transparency_banner is not None else summary.get("transparency_banner")
    weekly_header = format_weekly_transparency_header(banner)

    msg = f"""{weekly_header}📊 <b>تسوية الجولة وإحصائيات النتائج الحقيقية</b>
• الصفقات المسوّاة حديثاً: <b>{settled_now}</b>
• سجل النتائج الفعلية ({total_settled} صفقة): <b>{wins} فوز / {half_wins} نصف فوز / {losses} خسارة</b> (الدقة: <b>{hit_rate:.1f}%</b>)
{roi_line}
• العمليات المعلقة: <b>{pending}</b>
• معايرة MLE: <code>{_mle_txt}</code>
{_critic_block}
🚀 <b>توصيات النموذج المعايَر للجولة القادمة</b>
"""

    if df_opps is None or df_opps.empty:
        msg += "<i>لا توجد فرص مستوفية لمعايير الثقة والقيمة لهذه الجولة.</i>\n"
    else:
        _any_model = False
        _any_live = False
        for _, r in df_opps.iterrows():
            d_str = str(r["date"])[:10]
            league = r["league"]
            home = r["home"]
            away = r["away"]
            market = r["market"]
            side = str(r.get("selection_side", "") or "").strip()
            side_tag = f" [{side}]" if side else ""
            p_cal = float(r.get("p_calibrated", r.get("p_model", 0.58))) * 100.0
            stake = float(r.get("stake_kelly", 0.02)) * 100.0
            pt = str(r.get("price_type", "model"))
            l_best = r.get("live_best_odds")
            l_bk = r.get("live_best_bookmaker")
            l_pin = r.get("live_pinnacle_odds")
            l_ev = r.get("live_ev_pct")
            l_act = r.get("is_live_actionable")
            has_live = l_best is not None and not pd.isna(l_best) and float(l_best) > 1.0

            if pt == "verified":
                odds = float(r["market_odds"])
                ev = float(r.get("ev_percent", 0.0))
                price_line = f"  🎯 <b>{market}{side_tag}</b> @ <b>{odds:.2f}</b> (سعر سوق مؤكد | EV <code>{ev:+.1f}%</code>)\n"
            else:
                _any_model = True
                mo = float(r.get("min_odds") or 0.0)
                price_line = (
                    f"  🎯 <b>{market}{side_tag}</b> | الثقة المعايَرة <b>{p_cal:.1f}%</b> "
                    f"→ الحد الأدنى ≥ <b>{mo:.2f}</b> <i>(+6% أمان)</i>\n"
                )
            live_line = ""
            if has_live:
                _any_live = True
                bk_txt = f" @ <b>{l_bk}</b>" if (l_bk and str(l_bk).strip() and str(l_bk).lower() != "nan") else ""
                pin_txt = (
                    f" | Pinnacle: <code>{float(l_pin):.2f}</code>"
                    if (l_pin is not None and not pd.isna(l_pin) and float(l_pin) > 1.0)
                    else ""
                )
                ev_txt = (
                    f" | EV حي: <code>{float(l_ev):+.1f}%</code>"
                    if (l_ev is not None and not pd.isna(l_ev))
                    else ""
                )
                status_tag = "✅ <b>LIVE +EV</b>" if bool(l_act) else "⏳ <b>WAIT FOR ODDS</b>"
                live_line = f"  📡 السعر الحي: <b>{float(l_best):.2f}</b>{bk_txt}{pin_txt}{ev_txt} | {status_tag}\n"

            msg += f"• <b>{d_str} | {league}</b>\n"
            msg += f"  ⚽ {home} vs {away}\n"
            msg += price_line
            if live_line:
                msg += live_line
            msg += f"  💼 حجم الرهان المقترح: <b>{stake:.2f}%</b>\n\n"
        if _any_live:
            msg += (
                "📡 <i>الأسعار الحية مدمجة من The Odds API v4: ادخل فقط عند شارة ✅ LIVE +EV "
                "وتوفّر سعر ≥ الحد الأدنى (+6% هامش أمان).</i>\n"
            )
        elif _any_model:
            msg += (
                "⚠️ <i>وضع الوسيط (بدون THE_ODDS_API_KEY): الحد الأدنى للسعر يشمل هامش أمان +6% "
                "فوق السعر العادل المعايَر. لا تراهن بأقل من الحد الأدنى المذكور.</i>\n"
            )
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += "⚡ <i>المحرك الكمي v4 (Matchweek Window + Time Decay xG + Platt Calibration + Live Fixtures)</i>"
    return msg.strip()


def send_telegram_alert(message: str) -> bool:
    import requests

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        try:
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            r = requests.post(
                url,
                data={
                    "chat_id": tg_chat,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            if r.status_code == 200:
                print("[TELEGRAM] Alert sent successfully (status: 200).")
                return True
            print(f"[TELEGRAM ERROR] status {r.status_code}: {r.text[:200]}")
            return False
        except Exception as e:
            print(f"[TELEGRAM EXCEPTION] {e}")
            return False
    print("[TELEGRAM] No TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID detected. Running in local/dry mode.")
    return False
