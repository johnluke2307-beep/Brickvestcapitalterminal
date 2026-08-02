"""
Brickvestcapitalterminal — Streamlit control surface.

The dashboard is a *view* over the engine and the bot: it computes nothing on its
own beyond formatting, so the numbers on screen are the same numbers the
execution loop trades on.

Six tabs:
    Deck       — status strip, target acquisition, risk manager, operating rules
    Scanner    — the VRP edge, per symbol, with the reason anything was rejected
    Positions  — live short premium with its managed bracket levels
    Trade log  — realised record and the expectancy formula it produces
    Backtest   — the same rules replayed over history, premium assumption exposed
    Bot        — start/stop, halt state, and the event feed
    Settings   — every guardrail currently in force

Run locally with ``streamlit run app.py``; on Streamlit Community Cloud point the
app at this file and put the Alpaca keys in the Secrets panel.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import engine
from bot import BotState, TradingBot
from broker_client import BrokerError, build_client

st.set_page_config(
    page_title="Brickvestcapitalterminal",
    page_icon="📈",
    layout="wide",
    # "auto" collapses the sidebar on narrow screens. Forcing it open makes it
    # overlay the deck on a phone and swallow taps meant for the tabs.
    initial_sidebar_state="auto",
)

OPTION_MULTIPLIER = 100

# ======================================================================================
# Chart theme
# ======================================================================================
# Slots from a colourblind-validated palette. Series identity is fixed: implied
# volatility is always blue, realised volatility always orange — the colour follows
# the entity, never its rank, so a filtered chart never repaints its survivors.
THEMES = {
    # The deck runs dark by default; the light steps stay available for anyone
    # who wants to read it in daylight. Both were validated against their own
    # surface, so the series colours clear 3:1 and stay CVD-separable in each.
    "dark": {
        "surface": "#05070a",
        "panel": "#0a0d12",
        "panel_alt": "#0d1117",     # zebra stripe
        "border": "#1a2029",
        "text": "#e8edf2",
        "muted": "#78828f",
        # Amber is the terminal's chrome voice — labels, rails, column heads.
        # It never carries data, so it cannot be confused with a series colour.
        "chrome": "#ffa028",
        "chrome_dim": "#8a5a1b",
        "grid": "#151b23",
        "axis": "#232b36",
        "series_1": "#3987e5",  # blue   — implied volatility, income, equity
        "series_2": "#d95926",  # orange — realised volatility
        "series_1_fill": "rgba(57,135,229,0.16)",
        "positive": "#3987e5",
        "negative": "#d03b3b",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
    "light": {
        "surface": "#fcfcfb",
        "panel": "#f4f4f1",
        "panel_alt": "#efefeb",
        "border": "#e1e0d9",
        "text": "#0b0b0b",
        "muted": "#898781",
        "chrome": "#8a5a1b",
        "chrome_dim": "#b08a52",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series_1": "#2a78d6",
        "series_2": "#eb6834",
        "series_1_fill": "rgba(42,120,214,0.12)",
        "positive": "#2a78d6",
        "negative": "#d03b3b",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
}

MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def theme() -> dict:
    """Chart palette matching the active deck theme."""
    base = st.session_state.get("chart_theme")
    if base not in THEMES:
        try:
            base = st.get_option("theme.base") or "dark"
        except Exception:
            base = "dark"
    return THEMES.get(base, THEMES["dark"])


def inject_terminal_css() -> None:
    """Terminal chrome: panel rules, monospace figures, status cells.

    Streamlit\'s own theme (see .streamlit/config.toml) sets the base colours;
    this adds the console furniture on top — hairline-bordered panels, uppercase
    section rails, and tabular figures so columns of numbers line up.
    """
    p = theme()
    st.markdown(
        f"""
        <style>
          /* ---- surface -------------------------------------------------
             Streamlit resolves config.toml against the working directory, so a
             run started elsewhere would keep light chrome. These rules hold the
             terminal look however it was launched. */
          .stApp, [data-testid="stAppViewContainer"] {{ background: {p["surface"]}; }}
          [data-testid="stHeader"] {{ background: transparent; height: 2.2rem; }}
          .block-container {{ padding-top: 2.4rem !important; padding-bottom: 2rem; }}
          [data-testid="stSidebar"] {{
              background: {p["panel"]}; border-right: 1px solid {p["border"]};
          }}
          [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] li {{ color: {p["text"]}; }}
          [data-testid="stSidebar"] h3 {{
              color: {p["chrome"]}; letter-spacing: .14em; text-transform: uppercase;
              font-size: .8rem;
          }}
          [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
          [data-testid="stSidebar"] small {{ color: {p["muted"]} !important; }}
          [data-testid="stSidebar"] strong {{ color: {p["chrome"]}; }}

          /* ---- controls: boxy, monospace, amber on hover ---------------- */
          .stButton > button, .stDownloadButton > button {{
              background: {p["panel"]}; color: {p["text"]};
              border: 1px solid {p["border"]}; border-radius: 0;
              font-family: {MONO}; font-size: .72rem;
              letter-spacing: .1em; text-transform: uppercase; padding: .3rem .7rem;
          }}
          .stButton > button:hover, .stDownloadButton > button:hover {{
              border-color: {p["chrome"]}; color: {p["chrome"]}; background: {p["panel_alt"]};
          }}
          .stButton > button:disabled {{ color: {p["muted"]}; border-color: {p["border"]}; }}
          .stButton > button[kind="primary"] {{
              background: {p["chrome"]}; color: {p["surface"]};
              border-color: {p["chrome"]}; font-weight: 700;
          }}
          [data-baseweb="select"] > div, [data-baseweb="input"] > div,
          [data-testid="stSelectbox"] div[role="combobox"],
          [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
          [data-testid="stSelectbox"] div[data-baseweb="select"] div {{
              background-color: {p["panel"]} !important;
              border-color: {p["border"]} !important;
              color: {p["text"]} !important;
              border-radius: 0 !important; font-family: {MONO};
          }}
          [data-baseweb="popover"] li {{
              background-color: {p["panel"]} !important; color: {p["text"]} !important;
              font-family: {MONO}; font-size: .78rem;
          }}
          [data-baseweb="select"] svg {{ fill: {p["muted"]}; }}
          [data-baseweb="tag"] {{ border-radius: 0 !important; font-family: {MONO}; }}
          label p {{
              font-family: {MONO} !important; font-size: .64rem !important;
              text-transform: uppercase; letter-spacing: .11em;
              color: {p["chrome_dim"]} !important;
          }}

          /* ---- tabs: a function rail, not pills ------------------------- */
          [data-testid="stTabs"] [role="tablist"] {{
              flex-wrap: wrap !important; overflow-x: visible !important;
              row-gap: 0; gap: 0; border-bottom: 1px solid {p["border"]};
          }}
          [data-testid="stTabs"] [role="tab"] {{
              white-space: nowrap; font-family: {MONO};
              font-size: .72rem; letter-spacing: .11em; text-transform: uppercase;
              padding: .35rem .8rem; color: {p["muted"]};
              border-bottom: 2px solid transparent;
          }}
          [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
              color: {p["chrome"]}; border-bottom-color: {p["chrome"]}; background: {p["panel"]};
          }}

          /* ---- figures -------------------------------------------------- */
          [data-testid="stMetricValue"], [data-testid="stDataFrame"], .bvc-mono {{
              font-family: {MONO}; font-variant-numeric: tabular-nums;
          }}
          [data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
          [data-testid="stMetricLabel"] p {{
              text-transform: uppercase; letter-spacing: .1em;
              font-size: .62rem; color: {p["chrome_dim"]};
          }}

          /* ---- panels: boxed, amber title rail with a numbered chip ----- */
          .bvc-panel {{
              border: 1px solid {p["border"]}; border-radius: 0;
              background: {p["panel"]}; padding: 0 .65rem .5rem; margin-bottom: .5rem;
          }}
          .bvc-panel-title {{
              font-family: {MONO}; font-size: .66rem; font-weight: 700;
              text-transform: uppercase; letter-spacing: .16em; color: {p["chrome"]};
              border-bottom: 1px solid {p["border"]};
              padding: .34rem 0 .28rem; margin: 0 0 .4rem;
          }}
          .bvc-panel-title .idx {{
              color: {p["surface"]}; background: {p["chrome"]};
              padding: 0 .32rem; margin-right: .45rem; font-weight: 700;
          }}

          /* ---- status strip --------------------------------------------- */
          .bvc-strip {{
              display: flex; flex-wrap: wrap; gap: 0;
              border: 1px solid {p["border"]}; border-radius: 0;
              background: {p["panel"]}; overflow: hidden; margin-bottom: .5rem;
          }}
          .bvc-cell {{
              flex: 1 1 116px; padding: .32rem .7rem;
              border-right: 1px solid {p["border"]};
          }}
          .bvc-cell:last-child {{ border-right: none; }}
          .bvc-cell .k {{
              font-family: {MONO}; font-size: .57rem; letter-spacing: .15em;
              text-transform: uppercase; color: {p["chrome_dim"]};
          }}
          .bvc-cell .v {{
              font-family: {MONO}; font-size: 1rem; font-weight: 700;
              color: {p["text"]}; font-variant-numeric: tabular-nums;
          }}

          /* ---- data rows: zebra, tight, tabular -------------------------- */
          .bvc-row {{
              display: grid; align-items: center; gap: .4rem;
              font-family: {MONO}; font-size: .75rem;
              padding: .2rem .3rem; border-bottom: 1px solid {p["border"]};
              font-variant-numeric: tabular-nums;
          }}
          .bvc-row:nth-of-type(even) {{ background: {p["panel_alt"]}; }}
          .bvc-row:last-child {{ border-bottom: none; }}
          .bvc-head {{
              color: {p["chrome"]}; font-size: .59rem; letter-spacing: .13em;
              text-transform: uppercase; background: transparent !important;
              border-bottom: 1px solid {p["border"]};
          }}
          .bvc-tag {{
              font-family: {MONO}; font-size: .65rem; font-weight: 700;
              letter-spacing: .06em; padding: .05rem .4rem; border-radius: 0;
              border: 1px solid currentColor; white-space: nowrap;
          }}
          /* Anything that is not MONITOR wants a decision now, so it blinks. */
          @keyframes bvc-blink {{ 0%, 55% {{ opacity: 1; }} 56%, 100% {{ opacity: .28; }} }}
          .bvc-tag.bvc-act {{ animation: bvc-blink 1.1s steps(1, end) infinite; }}
          @media (prefers-reduced-motion: reduce) {{
              .bvc-tag.bvc-act {{ animation: none; text-decoration: underline; }}
          }}
          .t-good {{ color: {p["good"]}; }}
          .t-warn {{ color: {p["warning"]}; }}
          .t-crit {{ color: {p["critical"]}; }}
          .t-idle {{ color: {p["muted"]}; }}
          .t-accent {{ color: {p["series_1"]}; }}
          .t-chrome {{ color: {p["chrome"]}; }}
          .bvc-footer {{
              font-family: {MONO}; font-size: .65rem; color: {p["muted"]};
              border-top: 1px solid {p["border"]}; padding-top: .38rem;
              margin-top: .3rem; letter-spacing: .06em;
          }}
          .bvc-footer b {{ color: {p["chrome"]}; }}

          /* ---- misc ------------------------------------------------------ */
          [data-testid="stDataFrame"] {{ border: 1px solid {p["border"]}; }}
          [data-testid="stAlert"] {{ border-radius: 0; font-family: {MONO}; font-size: .77rem; }}
          hr {{ border-color: {p["border"]}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 320, *, showlegend: bool = False) -> go.Figure:
    """Recessive chrome, transparent surface, one axis, hover always on."""
    palette = theme()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=13, color=palette["text"]),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=palette["muted"])),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=palette["axis"], tickfont=dict(color=palette["muted"]))
    fig.update_yaxes(
        showgrid=True,
        gridcolor=palette["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=palette["axis"],
        tickfont=dict(color=palette["muted"]),
    )
    return fig


# ======================================================================================
# Shared runtime objects
# ======================================================================================
@st.cache_resource(show_spinner=False)
def get_bot() -> TradingBot:
    """One bot (and one broker connection) shared across every browser session."""
    settings = config.load_settings()
    client = build_client(settings)
    return TradingBot(client=client, settings=settings)


def md_escape(text: str) -> str:
    """Escape markdown that Streamlit would otherwise interpret.

    Dollar figures are the live case: Streamlit reads ``$…$`` as LaTeX, so a
    message like "needs $48,500 but the pool is $25,000" renders as italic maths
    instead of the sentence. Engine and backtest messages are full of them.
    """
    return text.replace("$", "\\$")


def fmt_usd(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"${value:,.{digits}f}"


def fmt_zar(value: Optional[float], digits: int = 0) -> str:
    return "—" if value is None else f"R{value:,.{digits}f}"


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


# ======================================================================================
# Sidebar — connection, controls, currency
# ======================================================================================
def render_sidebar(bot: TradingBot) -> None:
    settings = bot.settings
    with st.sidebar:
        st.markdown("### BVC Terminal")
        st.caption("Brickvestcapitalterminal · variance risk premium · Alpaca")

        # ---- link state --------------------------------------------------
        health = bot.client.health
        status_label = {
            "ok": ("🟢", "Connected"),
            "degraded": ("🟡", "Degraded"),
            "down": ("🔴", "Disconnected"),
        }[health.status]
        st.markdown(f"**{status_label[0]} {status_label[1]}** · {'paper' if settings.paper else 'LIVE'}")
        if health.last_ok:
            st.caption(f"Last good call {health.last_ok.strftime('%H:%M:%S')} UTC")
        if health.last_error:
            st.caption(f"⚠️ {health.last_error[:160]}")

        if not settings.credentials_present:
            st.error("Alpaca keys not configured — see the Settings tab.")
        elif st.button("Reconnect", use_container_width=True):
            bot.client.connect()
            st.rerun()

        st.divider()

        # ---- bot control -------------------------------------------------
        st.markdown("**Automation**")
        if bot.state.is_halted:
            st.error(f"HALTED — {bot.state.halt_reason}")
            if st.button("Clear halt & resume", type="primary", use_container_width=True):
                bot.resume()
                st.rerun()
        else:
            no_keys = not settings.credentials_present
            columns = st.columns(2)
            if columns[0].button("Start", use_container_width=True, disabled=bot.is_running or no_keys):
                bot.start()
                st.rerun()
            if columns[1].button("Stop", use_container_width=True, disabled=not bot.is_running):
                bot.stop()
                st.rerun()
            if st.button("Run one cycle now", use_container_width=True, disabled=no_keys):
                with st.spinner("Running cycle…"):
                    result = bot.run_once()
                st.toast(result.summary())
                st.rerun()

        st.caption(f"State: **{bot.state.mode}** · {bot.state.cycles} cycles")
        if bot.state.last_cycle_at:
            st.caption(f"Last cycle {bot.state.last_cycle_at} — {bot.state.last_cycle_summary}")

        st.divider()

        # ---- currency ----------------------------------------------------
        st.markdown("**USD / ZAR**")
        fx_quote = bot.fx.get_rate()
        st.metric("Rate", f"{fx_quote.rate:.4f}", help=f"Source: {fx_quote.source}")
        if fx_quote.stale:
            st.caption("⚠️ Live FX unavailable — cached/fallback rate in use.")
        else:
            st.caption(f"Live via {fx_quote.source}")
        with st.expander("Override rate"):
            manual = st.number_input("USD/ZAR", value=float(fx_quote.rate), min_value=1.0, max_value=100.0, step=0.05)
            if st.button("Pin rate", use_container_width=True):
                bot.fx.set_manual_rate(manual)
                st.rerun()
            if st.button("Refresh live", use_container_width=True):
                bot.fx.get_rate(force_refresh=True)
                st.rerun()

        st.divider()
        # Chart colours only — the deck chrome follows .streamlit/config.toml.
        st.session_state["chart_theme"] = st.selectbox(
            "Chart palette", ["dark", "light"], index=0 if theme() is THEMES["dark"] else 1
        )


# ======================================================================================
# Tab 1 — Deck
# ======================================================================================
def render_deck(bot: TradingBot) -> None:
    """The command deck: status strip, target acquisition, risk manager, logic rail.

    Layout mirrors a console terminal — a scan panel on the left, live risk on
    the right at twice the width, and the operating rules pinned along the
    bottom so the parameters the bot is enforcing are never off-screen.
    """
    settings = bot.settings
    palette = theme()
    fx_quote = bot.fx.get_rate()

    account = None
    account_error = None
    try:
        account = bot.client.get_account()
    except BrokerError as exc:
        account_error = str(exc)

    closed = bot.trade_log.closed_trades()
    expectancy = engine.compute_expectancy(closed)
    zar_by_month = engine.monthly_pnl(closed, "zar")
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    month_zar = zar_by_month.get(month_key, 0.0)
    progress = month_zar / settings.monthly_target_zar if settings.monthly_target_zar else 0.0

    render_status_strip(bot, account, fx_quote, month_zar, progress, expectancy)
    render_deck_controls(bot)

    if account_error:
        st.error(f"ACCOUNT FEED DOWN — {account_error}")
        st.caption("The bot halts rather than trade on stale data. Fix the link, then resume from the sidebar.")

    left, right = st.columns([1, 2], gap="small")
    with left:
        render_target_acquisition(bot)
    with right:
        render_risk_manager(bot, fx_quote)

    stop_pct = settings.stop_loss_multiple * 100
    st.markdown(
        f'<div class="bvc-footer">LOGIC: {settings.profit_target_pct:.0%} PROFIT TAKER ACTIVE'
        f" &nbsp;|&nbsp; STOP LOSS: {stop_pct:.0f}% OF CREDIT ({settings.stop_loss_price_multiple:.0f}x BUYBACK)"
        f" &nbsp;|&nbsp; TIME EXIT: {settings.time_exit_dte} DTE"
        f" &nbsp;|&nbsp; MAX MARGIN: {settings.max_margin_utilization:.0%} NAV"
        f" &nbsp;|&nbsp; ENTRY: {settings.target_dte} DTE @ {settings.target_delta:.2f}Δ, IVR ≥ {settings.min_iv_rank:.0f}"
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- analytics below the fold ----------------------------------------
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="bvc-panel-title"><span class="idx">3</span>Realised income by month · ZAR</div>', unsafe_allow_html=True)
        render_monthly_income_chart(zar_by_month, settings.monthly_target_zar)
    with right:
        st.markdown('<div class="bvc-panel-title"><span class="idx">4</span>Cumulative realised P&L · ZAR</div>', unsafe_allow_html=True)
        render_cumulative_pnl_chart(closed)

    render_expectancy_panel(expectancy, account, settings, palette)


def latest_scan(bot: TradingBot) -> tuple[List[engine.VRPSnapshot], Optional[datetime], str]:
    """The freshest scan available, from either the operator or the bot.

    A running bot scans every cycle, so the deck prefers that result whenever it
    is newer than the last manual scan — the panel stays live without anyone
    pressing anything.
    """
    manual = st.session_state.get("scan", [])
    manual_at = st.session_state.get("scan_at")

    auto, auto_at = [], None
    if bot.last_result and bot.last_result.scanned:
        auto, auto_at = bot.last_result.scanned, bot.last_result.started_at

    if auto and (manual_at is None or (auto_at and auto_at > manual_at)):
        return auto, auto_at, "bot cycle"
    if manual:
        return manual, manual_at, "manual scan"
    return [], None, "none"


def render_deck_controls(bot: TradingBot) -> None:
    """Scan and automation controls, so the deck needs no other tab to drive it."""
    _, scanned_at, source = latest_scan(bot)
    halted = bot.state.is_halted
    running = bot.is_running
    # Without credentials the first cycle would halt on an unreachable broker,
    # leaving a fresh deployment with a halt to clear before anything happened.
    no_keys = not bot.settings.credentials_present
    blocked = halted or no_keys

    columns = st.columns([1.1, 1.1, 1.1, 4.7])

    if columns[0].button("◆ RUN SCAN", use_container_width=True, disabled=blocked, key="deck_scan"):
        with st.spinner("Pricing chains…"):
            try:
                st.session_state["scan"] = bot.scan()
                st.session_state["scan_at"] = datetime.now(timezone.utc)
                st.rerun()
            except BrokerError as exc:
                st.error(md_escape(f"SCAN FAILED — {exc}"))

    if running:
        if columns[1].button("■ STOP BOT", use_container_width=True, key="deck_stop"):
            bot.stop()
            st.rerun()
    elif columns[1].button("▶ START BOT", use_container_width=True, disabled=blocked, key="deck_start"):
        bot.start()
        st.rerun()

    if columns[2].button("↻ RUN CYCLE", use_container_width=True, disabled=blocked, key="deck_cycle"):
        with st.spinner("Running cycle…"):
            result = bot.run_once()
        st.toast(result.summary())
        st.rerun()

    if halted:
        note = f"HALTED — {bot.state.halt_reason}"
    elif no_keys:
        note = "NO CREDENTIALS — add ALPACA_API_KEY and ALPACA_SECRET_KEY, then reload (see Settings)"
    else:
        stamp = f"{scanned_at:%H:%M:%S} UTC ({source})" if scanned_at else "never"
        note = (
            f"BOT {'RUNNING' if running else 'IDLE'} · {bot.state.cycles} cycles · "
            f"{bot.state.entries_today_count()} entries today · last scan {stamp}"
        )
    columns[3].markdown(f'<div class="bvc-footer" style="border:none;padding-top:.55rem">{note}</div>',
                        unsafe_allow_html=True)


def render_status_strip(bot, account, fx_quote, month_zar, progress, expectancy) -> None:
    """One row of console cells: link, capital, margin headroom, target progress."""
    settings = bot.settings
    palette = theme()
    health = bot.client.health

    link_class, link_text = {
        "ok": ("t-good", "ONLINE"),
        "degraded": ("t-warn", "DEGRADED"),
        "down": ("t-crit", "OFFLINE"),
    }[health.status]
    if bot.state.is_halted:
        link_class, link_text = "t-crit", "HALTED"

    if account:
        util = account.margin_utilization
        util_class = "t-crit" if util > settings.max_margin_utilization else "t-good"
        cells = [
            ("LINK", f'<span class="{link_class}">{link_text}</span> · {"PAPER" if settings.paper else "LIVE"}'),
            ("NAV USD", f"${account.equity:,.0f}"),
            ("NAV ZAR", f"R{account.equity * fx_quote.rate:,.0f}"),
            ("DAY P&L", _signed(account.day_pnl, palette, "$")),
            ("MARGIN", f'<span class="{util_class}">{util:.0%}</span> / {settings.max_margin_utilization:.0%}'),
            (f"{datetime.now(timezone.utc):%b} ZAR", f"R{month_zar:,.0f}"),
            ("TARGET", f'<span class="t-accent">{progress:.0%}</span> of R{settings.monthly_target_zar:,.0f}'),
            ("EXPECTANCY", _signed(expectancy.expectancy, palette, "$") if expectancy.has_data else "—"),
        ]
    else:
        cells = [
            ("LINK", f'<span class="{link_class}">{link_text}</span>'),
            ("NAV USD", "—"), ("NAV ZAR", "—"), ("DAY P&L", "—"), ("MARGIN", "—"),
            (f"{datetime.now(timezone.utc):%b} ZAR", f"R{month_zar:,.0f}"),
            ("TARGET", f"{progress:.0%}"),
            ("EXPECTANCY", _signed(expectancy.expectancy, palette, "$") if expectancy.has_data else "—"),
        ]

    html = "".join(f'<div class="bvc-cell"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells)
    st.markdown(f'<div class="bvc-strip">{html}</div>', unsafe_allow_html=True)


def _signed(value: float, palette: dict, prefix: str = "") -> str:
    colour = palette["good"] if value >= 0 else palette["critical"]
    return f'<span style="color:{colour}">{prefix}{value:+,.2f}</span>'


def render_target_acquisition(bot: TradingBot) -> None:
    """Left panel — the VRP scan, ranked, with each symbol's verdict."""
    snapshots, scanned_at, _ = latest_scan(bot)
    rows = ""
    if snapshots:
        ranked = sorted(
            snapshots, key=lambda s: (s.is_tradeable, s.vrp if s.vrp is not None else -9), reverse=True
        )
        for rank, snap in enumerate(ranked[:10], start=1):
            if snap.is_tradeable:
                tag, cls = "SELL", "t-good"
            else:
                tag, cls = "PASS", "t-idle"
            vrp = f"{snap.vrp * 100:+.1f}" if snap.vrp is not None else "  n/a"
            ivr = f"{snap.iv_rank.value:.0f}" if snap.iv_rank.value is not None else "--"
            rows += (
                '<div class="bvc-row" style="grid-template-columns:1.2rem 3.2rem 3rem 2.4rem 3rem;">'
                f'<span class="t-idle">{rank}</span>'
                f'<span class="t-accent">{snap.symbol}</span>'
                f"<span>{vrp}</span><span>{ivr}</span>"
                f'<span class="{cls}">{tag}</span></div>'
            )
    else:
        rows = '<div class="bvc-row t-idle">no scan yet — press RUN SCAN</div>'

    header = (
        '<div class="bvc-row bvc-head" style="grid-template-columns:1.2rem 3.2rem 3rem 2.4rem 3rem;">'
        "<span>#</span><span>TKR</span><span>VRP</span><span>IVR</span><span>SIG</span></div>"
    )
    stamp = f"{scanned_at:%H:%M:%S}" if scanned_at else "--:--:--"
    st.markdown(
        f'<div class="bvc-panel"><div class="bvc-panel-title"><span class="idx">1</span>Target acquisition · IV−RV'
        f'<span style="float:right;letter-spacing:.06em">{stamp}</span></div>{header}{rows}</div>',
        unsafe_allow_html=True,
    )


def render_risk_manager(bot: TradingBot, fx_quote) -> None:
    """Right panel — open short premium and the action the bot will take next.

    The action column is computed by ``bot.position_action``, the same rules the
    execution loop runs, so this panel cannot drift away from what will happen.
    """
    try:
        positions = [p for p in bot.client.get_option_positions() if p.is_short]
    except BrokerError as exc:
        st.markdown(
            f'<div class="bvc-panel"><div class="bvc-panel-title"><span class="idx">2</span>Risk manager</div>'
            f'<div class="bvc-row t-crit">position feed down — {exc}</div></div>',
            unsafe_allow_html=True,
        )
        return

    grid = "grid-template-columns:4.2rem 3.4rem 3.4rem 3.2rem 2.4rem 8rem;"
    header = (
        f'<div class="bvc-row bvc-head" style="{grid}">'
        "<span>TICKER</span><span>CREDIT</span><span>MARK</span><span>CAPT</span><span>DTE</span>"
        "<span style='text-align:right'>ACTION</span></div>"
    )

    if not positions:
        body = '<div class="bvc-row t-idle">flat — no open short premium</div>'
    else:
        body = ""
        severity_class = {"good": "t-good", "warning": "t-warn", "critical": "t-crit", "idle": "t-idle"}
        for position in positions:
            verdict = bot.position_action(position)
            cls = severity_class[verdict["severity"]]
            if verdict["severity"] != "idle":
                cls += " bvc-act"  # blinks — this one needs a decision
            captured = verdict["captured"]
            cap_cls = "t-good" if captured >= 0 else "t-crit"
            body += (
                f'<div class="bvc-row" style="{grid}">'
                f'<span class="t-accent">{position.underlying or position.symbol}'
                f'<span class="t-idle"> {position.option_type[0].upper() if position.option_type else ""}'
                f'{position.strike:g}</span></span>'
                f'<span>${verdict["credit"]:.2f}</span>'
                f'<span>${position.current_price:.2f}</span>'
                f'<span class="{cap_cls}">{captured:+.0%}</span>'
                f'<span>{position.dte if position.dte is not None else "--"}</span>'
                f'<span style="text-align:right"><span class="bvc-tag {cls}">{verdict["action"]}</span></span>'
                "</div>"
            )

    st.markdown(
        f'<div class="bvc-panel"><div class="bvc-panel-title"><span class="idx">2</span>Risk manager · managed brackets</div>'
        f"{header}{body}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Brackets are enforced by the bot each cycle, not resting at the exchange — "
        "Alpaca does not accept bracket orders on option legs. A stopped bot means unmanaged positions."
    )


def render_expectancy_panel(expectancy, account, settings, palette) -> None:
    """The expectancy formula, spelled out, with the bracket's breakeven hurdle."""
    hurdle = engine.breakeven_win_rate()
    st.markdown('<div class="bvc-panel-title"><span class="idx">5</span>Expectancy</div>', unsafe_allow_html=True)
    if expectancy.has_data:
        grid = st.columns(4)
        grid[0].metric("Win rate", fmt_pct(expectancy.p_win, 0), delta=f"{(expectancy.p_win - hurdle) * 100:+.0f}pts vs breakeven")
        grid[1].metric("Avg win", fmt_usd(expectancy.avg_win))
        grid[2].metric("Avg loss", fmt_usd(-expectancy.avg_loss))
        grid[3].metric("Profit factor", f"{expectancy.profit_factor:.2f}" if expectancy.profit_factor else "—")
        st.markdown(
            f'<div class="bvc-mono">E = ({expectancy.p_win:.3f} × {expectancy.avg_win:,.2f}) − '
            f"({expectancy.p_loss:.3f} × {expectancy.avg_loss:,.2f}) = "
            f"<b>{expectancy.expectancy:,.2f} USD</b> per trade over {expectancy.trades} closed</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No closed trades yet — expectancy appears once the first position settles.")
    st.caption(
        f"The {settings.profit_target_pct:.0%} target / {settings.stop_loss_multiple:.0%} stop geometry breaks even at a "
        f"{hurdle:.0%} win rate. Bracket placement alone cannot beat that — only selling volatility richer than what realises can."
    )


def render_monthly_income_chart(zar_by_month: dict, target: float) -> None:
    palette = theme()
    if not zar_by_month:
        st.caption("No settled trades yet.")
        return

    months = list(zar_by_month.keys())
    values = [zar_by_month[m] for m in months]
    colors = [palette["positive"] if v >= 0 else palette["negative"] for v in values]

    fig = go.Figure(
        go.Bar(
            x=months,
            y=values,
            marker=dict(color=colors, line=dict(width=2, color=palette["surface"])),
            text=[f"R{v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(color=palette["muted"], size=11),
            hovertemplate="%{x}<br>R%{y:,.0f}<extra></extra>",
        )
    )
    # The R10,000 baseline is drawn as a reference line, not a second series —
    # one measure, one axis.
    fig.add_hline(
        y=target,
        line=dict(color=palette["muted"], width=1, dash="dot"),
        annotation_text=f"Target R{target:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color=palette["muted"], size=11),
    )
    st.plotly_chart(style_figure(fig), use_container_width=True, config={"displayModeBar": False})


def render_cumulative_pnl_chart(closed: List[dict]) -> None:
    palette = theme()
    rows = [row for row in closed if row.get("closed_at") and row.get("pnl_zar")]
    if not rows:
        st.caption("No settled trades yet.")
        return

    rows.sort(key=lambda r: r["closed_at"])
    dates, cumulative, running = [], [], 0.0
    for row in rows:
        try:
            running += float(row["pnl_zar"])
        except (TypeError, ValueError):
            continue
        dates.append(row["closed_at"][:10])
        cumulative.append(running)

    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=cumulative,
            mode="lines+markers",
            line=dict(color=palette["series_1"], width=2),
            marker=dict(size=8, color=palette["series_1"], line=dict(width=2, color=palette["surface"])),
            fill="tozeroy",
            fillcolor=palette["series_1_fill"],
            hovertemplate="%{x}<br>R%{y:,.0f} cumulative<extra></extra>",
        )
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style_figure(fig), use_container_width=True, config={"displayModeBar": False})


# ======================================================================================
# Tab 2 — Scanner
# ======================================================================================
def render_scanner(bot: TradingBot) -> None:
    settings = bot.settings
    st.markdown("**Variance risk premium scan**")
    st.caption(
        f"Sell only when implied volatility is rich against realised: "
        f"IV Rank ≥ {settings.min_iv_rank:.0f} and IV − RV ≥ {settings.min_vrp * 100:.1f} vol points, "
        f"at {settings.target_dte} DTE and {settings.target_delta:.2f} delta."
    )

    if st.button("Run scan", type="primary"):
        with st.spinner("Pricing chains…"):
            try:
                st.session_state["scan"] = bot.scan()
                st.session_state["scan_at"] = datetime.now(timezone.utc)
                # The Deck tab is rendered earlier in this same pass, so it still
                # holds the previous result — rerun so both tabs agree.
                st.rerun()
            except BrokerError as exc:
                st.error(f"Scan failed: {exc}")

    # Same source as the deck, so a scan run from either place shows in both.
    snapshots, scanned_at, source = latest_scan(bot)
    if not snapshots:
        st.info("No scan yet. Run one above, or start the bot to populate it automatically.")
        return

    if scanned_at:
        st.caption(f"Scanned {scanned_at.strftime('%Y-%m-%d %H:%M:%S')} UTC · {source}")

    qualified = [s for s in snapshots if s.is_tradeable]
    columns = st.columns(3)
    columns[0].metric("Symbols scanned", len(snapshots))
    columns[1].metric("Qualified", len(qualified))
    best = max((s for s in snapshots if s.vrp is not None), key=lambda s: s.vrp, default=None)
    columns[2].metric("Widest VRP", f"{best.symbol} {best.vrp * 100:+.1f}pts" if best else "—")

    st.markdown("**IV − RV by symbol (annualised vol points)**")
    render_vrp_chart(snapshots)

    table = pd.DataFrame(
        [
            {
                "Symbol": s.symbol,
                "Spot": s.spot,
                "Expiry": s.expiration.isoformat() if s.expiration else "—",
                "DTE": s.dte,
                # Vols are stored as decimals; column_config formats the raw
                # value without scaling, so they are converted here.
                "IV %": (s.implied_vol * 100) if s.implied_vol is not None else None,
                "RV %": (s.reference_rv * 100) if s.reference_rv is not None else None,
                "VRP (pts)": (s.vrp * 100) if s.vrp is not None else None,
                "IV/RV": s.vrp_ratio,
                "IV Rank": s.iv_rank.value,
                "Rank source": s.iv_rank.label,
                "Status": "QUALIFIED" if s.is_tradeable else (s.reject_reason() or "—"),
            }
            for s in snapshots
        ]
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Spot": st.column_config.NumberColumn(format="$%.2f"),
            "IV %": st.column_config.NumberColumn(format="%.1f", help="ATM implied volatility, annualised"),
            "RV %": st.column_config.NumberColumn(format="%.1f", help="Yang-Zhang realised volatility, annualised"),
            "VRP (pts)": st.column_config.NumberColumn(format="%+.1f", help="IV − RV in annualised vol points"),
            "IV/RV": st.column_config.NumberColumn(format="%.2f"),
            # A plain number, not a progress bar: Streamlit renders ProgressColumn
            # in its accent red, which would read as "bad" for a high IV Rank —
            # exactly the value the strategy wants. Status carries the verdict.
            "IV Rank": st.column_config.NumberColumn(format="%.0f", help="0–100 within the trailing IV range"),
        },
    )

    proxy_count = sum(1 for s in snapshots if s.iv_rank.is_proxy and s.iv_rank.value is not None)
    if proxy_count:
        st.caption(
            f"ℹ️ {proxy_count} symbol(s) are ranking against a realised-volatility proxy while the local IV "
            "history builds. Each scan stores today's ATM IV; the rank becomes a true IV Rank once "
            f"{settings.iv_rank_min_samples} observations exist."
        )

    # IV vs RV history for one symbol — two series, so it carries a legend.
    st.divider()
    symbols = [s.symbol for s in snapshots]
    if symbols:
        chosen = st.selectbox("IV vs RV history", symbols)
        render_iv_rv_history(bot, chosen)


def render_vrp_chart(snapshots: List[engine.VRPSnapshot]) -> None:
    palette = theme()
    rows = [s for s in snapshots if s.vrp is not None]
    if not rows:
        st.caption("No symbol produced both an implied and a realised volatility.")
        return

    rows.sort(key=lambda s: s.vrp)
    values = [s.vrp * 100 for s in rows]
    # Diverging by sign: blue = we are paid for variance, red = we would be paying.
    colors = [palette["positive"] if v >= 0 else palette["negative"] for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=[s.symbol for s in rows],
            orientation="h",
            marker=dict(color=colors, line=dict(width=2, color=palette["surface"])),
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
            textfont=dict(color=palette["muted"], size=11),
            hovertemplate="%{y}<br>VRP %{x:+.1f} vol points<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=palette["axis"], width=1))
    fig.update_xaxes(showgrid=True, gridcolor=palette["grid"])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(style_figure(fig, height=max(240, 34 * len(rows))), use_container_width=True, config={"displayModeBar": False})


def render_iv_rv_history(bot: TradingBot, symbol: str) -> None:
    palette = theme()
    history = bot.iv_store.series(symbol)
    if len(history) < 2:
        st.caption(f"Only {len(history)} stored observation(s) for {symbol}. History accumulates one point per scan day.")
        return

    dates = [d for d, _ in history]
    ivs = [iv * 100 for _, iv in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ivs,
            name="Implied volatility",
            mode="lines",
            line=dict(color=palette["series_1"], width=2),
            hovertemplate="IV %{y:.1f}%<extra></extra>",
        )
    )

    # The stored realised vol for the same days, where it exists.
    rv_points = []
    for row in bot.iv_store.records(symbol):
        if row.get("rv"):
            try:
                rv_points.append((row["date"], float(row["rv"]) * 100))
            except (TypeError, ValueError):
                continue
    if rv_points:
        rv_points.sort()
        fig.add_trace(
            go.Scatter(
                x=[d for d, _ in rv_points],
                y=[v for _, v in rv_points],
                name="Realised volatility",
                mode="lines",
                line=dict(color=palette["series_2"], width=2),
                hovertemplate="RV %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(
        style_figure(fig, showlegend=len(fig.data) > 1),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption("The gap between the two lines is the premium the strategy is paid to carry.")


# ======================================================================================
# Tab 3 — Positions
# ======================================================================================
def render_positions(bot: TradingBot) -> None:
    settings = bot.settings
    fx_quote = bot.fx.get_rate()

    try:
        positions = bot.client.get_option_positions()
    except BrokerError as exc:
        st.error(f"Could not read positions: {exc}")
        return

    if not positions:
        st.info("No open option positions.")
        return

    live_records = {row["symbol"]: row for row in bot.trade_log.live_trades()}
    rows = []
    for position in positions:
        record = live_records.get(position.symbol)
        try:
            credit = float(record["credit"]) if record and record.get("credit") else abs(position.avg_entry_price)
        except (TypeError, ValueError):
            credit = abs(position.avg_entry_price)

        rows.append(
            {
                "Symbol": position.symbol,
                "Underlying": position.underlying or "—",
                "Type": (position.option_type or "").upper(),
                "Strike": position.strike,
                "Expiry": position.expiration.isoformat() if position.expiration else "—",
                "DTE": position.dte,
                "Qty": position.qty,
                "Credit": credit,
                "Mark": position.current_price,
                "Take profit": credit * (1 - settings.profit_target_pct),
                "Stop": credit * settings.stop_loss_price_multiple,
                "P&L (USD)": position.unrealized_pl,
                "P&L (ZAR)": position.unrealized_pl * fx_quote.rate,
            }
        )

    total_usd = sum(r["P&L (USD)"] for r in rows)
    columns = st.columns(4)
    columns[0].metric("Open positions", len(rows))
    columns[1].metric("Open P&L (USD)", fmt_usd(total_usd))
    columns[2].metric("Open P&L (ZAR)", fmt_zar(total_usd * fx_quote.rate))
    nearest = min((r["DTE"] for r in rows if r["DTE"] is not None), default=None)
    columns[3].metric("Nearest expiry", f"{nearest} DTE" if nearest is not None else "—")

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format="$%.2f"),
            "Credit": st.column_config.NumberColumn(format="$%.2f"),
            "Mark": st.column_config.NumberColumn(format="$%.2f"),
            "Take profit": st.column_config.NumberColumn(format="$%.2f", help=f"{settings.profit_target_pct:.0%} of credit"),
            "Stop": st.column_config.NumberColumn(format="$%.2f", help=f"{settings.stop_loss_multiple:.0%} of credit as loss"),
            "P&L (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "P&L (ZAR)": st.column_config.NumberColumn(format="R%.0f"),
        },
    )

    st.caption(
        "⚠️ Alpaca does not accept resting bracket orders on option legs. The take-profit and stop levels above "
        f"are enforced by the bot each cycle (every {settings.loop_interval_seconds}s while running), not by the "
        "exchange. If the bot is stopped, these positions are unmanaged."
    )

    st.divider()
    st.markdown("**Manual close**")
    left, right = st.columns([3, 1])
    target = left.selectbox("Position", [r["Symbol"] for r in rows], label_visibility="collapsed")
    if right.button("Close at market", type="secondary", use_container_width=True):
        try:
            bot.client.close_position(target)
            st.success(f"Close order sent for {target}.")
        except BrokerError as exc:
            st.error(f"Close failed: {exc}")


# ======================================================================================
# Tab 4 — Trade log
# ======================================================================================
def render_trade_log(bot: TradingBot) -> None:
    trades = bot.trade_log.all()
    if not trades:
        st.info("No trades recorded yet.")
        return

    closed = [t for t in trades if t.get("status") == "closed"]
    expectancy = engine.compute_expectancy(closed)

    columns = st.columns(5)
    columns[0].metric("Trades", len(trades))
    columns[1].metric("Closed", len(closed))
    columns[2].metric("Win rate", fmt_pct(expectancy.p_win, 0) if expectancy.has_data else "—")
    columns[3].metric("Total P&L", fmt_usd(expectancy.total_pnl) if expectancy.has_data else "—")
    columns[4].metric(
        "Profit factor",
        f"{expectancy.profit_factor:.2f}" if expectancy.profit_factor else "—",
        help="Gross wins ÷ gross losses",
    )

    if expectancy.has_data:
        st.markdown(
            f"**E = (P_win × W) − (P_loss × L)** = "
            f"({expectancy.p_win:.3f} × {expectancy.avg_win:,.2f}) − "
            f"({expectancy.p_loss:.3f} × {expectancy.avg_loss:,.2f}) = "
            f"**{expectancy.expectancy:,.2f} USD per trade**"
        )
        st.caption(
            f"Largest win {fmt_usd(expectancy.largest_win)} · largest loss {fmt_usd(-expectancy.largest_loss)}. "
            "A mechanical 30-delta seller should show a high win rate with a larger average loss — "
            "the edge lives in the expectancy, not the hit rate."
        )

    frame = pd.DataFrame(trades)
    display_columns = [
        "opened_at", "closed_at", "underlying", "symbol", "contracts", "strike", "expiration",
        "entry_delta", "entry_iv", "entry_iv_rank", "credit", "exit_debit", "pnl_usd", "pnl_zar",
        "status", "exit_reason",
    ]
    st.dataframe(
        frame[[c for c in display_columns if c in frame.columns]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download trade log (CSV)",
        data=frame.to_csv(index=False).encode(),
        file_name="brickvest_trade_log.csv",
        mime="text/csv",
    )


# ======================================================================================
# Tab 5 — Bot console
# ======================================================================================
def render_bot_console(bot: TradingBot) -> None:
    state: BotState = bot.state

    columns = st.columns(4)
    columns[0].metric("Mode", state.mode.upper())
    columns[1].metric("Cycles", state.cycles)
    columns[2].metric("Entries today", state.entries_today_count())
    columns[3].metric("Thread", "running" if bot.is_running else "stopped")

    if state.is_halted:
        st.error(
            f"**Emergency halt** — {state.halt_reason}\n\n"
            f"Halted at {state.halted_at}. No orders will be sent, and open positions are **not** being managed. "
            "Check the account at the broker, then clear the halt from the sidebar."
        )

    result = bot.last_result
    if result:
        st.markdown("**Last cycle**")
        st.caption(f"{result.started_at.strftime('%Y-%m-%d %H:%M:%S')} UTC — {result.summary()}")
        if result.blocked:
            for blocker in result.blocked:
                st.write(f"• {md_escape(blocker)}")
        if result.entries:
            st.success(f"Opened: {', '.join(e.get('symbol', '?') for e in result.entries)}")
        if result.exits:
            closed_labels = ", ".join("{} ({})".format(e.get("symbol"), e.get("reason")) for e in result.exits)
            st.info(f"Closed: {closed_labels}")

    st.divider()
    st.markdown("**Event feed**")
    if not state.events:
        st.caption("No events yet.")
        return

    icons = {"critical": "🔴", "error": "🟠", "info": "•"}
    for event in reversed(state.events[-60:]):
        st.write(f"{icons.get(event['level'], '•')} `{event['ts']}` {md_escape(event['message'])}")


# ======================================================================================
# Tab — Backtest
# ======================================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_history(symbols: tuple, start: str, end: str, _broker=None):
    """Price history for the backtester, cached for an hour.

    Two reasons this matters on a hosted deployment: the null-hypothesis run
    needs the same frames as the main run and must not fetch them twice, and
    Yahoo rate-limits shared cloud IP ranges — so every avoided download is one
    less chance of a 429 on a rerun.
    """
    import backtest as bt

    return bt.load_history(list(symbols), start, end, broker=_broker)


def render_backtest(bot: TradingBot) -> None:
    """Replay the configured rules over history, with the VRP assumption exposed."""
    import backtest as bt

    st.markdown('<div class="bvc-panel-title"><span class="idx">1</span>Historical replay</div>', unsafe_allow_html=True)
    st.caption(
        "The same rules the bot trades, run over historical prices. Option prices are modelled — "
        "no free source carries years of historical implied volatility — so the premium assumption "
        "below is the single most important input on this page."
    )

    defaults = bt.BacktestConfig.from_settings(bot.settings)
    universe = sorted(set(bot.settings.universe) | {"SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "EEM", "XLF"})

    row1 = st.columns([2.2, 1, 1, 1.4])
    symbols = row1[0].multiselect("Symbols", universe, default=[s for s in defaults.symbols if s in universe][:2])
    start = row1[1].text_input("Start", defaults.start_date)
    end = row1[2].text_input("End", defaults.end_date)
    strategy_options = ["ALL — compare", "short_put", "put_credit_spread", "iron_condor"]
    strategy = row1[3].selectbox(
        "Strategy", strategy_options,
        index=strategy_options.index(defaults.strategy) if defaults.strategy in strategy_options else 1,
    )
    compare_all = strategy == "ALL — compare"

    row2 = st.columns(4)
    short_delta = row2[0].slider("Short delta", 0.05, 0.45, float(defaults.short_delta), 0.01)
    dte_entry = row2[1].slider("Entry DTE", 20, 60, int(defaults.dte_entry), 1)
    dte_exit = row2[2].slider("Time exit DTE", 0, 30, int(defaults.dte_exit), 1)
    vol_rank = row2[3].slider("Min vol rank", 0.0, 90.0, float(defaults.vol_rank_threshold), 5.0)

    row3 = st.columns(4)
    profit_target = row3[0].slider("Profit target", 0.10, 0.90, float(defaults.profit_target), 0.05)
    stop_loss = row3[1].slider("Stop (× credit)", 0.5, 5.0, float(defaults.stop_loss), 0.25)
    capital = row3[2].number_input("Capital ($)", 1_000, 10_000_000, int(defaults.initial_capital), 1_000,
                                   help="A cash-secured put ties up strike × 100 — roughly $50k on SPY. "
                                        "Small accounts can only reach the defined-risk structures.")
    per_asset = row3[3].slider("Max per asset", 0.05, 0.50, float(defaults.max_allocation_per_asset), 0.05)

    with st.expander("Intraday entry window (live bot)"):
        win_cols = st.columns([1, 2, 2])
        window_on = win_cols[0].checkbox("Enabled", value=defaults.entry_window_enabled)
        win_start = win_cols[1].slider("Open trades from (min after bell)", 0, 240,
                                       int(defaults.entry_window_start_min), 5)
        win_end = win_cols[2].slider("…until (min after bell)", 0, 390,
                                     int(defaults.entry_window_end_min), 5)
        st.caption(
            "Enforced by the live bot against the exchange calendar, so half-days and DST are handled. "
            "**It cannot be replayed here** — this backtest runs on daily bars, which carry one price per "
            "day and no intraday timestamps. Free intraday history reaches back about 60 days, well short "
            "of a single 45-DTE cycle."
        )

    # The assumption that decides whether this measures an edge or measures noise.
    row4 = st.columns([2, 2, 2])
    vrp_points = row4[0].slider(
        "IV premium over RV (vol points)", 0.0, 8.0, float(defaults.vrp_points * 100), 0.5,
        help="Implied vol is modelled as realised vol plus this. Index options have historically "
             "paid 2–4 points. Zero means no variance risk premium exists.",
    ) / 100.0
    compare_null = row4[1].checkbox(
        "Also run the null hypothesis (0 points)", value=True,
        help="Runs the identical backtest with no premium. If the two curves look alike, "
             "the result is path luck rather than edge.",
    )
    run = row4[2].button("◈ RUN BACKTEST", use_container_width=True, type="primary")

    if run:
        if not symbols:
            st.error("Pick at least one symbol.")
            return
        cfg = bt.BacktestConfig(
            symbols=symbols, start_date=start, end_date=end,
            strategy="short_put" if compare_all else strategy,
            short_delta=short_delta, dte_entry=dte_entry, dte_exit=dte_exit,
            vol_rank_threshold=vol_rank, profit_target=profit_target, stop_loss=stop_loss,
            initial_capital=float(capital), max_allocation_per_asset=per_asset,
            vrp_points=vrp_points, risk_free_rate=bot.settings.risk_free_rate,
            max_margin_utilization=bot.settings.max_margin_utilization,
            entry_window_enabled=window_on,
            entry_window_start_min=win_start, entry_window_end_min=win_end,
        )
        try:
            broker = bot.client if bot.client.is_connected else None
            with st.spinner("Loading history…"):
                prices = load_backtest_history(tuple(symbols), start, end, _broker=broker)
            missing = [s for s in symbols if s not in prices]
            with st.spinner("Replaying…"):
                if compare_all:
                    # One price set, one balance, one rule set — only the
                    # structure differs, so the comparison isolates it.
                    comparison = bt.compare_strategies(cfg, prices)
                    st.session_state["bt_comparison"] = comparison
                    result = max(comparison.values(), key=lambda r: r.metrics.get("total_return", -9))
                    null = None
                else:
                    st.session_state["bt_comparison"] = None
                    result = bt.Backtester(cfg, prices).run()
                    null = None
                    if compare_null and vrp_points > 0:
                        # Same frames, same rules, no premium — the honest control.
                        null = bt.Backtester(replace(cfg, vrp_points=0.0), prices).run()
                if missing:
                    result.warnings.insert(0, f"No history for {', '.join(missing)} — excluded from the run.")
            st.session_state["bt_result"] = result
            st.session_state["bt_null"] = null
            st.session_state["bt_prices"] = prices
        except Exception as exc:
            st.error(md_escape(f"BACKTEST FAILED — {exc}"))
            st.caption(
                "History comes from yfinance, falling back to the broker feed. If this host blocks "
                "outbound HTTP, neither is reachable."
            )
            return

    result = st.session_state.get("bt_result")
    if result is None:
        st.info("Set the parameters above and press RUN BACKTEST.")
        return

    comparison = st.session_state.get("bt_comparison")
    if comparison:
        render_strategy_comparison(comparison, bot)
        st.divider()
    render_backtest_results(result, st.session_state.get("bt_null"), bot)

    prices = st.session_state.get("bt_prices")
    if prices:
        st.divider()
        render_robustness(result, prices, bot)


def render_strategy_comparison(comparison: dict, bot: TradingBot) -> None:
    """Every structure over the same history, ranked, with overlaid equity curves."""
    import backtest as bt

    palette = theme()
    rows = bt.comparison_table(comparison)
    if not rows:
        st.info("No comparable results.")
        return

    st.markdown('<div class="bvc-panel-title"><span class="idx">A</span>Strategy comparison</div>',
                unsafe_allow_html=True)

    grid = "grid-template-columns:8.5rem 5rem 4.5rem 4.5rem 5rem 4rem 4rem 6rem;"
    header = (
        f'<div class="bvc-row bvc-head" style="{grid}">'
        "<span>STRUCTURE</span><span style='text-align:right'>RETURN</span>"
        "<span style='text-align:right'>CAGR</span><span style='text-align:right'>SHARPE</span>"
        "<span style='text-align:right'>MAX DD</span><span style='text-align:right'>TRADES</span>"
        "<span style='text-align:right'>WIN</span><span style='text-align:right'>E/TRADE</span></div>"
    )
    body = ""
    for i, row in enumerate(rows):
        rank_cls = "t-chrome" if i == 0 else "t-accent"
        ret_cls = "t-good" if row["total_return"] >= 0 else "t-crit"
        if row["trades"] == 0:
            body += (
                f'<div class="bvc-row" style="{grid}">'
                f'<span class="{rank_cls}">{row["strategy"]}</span>'
                f'<span class="t-idle" style="grid-column:span 7">no trades — see the note below</span></div>'
            )
            continue
        body += (
            f'<div class="bvc-row" style="{grid}">'
            f'<span class="{rank_cls}">{row["strategy"]}</span>'
            f'<span class="{ret_cls}" style="text-align:right">{row["total_return"] * 100:+.1f}%</span>'
            f'<span style="text-align:right">{row["cagr"] * 100:.1f}%</span>'
            f'<span style="text-align:right">{row["sharpe"]:.2f}</span>'
            f'<span class="t-crit" style="text-align:right">{row["max_drawdown"] * 100:.1f}%</span>'
            f'<span style="text-align:right">{row["trades"]}</span>'
            f'<span style="text-align:right">{row["win_rate"] * 100:.0f}%</span>'
            f'<span style="text-align:right">${row["expectancy"]:,.0f}</span></div>'
        )
    st.markdown(f'<div class="bvc-panel">{header}{body}</div>', unsafe_allow_html=True)

    # Overlaid curves — same capital, same history, so the axis is shared and
    # the comparison is direct. Colour follows the structure, not its rank.
    series_colours = {
        "short_put": palette["series_1"],
        "put_credit_spread": palette["series_2"],
        "iron_condor": "#199e70",
    }
    fig = go.Figure()
    for name, result in comparison.items():
        if result.nav.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=result.nav.index, y=result.nav.values, mode="lines", name=name,
                line=dict(color=series_colours.get(name, palette["muted"]), width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
            )
        )
    first = next(iter(comparison.values()))
    fig.add_hline(
        y=first.config.initial_capital,
        line=dict(color=palette["muted"], width=1, dash="dot"),
        annotation_text="Starting capital", annotation_position="bottom right",
        annotation_font=dict(color=palette["muted"], size=11),
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style_figure(fig, height=360, showlegend=True),
                    use_container_width=True, config={"displayModeBar": False})

    notes = []
    for name, result in comparison.items():
        for note in result.warnings:
            notes.append(f"**{name}** — {note}")
    for note in notes[:6]:
        st.caption(md_escape(note))
    st.caption(
        "Same prices, same starting capital, same rules — only the structure differs. Return on capital "
        "is the honest comparison here: a cash-secured put posts the full strike as collateral, so it can "
        "look safe and still be the worst use of the money."
    )


SENSITIVITY_SWEEPS = {
    "short_delta": [0.16, 0.20, 0.25, 0.30, 0.35, 0.40],
    "profit_target": [0.25, 0.35, 0.50, 0.65, 0.75],
    "stop_loss": [1.0, 1.5, 2.0, 3.0, 4.0],
    "vol_rank_threshold": [0.0, 20.0, 40.0, 50.0, 60.0, 80.0],
    "dte_entry": [30, 38, 45, 52, 60],
    "vrp_points": [0.0, 0.01, 0.02, 0.03, 0.05, 0.07],
}


def render_robustness(result, prices, bot: TradingBot) -> None:
    """Overfitting diagnostics: sample size, out-of-sample, walk-forward, sweeps."""
    import backtest as bt

    palette = theme()
    cfg = result.config

    st.markdown('<div class="bvc-panel-title"><span class="idx">R</span>Robustness · is this an edge or a fit?</div>',
                unsafe_allow_html=True)

    sweep_param = st.selectbox("Sweep parameter", list(SENSITIVITY_SWEEPS), index=0)
    if not st.button("◈ RUN ROBUSTNESS CHECKS", use_container_width=False):
        st.caption(
            "Runs about a dozen extra replays: a chronological in/out-of-sample split, four "
            "walk-forward folds, and a sweep of the chosen parameter."
        )
        return

    with st.spinner("Splitting, walking forward and sweeping…"):
        adequacy = bt.sample_adequacy(result)
        split = bt.split_sample(cfg, prices)
        folds = bt.walk_forward(cfg, prices, folds=4)
        sweep = bt.sensitivity(cfg, prices, sweep_param, SENSITIVITY_SWEEPS[sweep_param])
        verdict = bt.plateau_score(sweep, getattr(cfg, sweep_param))

    # ---- sample adequacy --------------------------------------------------
    verdict_class = {"adequate": "t-good", "thin": "t-warn",
                     "insufficient": "t-crit", "no trades": "t-crit"}[adequacy["verdict"]]
    cells = [
        ("TRADES", f'{adequacy["trades"]}'),
        ("CONCURRENCY", f'{adequacy.get("concurrency", 0):.2f}×'),
        ("EFFECTIVE N", f'{adequacy.get("effective_trades", 0):.0f}'),
        ("FREE PARAMS", f'{adequacy["parameters"]}'),
        ("N / PARAM", f'<span class="{verdict_class}">{adequacy["trades_per_parameter"]:.1f}</span>'),
        ("EVIDENCE", f'<span class="{verdict_class}">{adequacy["verdict"].upper()}</span>'),
    ]
    html = "".join(f'<div class="bvc-cell"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells)
    st.markdown(f'<div class="bvc-strip">{html}</div>', unsafe_allow_html=True)
    st.caption(
        "Overlapping positions are not independent observations, so the effective count divides the "
        "trade count by average concurrency. Below ~10 trades per free parameter, the result cannot "
        "distinguish an edge from noise no matter how good it looks."
    )

    # ---- in / out of sample ----------------------------------------------
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="bvc-panel-title">In-sample vs out-of-sample</div>', unsafe_allow_html=True)
        grid = "grid-template-columns:7rem 5rem 5rem 5rem 4rem;"
        rows = (
            f'<div class="bvc-row bvc-head" style="{grid}"><span>SLICE</span>'
            "<span style='text-align:right'>CAGR</span><span style='text-align:right'>SHARPE</span>"
            "<span style='text-align:right'>MAX DD</span><span style='text-align:right'>N</span></div>"
        )
        for label, key in (("in-sample", "in_sample"), ("out-of-sample", "out_of_sample")):
            d = split[key]
            rows += (
                f'<div class="bvc-row" style="{grid}"><span class="t-accent">{label}</span>'
                f'<span style="text-align:right">{d["cagr"] * 100:.1f}%</span>'
                f'<span style="text-align:right">{d["sharpe"]:.2f}</span>'
                f'<span class="t-crit" style="text-align:right">{d["max_drawdown"] * 100:.1f}%</span>'
                f'<span style="text-align:right">{d["trades"]}</span></div>'
            )
        st.markdown(f'<div class="bvc-panel">{rows}</div>', unsafe_allow_html=True)
        decay = split["cagr_decay"]
        if decay is None:
            st.caption("One of the slices took no trades — the split is inconclusive.")
        elif decay < -0.05:
            st.markdown(
                f'<span class="t-crit bvc-mono">CAGR fell {abs(decay) * 100:.1f} points out of sample — '
                "the classic overfitting signature.</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<span class="t-good bvc-mono">CAGR held up out of sample ({decay * 100:+.1f} pts).</span>',
                unsafe_allow_html=True)

    with right:
        st.markdown('<div class="bvc-panel-title">Walk-forward folds</div>', unsafe_allow_html=True)
        grid = "grid-template-columns:3rem 9rem 5rem 5rem 4rem;"
        rows = (
            f'<div class="bvc-row bvc-head" style="{grid}"><span>#</span><span>PERIOD</span>'
            "<span style='text-align:right'>RETURN</span><span style='text-align:right'>SHARPE</span>"
            "<span style='text-align:right'>N</span></div>"
        )
        for f in folds:
            cls = "t-good" if f["total_return"] >= 0 else "t-crit"
            rows += (
                f'<div class="bvc-row" style="{grid}"><span class="t-idle">{f["fold"]}</span>'
                f'<span>{f["start"]} → {f["end"]}</span>'
                f'<span class="{cls}" style="text-align:right">{f["total_return"] * 100:+.1f}%</span>'
                f'<span style="text-align:right">{f["sharpe"]:.2f}</span>'
                f'<span style="text-align:right">{f["trades"]}</span></div>'
            )
        st.markdown(f'<div class="bvc-panel">{rows}</div>', unsafe_allow_html=True)
        losers = sum(1 for f in folds if f["total_return"] < 0)
        st.caption(
            f"{len(folds) - losers} of {len(folds)} folds profitable. One good regime can carry a "
            "multi-year total — an edge should recur across folds, not live in one of them."
        )

    # ---- sensitivity sweep -------------------------------------------------
    st.markdown(f'<div class="bvc-panel-title">Sensitivity · {sweep_param}</div>', unsafe_allow_html=True)
    chosen = getattr(cfg, sweep_param)
    xs = [r["value"] for r in sweep]
    ys = [r["total_return"] * 100 for r in sweep]
    fig = go.Figure(
        go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=palette["series_1"], width=2),
            marker=dict(size=9, color=[palette["chrome"] if abs(x - chosen) < 1e-9 else palette["series_1"] for x in xs],
                        line=dict(width=2, color=palette["surface"])),
            hovertemplate=f"{sweep_param}=%{{x}}<br>%{{y:.1f}}%<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=palette["muted"], width=1, dash="dot"))
    fig.update_yaxes(title_text="total return %", title_font=dict(size=11, color=palette["muted"]))
    st.plotly_chart(style_figure(fig, height=300), use_container_width=True, config={"displayModeBar": False})

    tone = ("t-crit" if "SPIKE" in verdict["verdict"] else
            "t-warn" if "fragile" in verdict["verdict"] else "t-good")
    st.markdown(
        f'<span class="{tone} bvc-mono">{verdict["verdict"].upper()}</span>'
        f'<span class="t-idle bvc-mono"> · your setting ({chosen}) is highlighted amber · '
        f'{verdict.get("share_positive", 0) * 100:.0f}% of the sweep is profitable</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "A real edge sits on a plateau: nudging the parameter moves the result a little. A curve fit "
        "sits on a spike — the chosen value is a peak surrounded by much worse neighbours, which means "
        "it was picked to fit noise. If a filter's sweep is flat or downward-sloping, that filter is "
        "not earning its place."
    )


def render_backtest_results(result, null, bot: TradingBot) -> None:
    """Metric strip, equity curve, drawdown and the trade record."""
    palette = theme()
    m = result.metrics
    fx_quote = bot.fx.get_rate()

    cells = [
        ("TOTAL RETURN", _signed(m["total_return"] * 100, palette) + "%"),
        ("CAGR", f'{m["cagr"] * 100:.2f}%'),
        ("SHARPE", f'{m["sharpe"]:.2f}'),
        ("MAX DD", f'<span class="t-crit">{m["max_drawdown"] * 100:.1f}%</span>'),
        ("TRADES", f'{m["trades"]}'),
        ("WIN RATE", f'{m["win_rate"] * 100:.0f}% / {m["breakeven_win_rate"] * 100:.0f}%'),
        ("E / TRADE", _signed(m["expectancy"], palette, "$")),
        ("FINAL NAV", f'${m["final_nav"]:,.0f}'),
    ]
    html = "".join(f'<div class="bvc-cell"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in cells)
    st.markdown(f'<div class="bvc-strip">{html}</div>', unsafe_allow_html=True)

    monthly_zar = (m["expectancy"] * m["trades"] / max(m["years"] * 12, 1e-9)) * fx_quote.rate
    st.markdown(
        f'<div class="bvc-footer">{m["start"]} → {m["end"]} · {m["years"]:.1f} yrs · '
        f'avg hold {m["avg_days_held"]:.0f} days · '
        f'implied ≈ <b>R{monthly_zar:,.0f}/month</b> at {fx_quote.rate:.2f} '
        f'(target R{bot.settings.monthly_target_zar:,.0f})</div>',
        unsafe_allow_html=True,
    )

    if m["trades"] and m["win_rate"] < m["breakeven_win_rate"] and m["expectancy"] > 0:
        # Not a contradiction: the breakeven rate assumes every trade ends at
        # the target or the stop. The time exit is a third outcome that closes
        # positions at a partial profit or loss, so the realised win/loss
        # magnitudes differ from the nominal geometry.
        st.caption(
            f"Win rate {m['win_rate']:.0%} sits below the {m['breakeven_win_rate']:.0%} nominal hurdle yet "
            "expectancy is positive — the hurdle assumes every trade ends at the target or the stop, while "
            f"the {result.config.dte_exit}-DTE time exit closes positions partway, changing the average "
            "win and loss."
        )

    for note in result.warnings:
        st.warning(md_escape(note))

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="bvc-panel-title"><span class="idx">2</span>Equity curve</div>', unsafe_allow_html=True)
        render_equity_curve(result, null)
    with right:
        st.markdown('<div class="bvc-panel-title"><span class="idx">3</span>Drawdown</div>', unsafe_allow_html=True)
        render_drawdown(result)

    if not result.trades:
        return

    st.markdown('<div class="bvc-panel-title"><span class="idx">4</span>Trades</div>', unsafe_allow_html=True)
    mix = {}
    for trade in result.trades:
        mix[trade["exit_reason"]] = mix.get(trade["exit_reason"], 0) + 1
    st.caption(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1])))

    frame = pd.DataFrame(result.trades)
    st.dataframe(frame, use_container_width=True, hide_index=True, height=320)
    st.download_button(
        "Download backtest trades (CSV)",
        data=frame.to_csv(index=False).encode(),
        file_name="brickvest_backtest_trades.csv",
        mime="text/csv",
    )


def render_equity_curve(result, null) -> None:
    """NAV over time. Two series when the null hypothesis was run alongside."""
    palette = theme()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.nav.index, y=result.nav.values, mode="lines", name="With VRP",
            line=dict(color=palette["series_1"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>With VRP</extra>",
        )
    )
    if null is not None and not null.nav.empty:
        fig.add_trace(
            go.Scatter(
                x=null.nav.index, y=null.nav.values, mode="lines", name="Null (no VRP)",
                line=dict(color=palette["series_2"], width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Null</extra>",
            )
        )
    fig.add_hline(
        y=result.config.initial_capital,
        line=dict(color=palette["muted"], width=1, dash="dot"),
        annotation_text="Starting capital", annotation_position="bottom right",
        annotation_font=dict(color=palette["muted"], size=11),
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(
        style_figure(fig, height=340, showlegend=len(fig.data) > 1),
        use_container_width=True, config={"displayModeBar": False},
    )
    if null is not None:
        st.caption(
            "Two curves, one difference: the assumed premium. If they track each other, the strategy "
            "is not harvesting an edge in this window."
        )


def render_drawdown(result) -> None:
    palette = theme()
    series = result.metrics.get("drawdown_series")
    if series is None or series.empty:
        st.caption("No drawdown data.")
        return
    fig = go.Figure(
        go.Scatter(
            x=series.index, y=series.values * 100, mode="lines",
            line=dict(color=palette["critical"], width=1.5),
            fill="tozeroy", fillcolor="rgba(208,59,59,0.18)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
        )
    )
    st.plotly_chart(style_figure(fig, height=340), use_container_width=True, config={"displayModeBar": False})


# ======================================================================================
# Tab 6 — Settings
# ======================================================================================
def render_settings(bot: TradingBot) -> None:
    settings = bot.settings

    if not settings.credentials_present:
        st.warning("**No Alpaca credentials found.** The terminal runs read-only until they are set.")
        st.code(
            "# .streamlit/secrets.toml  (Streamlit Community Cloud → Settings → Secrets)\n"
            'ALPACA_API_KEY = "PK…"\n'
            'ALPACA_SECRET_KEY = "…"\n'
            'ALPACA_PAPER = "true"\n\n'
            "# or as environment variables (Hugging Face Spaces → Settings → Variables & secrets)\n"
            "export ALPACA_API_KEY=PK…\n"
            "export ALPACA_SECRET_KEY=…",
            language="toml",
        )

    st.markdown("**Guardrails currently in force**")
    guardrails = pd.DataFrame(
        [
            ("Margin utilisation ceiling", f"{settings.max_margin_utilization:.0%} of equity, checked before every entry"),
            ("Equity floor", fmt_usd(settings.min_equity_usd)),
            ("Max open positions", str(settings.max_open_positions)),
            ("Max new positions / day", str(settings.max_new_positions_per_day)),
            ("One position per underlying", "yes" if settings.one_position_per_underlying else "no"),
            ("Liquidity filter", f"credit ≥ {fmt_usd(settings.min_credit_usd)}, spread ≤ {settings.max_spread_pct:.0%} of mid"),
            ("Fail-safe", "any broker or data failure halts the bot and flags the UI"),
            ("Dry run", "ON — no orders sent" if settings.dry_run else "off"),
        ],
        columns=["Guardrail", "Setting"],
    )
    st.dataframe(guardrails, use_container_width=True, hide_index=True)

    st.markdown("**Strategy**")
    strategy = pd.DataFrame(
        [
            ("Strategy", settings.strategy),
            ("Target DTE", f"{settings.target_dte} (window {settings.dte_min}–{settings.dte_max})"),
            ("Target delta", f"{settings.target_delta:.2f} ± {settings.delta_tolerance:.2f}"),
            ("Minimum IV Rank", f"{settings.min_iv_rank:.0f}"),
            ("Minimum VRP", f"{settings.min_vrp * 100:.1f} vol points"),
            ("Profit target", f"{settings.profit_target_pct:.0%} of credit"),
            ("Stop loss", f"{settings.stop_loss_multiple:.0%} of credit ⇒ buy back at {settings.stop_loss_price_multiple:.1f}× credit"),
            ("Time exit", f"{settings.time_exit_dte} DTE"),
            ("Contracts per trade", str(settings.contracts_per_trade)),
            ("Universe", ", ".join(settings.universe)),
        ],
        columns=["Parameter", "Value"],
    )
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    with st.expander("Full resolved configuration (secrets redacted)"):
        st.json(settings.as_dict())

    st.caption(
        "Every value above is read from environment variables or `st.secrets` at startup — see `config.py` for "
        "the variable names. Change them at the host and restart the app."
    )


# ======================================================================================
# Main
# ======================================================================================
def main() -> None:
    bot = get_bot()
    inject_terminal_css()
    render_sidebar(bot)

    palette = theme()
    health = bot.client.health
    if bot.state.is_halted:
        banner, colour = "HALTED", palette["critical"]
    else:
        banner, colour = (
            {"ok": ("ONLINE", palette["good"]),
             "degraded": ("DEGRADED", palette["warning"]),
             "down": ("OFFLINE", palette["critical"])}[health.status]
        )

    now = datetime.now(timezone.utc)
    try:
        market_open = bot.client.is_market_open() if bot.client.is_connected else None
    except BrokerError:
        market_open = None
    if market_open is None:
        session, session_colour = "SESSION —", palette["muted"]
    elif market_open:
        session, session_colour = "MKT OPEN", palette["good"]
    else:
        session, session_colour = "MKT CLOSED", palette["muted"]

    st.markdown(
        f"""
        <div style="font-family:{MONO};background:{palette['panel']};
                    border:1px solid {palette['border']};border-left:3px solid {palette['chrome']};
                    padding:.4rem .8rem;margin-bottom:.5rem;display:flex;
                    justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;">
          <span style="font-weight:700;letter-spacing:.28em;font-size:.92rem;color:{palette['chrome']};">
            BRICKVEST&nbsp;CAPITAL&nbsp;TERMINAL
          </span>
          <span style="font-size:.63rem;letter-spacing:.13em;color:{palette['muted']};">
            VRP&nbsp;·&nbsp;{bot.settings.target_dte}D&nbsp;·&nbsp;{bot.settings.target_delta:.2f}&Delta;
            &nbsp;·&nbsp;IVR&gt;{bot.settings.min_iv_rank:.0f}
            &nbsp;·&nbsp;ALPACA&nbsp;{'PAPER' if bot.settings.paper else 'LIVE'}
          </span>
          <span style="font-size:.68rem;letter-spacing:.12em;font-variant-numeric:tabular-nums;">
            <span style="color:{session_colour};font-weight:700;">{session}</span>
            <span style="color:{palette['muted']};">&nbsp;|&nbsp;{now:%Y-%m-%d %H:%M:%S}Z&nbsp;|&nbsp;</span>
            <span style="color:{colour};font-weight:700;">{banner}</span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bot.state.is_halted:
        st.error(f"EMERGENCY HALT — {bot.state.halt_reason}. Positions are not being managed.")

    if not bot.settings.paper:
        st.warning("⚠️ Live trading mode is enabled. Orders will be sent to a funded account.")

    tabs = st.tabs(["Deck", "Scanner", "Positions", "Trade log", "Backtest", "Bot", "Settings"])
    with tabs[0]:
        render_deck(bot)
    with tabs[1]:
        render_scanner(bot)
    with tabs[2]:
        render_positions(bot)
    with tabs[3]:
        render_trade_log(bot)
    with tabs[4]:
        render_backtest(bot)
    with tabs[5]:
        render_bot_console(bot)
    with tabs[6]:
        render_settings(bot)


if __name__ == "__main__":
    main()
