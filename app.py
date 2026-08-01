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
    Bot        — start/stop, halt state, and the event feed
    Settings   — every guardrail currently in force

Run locally with ``streamlit run app.py``; on Streamlit Community Cloud point the
app at this file and put the Alpaca keys in the Secrets panel.
"""

from __future__ import annotations

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
    initial_sidebar_state="expanded",
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
        "surface": "#0b0e13",
        "panel": "#11151c",
        "border": "#1e2632",
        "text": "#dfe6ee",
        "muted": "#7d8896",
        "grid": "#1a212b",
        "axis": "#2a3340",
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
        "border": "#e1e0d9",
        "text": "#0b0b0b",
        "muted": "#898781",
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
          /* Streamlit reads .streamlit/config.toml relative to the working
             directory, so a run started from elsewhere would keep the light
             chrome. These rules make the deck dark regardless of how it was
             launched. */
          .stApp, [data-testid="stAppViewContainer"] {{ background: {p["surface"]}; }}
          [data-testid="stHeader"] {{ background: transparent; }}
          [data-testid="stSidebar"] {{
              background: {p["panel"]};
              border-right: 1px solid {p["border"]};
          }}
          [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] label, [data-testid="stSidebar"] li {{
              color: {p["text"]};
          }}
          [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
          [data-testid="stSidebar"] small {{ color: {p["muted"]} !important; }}
          /* Controls need their own colours — inheriting the text colour alone
             leaves light-on-light buttons that cannot be read. */
          .stButton > button, .stDownloadButton > button {{
              background: {p["surface"]}; color: {p["text"]};
              border: 1px solid {p["border"]}; font-family: {MONO};
              font-size: .78rem; letter-spacing: .04em;
          }}
          .stButton > button:hover, .stDownloadButton > button:hover {{
              border-color: {p["series_1"]}; color: {p["series_1"]};
          }}
          .stButton > button:disabled {{ color: {p["muted"]}; border-color: {p["border"]}; }}
          [data-baseweb="select"] > div, [data-baseweb="input"] > div,
          [data-testid="stSelectbox"] div[role="combobox"],
          [data-testid="stNumberInput"] input,
          [data-testid="stSelectbox"] div[data-baseweb="select"] div {{
              background-color: {p["surface"]} !important;
              border-color: {p["border"]} !important;
              color: {p["text"]} !important;
          }}
          [data-baseweb="popover"] li {{
              background-color: {p["panel"]} !important; color: {p["text"]} !important;
          }}
          [data-baseweb="select"] svg {{ fill: {p["muted"]}; }}
          /* Figures in tables, metrics and code align only with tabular numerals. */
          [data-testid="stMetricValue"], [data-testid="stDataFrame"], .bvc-mono {{
              font-family: {MONO};
              font-variant-numeric: tabular-nums;
          }}
          [data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
          [data-testid="stMetricLabel"] p {{
              text-transform: uppercase; letter-spacing: .08em;
              font-size: .68rem; color: {p["muted"]};
          }}
          .bvc-panel {{
              border: 1px solid {p["border"]}; border-radius: 6px;
              background: {p["panel"]}; padding: .55rem .8rem .7rem;
              margin-bottom: .6rem;
          }}
          .bvc-panel-title {{
              font-family: {MONO}; font-size: .7rem; font-weight: 700;
              text-transform: uppercase; letter-spacing: .14em;
              color: {p["muted"]}; border-bottom: 1px solid {p["border"]};
              padding-bottom: .35rem; margin-bottom: .5rem;
          }}
          .bvc-strip {{
              display: flex; flex-wrap: wrap; gap: 0;
              border: 1px solid {p["border"]}; border-radius: 6px;
              background: {p["panel"]}; overflow: hidden; margin-bottom: .75rem;
          }}
          .bvc-cell {{
              flex: 1 1 118px; padding: .5rem .8rem;
              border-right: 1px solid {p["border"]};
          }}
          .bvc-cell:last-child {{ border-right: none; }}
          .bvc-cell .k {{
              font-family: {MONO}; font-size: .62rem; letter-spacing: .12em;
              text-transform: uppercase; color: {p["muted"]};
          }}
          .bvc-cell .v {{
              font-family: {MONO}; font-size: 1.05rem; font-weight: 700;
              color: {p["text"]}; font-variant-numeric: tabular-nums;
          }}
          .bvc-row {{
              display: grid; align-items: center; gap: .5rem;
              font-family: {MONO}; font-size: .78rem;
              padding: .3rem 0; border-bottom: 1px solid {p["border"]};
              font-variant-numeric: tabular-nums;
          }}
          .bvc-row:last-child {{ border-bottom: none; }}
          .bvc-head {{
              color: {p["muted"]}; font-size: .64rem; letter-spacing: .1em;
              text-transform: uppercase; border-bottom: 1px solid {p["border"]};
          }}
          .bvc-tag {{
              font-family: {MONO}; font-size: .68rem; font-weight: 700;
              letter-spacing: .06em; padding: .1rem .45rem; border-radius: 3px;
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
          .bvc-footer {{
              font-family: {MONO}; font-size: .7rem; color: {p["muted"]};
              border-top: 1px solid {p["border"]}; padding-top: .5rem;
              margin-top: .4rem; letter-spacing: .04em;
          }}
          /* A live risk figure must never blink — motion on a number you are
             about to act on costs legibility exactly when it matters most. */
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
        st.markdown("### Brickvestcapitalterminal")
        st.caption("Variance risk premium harvesting · Alpaca paper")

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
        st.markdown('<div class="bvc-panel-title">Realised income by month (ZAR)</div>', unsafe_allow_html=True)
        render_monthly_income_chart(zar_by_month, settings.monthly_target_zar)
    with right:
        st.markdown('<div class="bvc-panel-title">Cumulative realised P&L (ZAR)</div>', unsafe_allow_html=True)
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
                st.error(f"SCAN FAILED — {exc}")

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
        f'<div class="bvc-panel"><div class="bvc-panel-title">◆ Target acquisition · IV−RV'
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
            f'<div class="bvc-panel"><div class="bvc-panel-title">▣ Risk manager</div>'
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
        f'<div class="bvc-panel"><div class="bvc-panel-title">▣ Risk manager · managed brackets</div>'
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
    st.markdown('<div class="bvc-panel-title">Σ Expectancy</div>', unsafe_allow_html=True)
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
                st.write(f"• {blocker}")
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
        st.write(f"{icons.get(event['level'], '•')} `{event['ts']}` {event['message']}")


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

    st.markdown(
        f"""
        <div style="font-family:{MONO};border:1px solid {palette['border']};border-radius:6px;
                    background:{palette['panel']};padding:.55rem .9rem;margin-bottom:.75rem;
                    display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;">
          <span style="font-weight:700;letter-spacing:.22em;font-size:.95rem;color:{palette['text']};">
            BRICKVEST CAPITAL TERMINAL
          </span>
          <span style="font-size:.7rem;letter-spacing:.1em;color:{palette['muted']};">
            VARIANCE RISK PREMIUM · {bot.settings.target_dte} DTE · {bot.settings.target_delta:.2f}Δ ·
            IVR &gt; {bot.settings.min_iv_rank:.0f} · ALPACA {'PAPER' if bot.settings.paper else 'LIVE'}
          </span>
          <span style="font-size:.8rem;font-weight:700;letter-spacing:.12em;color:{colour};">
            STATUS: {banner}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bot.state.is_halted:
        st.error(f"EMERGENCY HALT — {bot.state.halt_reason}. Positions are not being managed.")

    if not bot.settings.paper:
        st.warning("⚠️ Live trading mode is enabled. Orders will be sent to a funded account.")

    tabs = st.tabs(["Deck", "Scanner", "Positions", "Trade log", "Bot", "Settings"])
    with tabs[0]:
        render_deck(bot)
    with tabs[1]:
        render_scanner(bot)
    with tabs[2]:
        render_positions(bot)
    with tabs[3]:
        render_trade_log(bot)
    with tabs[4]:
        render_bot_console(bot)
    with tabs[5]:
        render_settings(bot)


if __name__ == "__main__":
    main()
