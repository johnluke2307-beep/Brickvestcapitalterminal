"""
Brickvestcapitalterminal — Streamlit control surface.

The dashboard is a *view* over the engine and the bot: it computes nothing on its
own beyond formatting, so the numbers on screen are the same numbers the
execution loop trades on.

Six tabs:
    Overview   — capital, margin headroom, and Rand progress toward R10,000/month
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
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series_1": "#2a78d6",  # blue  — implied vol / positive VRP / income
        "series_2": "#eb6834",  # orange— realised vol
        "series_1_fill": "rgba(42,120,214,0.12)",
        "positive": "#2a78d6",
        "negative": "#d03b3b",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series_1": "#3987e5",
        "series_2": "#d95926",
        "series_1_fill": "rgba(57,135,229,0.16)",
        "positive": "#3987e5",
        "negative": "#d03b3b",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
}


def theme() -> dict:
    """Chart palette matching the active Streamlit theme."""
    base = st.session_state.get("chart_theme")
    if base not in THEMES:
        try:
            base = st.get_option("theme.base") or "light"
        except Exception:
            base = "light"
    return THEMES.get(base, THEMES["light"])


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
            columns = st.columns(2)
            if columns[0].button("Start", use_container_width=True, disabled=bot.is_running):
                bot.start()
                st.rerun()
            if columns[1].button("Stop", use_container_width=True, disabled=not bot.is_running):
                bot.stop()
                st.rerun()
            if st.button("Run one cycle now", use_container_width=True):
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
        st.session_state["chart_theme"] = st.selectbox(
            "Chart theme", ["light", "dark"], index=0 if theme() is THEMES["light"] else 1
        )


# ======================================================================================
# Tab 1 — Overview
# ======================================================================================
def render_overview(bot: TradingBot) -> None:
    settings = bot.settings
    fx_quote = bot.fx.get_rate()

    try:
        account = bot.client.get_account()
    except BrokerError as exc:
        st.error(f"Could not read the account: {exc}")
        st.info("The bot halts rather than trading on stale data. Fix the connection, then resume from the sidebar.")
        account = None

    closed = bot.trade_log.closed_trades()
    expectancy = engine.compute_expectancy(closed)
    zar_by_month = engine.monthly_pnl(closed, "zar")
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    month_zar = zar_by_month.get(month_key, 0.0)
    progress = month_zar / settings.monthly_target_zar if settings.monthly_target_zar else 0.0

    # ---- headline tiles ---------------------------------------------------
    columns = st.columns(4)
    if account:
        columns[0].metric(
            "Account equity",
            fmt_usd(account.equity),
            delta=f"{account.day_pnl:+,.2f} today" if account.last_equity else None,
        )
        columns[1].metric("Equity in Rand", fmt_zar(account.equity * fx_quote.rate))
    else:
        columns[0].metric("Account equity", "—")
        columns[1].metric("Equity in Rand", "—")

    columns[2].metric(
        f"{datetime.now(timezone.utc).strftime('%B')} realised",
        fmt_zar(month_zar),
        delta=f"{progress:.0%} of target",
    )
    columns[3].metric(
        "Expectancy / trade",
        fmt_usd(expectancy.expectancy) if expectancy.has_data else "—",
        help="E = (P_win × W) − (P_loss × L) over closed trades",
    )

    # ---- the R10,000 baseline --------------------------------------------
    st.markdown(f"**Monthly target — {fmt_zar(settings.monthly_target_zar)}**")
    st.progress(min(max(progress, 0.0), 1.0))
    remaining = max(settings.monthly_target_zar - month_zar, 0.0)
    st.caption(
        f"{fmt_zar(month_zar)} banked · {fmt_zar(remaining)} to go · "
        f"≈ {fmt_usd(remaining / fx_quote.rate if fx_quote.rate else 0)} of net premium at {fx_quote.rate:.2f}"
    )

    # ---- margin guardrail -------------------------------------------------
    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Margin guardrail**")
        if account:
            utilisation = account.margin_utilization
            ceiling = settings.max_margin_utilization
            palette = theme()
            over = utilisation > ceiling
            st.progress(min(utilisation / max(ceiling, 1e-9), 1.0))
            st.markdown(
                f"<span style='color:{palette['critical'] if over else palette['good']};font-weight:600'>"
                f"{'⛔ BREACHED' if over else '✅ Within limit'}</span> — "
                f"maintenance margin {fmt_usd(account.maintenance_margin)} of {fmt_usd(account.equity)} equity "
                f"({utilisation:.1%} of a {ceiling:.0%} ceiling)",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Options buying power {fmt_usd(account.options_buying_power)} · "
                f"Cash {fmt_usd(account.cash)} · "
                f"Options level {account.options_trading_level if account.options_trading_level is not None else '—'}"
            )
        else:
            st.caption("Account unavailable.")

    with right:
        st.markdown("**Realised expectancy**")
        hurdle = engine.breakeven_win_rate()
        if expectancy.has_data:
            grid = st.columns(3)
            grid[0].metric(
                "Win rate",
                fmt_pct(expectancy.p_win, 0),
                delta=f"{(expectancy.p_win - hurdle) * 100:+.0f}pts vs breakeven",
            )
            grid[1].metric("Avg win", fmt_usd(expectancy.avg_win))
            grid[2].metric("Avg loss", fmt_usd(-expectancy.avg_loss))
            st.caption(
                f"E = ({expectancy.p_win:.2f} × {expectancy.avg_win:,.2f}) − "
                f"({expectancy.p_loss:.2f} × {expectancy.avg_loss:,.2f}) = "
                f"**{expectancy.expectancy:,.2f} USD** over {expectancy.trades} closed trades"
            )
        else:
            st.caption("No closed trades yet — expectancy appears after the first position is settled.")
        st.caption(
            f"The 50% target / 200% stop geometry breaks even at a **{hurdle:.0%}** win rate. "
            "Bracket placement alone cannot beat that — only selling volatility richer than what realises can."
        )

    # ---- charts -----------------------------------------------------------
    st.divider()
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("**Realised income by month (ZAR)**")
        render_monthly_income_chart(zar_by_month, settings.monthly_target_zar)
    with chart_right:
        st.markdown("**Cumulative realised P&L (ZAR)**")
        render_cumulative_pnl_chart(closed)


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
            except BrokerError as exc:
                st.error(f"Scan failed: {exc}")

    snapshots: List[engine.VRPSnapshot] = st.session_state.get("scan", [])
    if not snapshots:
        st.info("No scan yet. Run one above, or start the bot to populate it automatically.")
        return

    scanned_at = st.session_state.get("scan_at")
    if scanned_at:
        st.caption(f"Scanned {scanned_at.strftime('%Y-%m-%d %H:%M:%S')} UTC")

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
    render_sidebar(bot)

    st.title("Brickvestcapitalterminal")
    st.caption(
        "Mechanical variance-risk-premium harvesting · 45 DTE · 30 delta · IV Rank > 50 · "
        "50% profit target / 200% stop · Alpaca"
    )

    if bot.state.is_halted:
        st.error(f"🔴 **EMERGENCY HALT** — {bot.state.halt_reason}. Positions are not being managed.")

    if not bot.settings.paper:
        st.warning("⚠️ Live trading mode is enabled. Orders will be sent to a funded account.")

    tabs = st.tabs(["Overview", "Scanner", "Positions", "Trade log", "Bot", "Settings"])
    with tabs[0]:
        render_overview(bot)
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
