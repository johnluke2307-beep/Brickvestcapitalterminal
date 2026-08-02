"""
Brickvestcapitalterminal — console deck.

The same platform as ``app.py``, rendered in a terminal with Rich instead of a
browser. It exists for the case the Streamlit tier cannot cover: a machine that
stays awake. Streamlit Community Cloud sleeps on inactivity, and a sleeping app
is a stopped bot, which is unmanaged positions — so on a VPS or a Pi you run
this next to ``python bot.py --loop`` (or let this process drive the loop
itself with ``--live``) and keep the browser dashboard for analysis.

    python terminal.py                 # read-only monitor, 5s refresh
    python terminal.py --live          # also run the trading loop
    python terminal.py --interval 15   # slower refresh

Every figure comes from the same engine the executor trades on. There are no
placeholder rows: when a panel has no data it says so, because a risk display
that invents plausible numbers is worse than one that shows nothing.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
import engine
from bot import TradingBot
from broker_client import BrokerError

BRAND = "BRICKVEST CAPITAL TERMINAL"

# Console styling — mirrors the web deck's semantics so the two read alike.
S_GOOD = "bold green"
S_WARN = "bold yellow"
S_CRIT = "bold red"
S_IDLE = "dim"
S_ACCENT = "bold cyan"


class ConsoleDeck:
    """Rich rendering of the account, the scan and the managed brackets."""

    def __init__(self, bot: TradingBot, drive_bot: bool = False) -> None:
        self.bot = bot
        self.drive_bot = drive_bot
        self.console = Console()
        self.scan: List[engine.VRPSnapshot] = []
        self.last_scan_at: Optional[datetime] = None
        self.last_error: Optional[str] = None

    # ---------------------------------------------------------------- data
    def refresh(self) -> None:
        """Pull one round of state. Never raises — failures land in the header."""
        try:
            if self.drive_bot and not self.bot.state.is_halted:
                result = self.bot.run_once()
                self.scan = result.scanned or self.scan
                if result.scanned:
                    self.last_scan_at = datetime.now(timezone.utc)
            else:
                # Read-only: refresh the scan on the first pass and every 15 min.
                stale = (
                    self.last_scan_at is None
                    or (datetime.now(timezone.utc) - self.last_scan_at).total_seconds() > 900
                )
                if stale:
                    self.scan = self.bot.scan()
                    self.last_scan_at = datetime.now(timezone.utc)
            self.last_error = None
        except BrokerError as exc:
            self.last_error = str(exc)
        except Exception as exc:  # a display must not take the process down
            self.last_error = f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------- layout
    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="strip", size=3),
            Layout(name="body", ratio=1),
            # Four rows: the rules line, the run-state line, and the panel border.
            Layout(name="footer", size=4),
        )
        layout["body"].split_row(
            Layout(name="scanner", ratio=1),
            Layout(name="portfolio", ratio=2),
        )
        layout["header"].update(self.header())
        layout["strip"].update(self.status_strip())
        layout["scanner"].update(self.target_panel())
        layout["portfolio"].update(self.risk_panel())
        layout["footer"].update(self.footer())
        return layout

    def header(self) -> Panel:
        state = self.bot.state
        health = self.bot.client.health
        if state.is_halted:
            status, style = "HALTED", S_CRIT
        else:
            status, style = {
                "ok": ("ONLINE", S_GOOD),
                "degraded": ("DEGRADED", S_WARN),
                "down": ("OFFLINE", S_CRIT),
            }[health.status]

        mode = "PAPER" if self.bot.settings.paper else "LIVE"
        line = Text(justify="center")
        line.append(f"{BRAND}  ", style="bold white")
        line.append("| ", style=S_IDLE)
        line.append(f"{mode} ", style=S_ACCENT)
        line.append("| STATUS: ", style=S_IDLE)
        line.append(status, style=style)
        line.append(f"  | {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC", style=S_IDLE)
        return Panel(line, border_style="blue", padding=(0, 1))

    def status_strip(self) -> Panel:
        """Capital, margin headroom and Rand progress on one line."""
        settings = self.bot.settings
        fx_quote = self.bot.fx.get_rate()
        closed = self.bot.trade_log.closed_trades()
        expectancy = engine.compute_expectancy(closed)
        month_zar = engine.monthly_pnl(closed, "zar").get(datetime.now(timezone.utc).strftime("%Y-%m"), 0.0)
        progress = month_zar / settings.monthly_target_zar if settings.monthly_target_zar else 0.0

        table = Table.grid(expand=True)
        for _ in range(6):
            table.add_column(justify="left", ratio=1)

        try:
            account = self.bot.client.get_account()
            util = account.margin_utilization
            util_style = S_CRIT if util > settings.max_margin_utilization else S_GOOD
            nav = f"${account.equity:,.0f}"
            nav_zar = f"R{account.equity * fx_quote.rate:,.0f}"
            day = Text(f"{account.day_pnl:+,.2f}", style=S_GOOD if account.day_pnl >= 0 else S_CRIT)
            margin = Text.assemble(
                (f"{util:.0%}", util_style), (f" / {settings.max_margin_utilization:.0%}", S_IDLE)
            )
        except BrokerError:
            nav, nav_zar, day = "—", "—", Text("—", style=S_IDLE)
            margin = Text("—", style=S_IDLE)

        table.add_row(
            _cell("NAV USD", nav),
            _cell("NAV ZAR", nav_zar),
            _cell("DAY P&L", day),
            _cell("MARGIN", margin),
            _cell(f"{datetime.now(timezone.utc):%b} ZAR", f"R{month_zar:,.0f}"),
            _cell(
                "TARGET",
                Text.assemble((f"{progress:.0%}", S_ACCENT), (f" of R{settings.monthly_target_zar:,.0f}", S_IDLE)),
            ),
        )
        subtitle = f"E ${expectancy.expectancy:,.2f}/trade" if expectancy.has_data else "no closed trades yet"
        return Panel(table, border_style="grey37", padding=(0, 1), subtitle=subtitle, subtitle_align="right")

    def target_panel(self) -> Panel:
        """Left — the VRP scan, ranked, with each symbol's verdict.

        Ranked by IV − RV, not by absolute implied volatility: the highest-IV
        name on the board is usually just the most volatile one, which is priced
        accordingly. The edge is IV *above what realises*, so that is the sort.
        """
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("#", style=S_IDLE, width=2)
        table.add_column("TKR", style=S_ACCENT, width=5)
        table.add_column("VRP", justify="right", width=6)
        table.add_column("IVR", justify="right", width=4)
        table.add_column("SIG", justify="right", width=5)

        if not self.scan:
            table.add_row("", Text("awaiting scan", style=S_IDLE), "", "", "")
        else:
            ranked = sorted(
                self.scan, key=lambda s: (s.is_tradeable, s.vrp if s.vrp is not None else -9), reverse=True
            )
            for rank, snap in enumerate(ranked[:12], start=1):
                vrp = f"{snap.vrp * 100:+.1f}" if snap.vrp is not None else "n/a"
                ivr = f"{snap.iv_rank.value:.0f}" if snap.iv_rank.value is not None else "--"
                signal = Text("SELL", style=S_GOOD) if snap.is_tradeable else Text("PASS", style=S_IDLE)
                table.add_row(str(rank), snap.symbol, vrp, ivr, signal)

        stamp = f"{self.last_scan_at:%H:%M:%S} UTC" if self.last_scan_at else "never"
        return Panel(
            table, title="[bold]◆ TARGET ACQUISITION[/bold] · IV−RV",
            subtitle=f"scanned {stamp}", subtitle_align="right", border_style="grey37",
        )

    def risk_panel(self) -> Panel:
        """Right — open short premium and the action the bot will take next.

        The action column calls ``bot.position_action``, the same rules the
        executor applies, so the panel cannot drift from what actually happens.
        """
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("TICKER", style="bold white", width=14)
        table.add_column("CREDIT", justify="right", width=7)
        table.add_column("MARK", justify="right", width=7)
        table.add_column("CAPT", justify="right", width=6)
        table.add_column("DTE", justify="right", width=4)
        table.add_column("ACTION", justify="right", width=16)

        try:
            positions = [p for p in self.bot.client.get_option_positions() if p.is_short]
        except BrokerError as exc:
            return Panel(
                Align.center(Text(f"position feed down — {exc}", style=S_CRIT)),
                title="[bold]▣ RISK MANAGER[/bold]", border_style="red",
            )

        if not positions:
            table.add_row(Text("flat — no open short premium", style=S_IDLE), "", "", "", "", "")
        else:
            styles = {"good": S_GOOD, "warning": S_WARN, "critical": S_CRIT, "idle": S_IDLE}
            for position in positions:
                verdict = self.bot.position_action(position)
                # Anything that is not MONITOR wants a decision now, so it blinks.
                action_style = styles[verdict["severity"]]
                if verdict["severity"] != "idle":
                    action_style += " blink"
                captured = verdict["captured"]
                ticker = f"{position.underlying or position.symbol} " + (
                    f"{position.option_type[0].upper()}{position.strike:g}" if position.option_type else ""
                )
                table.add_row(
                    ticker,
                    f"${verdict['credit']:.2f}",
                    f"${position.current_price:.2f}",
                    Text(f"{captured:+.0%}", style=S_GOOD if captured >= 0 else S_CRIT),
                    str(position.dte if position.dte is not None else "--"),
                    Text(verdict["action"], style=action_style),
                )

        note = "brackets enforced by the bot, not resting at the exchange"
        return Panel(
            table, title="[bold]▣ RISK MANAGER[/bold] · managed brackets",
            subtitle=note, subtitle_align="right", border_style="grey37",
        )

    def footer(self) -> Panel:
        settings = self.bot.settings
        state = self.bot.state
        rules = Text(justify="left")
        rules.append("LOGIC: ", style=S_IDLE)
        rules.append(f"{settings.profit_target_pct:.0%} PROFIT TAKER", style=S_GOOD)
        rules.append(" | STOP ", style=S_IDLE)
        rules.append(f"{settings.stop_loss_multiple:.0%} OF CREDIT", style=S_CRIT)
        rules.append(f" ({settings.stop_loss_price_multiple:.0f}x BUYBACK)", style=S_IDLE)
        rules.append(f" | TIME EXIT {settings.time_exit_dte} DTE", style=S_WARN)
        rules.append(f" | MAX MARGIN {settings.max_margin_utilization:.0%} NAV", style=S_IDLE)
        rules.append(f" | ENTRY {settings.target_dte} DTE @ {settings.target_delta:.2f}D", style=S_IDLE)

        if state.is_halted:
            status = Text(f"HALTED — {state.halt_reason}", style=S_CRIT)
        elif self.last_error:
            status = Text(f"last error: {self.last_error[:110]}", style=S_WARN)
        else:
            mode = "EXECUTING" if self.drive_bot else "MONITOR ONLY"
            status = Text(
                f"{mode} · cycle {state.cycles} · {state.entries_today_count()} entries today · "
                f"{state.last_cycle_summary or 'idle'}",
                style=S_IDLE,
            )
        return Panel(Group(rules, status), border_style="grey37", padding=(0, 1))


def _cell(label: str, value) -> Text:
    """One status cell, on a single line — the strip is one row tall."""
    text = Text()
    text.append(f"{label} ", style="dim")
    text.append(value if isinstance(value, Text) else Text(str(value), style="bold white"))
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Brickvestcapitalterminal console deck")
    parser.add_argument("--live", action="store_true", help="run the trading loop, not just monitor")
    parser.add_argument("--interval", type=int, default=5, help="seconds between refreshes (default 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.CRITICAL)  # keep the deck clean; bot.py logs properly
    settings = config.load_settings()
    console = Console()

    if not settings.credentials_present:
        console.print("[bold red]ALPACA_API_KEY / ALPACA_SECRET_KEY are not set[/bold red] — see README.md")
        return 2

    bot = TradingBot(settings=settings)
    if not bot.client.is_connected:
        console.print(f"[bold red]Connection failed:[/bold red] {bot.client.health.last_error}")
        console.print("[dim]Check the keys, and that this host can reach paper-api.alpaca.markets.[/dim]")
        return 1

    deck = ConsoleDeck(bot, drive_bot=args.live)
    if args.live:
        console.print("[bold yellow]LIVE MODE[/bold yellow] — this process will place and manage orders.")
        time.sleep(2)

    with Live(deck.render(), refresh_per_second=4, screen=True) as live:
        while True:
            deck.refresh()
            live.update(deck.render())
            if bot.state.is_halted and args.live:
                live.update(deck.render())
                time.sleep(args.interval)  # keep displaying the halt reason
                continue
            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        Console().print("\n[dim]System shutdown.[/dim]")
