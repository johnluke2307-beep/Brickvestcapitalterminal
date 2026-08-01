"""
Brickvestcapitalterminal — automated execution loop.

The mechanical rule set, in one sentence: *sell the 30-delta option roughly 45
days out, only when implied volatility is objectively rich (IV Rank > 50 and
IV > RV), take profit at 50% of the credit, stop at 200% of the credit, and stand
down entirely if margin, data or the broker link says so.*

Each cycle does exactly four things, in this order:

1. **Preflight** — broker health, account health, margin utilisation, halt state.
2. **Manage** — every open position is checked against the profit target, the
   stop and the 21-DTE time exit *before* any new risk is considered.
3. **Scan** — the universe is scored for VRP and IV Rank.
4. **Enter** — the best qualifying candidate is sold, subject to every guardrail.

Synthetic brackets
------------------
Alpaca does not accept bracket/OCO orders on option legs, so the 50%/200% pair is
enforced here as a **managed bracket**: the bot re-prices each open contract every
cycle and fires the closing order when a trigger is crossed. The trigger levels are
computed at entry, written to the trade log, and surfaced in the UI, so the
intended exits are visible and auditable even though they are not resting at the
exchange. A cycle interval shorter than the time it takes price to travel from
target to stop is therefore a real parameter, not a cosmetic one — the default is
five minutes.

Run it head-less with ``python bot.py --loop``, or drive it from the dashboard.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import config
import engine
from broker_client import (
    AccountSnapshot,
    BrokerClient,
    BrokerError,
    OptionQuote,
    PositionView,
    build_client,
)

logger = logging.getLogger("brickvest.bot")

OPTION_MULTIPLIER = 100


# ======================================================================================
# Persistent bot state
# ======================================================================================
@dataclass
class BotState:
    """Everything the bot must remember across restarts, including the halt flag."""

    mode: str = "idle"  # idle | running | halted
    halt_reason: Optional[str] = None
    halted_at: Optional[str] = None
    last_cycle_at: Optional[str] = None
    last_cycle_summary: str = ""
    cycles: int = 0
    entries_today: int = 0
    entries_date: str = ""
    events: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------------- state
    @property
    def is_halted(self) -> bool:
        return self.mode == "halted"

    def log_event(self, level: str, message: str, **extra) -> None:
        """Append to the rolling event feed the dashboard renders."""
        self.events.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "level": level,
                "message": message,
                **extra,
            }
        )
        self.events = self.events[-200:]  # bounded — this file lives on a free tier

    def note_entry(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.entries_date != today:
            self.entries_date = today
            self.entries_today = 0
        self.entries_today += 1

    def entries_today_count(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        return self.entries_today if self.entries_date == today else 0

    # ------------------------------------------------------------ persistence
    def save(self, path: Optional[Path] = None) -> None:
        path = Path(path or config.BOT_STATE_PATH)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(self), indent=2))
        except OSError as exc:
            logger.warning("could not persist bot state: %s", exc)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BotState":
        path = Path(path or config.BOT_STATE_PATH)
        try:
            payload = json.loads(path.read_text())
            known = {f for f in cls.__dataclass_fields__}  # tolerate schema drift
            return cls(**{k: v for k, v in payload.items() if k in known})
        except Exception:
            return cls()


# ======================================================================================
# Cycle result
# ======================================================================================
@dataclass
class CycleResult:
    """Structured outcome of one bot cycle — rendered verbatim by the dashboard."""

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ok: bool = True
    halted: bool = False
    reason: Optional[str] = None
    scanned: List[engine.VRPSnapshot] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)
    entries: List[dict] = field(default_factory=list)
    exits: List[dict] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)
    account: Optional[AccountSnapshot] = None

    def summary(self) -> str:
        if self.halted:
            return f"HALTED — {self.reason}"
        if not self.ok:
            return f"error — {self.reason}"
        bits = [f"{len(self.scanned)} scanned", f"{len(self.candidates)} qualified"]
        if self.entries:
            bits.append(f"{len(self.entries)} opened")
        if self.exits:
            bits.append(f"{len(self.exits)} closed")
        if self.blocked:
            bits.append(f"blocked: {self.blocked[0]}")
        return ", ".join(bits)


# ======================================================================================
# The bot
# ======================================================================================
class TradingBot:
    """Mechanical 45-DTE / 30-delta premium seller with hard risk guardrails."""

    def __init__(
        self,
        client: Optional[BrokerClient] = None,
        settings: Optional[config.Settings] = None,
        *,
        trade_log: Optional[engine.TradeLog] = None,
        iv_store: Optional[engine.IVHistoryStore] = None,
        fx: Optional[engine.ForexConverter] = None,
    ) -> None:
        self.settings = settings or config.load_settings()
        self.client = client or build_client(self.settings)
        self.trade_log = trade_log or engine.TRADE_LOG
        self.iv_store = iv_store or engine.IV_STORE
        self.fx = fx or engine.FX
        self.state = BotState.load()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_result: Optional[CycleResult] = None

    # ---------------------------------------------------------------- fail-safe
    def halt(self, reason: str) -> None:
        """Enter the emergency state. Only :meth:`resume` clears it.

        Called on any broker/market-data failure, margin breach or account block.
        The bot will refuse to open or manage positions until a human resumes it,
        which is the intended behaviour: an unattended seller of naked premium
        with a broken data feed is the single most expensive failure mode here.
        """
        self.state.mode = "halted"
        self.state.halt_reason = reason
        self.state.halted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.state.log_event("critical", f"HALTED: {reason}")
        self.state.save()
        logger.critical("bot halted: %s", reason)

    def resume(self) -> None:
        """Clear the emergency state after the operator has checked the account."""
        self.state.mode = "idle"
        self.state.halt_reason = None
        self.state.halted_at = None
        self.state.log_event("info", "halt cleared by operator")
        self.state.save()

    # ------------------------------------------------------------------- cycle
    def run_once(self) -> CycleResult:
        """Execute one full preflight → manage → scan → enter cycle."""
        result = CycleResult()
        with self._lock:
            try:
                if self.state.is_halted:
                    result.halted = True
                    result.ok = False
                    result.reason = self.state.halt_reason or "halted"
                    return result

                # ---------------- 1. preflight -------------------------------
                account = self._preflight(result)
                if account is None:
                    return result
                result.account = account

                # ---------------- 2. manage existing risk first ---------------
                result.exits = self._manage_open_positions(account)

                # ---------------- 3. scan for edge ----------------------------
                result.scanned = self._scan_universe()
                result.candidates = [s for s in result.scanned if s.is_tradeable]

                # ---------------- 4. enter ------------------------------------
                blockers = self._entry_blockers(account)
                if blockers:
                    result.blocked = blockers
                    self.state.log_event("info", f"no new entries — {blockers[0]}")
                elif result.candidates:
                    entry, rejections = self._enter_best(result.candidates, account)
                    if entry:
                        result.entries.append(entry)
                    else:
                        # A qualifying VRP that produced no trade must say why —
                        # silence here is indistinguishable from a broken scanner.
                        result.blocked.extend(rejections or ["no contract passed the entry checks"])
                else:
                    result.blocked.append("no candidate cleared the VRP / IV Rank filter")

                return result

            except BrokerError as exc:
                # Broker or data failure → emergency state, never an exception
                # that would leave positions unmanaged and the UI blank.
                self.halt(f"broker failure: {exc}")
                result.ok = False
                result.halted = True
                result.reason = str(exc)
                return result
            except Exception as exc:  # defensive: an unexpected bug must not trade on
                logger.exception("unhandled error in bot cycle")
                self.halt(f"unhandled error: {type(exc).__name__}: {exc}")
                result.ok = False
                result.halted = True
                result.reason = str(exc)
                return result
            finally:
                self.state.cycles += 1
                self.state.last_cycle_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self.state.last_cycle_summary = result.summary()
                self.state.save()
                self.last_result = result

    # ------------------------------------------------------------- 1. preflight
    def _preflight(self, result: CycleResult) -> Optional[AccountSnapshot]:
        """Verify the link, the account and the margin ceiling before trading."""
        if not self.client.is_connected and not self.client.connect():
            self.halt(f"broker unreachable: {self.client.health.last_error}")
            result.halted, result.ok = True, False
            result.reason = self.client.health.last_error
            return None

        account = self.client.get_account()  # raises BrokerError → caught upstream

        if not account.is_healthy:
            self.halt("account is blocked or trading-suspended at the broker")
            result.halted, result.ok = True, False
            result.reason = "account blocked"
            return None

        if not self.client.health.is_tradeable:
            self.halt(f"market data degraded: {self.client.health.last_error}")
            result.halted, result.ok = True, False
            result.reason = "degraded data feed"
            return None

        self.state.mode = "running"
        return account

    # ---------------------------------------------------------------- 2. manage
    def _manage_open_positions(self, account: AccountSnapshot) -> List[dict]:
        """Apply the synthetic bracket to every short option position we hold."""
        exits: List[dict] = []
        try:
            positions = self.client.get_option_positions()
        except BrokerError as exc:
            self.halt(f"could not read positions: {exc}")
            return exits

        # Settle anything the broker no longer shows before deciding anything new.
        exits.extend(self._reconcile(positions))
        working = self._working_order_symbols()

        for position in positions:
            if not position.is_short:
                continue  # long wings are closed with their short leg, not alone
            decision = self._exit_decision(position)
            if not decision:
                continue
            if position.symbol in working:
                # A close is already resting at the broker; re-sending would
                # double the order and buy back more than we are short.
                self.state.log_event("info", f"{position.symbol}: close order already working")
                continue

            reason, trigger_price = decision
            try:
                order = self._close_position(position, reason, trigger_price)
                exits.append(order)
                self.state.log_event(
                    "info",
                    f"exit {position.symbol}: {reason} at {position.current_price:.2f}",
                    symbol=position.symbol,
                    reason=reason,
                )
            except BrokerError as exc:
                self.state.log_event("error", f"exit failed for {position.symbol}: {exc}")
                self.halt(f"exit order failed for {position.symbol}: {exc}")
                break
        return exits

    def _working_order_symbols(self) -> set:
        """Symbols with an unfilled order resting at the broker."""
        try:
            return {order.get("symbol") for order in self.client.get_orders("open") if order.get("symbol")}
        except BrokerError:
            return set()

    def _reconcile(self, positions: List[PositionView]) -> List[dict]:
        """Settle logged trades whose position has actually left the account.

        An exit order is *submitted*, not filled, so the trade log marks it
        ``closing`` and waits. When the broker stops reporting the position the
        trade is genuinely flat and is booked at the recorded exit price. This is
        what keeps the expectancy figures honest: nothing is counted as realised
        P&L until the position is really gone.
        """
        held = {position.symbol for position in positions}
        settled: List[dict] = []
        for record in self.trade_log.all():
            if record.get("status") not in {"open", "closing"} or record.get("symbol") in held:
                continue
            try:
                credit = float(record.get("credit") or 0.0)
                debit = float(record.get("exit_debit") or 0.0)
                contracts = int(float(record.get("contracts") or 1))
            except (TypeError, ValueError):
                credit, debit, contracts = 0.0, 0.0, 1

            reason = record.get("exit_reason") or "expired_or_closed_externally"
            pnl_usd = (credit - debit) * OPTION_MULTIPLIER * contracts
            quote = self.fx.get_rate()
            self.trade_log.update(
                record["trade_id"],
                closed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                exit_debit=f"{debit:.4f}",
                pnl_usd=f"{pnl_usd:.2f}",
                pnl_zar=f"{pnl_usd * quote.rate:.2f}",
                usd_zar=f"{quote.rate:.4f}",
                status="closed",
                exit_reason=reason,
            )
            self.state.log_event(
                "info", f"settled {record.get('symbol')}: {reason}, P&L ${pnl_usd:,.2f}", symbol=record.get("symbol")
            )
            settled.append({"symbol": record.get("symbol"), "reason": reason, "debit": debit, "pnl_usd": pnl_usd})
        return settled

    def _exit_decision(self, position: PositionView) -> Optional[tuple[str, float]]:
        """Return ``(reason, trigger_price)`` when an exit rule fires.

        Entry credit comes from the trade log when the position was opened by
        this bot, and from the broker's average entry price otherwise — so
        positions opened by hand are still managed rather than ignored.
        """
        settings = self.settings
        record = self._live_record(position.symbol)
        try:
            credit = float(record["credit"]) if record and record.get("credit") else abs(position.avg_entry_price)
        except (TypeError, ValueError):
            credit = abs(position.avg_entry_price)
        if credit <= 0:
            return None

        price = position.current_price
        profit_trigger = credit * (1.0 - settings.profit_target_pct)
        stop_trigger = credit * settings.stop_loss_price_multiple

        if price > 0 and price <= profit_trigger:
            return ("profit_target", profit_trigger)
        if price >= stop_trigger:
            return ("stop_loss", stop_trigger)
        if position.dte is not None and position.dte <= settings.time_exit_dte:
            return ("time_exit", price)
        return None

    def _close_position(self, position: PositionView, reason: str, trigger_price: float) -> dict:
        """Buy the short contract back and mark the trade as closing."""
        qty = int(abs(position.qty))
        if self.settings.dry_run:
            order = {"symbol": position.symbol, "status": "dry_run", "qty": qty}
        elif reason == "stop_loss":
            # A stop must actually get out; marketable close rather than a limit
            # that may never fill while the loss keeps widening.
            order = self.client.close_position(position.symbol, qty)
        else:
            order = self.client.submit_option_order(
                symbol=position.symbol,
                qty=qty,
                side="buy",
                position_intent="buy_to_close",
                limit_price=max(round(position.current_price, 2), 0.01),
                client_order_id=f"bvc-x-{uuid.uuid4().hex[:12]}",
            )

        debit = float(order.get("filled_avg_price") or position.current_price or trigger_price)
        self._mark_closing(position.symbol, debit, reason)
        return {"symbol": position.symbol, "reason": reason, "debit": debit, "order": order}

    def _mark_closing(self, symbol: str, debit: float, reason: str) -> None:
        """Record the intended exit. :meth:`_reconcile` books it once it is flat."""
        record = self._live_record(symbol)
        if not record:
            return
        self.trade_log.update(
            record["trade_id"],
            exit_debit=f"{debit:.4f}",
            exit_reason=reason,
            status="closing",
        )

    def _live_record(self, symbol: str) -> Optional[dict]:
        """The trade-log row for a position that is open or closing."""
        return self.trade_log.find_by_symbol(symbol, status="open") or self.trade_log.find_by_symbol(
            symbol, status="closing"
        )

    # ------------------------------------------------------------------ 3. scan
    def scan(self) -> List[engine.VRPSnapshot]:
        """Score the universe without touching the account — safe for the UI.

        Returns whatever it managed to compute; per-symbol failures are attached
        to that symbol's snapshot rather than raised, so one bad ticker never
        blanks the scanner.
        """
        if not self.client.is_connected and not self.client.connect():
            raise BrokerError(self.client.health.last_error or "broker unreachable")
        return self._scan_universe()

    def _scan_universe(self) -> List[engine.VRPSnapshot]:
        """Score every symbol for the variance risk premium."""
        snapshots: List[engine.VRPSnapshot] = []
        try:
            bars_by_symbol = self.client.get_daily_bars(self.settings.universe, lookback_days=400)
        except BrokerError as exc:
            self.state.log_event("error", f"bar download failed: {exc}")
            raise

        for symbol in self.settings.universe:
            bars = bars_by_symbol.get(symbol) or {}
            closes = bars.get("close") or []
            if len(closes) < self.settings.rv_window + 2:
                snapshots.append(
                    engine.build_vrp_snapshot(symbol, None, None, bars, self.iv_store, error="insufficient price history")
                )
                continue

            spot = closes[-1]
            try:
                expiration = self._select_expiration(symbol)
                if expiration is None:
                    snapshots.append(
                        engine.build_vrp_snapshot(symbol, spot, None, bars, self.iv_store, error="no expiry in DTE window")
                    )
                    continue
                chain = self._load_chain(symbol, expiration, spot)
                atm_iv = self._atm_implied_vol(chain, spot, expiration)
            except BrokerError as exc:
                snapshots.append(engine.build_vrp_snapshot(symbol, spot, None, bars, self.iv_store, error=str(exc)))
                continue

            snapshot = engine.build_vrp_snapshot(
                symbol, spot, atm_iv, bars, self.iv_store, expiration=expiration
            )
            snapshots.append(snapshot)
        return snapshots

    def _select_expiration(self, symbol: str) -> Optional[date]:
        """Pick the listed expiry closest to the 45-DTE target inside the window."""
        expirations = self.client.get_expirations(symbol, self.settings.dte_min, self.settings.dte_max)
        if not expirations:
            return None
        today = datetime.now(timezone.utc).date()
        return min(expirations, key=lambda d: abs((d - today).days - self.settings.target_dte))

    def _load_chain(self, symbol: str, expiration: date, spot: float) -> List[OptionQuote]:
        """Fetch the put chain around the money and fill in any missing greeks."""
        chain = self.client.get_chain(
            symbol,
            expiration,
            option_type="put",
            strike_low=spot * 0.70,
            strike_high=spot * 1.05,
        )
        return self._enrich(chain, spot, expiration)

    def _enrich(self, chain: List[OptionQuote], spot: float, expiration: date) -> List[OptionQuote]:
        """Backfill implied volatility and delta the free feed did not supply.

        The indicative feed often returns quotes without greeks. Rather than skip
        those contracts — which would frequently empty the chain — the engine
        inverts Black-Scholes off the mid price and derives delta from it.
        """
        t = engine.year_fraction(expiration)
        rate = self.settings.risk_free_rate
        for quote in chain:
            if t <= 0 or not quote.mid:
                continue
            is_call = quote.option_type == "call"
            if quote.implied_volatility is None:
                quote.implied_volatility = engine.implied_vol(quote.mid, spot, quote.strike, t, rate, is_call)
            if quote.delta is None and quote.implied_volatility:
                quote.delta = engine.bs_delta(spot, quote.strike, t, quote.implied_volatility, rate, is_call)
        return chain

    def _atm_implied_vol(self, chain: List[OptionQuote], spot: float, expiration: date) -> Optional[float]:
        """At-the-money implied volatility: the strike nearest spot with a valid IV."""
        candidates = [q for q in chain if q.implied_volatility and 0.01 < q.implied_volatility < 5.0]
        if not candidates:
            return None
        nearest = sorted(candidates, key=lambda q: abs(q.strike - spot))[:3]
        return sum(q.implied_volatility for q in nearest) / len(nearest)

    # ----------------------------------------------------------------- 4. enter
    def _entry_blockers(self, account: AccountSnapshot) -> List[str]:
        """Every reason the bot is not allowed to add risk right now."""
        settings = self.settings
        blockers: List[str] = []

        # The margin safeguard. Checked before anything else that costs money.
        if account.margin_utilization > settings.max_margin_utilization:
            blockers.append(
                f"margin utilisation {account.margin_utilization:.0%} exceeds the "
                f"{settings.max_margin_utilization:.0%} ceiling"
            )
        if account.equity < settings.min_equity_usd:
            blockers.append(f"equity ${account.equity:,.0f} below the ${settings.min_equity_usd:,.0f} floor")

        open_positions = len({p.symbol for p in self._safe_positions() if p.is_short})
        if open_positions >= settings.max_open_positions:
            blockers.append(f"{open_positions} open positions at the {settings.max_open_positions} limit")
        if self.state.entries_today_count() >= settings.max_new_positions_per_day:
            blockers.append(f"daily entry limit ({settings.max_new_positions_per_day}) reached")

        try:
            if not self.client.is_market_open():
                blockers.append("market closed")
        except BrokerError as exc:
            blockers.append(f"clock unavailable: {exc}")

        return blockers

    def _safe_positions(self) -> List[PositionView]:
        try:
            return self.client.get_option_positions()
        except BrokerError:
            return []

    def _enter_best(
        self, candidates: List[engine.VRPSnapshot], account: AccountSnapshot
    ) -> tuple[Optional[dict], List[str]]:
        """Sell the richest qualifying candidate that also passes the contract checks.

        Returns the entry (or ``None``) together with the reason each candidate
        was passed over, so the dashboard can explain an empty cycle.
        """
        held = {p.underlying for p in self._safe_positions() if p.is_short and p.underlying}
        ranked = sorted(candidates, key=lambda s: (s.vrp or 0.0, s.iv_rank.value or 0.0), reverse=True)
        rejections: List[str] = []

        def reject(symbol: str, why: str) -> None:
            rejections.append(f"{symbol}: {why}")
            self.state.log_event("info", f"{symbol}: {why}")

        for snapshot in ranked:
            if self.settings.one_position_per_underlying and snapshot.symbol in held:
                rejections.append(f"{snapshot.symbol}: already holding a position in this underlying")
                continue
            if not snapshot.expiration or not snapshot.spot:
                rejections.append(f"{snapshot.symbol}: no expiry or spot price")
                continue
            try:
                chain = self._load_chain(snapshot.symbol, snapshot.expiration, snapshot.spot)
                contract = self._select_short_strike(chain)
                if contract is None:
                    reject(snapshot.symbol, f"no contract within {self.settings.delta_tolerance:.2f} of "
                                            f"{self.settings.target_delta:.2f} delta")
                    continue
                illiquid = self._liquidity_reject_reason(contract)
                if illiquid:
                    reject(snapshot.symbol, f"{contract.symbol} {illiquid}")
                    continue
                capital = self._capital_reject_reason(contract, account)
                if capital:
                    reject(snapshot.symbol, capital)
                    continue
                entry = self._open_position(snapshot, contract, chain)
                if entry:
                    return entry, rejections
                rejections.append(f"{snapshot.symbol}: net credit did not clear the minimum")
            except BrokerError as exc:
                self.state.log_event("error", f"entry failed for {snapshot.symbol}: {exc}")
                raise
        return None, rejections

    def _select_short_strike(self, chain: List[OptionQuote]) -> Optional[OptionQuote]:
        """The put whose delta is closest to the 30-delta target, within tolerance."""
        target = abs(self.settings.target_delta)
        tolerance = self.settings.delta_tolerance
        scored = [
            (abs(abs(q.delta) - target), q)
            for q in chain
            if q.delta is not None and abs(abs(q.delta) - target) <= tolerance and q.mid
        ]
        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    def _liquidity_reject_reason(self, contract: OptionQuote) -> Optional[str]:
        """Reject wide or thin quotes — slippage is the tax on a small edge."""
        settings = self.settings
        if not contract.has_two_sided_quote:
            return "no two-sided quote"
        mid = contract.mid or 0.0
        if mid < settings.min_credit_usd:
            return f"credit ${mid:.2f} below the ${settings.min_credit_usd:.2f} minimum"
        spread = contract.spread_pct
        if spread is None or spread > settings.max_spread_pct:
            return f"bid/ask spread {spread:.0%} exceeds {settings.max_spread_pct:.0%} of mid"
        return None

    def _capital_reject_reason(self, contract: OptionQuote, account: AccountSnapshot) -> Optional[str]:
        """Check the trade fits *and* leaves the margin ceiling intact afterwards.

        For a cash-secured put the requirement is the full strike notional; a
        defined-risk spread only needs the width. The projected post-trade margin
        utilisation must still sit under the ceiling — the guardrail is forward
        looking, so a trade that would breach it is never sent in the first place.
        """
        settings = self.settings
        contracts = max(settings.contracts_per_trade, 1)
        if settings.strategy == "put_credit_spread":
            requirement = settings.spread_width * OPTION_MULTIPLIER * contracts
        else:
            requirement = contract.strike * OPTION_MULTIPLIER * contracts

        if account.options_buying_power and requirement > account.options_buying_power:
            return (
                f"needs ${requirement:,.0f} of options buying power, "
                f"${account.options_buying_power:,.0f} available"
            )

        projected = (account.maintenance_margin + requirement) / account.equity if account.equity > 0 else 1.0
        if projected > settings.max_margin_utilization:
            return (
                f"${requirement:,.0f} requirement would take margin utilisation to "
                f"{projected:.0%}, above the {settings.max_margin_utilization:.0%} ceiling"
            )
        return None

    def _open_position(
        self, snapshot: engine.VRPSnapshot, contract: OptionQuote, chain: List[OptionQuote]
    ) -> Optional[dict]:
        """Sell the contract, record the managed bracket levels, log the trade."""
        settings = self.settings
        contracts = max(settings.contracts_per_trade, 1)
        credit = contract.mid or 0.0
        trade_id = uuid.uuid4().hex[:12]

        long_leg: Optional[OptionQuote] = None
        if settings.strategy == "put_credit_spread":
            long_leg = self._select_long_wing(chain, contract)
            if long_leg is None:
                self.state.log_event("info", f"{snapshot.symbol}: no long wing available for the spread")
                return None
            credit = max((contract.mid or 0.0) - (long_leg.mid or 0.0), 0.0)

        if credit < settings.min_credit_usd:
            self.state.log_event("info", f"{snapshot.symbol}: net credit ${credit:.2f} below the minimum")
            return None

        # ---- send the order ------------------------------------------------
        if settings.dry_run:
            order = {"status": "dry_run", "symbol": contract.symbol, "qty": contracts}
        elif long_leg is not None:
            order = self.client.submit_vertical_spread(
                short_symbol=contract.symbol,
                long_symbol=long_leg.symbol,
                qty=contracts,
                limit_price=-round(credit, 2),  # negative limit = net credit
                opening=True,
                client_order_id=f"bvc-o-{trade_id}",
            )
        else:
            order = self.client.submit_option_order(
                symbol=contract.symbol,
                qty=contracts,
                side="sell",
                position_intent="sell_to_open",
                limit_price=round(credit, 2),
                client_order_id=f"bvc-o-{trade_id}",
            )

        fill = float(order.get("filled_avg_price") or credit)
        quote = self.fx.get_rate()
        self.trade_log.append(
            {
                "trade_id": trade_id,
                "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "underlying": snapshot.symbol,
                "symbol": contract.symbol,
                "strategy": settings.strategy,
                "contracts": contracts,
                "strike": f"{contract.strike:.2f}",
                "expiration": contract.expiration.isoformat(),
                "entry_delta": f"{contract.delta:.4f}" if contract.delta is not None else "",
                "entry_iv": f"{snapshot.implied_vol:.4f}" if snapshot.implied_vol else "",
                "entry_rv": f"{snapshot.reference_rv:.4f}" if snapshot.reference_rv else "",
                "entry_iv_rank": f"{snapshot.iv_rank.value:.1f}" if snapshot.iv_rank.value is not None else "",
                "credit": f"{fill:.4f}",
                "usd_zar": f"{quote.rate:.4f}",
                "status": "open",
                "note": (
                    f"managed bracket: take profit {fill * (1 - settings.profit_target_pct):.2f}, "
                    f"stop {fill * settings.stop_loss_price_multiple:.2f}"
                    + (f", long wing {long_leg.symbol}" if long_leg else "")
                ),
            }
        )
        self.state.note_entry()

        expected = engine.theoretical_expectancy(fill, contract.delta or -settings.target_delta, contracts=contracts)
        self.state.log_event(
            "info",
            f"opened {contract.symbol} for ${fill:.2f} credit "
            f"(IVR {snapshot.iv_rank.value:.0f}, VRP {(snapshot.vrp or 0) * 100:.1f}pts, "
            f"delta-implied win rate {expected.p_win:.0%} vs {expected.breakeven_p_win:.0%} breakeven)",
            symbol=contract.symbol,
        )
        return {
            "trade_id": trade_id,
            "symbol": contract.symbol,
            "underlying": snapshot.symbol,
            "credit": fill,
            "contracts": contracts,
            "delta": contract.delta,
            "expiration": contract.expiration.isoformat(),
            "take_profit": round(fill * (1 - settings.profit_target_pct), 2),
            "stop_loss": round(fill * settings.stop_loss_price_multiple, 2),
            "expectancy_usd": expected.expectancy,
            "p_win_delta": expected.p_win,
            "p_win_breakeven": expected.breakeven_p_win,
            "order": order,
        }

    def _select_long_wing(self, chain: List[OptionQuote], short: OptionQuote) -> Optional[OptionQuote]:
        """The protective put ``spread_width`` below the short strike."""
        target = short.strike - self.settings.spread_width
        below = [q for q in chain if q.strike < short.strike and q.mid]
        if not below:
            return None
        return min(below, key=lambda q: abs(q.strike - target))

    # ------------------------------------------------------- background looping
    def start(self, interval_seconds: Optional[int] = None) -> bool:
        """Run cycles on a daemon thread so the Streamlit UI stays responsive."""
        if self._thread and self._thread.is_alive():
            return False
        interval = interval_seconds or self.settings.loop_interval_seconds
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True, name="bvc-bot")
        self._thread.start()
        self.state.log_event("info", f"bot started ({interval}s cycle)")
        self.state.save()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self.state.mode = "idle"
        self.state.log_event("info", "bot stopped by operator")
        self.state.save()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    def _loop(self, interval: int) -> None:
        while not self._stop_event.is_set():
            result = self.run_once()
            if result.halted:
                logger.critical("halting loop: %s", result.reason)
                break
            self._stop_event.wait(interval)

    def run_forever(self, interval_seconds: Optional[int] = None, on_cycle: Optional[Callable] = None) -> None:
        """Blocking loop for head-less hosting (``python bot.py --loop``)."""
        interval = interval_seconds or self.settings.loop_interval_seconds
        logger.info("starting head-less loop at %ss intervals", interval)
        while True:
            result = self.run_once()
            logger.info("cycle: %s", result.summary())
            if on_cycle:
                on_cycle(result)
            if result.halted:
                logger.critical("bot halted — manual intervention required: %s", result.reason)
                return
            time.sleep(interval)

    # --------------------------------------------------------------- reporting
    def portfolio_report(self) -> dict:
        """Account, positions, expectancy and Rand progress in one payload."""
        report: Dict[str, object] = {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        try:
            account = self.client.get_account()
            positions = self.client.get_option_positions()
            report["account"] = account
            report["positions"] = positions
            report["connection"] = self.client.health
        except BrokerError as exc:
            report["error"] = str(exc)
            report["connection"] = self.client.health

        closed = self.trade_log.closed_trades()
        fx_quote = self.fx.get_rate()
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        month_zar = engine.monthly_pnl(closed, "zar").get(month_key, 0.0)

        report["expectancy"] = engine.compute_expectancy(closed)
        report["fx"] = fx_quote
        report["month_zar"] = month_zar
        report["month_target_zar"] = self.settings.monthly_target_zar
        report["month_progress"] = (
            month_zar / self.settings.monthly_target_zar if self.settings.monthly_target_zar else 0.0
        )
        return report


# ======================================================================================
# CLI
# ======================================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Brickvestcapitalterminal trading bot")
    parser.add_argument("--loop", action="store_true", help="run continuously instead of a single cycle")
    parser.add_argument("--interval", type=int, default=None, help="seconds between cycles")
    parser.add_argument("--dry-run", action="store_true", help="scan and log but never send an order")
    parser.add_argument("--resume", action="store_true", help="clear a persisted halt state and exit")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    settings = config.load_settings()
    if args.dry_run:
        settings.dry_run = True

    if not settings.credentials_present:
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set — see README.md")
        return 2

    bot = TradingBot(settings=settings)

    if args.resume:
        bot.resume()
        logger.info("halt state cleared")
        return 0

    if bot.state.is_halted:
        logger.error("bot is halted (%s). Re-run with --resume once resolved.", bot.state.halt_reason)
        return 1

    if args.loop:
        bot.run_forever(args.interval)
    else:
        result = bot.run_once()
        logger.info("cycle: %s", result.summary())
        for snapshot in result.scanned:
            reason = snapshot.reject_reason()
            logger.info(
                "  %-5s IV %-6s RV %-6s VRP %-7s IVR %-5s %s",
                snapshot.symbol,
                f"{snapshot.implied_vol * 100:.1f}%" if snapshot.implied_vol else "n/a",
                f"{snapshot.reference_rv * 100:.1f}%" if snapshot.reference_rv else "n/a",
                f"{snapshot.vrp * 100:+.1f}pts" if snapshot.vrp is not None else "n/a",
                f"{snapshot.iv_rank.value:.0f}" if snapshot.iv_rank.value is not None else "n/a",
                reason or "QUALIFIED",
            )
        return 0 if result.ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
