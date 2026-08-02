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
import strategies
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
# Hard risk limits — outside the agent's reach by construction
# ======================================================================================
def _limit(key: str, default: float) -> float:
    try:
        return float(config.setting(key, default))
    except (TypeError, ValueError):
        return default


#: **The kill switch.** If the account's intraday loss reaches this fraction of
#: its start-of-day equity, the bot halts and a human has to clear it.
#:
#: This is deliberately *not* a field on :class:`config.Settings`, and therefore
#: not in ``HERMES_BOUNDS`` and not expressible in ``strategy_config.json``. The
#: research layer's only write verb is "put a number in that file", and this
#: number is not in that file. An agent tuning for return has every incentive to
#: widen a daily loss limit, and the design answer is not to trust it not to —
#: it is to make the limit unreachable from where the agent lives.
#:
#: The operator can still set it, at the process boundary, before anything runs.
DAILY_LOSS_LIMIT_PCT = _limit("BVC_DAILY_LOSS_LIMIT_PCT", 0.03)

#: On a kill-switch breach, also market-close every short option position.
#: Off by default: with a venue that rests real brackets at the exchange the
#: positions are already protected, and force-liquidating into a panic is its
#: own way to lose money. Turn it on for venues where the bracket is synthetic
#: and a halted bot means an unmanaged short.
KILL_SWITCH_FLATTENS = str(config.setting("BVC_KILL_SWITCH_FLATTEN", "false")).lower() in {
    "1", "true", "yes", "y", "on"
}

#: Drop-box for an out-of-process stop request. Anything that can write this
#: file can stop the bot; nothing that writes a file can start it. That
#: asymmetry is the point — see :meth:`TradingBot._consume_halt_request`. The
#: path is declared in ``config`` so the API can write it without importing
#: this module, which would drag the broker layer into that process.
HALT_REQUEST_PATH = config.HALT_REQUEST_PATH


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

    #: Surfaced on the instance so the observation payload can state the limit
    #: the agent cannot change, without the agent having to import this module.
    DAILY_LOSS_LIMIT_PCT = DAILY_LOSS_LIMIT_PCT

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
        self._iv_seeded = False
        self._config_mtime = self.settings.config_store().mtime()

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

    # ------------------------------------------------- agent parameter overlay
    def reload_strategy_config(self, *, force: bool = False) -> Dict[str, object]:
        """Re-read ``strategy_config.json`` and adopt whatever it is allowed to change.

        This is the entire mechanism by which the research layer influences
        trading: a file lands on disk, and the execution loop decides at a safe
        moment to read it. There is no callback, no socket and no shared object,
        so a hung, crashed or hostile research process cannot stall or reach
        into a live event loop.

        Everything the file asks for still goes through
        :func:`config.vet_changes`, measured against the operator's pre-overlay
        baseline — reloading a hundred times cannot achieve what one reload may
        not.
        """
        stamp = self.settings.config_store().mtime()
        if not force and stamp == self._config_mtime:
            return {}
        self._config_mtime = stamp

        before = self.settings.mutable_values()
        self.settings.apply_overlay()
        changed = {k: v for k, v in self.settings.mutable_values().items() if before.get(k) != v}

        if changed:
            self.state.log_event(
                "info",
                "strategy config reloaded — "
                + ", ".join(f"{k}={v}" for k, v in changed.items())
                + (f" (by {self.settings.overlay_updated_by})" if self.settings.overlay_updated_by else ""),
            )
        for name, why in self.settings.overlay_rejected.items():
            self.state.log_event("warning", f"strategy config refused {name}: {why}")
        if changed or self.settings.overlay_rejected:
            self.state.save()
        return changed

    def mark_config_current(self) -> None:
        """Note that the file on disk is already reflected in memory.

        Used by the in-process control surface, which applies a change and
        writes the file in one step; without this the next cycle would log a
        reload for a change it already made.
        """
        self._config_mtime = self.settings.config_store().mtime()

    def _consume_halt_request(self) -> Optional[str]:
        """Honour a stop requested by another process, then delete the request.

        The research layer and the HTTP API have no handle on this object — by
        design, since a shared handle is a shared failure. What they have is
        permission to write one small file, which this loop reads at a moment of
        its own choosing.

        The request is deleted once acted on. A halt file that survived being
        honoured would re-halt the bot the moment a human resumed it, and the
        operator would be arguing with a file rather than with a decision.
        """
        try:
            payload = json.loads(HALT_REQUEST_PATH.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            payload = {}
        HALT_REQUEST_PATH.unlink(missing_ok=True)
        who = str(payload.get("actor") or "external")
        return f"{who}: {payload.get('reason') or 'halt requested'}"

    # --------------------------------------------------------- the kill switch
    def _daily_loss_breach(self, account: AccountSnapshot) -> Optional[str]:
        """The hard-coded intraday loss limit. Returns a reason, or ``None``.

        Measured against start-of-day equity from the broker's own books rather
        than against the trade log, so it includes open positions marking
        against us. A limit that only counts realised losses would sit quiet
        through exactly the day it exists for.
        """
        if DAILY_LOSS_LIMIT_PCT <= 0:
            return None
        start_equity = account.last_equity or (account.equity - account.day_pnl)
        if start_equity <= 0:
            return None
        loss_pct = -account.day_pnl / start_equity
        if loss_pct < DAILY_LOSS_LIMIT_PCT:
            return None
        return (
            f"daily loss limit breached: ${account.day_pnl:,.0f} is {loss_pct:.2%} of "
            f"start-of-day equity ${start_equity:,.0f}, limit {DAILY_LOSS_LIMIT_PCT:.2%}"
        )

    def _trip_kill_switch(self, reason: str) -> None:
        """Flatten if configured to, then halt. Halting is not optional."""
        if KILL_SWITCH_FLATTENS:
            for position in self._safe_positions():
                if not position.is_short:
                    continue
                try:
                    self.client.close_position(position.symbol, int(abs(position.qty)))
                    self._mark_closing(position.symbol, position.current_price, "kill_switch")
                    self.state.log_event("critical", f"kill switch flattened {position.symbol}")
                except BrokerError as exc:
                    self.state.log_event("critical", f"kill switch could not flatten {position.symbol}: {exc}")
        elif not getattr(self.client.capabilities, "native_brackets", False):
            # Say it plainly rather than let the operator discover it later:
            # a halted bot with synthetic brackets is a bot no longer watching
            # its own stops.
            self.state.log_event(
                "critical",
                "kill switch halted the bot, but this venue has no resting brackets — "
                "open short positions are now unmanaged. Set BVC_KILL_SWITCH_FLATTEN=true "
                "or close them by hand.",
            )
        self.halt(reason)

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

                # ---------------- 0. out-of-process stop, then parameters -----
                requested = self._consume_halt_request()
                if requested:
                    self.halt(requested)
                    result.halted, result.ok = True, False
                    result.reason = requested
                    return result


                # Between cycles, never inside one: a cycle that scanned under
                # one delta target and entered under another is a cycle whose
                # log cannot be reconstructed.
                self.reload_strategy_config()

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

        # The kill switch runs before every other account check. It is the one
        # rule that does not consult Settings, does not consult the agent and
        # cannot be tuned from inside the process.
        breach = self._daily_loss_breach(account)
        if breach:
            self._trip_kill_switch(breach)
            result.halted, result.ok = True, False
            result.reason = breach
            return None

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

        # One backfill per process: a year of real IV is a big request and the
        # store already refuses to overwrite what it holds.
        if not self._iv_seeded:
            self._iv_seeded = True
            try:
                self._seed_iv_history()
            except Exception as exc:  # never let a backfill stop a trading cycle
                logger.warning("IV backfill skipped: %s", exc)

        self.state.mode = "running"
        return account

    # ---------------------------------------------------------------- 2. manage
    def _manage_open_positions(self, account: AccountSnapshot) -> List[dict]:
        """Apply the exit rules to every open structure, leg by leg or as a whole.

        Management is *trade*-centric, not position-centric. A condor's exit is a
        decision about four legs at one net price; evaluating each leg on its own
        would close the tested wing of a spread and leave the short naked, which
        is the single worst thing this loop could do.

        Positions with no trade-log record — opened by hand, or inherited — are
        still managed, one leg at a time, on the broker's own entry price.
        """
        exits: List[dict] = []
        try:
            positions = self.client.get_option_positions()
        except BrokerError as exc:
            self.halt(f"could not read positions: {exc}")
            return exits

        # Settle anything the broker no longer shows before deciding anything new.
        exits.extend(self._reconcile(positions))
        working = self._working_order_symbols()
        by_symbol = {p.symbol: p for p in positions}
        claimed: set = set()

        # ---- structures this bot opened -------------------------------------
        for record in self.trade_log.all():
            if record.get("status") != "open":
                continue
            legs = self._trade_legs(record)
            # Claim before deciding, and claim even when the structure is only
            # partly visible. A half-filled condor whose legs fall through to
            # the single-leg path below would have its long wing closed and its
            # short left naked — the exact failure this split exists to prevent.
            claimed.update(leg["symbol"] for leg in legs)
            if not all(leg["symbol"] in by_symbol for leg in legs):
                continue  # partially filled or partially closed — _reconcile owns it

            decision = self._trade_exit_decision(record, by_symbol)
            if not decision:
                continue
            if any(leg["symbol"] in working for leg in legs):
                self.state.log_event("info", f"{record.get('symbol')}: close order already working")
                continue

            reason, close_cost, pnl = decision
            try:
                exits.append(self._close_trade(record, legs, reason, close_cost, by_symbol))
                self.state.log_event(
                    "info",
                    f"exit {record.get('strategy', 'trade')} {record.get('underlying')}: "
                    f"{reason} at net {close_cost:.2f} (P&L ${pnl * OPTION_MULTIPLIER * int(float(record.get('contracts') or 1)):,.0f})",
                    symbol=record.get("symbol"),
                    reason=reason,
                )
            except BrokerError as exc:
                self.state.log_event("error", f"exit failed for {record.get('symbol')}: {exc}")
                self.halt(f"exit order failed for {record.get('symbol')}: {exc}")
                return exits

        # ---- anything short that this bot does not know about ---------------
        for position in positions:
            if not position.is_short or position.symbol in claimed:
                continue
            decision = self._exit_decision(position)
            if not decision or position.symbol in working:
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

    # ------------------------------------------------ structure-aware exits
    @staticmethod
    def _trade_legs(record: dict) -> List[dict]:
        """The legs of a logged trade, tolerating rows written before the library."""
        raw = record.get("legs")
        if raw:
            try:
                legs = json.loads(raw)
                if isinstance(legs, list) and legs:
                    return legs
            except (TypeError, json.JSONDecodeError):
                pass
        return [{"symbol": record.get("symbol"), "action": "sell", "ratio": 1}]

    @staticmethod
    def _close_cost(legs: List[dict], by_symbol: Dict[str, PositionView]) -> Optional[float]:
        """What it costs per share to flatten the structure right now.

        Positive = we pay to get out (the normal case for a credit structure).
        Negative = closing pays us, which is what a profitable debit trade does.
        """
        total = 0.0
        for leg in legs:
            position = by_symbol.get(leg["symbol"])
            if position is None:
                return None
            price = abs(position.current_price)
            total += price * int(leg.get("ratio", 1)) * (1 if leg["action"] == "sell" else -1)
        return round(total, 4)

    def _trade_exit_decision(
        self, record: dict, by_symbol: Dict[str, PositionView]
    ) -> Optional[tuple[str, float, float]]:
        """``(reason, close_cost, pnl_per_share)`` when a rule fires, else ``None``.

        Exits are decided on **P&L as a fraction of the premium at risk**, which
        is the one formulation that means the same thing for a credit structure
        and a debit one. "Close at 50% of the credit" and "take half the debit
        as profit" are the same rule written twice; this is the rule.
        """
        settings = self.settings
        legs = self._trade_legs(record)
        try:
            premium = float(record.get("credit") or 0.0)
        except (TypeError, ValueError):
            return None
        if premium == 0.0:
            return None

        close_cost = self._close_cost(legs, by_symbol)
        if close_cost is None:
            return None
        pnl = premium - close_cost
        at_risk = abs(premium)

        # A single leg resting at the exchange already has its bracket working.
        exchange_held = (
            getattr(self.client.capabilities, "native_brackets", False)
            and len(legs) == 1
            and "native bracket" in str(record.get("note", ""))
        )
        if not exchange_held:
            if pnl >= settings.profit_target_pct * at_risk:
                return ("profit_target", close_cost, pnl)
            if pnl <= -settings.stop_loss_multiple * at_risk:
                return ("stop_loss", close_cost, pnl)

        dtes = [by_symbol[leg["symbol"]].dte for leg in legs if by_symbol.get(leg["symbol"])]
        dtes = [d for d in dtes if d is not None]
        if dtes and min(dtes) <= settings.time_exit_dte:
            return ("time_exit", close_cost, pnl)
        return None

    def _close_trade(
        self,
        record: dict,
        legs: List[dict],
        reason: str,
        close_cost: float,
        by_symbol: Dict[str, PositionView],
    ) -> dict:
        """Flatten a whole structure in one order and mark the trade closing."""
        contracts = int(float(record.get("contracts") or 1))

        if self.settings.dry_run:
            order = {"status": "dry_run", "symbol": record.get("symbol"), "qty": contracts}
        elif len(legs) == 1:
            position = by_symbol[legs[0]["symbol"]]
            order = self._close_position(position, reason, close_cost)
            return {"symbol": position.symbol, "reason": reason, "debit": close_cost, "order": order["order"]}
        elif reason == "stop_loss":
            # A stop must actually get out. Market on the whole combo rather
            # than a limit that may never fill while the loss keeps widening.
            order = self.client.submit_combo(
                legs=legs, qty=contracts, limit_price=None, opening=False,
                client_order_id=f"bvc-x-{uuid.uuid4().hex[:12]}",
            )
        else:
            order = self.client.submit_combo(
                legs=legs, qty=contracts,
                limit_price=round(-close_cost, 2),  # we pay to close → negative cashflow
                opening=False,
                client_order_id=f"bvc-x-{uuid.uuid4().hex[:12]}",
            )

        self.trade_log.update(
            record["trade_id"],
            exit_debit=f"{close_cost:.4f}",
            exit_reason=reason,
            status="closing",
        )
        return {"symbol": record.get("symbol"), "reason": reason, "debit": close_cost, "order": order}

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

    def position_action(self, position: PositionView) -> dict:
        """What the exit rules say about this position *right now*.

        Deliberately routed through the same :meth:`_exit_decision` the trading
        loop uses, so the action shown on screen is the action the bot will take
        on its next cycle — a risk panel that disagrees with the engine is worse
        than no risk panel.
        """
        record = self._live_record(position.symbol)
        try:
            credit = float(record["credit"]) if record and record.get("credit") else abs(position.avg_entry_price)
        except (TypeError, ValueError):
            credit = abs(position.avg_entry_price)

        decision: Optional[tuple[str, float]] = None
        captured = 0.0
        if record:
            # Route through the structure-aware path so a leg of a condor shows
            # the condor's decision, not a decision about that leg alone.
            legs = self._trade_legs(record)
            by_symbol = {p.symbol: p for p in self._safe_positions()}
            if all(leg["symbol"] in by_symbol for leg in legs):
                verdict = self._trade_exit_decision(record, by_symbol)
                close_cost = self._close_cost(legs, by_symbol)
                if close_cost is not None and credit:
                    captured = (credit - close_cost) / abs(credit)
                if verdict:
                    decision = (verdict[0], verdict[1])
        if decision is None and not record:
            price = position.current_price
            captured = ((credit - price) / credit) if credit > 0 else 0.0
            decision = self._exit_decision(position)

        if decision is None:
            return {
                "action": "MONITOR", "severity": "idle", "captured": captured,
                "credit": credit, "reason": "inside the bracket",
            }

        reason, trigger = decision
        label, severity = {
            "profit_target": ("CLOSE (PROFIT)", "good"),
            "stop_loss": ("STOP OUT", "critical"),
            "time_exit": ("CLOSE (21 DTE)", "warning"),
        }[reason]
        return {
            "action": label, "severity": severity, "captured": captured,
            "credit": credit, "reason": reason, "trigger": trigger,
        }

    def _exit_decision(self, position: PositionView) -> Optional[tuple[str, float]]:
        """Return ``(reason, trigger_price)`` when an exit rule fires.

        Entry credit comes from the trade log when the position was opened by
        this bot, and from the broker's average entry price otherwise — so
        positions opened by hand are still managed rather than ignored.

        When the broker holds a real bracket, the profit target and stop are
        already resting at the exchange and must not be fired again here: doing
        so would buy the contract back twice. Only the time exit, which no
        exchange can express, stays with the bot.
        """
        settings = self.settings
        record_native = self._live_record(position.symbol)
        exchange_held = (
            getattr(self.client.capabilities, "native_brackets", False)
            and record_native is not None
            and "native bracket" in str(record_native.get("note", ""))
        )
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

        if not exchange_held:
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

    def _seed_iv_history(self) -> int:
        """Backfill the local IV store from the broker's own IV series.

        Turns IV Rank from a realised-vol proxy into a true trailing-range
        statistic the moment a venue that carries historical implied volatility
        is connected. No-op on venues that do not.
        """
        if not (self.settings.use_broker_iv_history
                and getattr(self.client.capabilities, "historical_iv", False)):
            return 0
        seeded = 0
        for symbol in self.settings.universe:
            try:
                series = self.client.get_iv_history(symbol, lookback_days=self.settings.iv_rank_window + 60)
            except (BrokerError, AttributeError) as exc:
                self.state.log_event("info", f"IV history unavailable for {symbol}: {exc}")
                continue
            for stamp, iv in series:
                if stamp and iv:
                    self.iv_store.record_on(symbol, stamp, float(iv))
                    seeded += 1
        if seeded:
            self.state.log_event("info", f"seeded {seeded} real IV observations from the broker")
        return seeded

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

    def _select_expiration(
        self,
        symbol: str,
        *,
        dte_min: Optional[int] = None,
        dte_max: Optional[int] = None,
        target: Optional[int] = None,
        after: Optional[date] = None,
    ) -> Optional[date]:
        """Pick the listed expiry closest to a DTE target inside a window.

        ``after`` is used for the back month of a calendar or diagonal: the two
        legs must be in genuinely different expiries, and on a chain with weekly
        listings the nearest match to the back-month target can otherwise land
        on the front month itself.
        """
        settings = self.settings
        expirations = self.client.get_expirations(
            symbol,
            settings.dte_min if dte_min is None else dte_min,
            settings.dte_max if dte_max is None else dte_max,
        )
        if after:
            expirations = [d for d in expirations if d > after]
        if not expirations:
            return None
        today = datetime.now(timezone.utc).date()
        goal = settings.target_dte if target is None else target
        return min(expirations, key=lambda d: abs((d - today).days - goal))

    def _load_chain(
        self,
        symbol: str,
        expiration: date,
        spot: float,
        *,
        option_type: Optional[str] = "put",
        wide: bool = False,
    ) -> List[OptionQuote]:
        """Fetch a chain around the money and fill in any missing greeks.

        ``option_type=None`` pulls both rights, which the two-sided strategies
        (condor, butterfly) need. ``wide`` widens the strike window for
        structures with legs far from the money — a 0.80-delta back-month call
        in a diagonal sits well below spot.
        """
        low, high = (0.55, 1.45) if wide else (0.70, 1.05)
        chain = self.client.get_chain(
            symbol,
            expiration,
            option_type=option_type,
            strike_low=spot * low,
            strike_high=spot * high,
        )
        return self._enrich(chain, spot, expiration)

    def _build_context(self, symbol: str, spot: float, expiration: date) -> strategies.BuildContext:
        """Assemble everything the active strategy's builder is allowed to see."""
        definition = strategies.get(self.settings.strategy)
        two_sided = definition.key in {"iron_condor", "iron_butterfly"}
        near = self._load_chain(
            symbol, expiration, spot,
            option_type=None if two_sided else self._primary_right(definition),
            wide=definition.needs_back_month,
        )
        far: List[OptionQuote] = []
        if definition.needs_back_month:
            back = self._select_expiration(
                symbol,
                dte_min=self.settings.back_month_dte - 45,
                dte_max=self.settings.back_month_dte + 60,
                target=self.settings.back_month_dte,
                after=expiration,
            )
            if back:
                far = self._load_chain(
                    symbol, back, spot,
                    option_type=self._primary_right(definition), wide=True,
                )
        return strategies.BuildContext(
            underlying=symbol, spot=spot, settings=self.settings, near=near, far=far
        )

    @staticmethod
    def _primary_right(definition: strategies.StrategyDefinition) -> str:
        """Which side of the chain a single-sided strategy trades."""
        return "call" if definition.key in {"covered_call", "call_credit_spread",
                                            "diagonal_spread"} else "put"

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
            elif settings.entry_window_enabled:
                elapsed = self.client.minutes_since_open()
                if elapsed is None:
                    blockers.append("session bounds unavailable")
                elif not (settings.entry_window_start_min <= elapsed <= settings.entry_window_end_min):
                    blockers.append(
                        f"outside the entry window — {elapsed:.0f} min after the open, "
                        f"window is {settings.entry_window_start_min}–{settings.entry_window_end_min} min"
                    )
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
                ctx = self._build_context(snapshot.symbol, snapshot.spot, snapshot.expiration)
                plan = strategies.build_plan(self.settings.strategy, ctx)
                if plan is None:
                    reject(snapshot.symbol, self._build_failure_reason(ctx))
                    continue
                illiquid = self._liquidity_reject_reason(plan)
                if illiquid:
                    reject(snapshot.symbol, illiquid)
                    continue
                capital = self._capital_reject_reason(plan, account)
                if capital:
                    reject(snapshot.symbol, capital)
                    continue
                entry = self._open_position(snapshot, plan)
                if entry:
                    return entry, rejections
                rejections.append(f"{snapshot.symbol}: net premium did not clear the minimum")
            except BrokerError as exc:
                self.state.log_event("error", f"entry failed for {snapshot.symbol}: {exc}")
                raise
        return None, rejections

    def _build_failure_reason(self, ctx: strategies.BuildContext) -> str:
        """Say which leg the chain could not supply, not just 'no trade'.

        A strategy that silently declines to build is indistinguishable from a
        broken scanner, and the two want completely different responses.
        """
        settings = self.settings
        definition = strategies.get(settings.strategy)
        if definition.needs_back_month and not ctx.far:
            return f"no back-month expiry near {settings.back_month_dte} DTE"
        return (
            f"chain has no {definition.label.lower()} within "
            f"{settings.delta_tolerance:.2f} of {settings.target_delta:.2f} delta"
            + (f" with a {settings.spread_width:g}-wide wing" if definition.is_multi_leg else "")
        )

    def _liquidity_reject_reason(self, plan: strategies.StrategyPlan) -> Optional[str]:
        """Reject wide or thin quotes — slippage is the tax on a small edge.

        Every leg is checked, not just the short one. A condor whose far wing is
        quoted 0.05 × 0.40 is a condor you cannot get out of, and the profit
        target will never be reachable at a price anyone will pay.
        """
        settings = self.settings
        for leg in plan.legs:
            quote = leg.quote
            if not quote.has_two_sided_quote:
                return f"{quote.symbol}: no two-sided quote"
            spread = quote.spread_pct
            if spread is None or spread > settings.max_spread_pct:
                return (
                    f"{quote.symbol}: bid/ask spread {spread:.0%} exceeds "
                    f"{settings.max_spread_pct:.0%} of mid"
                )
        if plan.premium_at_risk < settings.min_credit_usd:
            label = "credit" if plan.is_credit else "debit"
            return (
                f"net {label} ${plan.premium_at_risk:.2f} below the "
                f"${settings.min_credit_usd:.2f} minimum"
            )
        return None

    def _capital_reject_reason(
        self, plan: strategies.StrategyPlan, account: AccountSnapshot
    ) -> Optional[str]:
        """Check the trade fits *and* leaves the margin ceiling intact afterwards.

        The requirement comes from the strategy itself — strike notional for a
        cash-secured put, one wing's width for a condor, the debit for a
        calendar. The projected post-trade margin utilisation must still sit
        under the ceiling: the guardrail is forward looking, so a trade that
        would breach it is never sent in the first place.
        """
        settings = self.settings
        contracts = max(settings.contracts_per_trade, 1)
        requirement = plan.capital_required * contracts

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
        self, snapshot: engine.VRPSnapshot, plan: strategies.StrategyPlan
    ) -> Optional[dict]:
        """Send the plan as one order, record the exit levels, log the trade."""
        settings = self.settings
        definition = strategies.get(plan.strategy)
        contracts = max(settings.contracts_per_trade, 1)
        premium = plan.net_premium
        trade_id = uuid.uuid4().hex[:12]
        short_leg = plan.short_leg
        anchor = short_leg.quote if short_leg else plan.legs[0].quote

        if plan.premium_at_risk < settings.min_credit_usd:
            self.state.log_event(
                "info", f"{snapshot.symbol}: net premium ${plan.premium_at_risk:.2f} below the minimum"
            )
            return None

        legs = [{"symbol": leg.symbol, "action": leg.action, "ratio": leg.ratio} for leg in plan.legs]
        native = getattr(self.client.capabilities, "native_brackets", False)
        # Resting brackets only exist for a single leg. A four-leg condor's exit
        # is a net price on the whole structure, which no exchange will hold —
        # those stay managed by this loop, and the trade log says which is which.
        use_native = native and len(plan.legs) == 1 and plan.is_credit

        # ---- send the order ------------------------------------------------
        if settings.dry_run:
            order = {"status": "dry_run", "symbol": anchor.symbol, "qty": contracts}
        elif use_native:
            order = self.client.submit_bracketed_short(
                symbol=anchor.symbol,
                qty=contracts,
                credit=round(premium, 2),
                take_profit=max(round(premium * (1 - settings.profit_target_pct), 2), 0.01),
                stop_loss=round(premium * settings.stop_loss_price_multiple, 2),
                client_order_id=f"bvc-o-{trade_id}",
            )
        elif len(plan.legs) > 1:
            order = self.client.submit_combo(
                legs=legs,
                qty=contracts,
                limit_price=round(premium, 2),   # signed: + we are paid, − we pay
                opening=True,
                client_order_id=f"bvc-o-{trade_id}",
            )
        else:
            order = self.client.submit_option_order(
                symbol=anchor.symbol,
                qty=contracts,
                side="sell" if plan.legs[0].is_short else "buy",
                position_intent="sell_to_open" if plan.legs[0].is_short else "buy_to_open",
                limit_price=round(abs(premium), 2),
                client_order_id=f"bvc-o-{trade_id}",
            )

        # A single-leg fill price is the leg's price; a combo's is the net.
        fill = float(order.get("filled_avg_price") or abs(premium))
        fill = fill if premium >= 0 else -fill
        quote = self.fx.get_rate()
        self.trade_log.append(
            {
                "trade_id": trade_id,
                "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "underlying": snapshot.symbol,
                "symbol": anchor.symbol,
                "strategy": plan.strategy,
                "contracts": contracts,
                "strike": f"{anchor.strike:.2f}",
                "expiration": plan.nearest_expiration.isoformat(),
                "entry_delta": f"{anchor.delta:.4f}" if anchor.delta is not None else "",
                "entry_iv": f"{snapshot.implied_vol:.4f}" if snapshot.implied_vol else "",
                "entry_rv": f"{snapshot.reference_rv:.4f}" if snapshot.reference_rv else "",
                "entry_iv_rank": f"{snapshot.iv_rank.value:.1f}" if snapshot.iv_rank.value is not None else "",
                # ``credit`` stays the column name for backward compatibility with
                # existing logs; it is now the signed net premium.
                "credit": f"{fill:.4f}",
                "legs": json.dumps(legs),
                "capital_required": f"{plan.capital_required * contracts:.2f}",
                "max_loss": f"{plan.max_loss * contracts:.2f}" if plan.max_loss is not None else "",
                "usd_zar": f"{quote.rate:.4f}",
                "status": "open",
                "note": (
                    f"{'native' if order.get('bracket') else 'managed'} bracket: "
                    f"take profit at {settings.profit_target_pct:.0%} of "
                    f"${abs(fill):.2f} {'credit' if fill >= 0 else 'debit'}, "
                    f"stop at {settings.stop_loss_multiple:.0%} — {plan.describe()}"
                ),
            }
        )
        self.state.note_entry()

        expected = engine.theoretical_expectancy(
            abs(fill), anchor.delta or -settings.target_delta, contracts=contracts
        )
        self.state.log_event(
            "info",
            f"opened {definition.label} on {snapshot.symbol} "
            f"({plan.describe()}) for ${abs(fill):.2f} "
            f"{'credit' if fill >= 0 else 'debit'} "
            f"(IVR {snapshot.iv_rank.value:.0f}, VRP {(snapshot.vrp or 0) * 100:.1f}pts)",
            symbol=anchor.symbol,
        )
        return {
            "trade_id": trade_id,
            "symbol": anchor.symbol,
            "underlying": snapshot.symbol,
            "strategy": plan.strategy,
            "credit": fill,
            "net_premium": fill,
            "legs": legs,
            "contracts": contracts,
            "delta": anchor.delta,
            "net_delta": plan.net_delta,
            "expiration": plan.nearest_expiration.isoformat(),
            "capital_required": plan.capital_required * contracts,
            "max_loss": (plan.max_loss * contracts) if plan.max_loss is not None else None,
            "take_profit": round(abs(fill) * (1 - settings.profit_target_pct), 2),
            "stop_loss": round(abs(fill) * settings.stop_loss_price_multiple, 2),
            "expectancy_usd": expected.expectancy,
            "p_win_delta": expected.p_win,
            "p_win_breakeven": expected.breakeven_p_win,
            "native_bracket": bool(order.get("bracket")),
            "order": order,
        }

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
