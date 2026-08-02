"""
Brickvestcapitalterminal — broker and market-data layer (Alpaca).

The platform was specified against Interactive Brokers. Alpaca is used here
because it gives free market data *and* a free options-enabled paper account
with no local gateway process — IBKR requires a running TWS/IBC instance, which
no free Streamlit or Hugging Face host will keep alive.

Everything the rest of the codebase touches goes through :class:`BrokerClient`,
an intentionally small surface (account, positions, chain, bars, orders). A
future ``IBKRBrokerClient`` implementing the same methods can be dropped in
without changing ``bot.py``, ``engine.py`` or ``app.py``.

Design notes
------------
* **Connection safety.** Every outbound call runs through :meth:`_guard`, which
  retries transient failures with exponential backoff, records the outcome on a
  :class:`ConnectionHealth` object and re-raises as :class:`BrokerError`. Callers
  never see a raw SDK exception, and the UI can always render current link state.
* **Threading over asyncio.** Alpaca's trading and historical APIs are REST, so
  the concurrency win is in keeping Streamlit's script thread free, not in an
  event loop. The bot runs on a daemon thread; this client is guarded by a
  re-entrant lock and is safe to share between that thread and the UI.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import config

logger = logging.getLogger("brickvest.broker")


# ======================================================================================
# Errors and connection health
# ======================================================================================
class BrokerError(RuntimeError):
    """Any failure reaching the broker or market data, normalised."""


class MarketDataError(BrokerError):
    """The broker is reachable but the data it returned is unusable."""


@dataclass
class ConnectionHealth:
    """Rolling view of link state — this is what drives the UI's status light."""

    connected: bool = False
    last_ok: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0

    @property
    def status(self) -> str:
        """``ok`` → trade, ``degraded`` → watch, ``down`` → the bot must halt."""
        if self.consecutive_failures >= 3 or not self.connected:
            return "down"
        if self.consecutive_failures > 0:
            return "degraded"
        return "ok"

    @property
    def is_tradeable(self) -> bool:
        return self.status == "ok"

    def record_success(self) -> None:
        self.connected = True
        self.last_ok = datetime.now(timezone.utc)
        self.consecutive_failures = 0
        self.total_calls += 1

    def record_failure(self, message: str) -> None:
        self.last_error = message
        self.last_error_at = datetime.now(timezone.utc)
        self.consecutive_failures += 1
        self.total_calls += 1
        self.total_failures += 1
        if self.consecutive_failures >= 3:
            self.connected = False


# ======================================================================================
# Value objects
# ======================================================================================
@dataclass
class BrokerCapabilities:
    """What a given broker can actually do.

    The bot adapts rather than assuming. Alpaca cannot rest a bracket on an
    option leg, so the 50%/200% pair is enforced by the loop; IBKR can, so the
    exits sit at the exchange and survive the bot being stopped. That difference
    is the single biggest operational gap between the two, and it must be visible
    in code rather than buried in a comment.
    """

    name: str = "generic"
    native_brackets: bool = False        # resting OCO/bracket on option legs
    historical_iv: bool = False          # real implied-vol history, not a proxy
    multi_leg_orders: bool = True
    paper: bool = True
    requires_gateway: bool = False       # needs TWS/IB Gateway running locally


@dataclass
class AccountSnapshot:
    """Normalised account state with the margin guardrail pre-computed."""

    equity: float = 0.0
    last_equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    options_buying_power: float = 0.0
    maintenance_margin: float = 0.0
    initial_margin: float = 0.0
    long_market_value: float = 0.0
    short_market_value: float = 0.0
    portfolio_value: float = 0.0
    options_trading_level: Optional[int] = None
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False
    currency: str = "USD"
    account_number: str = ""
    asof: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def margin_utilization(self) -> float:
        """Maintenance margin as a fraction of equity — the fat-tail governor."""
        return (self.maintenance_margin / self.equity) if self.equity > 0 else 0.0

    @property
    def day_pnl(self) -> float:
        return self.equity - self.last_equity if self.last_equity else 0.0

    @property
    def is_healthy(self) -> bool:
        return not (self.trading_blocked or self.account_blocked)


@dataclass
class PositionView:
    """A broker position, with option contracts decoded from their OCC symbol."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    asset_class: str = "us_equity"
    underlying: Optional[str] = None
    expiration: Optional[date] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # "call" | "put"

    @property
    def is_option(self) -> bool:
        return self.option_type is not None

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def dte(self) -> Optional[int]:
        if not self.expiration:
            return None
        return (self.expiration - datetime.now(timezone.utc).date()).days


@dataclass
class OptionQuote:
    """One contract's tradeable state: quote, greeks and derived mid price."""

    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    open_interest: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None and self.ask > 0:
            return round((self.bid + self.ask) / 2.0, 4)
        return self.last

    @property
    def spread_pct(self) -> Optional[float]:
        """Bid/ask width as a fraction of the mid — the liquidity gate."""
        mid = self.mid
        if not mid or mid <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / mid

    @property
    def has_two_sided_quote(self) -> bool:
        return bool(self.bid and self.ask and self.ask >= self.bid > 0)

    @property
    def dte(self) -> int:
        return (self.expiration - datetime.now(timezone.utc).date()).days


# ======================================================================================
# OCC symbol handling
# ======================================================================================
_OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ_symbol(symbol: str) -> Optional[dict]:
    """Decode an OCC option symbol, e.g. ``SPY251219P00600000``.

    Returns ``None`` for equity symbols so callers can use it as a type test.
    """
    match = _OCC_RE.match(symbol.strip().upper())
    if not match:
        return None
    parts = match.groupdict()
    return {
        "underlying": parts["root"],
        "expiration": date(2000 + int(parts["yy"]), int(parts["mm"]), int(parts["dd"])),
        "option_type": "call" if parts["cp"] == "C" else "put",
        "strike": int(parts["strike"]) / 1000.0,
    }


# ======================================================================================
# The client
# ======================================================================================
class BrokerClient:
    """Thread-safe Alpaca trading + market-data client with health tracking."""

    #: Alpaca: no resting brackets on options, no historical IV.
    capabilities = BrokerCapabilities(
        name="alpaca", native_brackets=False, historical_iv=False,
        multi_leg_orders=True, requires_gateway=False,
    )

    def __init__(self, settings: Optional[config.Settings] = None) -> None:
        self.settings = settings or config.load_settings()
        self.capabilities = BrokerCapabilities(**{**self.capabilities.__dict__, "paper": self.settings.paper})
        self.health = ConnectionHealth()
        self._lock = threading.RLock()
        self._trading = None
        self._stock_data = None
        self._option_data = None
        self._contract_cache: Dict[str, tuple[float, list]] = {}

    # ------------------------------------------------------------------ wiring
    def connect(self) -> bool:
        """Instantiate the SDK clients and verify credentials with one call."""
        if not self.settings.credentials_present:
            self.health.record_failure("ALPACA_API_KEY / ALPACA_SECRET_KEY not configured")
            return False
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient

            with self._lock:
                self._trading = TradingClient(
                    api_key=self.settings.alpaca_api_key,
                    secret_key=self.settings.alpaca_secret_key,
                    paper=self.settings.paper,
                )
                self._stock_data = StockHistoricalDataClient(
                    api_key=self.settings.alpaca_api_key,
                    secret_key=self.settings.alpaca_secret_key,
                )
                self._option_data = OptionHistoricalDataClient(
                    api_key=self.settings.alpaca_api_key,
                    secret_key=self.settings.alpaca_secret_key,
                )
            self.get_account()  # credential smoke test; raises on bad keys
            return True
        except BrokerError:
            return False
        except Exception as exc:  # SDK import or construction failure
            self.health.record_failure(f"connect failed: {exc}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._trading is not None and self.health.connected

    def _require(self, client, name: str):
        if client is None:
            raise BrokerError(f"{name} client is not connected — call connect() first")
        return client

    def _guard(self, label: str, fn: Callable[[], Any], *, retries: int = 2, backoff: float = 1.5) -> Any:
        """Run an SDK call with retries, health accounting and error normalising.

        This is the single choke point that makes the fail-safe possible: nothing
        reaches the broker without its result being reflected in ``self.health``.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                with self._lock:
                    result = fn()
                self.health.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                message = f"{label}: {type(exc).__name__}: {exc}"
                logger.warning("broker call failed (attempt %s/%s) — %s", attempt + 1, retries + 1, message)
                if attempt < retries:
                    time.sleep(backoff ** (attempt + 1))
        self.health.record_failure(f"{label}: {last_exc}")
        raise BrokerError(f"{label} failed after {retries + 1} attempts: {last_exc}") from last_exc

    # ----------------------------------------------------------------- account
    def get_account(self) -> AccountSnapshot:
        """Current account state, including the maintenance-margin utilisation."""
        client = self._require(self._trading, "trading")
        raw = self._guard("get_account", client.get_account)

        def num(value, default=0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        level = getattr(raw, "options_trading_level", None)
        return AccountSnapshot(
            equity=num(raw.equity),
            last_equity=num(raw.last_equity),
            cash=num(raw.cash),
            buying_power=num(raw.buying_power),
            options_buying_power=num(getattr(raw, "options_buying_power", 0)),
            maintenance_margin=num(raw.maintenance_margin),
            initial_margin=num(raw.initial_margin),
            long_market_value=num(raw.long_market_value),
            short_market_value=num(raw.short_market_value),
            portfolio_value=num(raw.portfolio_value),
            options_trading_level=int(level) if level is not None else None,
            pattern_day_trader=bool(getattr(raw, "pattern_day_trader", False)),
            trading_blocked=bool(getattr(raw, "trading_blocked", False)),
            account_blocked=bool(getattr(raw, "account_blocked", False)),
            currency=str(getattr(raw, "currency", "USD") or "USD"),
            account_number=str(getattr(raw, "account_number", "") or ""),
        )

    def get_positions(self) -> List[PositionView]:
        """All open positions, with option contracts decoded from OCC symbols."""
        client = self._require(self._trading, "trading")
        raw_positions = self._guard("get_all_positions", client.get_all_positions)

        views: List[PositionView] = []
        for pos in raw_positions:
            def num(value, default=0.0) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            view = PositionView(
                symbol=str(pos.symbol),
                qty=num(pos.qty),
                avg_entry_price=num(pos.avg_entry_price),
                current_price=num(pos.current_price),
                market_value=num(pos.market_value),
                cost_basis=num(pos.cost_basis),
                unrealized_pl=num(pos.unrealized_pl),
                unrealized_plpc=num(pos.unrealized_plpc),
                asset_class=str(getattr(pos.asset_class, "value", pos.asset_class)),
            )
            decoded = parse_occ_symbol(view.symbol)
            if decoded:
                view.underlying = decoded["underlying"]
                view.expiration = decoded["expiration"]
                view.strike = decoded["strike"]
                view.option_type = decoded["option_type"]
            views.append(view)
        return views

    def get_option_positions(self) -> List[PositionView]:
        return [p for p in self.get_positions() if p.is_option]

    # ------------------------------------------------------------------- clock
    def is_market_open(self) -> bool:
        client = self._require(self._trading, "trading")
        clock = self._guard("get_clock", client.get_clock)
        return bool(getattr(clock, "is_open", False))

    def get_session_bounds(self, day: Optional[date] = None) -> Optional[tuple[datetime, datetime]]:
        """Today's real open and close, in UTC, or ``None`` if it is not a session day.

        Read from the exchange calendar rather than assumed to be 13:30–20:00
        UTC: half-days close at 18:00 UTC, and the US/Eastern offset shifts
        twice a year. An entry window measured from a guessed open would drift
        by an hour for months at a time.
        """
        from zoneinfo import ZoneInfo

        from alpaca.trading.requests import GetCalendarRequest

        client = self._require(self._trading, "trading")
        day = day or datetime.now(timezone.utc).date()
        sessions = self._guard(
            "get_calendar",
            lambda: client.get_calendar(GetCalendarRequest(start=day, end=day)),
        )
        for session in sessions or []:
            if getattr(session, "date", None) != day:
                continue
            eastern = ZoneInfo("America/New_York")
            open_dt = datetime.combine(session.date, session.open, tzinfo=eastern)
            close_dt = datetime.combine(session.date, session.close, tzinfo=eastern)
            return open_dt.astimezone(timezone.utc), close_dt.astimezone(timezone.utc)
        return None

    def minutes_since_open(self) -> Optional[float]:
        """Minutes elapsed since today's open, or ``None`` outside a session."""
        bounds = self.get_session_bounds()
        if not bounds:
            return None
        open_dt, close_dt = bounds
        now = datetime.now(timezone.utc)
        if not (open_dt <= now <= close_dt):
            return None
        return (now - open_dt).total_seconds() / 60.0

    def get_clock(self) -> dict:
        client = self._require(self._trading, "trading")
        clock = self._guard("get_clock", client.get_clock)
        return {
            "is_open": bool(getattr(clock, "is_open", False)),
            "next_open": getattr(clock, "next_open", None),
            "next_close": getattr(clock, "next_close", None),
            "timestamp": getattr(clock, "timestamp", None),
        }

    # ------------------------------------------------------------- equity data
    def get_daily_bars(self, symbols: Sequence[str], lookback_days: int = 400) -> Dict[str, dict]:
        """Daily OHLC history keyed by symbol, as parallel lists for the engine."""
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._require(self._stock_data, "stock data")
        symbols = [s.upper() for s in symbols]
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=lookback_days),
            feed=DataFeed(self.settings.stock_feed),
        )
        barset = self._guard("get_stock_bars", lambda: client.get_stock_bars(request))

        out: Dict[str, dict] = {}
        data = getattr(barset, "data", {}) or {}
        for symbol in symbols:
            bars = data.get(symbol) or []
            out[symbol] = {
                "timestamp": [getattr(b, "timestamp", None) for b in bars],
                "open": [float(b.open) for b in bars],
                "high": [float(b.high) for b in bars],
                "low": [float(b.low) for b in bars],
                "close": [float(b.close) for b in bars],
                "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
            }
        return out

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """Latest trade price, falling back to the most recent daily close.

        The IEX free feed can be thin outside regular hours, so the close is a
        legitimate second source rather than an error path.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockLatestTradeRequest

        client = self._require(self._stock_data, "stock data")
        symbol = symbol.upper()
        try:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=DataFeed(self.settings.stock_feed))
            trades = self._guard("get_stock_latest_trade", lambda: client.get_stock_latest_trade(request), retries=1)
            trade = trades.get(symbol) if isinstance(trades, dict) else trades
            price = float(getattr(trade, "price", 0) or 0)
            if price > 0:
                return price
        except BrokerError:
            pass

        bars = self.get_daily_bars([symbol], lookback_days=10).get(symbol, {})
        closes = bars.get("close") or []
        return float(closes[-1]) if closes else None

    # ------------------------------------------------------------- option data
    def get_expirations(self, underlying: str, dte_min: int, dte_max: int) -> List[date]:
        """Tradeable expiration dates for an underlying inside a DTE window."""
        from alpaca.trading.enums import AssetStatus
        from alpaca.trading.requests import GetOptionContractsRequest

        client = self._require(self._trading, "trading")
        today = datetime.now(timezone.utc).date()
        request = GetOptionContractsRequest(
            underlying_symbols=[underlying.upper()],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=today + timedelta(days=max(dte_min, 0)),
            expiration_date_lte=today + timedelta(days=max(dte_max, dte_min)),
            limit=10_000,
        )
        response = self._guard("get_option_contracts", lambda: client.get_option_contracts(request))
        contracts = getattr(response, "option_contracts", None) or []
        expirations = {c.expiration_date for c in contracts if getattr(c, "expiration_date", None)}
        return sorted(expirations)

    def get_chain(
        self,
        underlying: str,
        expiration: date,
        option_type: Optional[str] = None,
        *,
        strike_low: Optional[float] = None,
        strike_high: Optional[float] = None,
    ) -> List[OptionQuote]:
        """Option chain snapshots for one expiration, normalised to OptionQuote.

        The free "indicative" feed frequently omits greeks and implied volatility.
        Those gaps are filled by the engine (Black-Scholes inversion off the mid
        price) in :meth:`enrich_chain`, so the strategy never depends on a paid
        data entitlement.
        """
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType

        client = self._require(self._option_data, "option data")
        request = OptionChainRequest(
            underlying_symbol=underlying.upper(),
            feed=OptionsFeed(self.settings.options_feed),
            expiration_date=expiration,
            type=ContractType(option_type) if option_type else None,
            strike_price_gte=strike_low,
            strike_price_lte=strike_high,
        )
        snapshots = self._guard("get_option_chain", lambda: client.get_option_chain(request)) or {}

        quotes: List[OptionQuote] = []
        for symbol, snap in snapshots.items():
            decoded = parse_occ_symbol(symbol)
            if not decoded or decoded["expiration"] != expiration:
                continue
            if option_type and decoded["option_type"] != option_type:
                continue

            quote = getattr(snap, "latest_quote", None)
            trade = getattr(snap, "latest_trade", None)
            greeks = getattr(snap, "greeks", None)
            quotes.append(
                OptionQuote(
                    symbol=symbol,
                    underlying=decoded["underlying"],
                    expiration=decoded["expiration"],
                    strike=decoded["strike"],
                    option_type=decoded["option_type"],
                    bid=float(getattr(quote, "bid_price", 0) or 0) or None,
                    ask=float(getattr(quote, "ask_price", 0) or 0) or None,
                    last=float(getattr(trade, "price", 0) or 0) or None,
                    implied_volatility=_positive_or_none(getattr(snap, "implied_volatility", None)),
                    delta=_float_or_none(getattr(greeks, "delta", None)),
                    gamma=_float_or_none(getattr(greeks, "gamma", None)),
                    theta=_float_or_none(getattr(greeks, "theta", None)),
                    vega=_float_or_none(getattr(greeks, "vega", None)),
                )
            )
        quotes.sort(key=lambda q: q.strike)
        return quotes

    def get_option_quote(self, symbol: str) -> Optional[OptionQuote]:
        """Latest quote for a single contract — used by the exit manager."""
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionSnapshotRequest

        decoded = parse_occ_symbol(symbol)
        if not decoded:
            return None
        client = self._require(self._option_data, "option data")
        request = OptionSnapshotRequest(symbol_or_symbols=symbol, feed=OptionsFeed(self.settings.options_feed))
        snapshots = self._guard("get_option_snapshot", lambda: client.get_option_snapshot(request)) or {}
        snap = snapshots.get(symbol)
        if snap is None:
            return None

        quote = getattr(snap, "latest_quote", None)
        trade = getattr(snap, "latest_trade", None)
        greeks = getattr(snap, "greeks", None)
        return OptionQuote(
            symbol=symbol,
            underlying=decoded["underlying"],
            expiration=decoded["expiration"],
            strike=decoded["strike"],
            option_type=decoded["option_type"],
            bid=float(getattr(quote, "bid_price", 0) or 0) or None,
            ask=float(getattr(quote, "ask_price", 0) or 0) or None,
            last=float(getattr(trade, "price", 0) or 0) or None,
            implied_volatility=_positive_or_none(getattr(snap, "implied_volatility", None)),
            delta=_float_or_none(getattr(greeks, "delta", None)),
            gamma=_float_or_none(getattr(greeks, "gamma", None)),
            theta=_float_or_none(getattr(greeks, "theta", None)),
            vega=_float_or_none(getattr(greeks, "vega", None)),
        )

    # ------------------------------------------------------------------ orders
    def submit_option_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        position_intent: str,
        limit_price: Optional[float] = None,
        *,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Send a single-leg option order.

        A limit price is used whenever one is supplied — market orders on
        options routinely fill through a wide indicative spread, which would eat
        the very premium the strategy exists to collect.
        """
        from alpaca.trading.enums import OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        client = self._require(self._trading, "trading")
        common = dict(
            symbol=symbol,
            qty=abs(int(qty)),
            side=OrderSide(side),
            time_in_force=TimeInForce(time_in_force),
            position_intent=PositionIntent(position_intent),
            client_order_id=client_order_id,
        )
        if limit_price is not None and limit_price > 0:
            request = LimitOrderRequest(limit_price=round(float(limit_price), 2), **common)
        else:
            request = MarketOrderRequest(**common)

        order = self._guard("submit_order", lambda: client.submit_order(request), retries=1)
        return _order_to_dict(order)

    def submit_vertical_spread(
        self,
        short_symbol: str,
        long_symbol: str,
        qty: int,
        limit_price: Optional[float] = None,
        *,
        opening: bool = True,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Send a two-leg vertical as a single multi-leg (``mleg``) order.

        Both legs price together, so the credit is never legged into at a worse
        net price. Requires options level 3 on the account.
        """
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, OptionLegRequest

        client = self._require(self._trading, "trading")
        short_side = OrderSide.SELL if opening else OrderSide.BUY
        long_side = OrderSide.BUY if opening else OrderSide.SELL
        short_intent = PositionIntent.SELL_TO_OPEN if opening else PositionIntent.BUY_TO_CLOSE
        long_intent = PositionIntent.BUY_TO_OPEN if opening else PositionIntent.SELL_TO_CLOSE

        legs = [
            OptionLegRequest(symbol=short_symbol, ratio_qty=1, side=short_side, position_intent=short_intent),
            OptionLegRequest(symbol=long_symbol, ratio_qty=1, side=long_side, position_intent=long_intent),
        ]
        common = dict(
            qty=abs(int(qty)),
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            legs=legs,
            client_order_id=client_order_id,
        )
        # A net credit is submitted as a negative limit price on a multi-leg order.
        if limit_price is not None:
            request = LimitOrderRequest(limit_price=round(float(limit_price), 2), **common)
        else:
            request = MarketOrderRequest(**common)

        order = self._guard("submit_mleg_order", lambda: client.submit_order(request), retries=1)
        return _order_to_dict(order)

    def close_position(self, symbol: str, qty: Optional[int] = None) -> dict:
        """Flatten a position at market — the emergency and time-exit path."""
        from alpaca.trading.requests import ClosePositionRequest

        client = self._require(self._trading, "trading")
        close_options = ClosePositionRequest(qty=str(abs(int(qty)))) if qty else None
        order = self._guard(
            "close_position",
            lambda: client.close_position(symbol, close_options=close_options),
            retries=1,
        )
        return _order_to_dict(order)

    def get_orders(self, status: str = "open", limit: int = 50) -> List[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        client = self._require(self._trading, "trading")
        request = GetOrdersRequest(status=QueryOrderStatus(status), limit=limit)
        orders = self._guard("get_orders", lambda: client.get_orders(request))
        return [_order_to_dict(order) for order in (orders or [])]

    def cancel_order(self, order_id: str) -> None:
        client = self._require(self._trading, "trading")
        self._guard("cancel_order", lambda: client.cancel_order_by_id(order_id), retries=1)

    def cancel_all_orders(self) -> None:
        client = self._require(self._trading, "trading")
        self._guard("cancel_orders", client.cancel_orders, retries=1)


# ======================================================================================
# Helpers
# ======================================================================================
def _float_or_none(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _positive_or_none(value) -> Optional[float]:
    parsed = _float_or_none(value)
    return parsed if parsed and parsed > 0 else None


def _order_to_dict(order) -> dict:
    """Flatten an SDK order into plain JSON-able data for logs and the UI."""
    if order is None:
        return {}
    if isinstance(order, dict):
        return order

    def enum_value(attr):
        value = getattr(order, attr, None)
        return getattr(value, "value", value)

    return {
        "id": str(getattr(order, "id", "") or ""),
        "client_order_id": str(getattr(order, "client_order_id", "") or ""),
        "symbol": str(getattr(order, "symbol", "") or ""),
        "qty": _float_or_none(getattr(order, "qty", None)),
        "filled_qty": _float_or_none(getattr(order, "filled_qty", None)),
        "filled_avg_price": _float_or_none(getattr(order, "filled_avg_price", None)),
        "limit_price": _float_or_none(getattr(order, "limit_price", None)),
        "side": enum_value("side"),
        "order_class": enum_value("order_class"),
        "status": enum_value("status"),
        "submitted_at": str(getattr(order, "submitted_at", "") or ""),
    }


def build_client(settings: Optional[config.Settings] = None) -> BrokerClient:
    """Construct and connect the configured broker. Never raises — inspect ``.health``.

    ``BVC_BROKER`` selects the venue. IBKR is imported lazily so a deployment
    that only uses Alpaca never needs ib_async installed, and vice versa.
    """
    settings = settings or config.load_settings()
    broker = (settings.broker or "alpaca").lower()

    if broker in {"ibkr", "ib", "interactive_brokers"}:
        try:
            from ibkr_client import IBKRClient

            client: BrokerClient = IBKRClient(settings)
        except ImportError as exc:
            client = BrokerClient(settings)
            client.health.record_failure(
                f"BVC_BROKER=ibkr but ib_async is not installed ({exc}). "
                "pip install ib_async, or set BVC_BROKER=alpaca."
            )
            return client
    else:
        client = BrokerClient(settings)

    client.connect()
    return client
