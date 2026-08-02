"""
Brickvestcapitalterminal — Interactive Brokers connector.

Implements the same surface as :class:`broker_client.BrokerClient`, so
``bot.py``, ``engine.py``, ``backtest.py`` and ``app.py`` are unchanged: select
it with ``BVC_BROKER=ibkr`` and everything downstream carries on working.

Why IBKR is worth the gateway
-----------------------------
Two things it does that Alpaca's free tier cannot, and both of them repair a
weakness this platform has had to work around:

1. **Resting bracket orders on option legs.** The 50% profit target and 200%
   stop sit at the exchange rather than being re-evaluated by the loop every
   five minutes. A stopped bot no longer means unmanaged positions — the single
   largest operational risk in the Alpaca build.
2. **Real historical implied volatility** via ``OPTION_IMPLIED_VOLATILITY``
   bars. IV Rank stops being ranked against a realised-vol proxy, and the
   backtest can price on the volatility that actually traded instead of the
   modelled ``RV + vrp_points`` surface. That surface was the biggest assumption
   in the whole system; this removes it.

The cost is a running TWS or IB Gateway on the same host, which rules out free
Streamlit hosting. Run it beside ``terminal.py --live`` on a VPS.

Threading model
---------------
``ib_async`` is asyncio-native while the rest of the platform is threaded: the
bot on a daemon thread, Streamlit on its own. So this module owns a private
event loop running on a dedicated thread, and every public method marshals its
coroutine onto that loop with ``run_coroutine_threadsafe``. Callers stay
synchronous and see the same blocking API the Alpaca client presents. One loop,
one connection, one place where the async boundary lives.

Symbols
-------
Externally the platform speaks OCC (``SPY251219P00600000``). IBKR speaks
``Contract`` objects. Translation happens here and nowhere else, which is what
keeps this a drop-in replacement.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

import config
from broker_client import (
    AccountSnapshot,
    BrokerCapabilities,
    BrokerClient,
    BrokerError,
    ConnectionHealth,
    OptionQuote,
    PositionView,
    parse_occ_symbol,
)

logger = logging.getLogger("brickvest.ibkr")

#: IBKR account-summary tags mapped onto :class:`AccountSnapshot`.
ACCOUNT_TAGS = (
    "NetLiquidation", "TotalCashValue", "BuyingPower", "MaintMarginReq",
    "InitMarginReq", "GrossPositionValue", "AvailableFunds", "ExcessLiquidity",
    "PreviousDayEquityWithLoanValue",
)


def occ_to_parts(symbol: str) -> dict:
    """OCC string → the fields IBKR needs to build an ``Option`` contract."""
    parsed = parse_occ_symbol(symbol)
    if not parsed:
        raise BrokerError(f"not an option symbol: {symbol}")
    return parsed


def parts_to_occ(underlying: str, expiration: date, right: str, strike: float) -> str:
    """IBKR contract fields → the OCC string the rest of the platform uses."""
    return (
        f"{underlying.upper()}{expiration:%y%m%d}"
        f"{'C' if str(right).upper().startswith('C') else 'P'}"
        f"{int(round(strike * 1000)):08d}"
    )


class _LoopThread:
    """A private asyncio loop on its own thread.

    ib_async requires that every call touching the connection happen on the loop
    that owns it. Creating one here — rather than borrowing whatever loop the
    caller happens to be on — means Streamlit reruns, the bot thread and the
    console deck all share a single, stable connection.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True, name="bvc-ibkr-loop")
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro, timeout: float = 30.0):
        """Run a coroutine on the loop and block until it returns."""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            raise BrokerError(f"IBKR call timed out after {timeout:.0f}s")

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)


class IBKRClient(BrokerClient):
    """Interactive Brokers via ib_async, presenting the platform's broker API."""

    capabilities = BrokerCapabilities(
        name="ibkr",
        native_brackets=True,
        historical_iv=True,
        multi_leg_orders=True,
        requires_gateway=True,
    )

    def __init__(self, settings: Optional[config.Settings] = None) -> None:
        self.settings = settings or config.load_settings()
        self.capabilities = BrokerCapabilities(
            **{**IBKRClient.capabilities.__dict__, "paper": self.settings.paper}
        )
        self.health = ConnectionHealth()
        self._lock = threading.RLock()
        self._loop: Optional[_LoopThread] = None
        self._ib = None
        self._con_id_cache: Dict[str, int] = {}

    # ------------------------------------------------------------------ wiring
    def connect(self) -> bool:
        """Connect to TWS/Gateway. Never raises — inspect ``.health``."""
        try:
            from ib_async import IB
        except ImportError as exc:
            self.health.record_failure(f"ib_async not installed: {exc}")
            return False

        try:
            with self._lock:
                if self._loop is None:
                    self._loop = _LoopThread()
                if self._ib is None:
                    self._ib = IB()

                if not self._ib.isConnected():
                    self._loop.run(
                        self._ib.connectAsync(
                            host=self.settings.ibkr_host,
                            port=self.settings.ibkr_port,
                            clientId=self.settings.ibkr_client_id,
                            readonly=self.settings.ibkr_readonly,
                            timeout=self.settings.ibkr_timeout,
                        ),
                        timeout=self.settings.ibkr_timeout + 5,
                    )
            self.get_account()  # prove the link end to end, not just the socket
            return True
        except BrokerError:
            return False
        except Exception as exc:
            self.health.record_failure(
                f"connect failed: {exc}. Is TWS/IB Gateway running on "
                f"{self.settings.ibkr_host}:{self.settings.ibkr_port} with the API enabled?"
            )
            return False

    @property
    def is_connected(self) -> bool:
        return bool(self._ib is not None and self._ib.isConnected() and self.health.connected)

    def disconnect(self) -> None:
        try:
            if self._ib is not None and self._ib.isConnected():
                self._ib.disconnect()
        finally:
            if self._loop is not None:
                self._loop.stop()
                self._loop = None

    def _guard(self, label: str, fn: Callable[[], Any], *, retries: int = 1, backoff: float = 1.5) -> Any:
        """Same choke point as the Alpaca client: retry, record health, normalise."""
        import time as _time

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                with self._lock:
                    result = fn()
                self.health.record_success()
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning("IBKR %s failed (%s/%s): %s", label, attempt + 1, retries + 1, exc)
                if attempt < retries:
                    _time.sleep(backoff ** (attempt + 1))
        self.health.record_failure(f"{label}: {last_exc}")
        raise BrokerError(f"{label} failed after {retries + 1} attempts: {last_exc}") from last_exc

    def _run(self, coro, timeout: float = 30.0):
        if self._loop is None:
            raise BrokerError("IBKR loop is not running — call connect() first")
        return self._loop.run(coro, timeout=timeout)

    # ----------------------------------------------------------------- account
    def get_account(self) -> AccountSnapshot:
        """Account summary mapped onto the platform's snapshot.

        ``MaintMarginReq`` is the figure the 50% margin guardrail reads, and it
        is IBKR's own real-time requirement rather than a derived estimate.
        """
        ib = self._require_ib()
        values = self._guard(
            "accountSummary",
            lambda: self._run(ib.accountSummaryAsync(self.settings.ibkr_account or "")),
        )

        summary: Dict[str, float] = {}
        account_id = self.settings.ibkr_account or ""
        for value in values or []:
            if value.tag in ACCOUNT_TAGS and value.currency in ("USD", ""):
                try:
                    summary[value.tag] = float(value.value)
                except (TypeError, ValueError):
                    continue
            if not account_id and getattr(value, "account", ""):
                account_id = value.account

        equity = summary.get("NetLiquidation", 0.0)
        return AccountSnapshot(
            equity=equity,
            last_equity=summary.get("PreviousDayEquityWithLoanValue", equity),
            cash=summary.get("TotalCashValue", 0.0),
            buying_power=summary.get("BuyingPower", 0.0),
            # IBKR has no separate options buying power; available funds is the
            # honest analogue for "what can still be committed".
            options_buying_power=summary.get("AvailableFunds", 0.0),
            maintenance_margin=summary.get("MaintMarginReq", 0.0),
            initial_margin=summary.get("InitMarginReq", 0.0),
            long_market_value=summary.get("GrossPositionValue", 0.0),
            portfolio_value=equity,
            account_number=account_id,
            currency="USD",
        )

    def get_positions(self) -> List[PositionView]:
        self._require_ib()
        items = self._guard("portfolio", lambda: self._run(self._portfolio_async()))

        views: List[PositionView] = []
        for item in items or []:
            contract = item.contract
            sec_type = getattr(contract, "secType", "")
            if sec_type == "OPT":
                expiry = _parse_ib_date(contract.lastTradeDateOrContractMonth)
                symbol = parts_to_occ(contract.symbol, expiry, contract.right, float(contract.strike))
                option_type = "call" if str(contract.right).upper().startswith("C") else "put"
            else:
                symbol, expiry, option_type = contract.symbol, None, None

            qty = float(item.position)
            avg_cost = float(item.averageCost or 0.0)
            multiplier = float(getattr(contract, "multiplier", "") or 1) if sec_type == "OPT" else 1.0
            # IBKR quotes averageCost per contract (premium × multiplier) on
            # options; the platform works in per-share premium everywhere else.
            avg_entry = (avg_cost / multiplier) if multiplier else avg_cost

            views.append(
                PositionView(
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=abs(avg_entry),
                    current_price=float(item.marketPrice or 0.0),
                    market_value=float(item.marketValue or 0.0),
                    cost_basis=avg_cost * qty,
                    unrealized_pl=float(item.unrealizedPNL or 0.0),
                    unrealized_plpc=(
                        float(item.unrealizedPNL or 0.0) / abs(avg_cost * qty)
                        if avg_cost and qty else 0.0
                    ),
                    asset_class="us_option" if sec_type == "OPT" else "us_equity",
                    underlying=contract.symbol if sec_type == "OPT" else None,
                    expiration=expiry,
                    strike=float(contract.strike) if sec_type == "OPT" else None,
                    option_type=option_type,
                )
            )
        return views

    async def _portfolio_async(self):
        return self._ib.portfolio()

    def get_option_positions(self) -> List[PositionView]:
        return [p for p in self.get_positions() if p.is_option]

    # ------------------------------------------------------------------- clock
    def get_session_bounds(self, day: Optional[date] = None) -> Optional[tuple[datetime, datetime]]:
        """Today's real session from the exchange's own trading hours."""
        from zoneinfo import ZoneInfo

        from ib_async import Stock

        ib = self._require_ib()
        day = day or datetime.now(timezone.utc).date()
        details = self._guard(
            "contractDetails",
            lambda: self._run(ib.reqContractDetailsAsync(Stock("SPY", "SMART", "USD"))),
        )
        if not details:
            return None

        detail = details[0]
        hours = getattr(detail, "liquidHours", "") or getattr(detail, "tradingHours", "")
        zone = ZoneInfo(getattr(detail, "timeZoneId", None) or "America/New_York")
        # Format: "20260803:0930-20260803:1600;20260804:CLOSED"
        for chunk in str(hours).split(";"):
            if ":" not in chunk or "CLOSED" in chunk.upper():
                continue
            try:
                open_part, close_part = chunk.split("-")
                open_dt = datetime.strptime(open_part.strip(), "%Y%m%d:%H%M").replace(tzinfo=zone)
                close_raw = close_part.strip()
                if ":" not in close_raw:
                    continue
                close_dt = datetime.strptime(close_raw, "%Y%m%d:%H%M").replace(tzinfo=zone)
            except ValueError:
                continue
            if open_dt.date() == day:
                return open_dt.astimezone(timezone.utc), close_dt.astimezone(timezone.utc)
        return None

    def minutes_since_open(self) -> Optional[float]:
        bounds = self.get_session_bounds()
        if not bounds:
            return None
        open_dt, close_dt = bounds
        now = datetime.now(timezone.utc)
        if not (open_dt <= now <= close_dt):
            return None
        return (now - open_dt).total_seconds() / 60.0

    def is_market_open(self) -> bool:
        bounds = self.get_session_bounds()
        if not bounds:
            return False
        now = datetime.now(timezone.utc)
        return bounds[0] <= now <= bounds[1]

    def get_clock(self) -> dict:
        bounds = self.get_session_bounds()
        return {
            "is_open": self.is_market_open(),
            "next_open": bounds[0] if bounds else None,
            "next_close": bounds[1] if bounds else None,
            "timestamp": datetime.now(timezone.utc),
        }

    # ------------------------------------------------------------- equity data
    def get_daily_bars(self, symbols: Sequence[str], lookback_days: int = 400) -> Dict[str, dict]:
        from ib_async import Stock

        ib = self._require_ib()
        out: Dict[str, dict] = {}
        for symbol in [s.upper() for s in symbols]:
            contract = Stock(symbol, "SMART", "USD")
            bars = self._guard(
                f"historicalData:{symbol}",
                lambda c=contract: self._run(
                    ib.reqHistoricalDataAsync(
                        c, endDateTime="", durationStr=f"{max(lookback_days, 30)} D",
                        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True,
                    ),
                    timeout=60,
                ),
                retries=0,
            )
            out[symbol] = {
                "timestamp": [getattr(b, "date", None) for b in bars or []],
                "open": [float(b.open) for b in bars or []],
                "high": [float(b.high) for b in bars or []],
                "low": [float(b.low) for b in bars or []],
                "close": [float(b.close) for b in bars or []],
                "volume": [float(getattr(b, "volume", 0) or 0) for b in bars or []],
            }
        return out

    def get_iv_history(self, symbol: str, lookback_days: int = 400) -> List[tuple]:
        """Real daily implied volatility — the thing no free feed provides.

        Returns ``(date, iv)`` pairs from IBKR's ``OPTION_IMPLIED_VOLATILITY``
        series. With this, IV Rank is a true trailing-range statistic rather
        than the realised-vol proxy the Alpaca build has to fall back on.
        """
        from ib_async import Stock

        ib = self._require_ib()
        contract = Stock(symbol.upper(), "SMART", "USD")
        bars = self._guard(
            f"ivHistory:{symbol}",
            lambda: self._run(
                ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr=f"{max(lookback_days, 30)} D",
                    barSizeSetting="1 day", whatToShow="OPTION_IMPLIED_VOLATILITY", useRTH=True,
                ),
                timeout=60,
            ),
            retries=0,
        )
        series = []
        for bar in bars or []:
            close = float(getattr(bar, "close", 0) or 0)
            if close > 0:
                series.append((getattr(bar, "date", None), close))
        return series

    def get_spot_price(self, symbol: str) -> Optional[float]:
        from ib_async import Stock

        ib = self._require_ib()
        contract = Stock(symbol.upper(), "SMART", "USD")
        try:
            tickers = self._guard(
                f"spot:{symbol}",
                lambda: self._run(ib.reqTickersAsync(contract), timeout=20),
                retries=0,
            )
            for ticker in tickers or []:
                for field in ("last", "close", "marketPrice"):
                    value = getattr(ticker, field, None)
                    if callable(value):
                        value = value()
                    if value and value == value and value > 0:  # not NaN
                        return float(value)
        except BrokerError:
            pass

        closes = self.get_daily_bars([symbol], lookback_days=10).get(symbol, {}).get("close") or []
        return float(closes[-1]) if closes else None

    # ------------------------------------------------------------- option data
    def _underlying_con_id(self, underlying: str) -> int:
        from ib_async import Stock

        if underlying in self._con_id_cache:
            return self._con_id_cache[underlying]
        ib = self._require_ib()
        details = self._guard(
            f"conId:{underlying}",
            lambda: self._run(ib.reqContractDetailsAsync(Stock(underlying, "SMART", "USD"))),
        )
        if not details:
            raise BrokerError(f"no contract details for {underlying}")
        con_id = int(details[0].contract.conId)
        self._con_id_cache[underlying] = con_id
        return con_id

    def get_expirations(self, underlying: str, dte_min: int, dte_max: int) -> List[date]:
        ib = self._require_ib()
        underlying = underlying.upper()
        con_id = self._underlying_con_id(underlying)
        chains = self._guard(
            f"secDefOptParams:{underlying}",
            lambda: self._run(ib.reqSecDefOptParamsAsync(underlying, "", "STK", con_id), timeout=30),
        )

        today = datetime.now(timezone.utc).date()
        found: set = set()
        for chain in chains or []:
            if chain.exchange not in ("SMART", ""):
                continue
            for raw in chain.expirations:
                expiry = _parse_ib_date(raw)
                if expiry and dte_min <= (expiry - today).days <= dte_max:
                    found.add(expiry)
        return sorted(found)

    def get_chain(
        self,
        underlying: str,
        expiration: date,
        option_type: Optional[str] = None,
        *,
        strike_low: Optional[float] = None,
        strike_high: Optional[float] = None,
    ) -> List[OptionQuote]:
        """Option chain with IBKR's own model greeks — no Black-Scholes fallback needed."""
        from ib_async import Option

        ib = self._require_ib()
        underlying = underlying.upper()
        con_id = self._underlying_con_id(underlying)
        chains = self._guard(
            f"secDefOptParams:{underlying}",
            lambda: self._run(ib.reqSecDefOptParamsAsync(underlying, "", "STK", con_id), timeout=30),
        )

        strikes: List[float] = []
        for chain in chains or []:
            if chain.exchange in ("SMART", "") and expiration.strftime("%Y%m%d") in chain.expirations:
                strikes = sorted(chain.strikes)
                break
        if not strikes:
            raise BrokerError(f"no strikes for {underlying} {expiration}")

        strikes = [
            k for k in strikes
            if (strike_low is None or k >= strike_low) and (strike_high is None or k <= strike_high)
        ]
        rights = ["P"] if option_type == "put" else ["C"] if option_type == "call" else ["P", "C"]
        contracts = [
            Option(underlying, expiration.strftime("%Y%m%d"), k, right, "SMART", currency="USD")
            for k in strikes for right in rights
        ]
        if not contracts:
            return []

        qualified = self._guard(
            f"qualify:{underlying}",
            lambda: self._run(ib.qualifyContractsAsync(*contracts), timeout=60),
            retries=0,
        ) or []
        tickers = self._guard(
            f"tickers:{underlying}",
            lambda: self._run(ib.reqTickersAsync(*qualified), timeout=90),
            retries=0,
        ) or []

        quotes: List[OptionQuote] = []
        for ticker in tickers:
            contract = ticker.contract
            greeks = getattr(ticker, "modelGreeks", None)
            quotes.append(
                OptionQuote(
                    symbol=parts_to_occ(
                        contract.symbol, _parse_ib_date(contract.lastTradeDateOrContractMonth),
                        contract.right, float(contract.strike),
                    ),
                    underlying=contract.symbol,
                    expiration=expiration,
                    strike=float(contract.strike),
                    option_type="call" if str(contract.right).upper().startswith("C") else "put",
                    bid=_clean(getattr(ticker, "bid", None)),
                    ask=_clean(getattr(ticker, "ask", None)),
                    last=_clean(getattr(ticker, "last", None)) or _clean(getattr(ticker, "close", None)),
                    implied_volatility=_clean(getattr(greeks, "impliedVol", None)),
                    delta=_clean(getattr(greeks, "delta", None)),
                    gamma=_clean(getattr(greeks, "gamma", None)),
                    theta=_clean(getattr(greeks, "theta", None)),
                    vega=_clean(getattr(greeks, "vega", None)),
                )
            )
        quotes.sort(key=lambda q: q.strike)
        return quotes

    def get_option_quote(self, symbol: str) -> Optional[OptionQuote]:
        from ib_async import Option

        parts = occ_to_parts(symbol)
        ib = self._require_ib()
        contract = Option(
            parts["underlying"], parts["expiration"].strftime("%Y%m%d"), parts["strike"],
            "C" if parts["option_type"] == "call" else "P", "SMART", currency="USD",
        )
        qualified = self._guard(
            f"qualify:{symbol}", lambda: self._run(ib.qualifyContractsAsync(contract), timeout=30), retries=0,
        )
        if not qualified:
            return None
        tickers = self._guard(
            f"ticker:{symbol}", lambda: self._run(ib.reqTickersAsync(*qualified), timeout=30), retries=0,
        )
        if not tickers:
            return None

        ticker = tickers[0]
        greeks = getattr(ticker, "modelGreeks", None)
        return OptionQuote(
            symbol=symbol,
            underlying=parts["underlying"],
            expiration=parts["expiration"],
            strike=parts["strike"],
            option_type=parts["option_type"],
            bid=_clean(getattr(ticker, "bid", None)),
            ask=_clean(getattr(ticker, "ask", None)),
            last=_clean(getattr(ticker, "last", None)) or _clean(getattr(ticker, "close", None)),
            implied_volatility=_clean(getattr(greeks, "impliedVol", None)),
            delta=_clean(getattr(greeks, "delta", None)),
            gamma=_clean(getattr(greeks, "gamma", None)),
            theta=_clean(getattr(greeks, "theta", None)),
            vega=_clean(getattr(greeks, "vega", None)),
        )

    # ------------------------------------------------------------------ orders
    def _option_contract(self, symbol: str):
        from ib_async import Option

        parts = occ_to_parts(symbol)
        ib = self._require_ib()
        contract = Option(
            parts["underlying"], parts["expiration"].strftime("%Y%m%d"), parts["strike"],
            "C" if parts["option_type"] == "call" else "P", "SMART", currency="USD",
        )
        qualified = self._guard(
            f"qualify:{symbol}", lambda: self._run(ib.qualifyContractsAsync(contract), timeout=30), retries=0,
        )
        if not qualified:
            raise BrokerError(f"IBKR could not qualify {symbol}")
        return qualified[0]

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
        from ib_async import LimitOrder, MarketOrder

        self._require_ib()
        contract = self._option_contract(symbol)
        action = "BUY" if side.lower() == "buy" else "SELL"
        quantity = abs(int(qty))

        if limit_price is not None and limit_price > 0:
            order = LimitOrder(action, quantity, round(float(limit_price), 2))
        else:
            order = MarketOrder(action, quantity)
        order.tif = "DAY" if time_in_force.lower() == "day" else time_in_force.upper()
        if client_order_id:
            order.orderRef = client_order_id[:32]

        trade = self._guard(
            f"placeOrder:{symbol}", lambda: self._run(self._place_async(contract, order)), retries=0,
        )
        return _trade_to_dict(trade, symbol)

    def submit_bracketed_short(
        self,
        symbol: str,
        qty: int,
        credit: float,
        take_profit: float,
        stop_loss: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Sell to open with the profit target and stop resting at the exchange.

        This is what IBKR buys you. The 50%/200% pair no longer depends on the
        bot being alive: the child orders are held by IBKR as an OCA group, so a
        crashed process, a lost connection or a stopped container leaves the
        position protected rather than naked.
        """
        ib = self._require_ib()
        contract = self._option_contract(symbol)
        quantity = abs(int(qty))

        bracket = self._guard(
            f"bracket:{symbol}",
            lambda: ib.bracketOrder(
                "SELL", quantity,
                limitPrice=round(float(credit), 2),
                takeProfitPrice=round(float(take_profit), 2),
                stopLossPrice=round(float(stop_loss), 2),
            ),
            retries=0,
        )
        for order in bracket:
            order.tif = "GTC"  # the exits must outlive today's session
            if client_order_id:
                order.orderRef = client_order_id[:32]

        trades = []
        for order in bracket:
            trades.append(
                self._guard(
                    f"placeBracket:{symbol}",
                    lambda o=order: self._run(self._place_async(contract, o)),
                    retries=0,
                )
            )
        parent = _trade_to_dict(trades[0], symbol)
        parent["bracket"] = True
        parent["take_profit"] = round(float(take_profit), 2)
        parent["stop_loss"] = round(float(stop_loss), 2)
        parent["child_order_ids"] = [_trade_to_dict(t, symbol).get("id") for t in trades[1:]]
        return parent

    async def _place_async(self, contract, order):
        trade = self._ib.placeOrder(contract, order)
        await asyncio.sleep(0)  # let ib_async flush the request
        return trade

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
        """Two legs as one IBKR combo, so the spread prices as a unit."""
        from ib_async import Contract, ComboLeg, LimitOrder, MarketOrder

        self._require_ib()
        short_contract = self._option_contract(short_symbol)
        long_contract = self._option_contract(long_symbol)
        parts = occ_to_parts(short_symbol)

        combo = Contract(
            secType="BAG", symbol=parts["underlying"], exchange="SMART", currency="USD",
            comboLegs=[
                ComboLeg(conId=short_contract.conId, ratio=1,
                         action="SELL" if opening else "BUY", exchange="SMART"),
                ComboLeg(conId=long_contract.conId, ratio=1,
                         action="BUY" if opening else "SELL", exchange="SMART"),
            ],
        )
        quantity = abs(int(qty))
        # A combo sold for a credit is a SELL at a positive limit on IBKR —
        # unlike Alpaca, where a credit is expressed as a negative limit.
        action = "SELL" if opening else "BUY"
        if limit_price is not None:
            order = LimitOrder(action, quantity, round(abs(float(limit_price)), 2))
        else:
            order = MarketOrder(action, quantity)
        order.tif = "DAY"
        if client_order_id:
            order.orderRef = client_order_id[:32]

        trade = self._guard(
            f"placeCombo:{short_symbol}", lambda: self._run(self._place_async(combo, order)), retries=0,
        )
        return _trade_to_dict(trade, short_symbol)

    def close_position(self, symbol: str, qty: Optional[int] = None) -> dict:
        """Flatten at market — the stop and emergency path."""
        from ib_async import MarketOrder

        positions = {p.symbol: p for p in self.get_positions()}
        position = positions.get(symbol)
        if position is None:
            raise BrokerError(f"no open position in {symbol}")

        quantity = abs(int(qty)) if qty else int(abs(position.qty))
        action = "BUY" if position.qty < 0 else "SELL"
        contract = self._option_contract(symbol)
        order = MarketOrder(action, quantity)
        trade = self._guard(
            f"close:{symbol}", lambda: self._run(self._place_async(contract, order)), retries=0,
        )
        return _trade_to_dict(trade, symbol)

    def get_orders(self, status: str = "open", limit: int = 50) -> List[dict]:
        self._require_ib()
        trades = self._guard("trades", lambda: self._run(self._trades_async()))
        out = []
        for trade in trades or []:
            info = _trade_to_dict(trade)
            state = (info.get("status") or "").lower()
            is_open = state in {"presubmitted", "submitted", "pendingsubmit", "apipending", "queued"}
            if status == "open" and not is_open:
                continue
            if status == "closed" and is_open:
                continue
            out.append(info)
        return out[:limit]

    async def _trades_async(self):
        return self._ib.trades()

    def cancel_order(self, order_id: str) -> None:
        ib = self._require_ib()
        for trade in self._ib.trades():
            if str(getattr(trade.order, "orderId", "")) == str(order_id):
                self._guard("cancelOrder", lambda t=trade: ib.cancelOrder(t.order), retries=0)
                return
        raise BrokerError(f"order {order_id} not found")

    def cancel_all_orders(self) -> None:
        ib = self._require_ib()
        self._guard("reqGlobalCancel", lambda: ib.reqGlobalCancel(), retries=0)

    # ----------------------------------------------------------------- helpers
    def _require_ib(self):
        if self._ib is None or not self._ib.isConnected():
            raise BrokerError(
                "IBKR is not connected — start TWS or IB Gateway with the API enabled, then reconnect"
            )
        return self._ib


def _clean(value) -> Optional[float]:
    """IBKR uses NaN and -1 for 'no data'; both must become None, not a price."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0:  # NaN or sentinel
        return None
    return number or None


def _parse_ib_date(raw) -> Optional[date]:
    if isinstance(raw, date):
        return raw
    text = str(raw or "").strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:10] if "-" in text else text[:8], fmt).date()
        except ValueError:
            continue
    return None


def _trade_to_dict(trade, symbol: str = "") -> dict:
    """Flatten an ib_async Trade into the same shape the Alpaca client returns."""
    if trade is None:
        return {}
    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    return {
        "id": str(getattr(order, "orderId", "") or ""),
        "client_order_id": str(getattr(order, "orderRef", "") or ""),
        "symbol": symbol or getattr(getattr(trade, "contract", None), "localSymbol", ""),
        "qty": _clean(getattr(order, "totalQuantity", None)),
        "filled_qty": _clean(getattr(status, "filled", None)),
        "filled_avg_price": _clean(getattr(status, "avgFillPrice", None)),
        "limit_price": _clean(getattr(order, "lmtPrice", None)),
        "side": str(getattr(order, "action", "") or "").lower(),
        "order_class": "bracket" if getattr(order, "parentId", 0) else "simple",
        "status": str(getattr(status, "status", "") or ""),
        "submitted_at": str(getattr(trade, "log", [{}])[0].time) if getattr(trade, "log", None) else "",
    }
