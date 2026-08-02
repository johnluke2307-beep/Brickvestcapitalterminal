"""
Brickvestcapitalterminal — historical backtester.

Replays the terminal's own mechanical rules over history: sell at the target
DTE and delta when volatility is rich, take profit at 50% of the credit, stop at
200%, and close anything still open inside the time exit. It reuses
``engine.py`` for pricing, volatility and expectancy, so a backtest and a live
cycle are scored by the same code rather than two implementations that drift.

Strategies come from ``strategies.py`` — the same library the live bot trades,
so a comparison here is a comparison of the things that would actually run:

    cash_secured_put     the live default (options level 2)
    covered_call         against stock you already hold (level 1)
    put_credit_spread    defined-risk vertical (level 3)
    call_credit_spread   the same, on the upside
    iron_condor          both tails sold and bought back
    iron_butterfly       shorts collapsed to the money

Calendars and diagonals are in the library but not replayable here: this model
prices one expiry per trade, and a two-expiry structure would come out quietly
wrong. Asking for one raises rather than approximates.

The volatility assumption is the whole ballgame
-----------------------------------------------
No free data source carries years of historical *implied* volatility, so option
prices here are modelled. The naive approach — price every option at trailing
realised volatility — quietly destroys the thing being tested: if IV equals RV
then there is no variance risk premium, the seller collects exactly fair value,
and the backtest measures nothing but path luck.

So entries and marks are priced on a surface of

    IV(t) = RV(t) + vrp_points

where ``vrp_points`` is the premium the market has historically paid over
realised vol (2–4 points on index products; 3 by default). Profit then comes
from the *actual* forward path being calmer than that surface implied, which is
the real mechanism. Set ``vrp_points = 0`` to run the null hypothesis: no edge
exists. Always run both — the gap between them is how much of the result is your
assumption rather than the market's behaviour.

This is a model, not a fill log. It has no bid/ask spread, no early assignment,
no dividend or pin risk, and it assumes every strike is available and liquid.
Treat the output as an order-of-magnitude sanity check on the rules, never as a
promise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

if TYPE_CHECKING:  # annotations only — pandas is imported lazily at call sites
    import pandas as pd

import config
import engine
import strategies

OPTION_MULTIPLIER = 100
TRADING_DAYS = 252


# ======================================================================================
# Configuration
# ======================================================================================
@dataclass
class BacktestConfig:
    """Everything the replay needs. Defaults mirror the live settings."""

    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"

    initial_capital: float = 100_000.0
    #: Ceiling on collateral committed to any one underlying, as a share of NAV.
    #: A cash-secured put ties up strike × 100 — roughly $28k on a $280 ETF — so
    #: a 10% cap that suits defined-risk spreads cannot size a single CSP
    #: contract on a $100k account. The default is set for the CSP case.
    max_allocation_per_asset: float = 0.25
    #: Share of an underlying's free pool a single trade may take. A preference,
    #: not a hard cap: contracts are indivisible, so one contract is allowed
    #: whenever it fits the pool even if it exceeds this slice.
    trade_allocation_pct: float = 0.50
    #: Mirrors the live margin guardrail: total collateral vs NAV.
    max_margin_utilization: float = 0.50

    #: Any key from ``strategies.REGISTRY`` that this replay can price — see
    #: :data:`ALL_STRATEGIES`. ``short_put`` is accepted as a legacy alias.
    strategy: str = "cash_secured_put"
    dte_entry: int = 45
    dte_exit: int = 21
    short_delta: float = 0.30
    long_delta: float = 0.10
    profit_target: float = 0.50
    stop_loss: float = 2.00  # loss as a multiple of the credit

    vol_window: int = 30
    vol_lookback: int = 252
    vol_rank_threshold: float = 50.0

    risk_free_rate: float = 0.045
    #: Annualised vol points that implied trades above realised. The edge.
    vrp_points: float = 0.03
    #: Round strikes to this increment (SPY/QQQ trade $1 wide).
    strike_increment: float = 1.0
    #: One open position per underlying at a time, as the live bot enforces.
    one_position_per_underlying: bool = True
    #: Intraday entry window, in minutes after the opening bell. Carried so the
    #: backtest and the live bot share one config shape — but see the warning
    #: raised in :meth:`Backtester.run`: daily bars cannot honour it.
    entry_window_enabled: bool = False
    entry_window_start_min: int = 30
    entry_window_end_min: int = 120

    @classmethod
    def from_settings(cls, settings: Optional[config.Settings] = None) -> "BacktestConfig":
        """Seed the backtest from whatever the live terminal is configured to do."""
        s = settings or config.load_settings()
        return cls(
            symbols=list(s.universe[:4]),
            strategy=_canonical_strategy(s.strategy),
            dte_entry=s.target_dte,
            dte_exit=s.time_exit_dte,
            short_delta=abs(s.target_delta),
            profit_target=s.profit_target_pct,
            stop_loss=s.stop_loss_multiple,
            vol_rank_threshold=s.min_iv_rank,
            risk_free_rate=s.risk_free_rate,
            max_margin_utilization=s.max_margin_utilization,
            one_position_per_underlying=s.one_position_per_underlying,
            entry_window_enabled=s.entry_window_enabled,
            entry_window_start_min=s.entry_window_start_min,
            entry_window_end_min=s.entry_window_end_min,
        )


# ======================================================================================
# Price history
# ======================================================================================
def load_history(
    symbols: Sequence[str],
    start: str,
    end: str,
    broker=None,
    buffer_days: int = 420,
) -> Dict[str, "pd.DataFrame"]:
    """Daily OHLC per symbol, with a pre-roll so volatility is warm at the start.

    Prefers yfinance — it needs no credentials, so the Backtest tab works on a
    fresh deployment before any keys are set, and it carries far more history
    than a free brokerage feed. Falls back to the live broker client.
    """
    import pandas as pd

    start_dt = pd.to_datetime(start)
    buffered = (start_dt - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    frames: Dict[str, pd.DataFrame] = {}

    try:
        import yfinance as yf

        raw = yf.download(
            list(symbols), start=buffered, end=end, auto_adjust=False,
            progress=False, group_by="ticker", threads=False,
        )
        for symbol in symbols:
            frame = _extract_symbol(raw, symbol)
            if frame is not None and not frame.empty:
                frames[symbol] = frame
        if frames:
            return frames
    except Exception:
        pass  # fall through to the broker

    if broker is None:
        raise RuntimeError(
            "No price history available: yfinance is not installed or returned nothing, "
            "and no broker client was supplied."
        )

    days = (pd.to_datetime(end) - start_dt).days + buffer_days
    bars = broker.get_daily_bars(list(symbols), lookback_days=days)
    for symbol, series in bars.items():
        closes = series.get("close")
        if not closes:
            continue
        stamps = series.get("timestamp") or []
        if len(stamps) == len(closes) and all(s is not None for s in stamps):
            index = pd.to_datetime([pd.Timestamp(s).tz_localize(None) for s in stamps])
        else:
            # Some feeds omit timestamps; fall back to a business-day index
            # ending at the requested end date so the replay still lines up.
            index = pd.bdate_range(end=pd.to_datetime(end), periods=len(closes))
        frames[symbol] = pd.DataFrame(
            {k: series[k] for k in ("open", "high", "low", "close") if series.get(k)}, index=index
        )
    if not frames:
        raise RuntimeError("No price history could be loaded for the requested symbols.")
    return frames


def _extract_symbol(raw: "pd.DataFrame", symbol: str) -> Optional["pd.DataFrame"]:
    """Pull one symbol's OHLC out of a yfinance frame, whatever shape it came in.

    yfinance returns flat columns for a single ticker but a MultiIndex for
    several, and which level holds the ticker has moved between releases. Rather
    than branch on the version or the symbol count, inspect the frame.
    """
    import pandas as pd

    frame = raw
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(0):
            frame = raw.xs(symbol, axis=1, level=0)
        elif symbol in raw.columns.get_level_values(-1):
            frame = raw.xs(symbol, axis=1, level=-1)
        else:
            return None

    wanted = {c.lower(): c for c in frame.columns}
    if not {"open", "high", "low", "close"} <= set(wanted):
        return None
    frame = frame[[wanted[c] for c in ("open", "high", "low", "close")]].dropna()
    frame.columns = ["open", "high", "low", "close"]
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    return frame


def volatility_frame(prices: "pd.DataFrame", window: int, lookback: int) -> "pd.DataFrame":
    """Trailing realised volatility and its percentile rank within ``lookback``.

    The rank stands in for IV Rank: with no historical implied-vol series, the
    position of today's realised vol inside its own year is the best available
    proxy for "is volatility rich right now".
    """
    import numpy as np
    import pandas as pd

    log_returns = np.log(prices["close"] / prices["close"].shift(1))
    realised = log_returns.rolling(window).std() * math.sqrt(TRADING_DAYS)
    # Vectorised percentile rank — the per-row slice loop is needlessly O(n·k).
    rank = realised.rolling(lookback, min_periods=int(lookback * 0.8)).rank(pct=True) * 100.0
    return pd.DataFrame({"rv": realised, "vol_rank": rank})


# ======================================================================================
# Positions
# ======================================================================================
@dataclass
class Leg:
    """One option leg. ``qty`` is negative when short."""

    strike: float
    is_call: bool
    qty: int


@dataclass
class BacktestPosition:
    symbol: str
    entry_date: datetime
    expiration: datetime
    legs: List[Leg]
    entry_credit: float          # net credit per contract, in dollars
    contracts: int
    collateral: float            # total capital held against the position
    entry_iv: float
    entry_rank: float
    exit_date: Optional[datetime] = None
    exit_debit: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def value(self, spot: float, iv: float, t: float, rate: float) -> float:
        """Cost to close the whole position now, in dollars."""
        per_contract = 0.0
        for leg in self.legs:
            price = engine.bs_price(spot, leg.strike, t, iv, rate, leg.is_call)
            per_contract += -leg.qty * price  # short legs cost money to buy back
        return per_contract * self.contracts * OPTION_MULTIPLIER


# ======================================================================================
# The replay
# ======================================================================================
@dataclass
class BacktestResult:
    nav: "pd.Series"
    trades: List[dict]
    config: BacktestConfig
    metrics: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class Backtester:
    """Daily loop: exits first, then mark to market, then entries."""

    def __init__(self, cfg: BacktestConfig, prices: Dict[str, "pd.DataFrame"]) -> None:
        self.cfg = cfg
        self.prices = prices
        self.vols = {s: volatility_frame(df, cfg.vol_window, cfg.vol_lookback) for s, df in prices.items()}
        self.cash = cfg.initial_capital
        self.positions: List[BacktestPosition] = []
        self.trades: List[dict] = []
        self.nav_history: List[tuple] = []
        self.warnings: List[str] = []
        self.rejections: Dict[tuple, tuple] = {}

    # ------------------------------------------------------------------ helpers
    def _reject(self, symbol: str, code: str, message: str) -> None:
        """Tally why an entry was skipped, so an empty run can explain itself.

        Bucketed by (symbol, cause) rather than by the rendered text — the
        message carries changing dollar figures, and grouping on those would
        produce a page of near-identical lines instead of one clear reason.
        """
        key = (symbol, code)
        count, _ = self.rejections.get(key, (0, message))
        self.rejections[key] = (count + 1, message)

    def implied_vol(self, realised: float) -> float:
        """The modelled IV surface: realised vol plus the variance risk premium."""
        return max(realised + self.cfg.vrp_points, 0.01)

    def _round_strike(self, strike: float) -> float:
        step = self.cfg.strike_increment
        return round(strike / step) * step if step > 0 else strike

    def _build_legs(self, spot: float, t: float, iv: float) -> Optional[tuple[List[Leg], float]]:
        """Strikes for the configured structure, plus the collateral per contract.

        Mirrors ``strategies.py`` leg for leg, with one deliberate difference:
        strikes come from an inverse-delta solve on the modelled surface rather
        than from a listed chain, because there is no chain in a replay. Every
        structure here is single-expiry — the calendar and diagonal are refused
        up front rather than approximated, since this replay prices one expiry
        per trade and a two-expiry structure would come out quietly wrong.
        """
        cfg = self.cfg
        rate = cfg.risk_free_rate
        strategy = _canonical_strategy(cfg.strategy)

        if strategy in TWO_EXPIRY_STRATEGIES:
            raise ValueError(
                f"{strategy} needs two expiries; this backtester models one expiry per "
                "trade. Compare it live, or extend Backtester to carry a back month."
            )

        def put_at(delta: float) -> float:
            return self._round_strike(
                engine.strike_from_delta(spot, t, iv, rate, -abs(delta), is_call=False)
            )

        def call_at(delta: float) -> float:
            return self._round_strike(
                engine.strike_from_delta(spot, t, iv, rate, abs(delta), is_call=True)
            )

        # ---- single-leg -----------------------------------------------------
        if strategy == "cash_secured_put":
            short_put = put_at(cfg.short_delta)
            if short_put >= spot:
                return None
            # Cash-secured: the collateral is the full strike notional.
            return [Leg(short_put, False, -1)], short_put * OPTION_MULTIPLIER

        if strategy == "covered_call":
            short_call = call_at(cfg.short_delta)
            if short_call <= spot:
                return None
            # The capital is the 100 shares standing behind it, not the option.
            return [Leg(short_call, True, -1)], spot * OPTION_MULTIPLIER

        # ---- verticals ------------------------------------------------------
        if strategy == "put_credit_spread":
            short_put = put_at(cfg.short_delta)
            long_put = min(put_at(cfg.long_delta), short_put - cfg.strike_increment)
            if short_put >= spot or long_put <= 0:
                return None
            return (
                [Leg(short_put, False, -1), Leg(long_put, False, 1)],
                (short_put - long_put) * OPTION_MULTIPLIER,
            )

        if strategy == "call_credit_spread":
            short_call = call_at(cfg.short_delta)
            long_call = max(call_at(cfg.long_delta), short_call + cfg.strike_increment)
            if short_call <= spot:
                return None
            return (
                [Leg(short_call, True, -1), Leg(long_call, True, 1)],
                (long_call - short_call) * OPTION_MULTIPLIER,
            )

        # ---- four-leg -------------------------------------------------------
        if strategy == "iron_butterfly":
            body = self._round_strike(spot)
            wing = max(cfg.strike_increment, body - put_at(cfg.long_delta))
            legs = [
                Leg(body, False, -1), Leg(body - wing, False, 1),
                Leg(body, True, -1), Leg(body + wing, True, 1),
            ]
            return legs, wing * OPTION_MULTIPLIER

        if strategy == "iron_condor":
            short_put = put_at(cfg.short_delta)
            long_put = min(put_at(cfg.long_delta), short_put - cfg.strike_increment)
            short_call = max(call_at(cfg.short_delta), spot + cfg.strike_increment)
            long_call = max(call_at(cfg.long_delta), short_call + cfg.strike_increment)
            if short_put >= spot or long_put <= 0:
                return None
            legs = [
                Leg(short_put, False, -1), Leg(long_put, False, 1),
                Leg(short_call, True, -1), Leg(long_call, True, 1),
            ]
            # Only one side can lose at expiry, so margin is the wider wing.
            return legs, max(short_put - long_put, long_call - short_call) * OPTION_MULTIPLIER

        raise ValueError(f"unknown strategy {cfg.strategy!r}; available: {', '.join(ALL_STRATEGIES)}")

    # --------------------------------------------------------------------- loop
    def run(self) -> BacktestResult:
        import pandas as pd

        calendar = None
        for frame in self.prices.values():
            calendar = frame.index if calendar is None else calendar.union(frame.index)
        start = pd.to_datetime(self.cfg.start_date)
        end = pd.to_datetime(self.cfg.end_date)
        days = [d for d in calendar if start <= d <= end]
        if not days:
            raise RuntimeError("No trading days in the requested window.")

        for today in days:
            spot, realised, rank = {}, {}, {}
            for symbol in self.prices:
                frame, vol = self.prices[symbol], self.vols[symbol]
                if today not in frame.index:
                    continue
                rv = vol.loc[today, "rv"]
                if not rv or rv != rv:  # NaN check without numpy
                    continue
                spot[symbol] = float(frame.loc[today, "close"])
                realised[symbol] = float(rv)
                rank[symbol] = float(vol.loc[today, "vol_rank"]) if vol.loc[today, "vol_rank"] == vol.loc[today, "vol_rank"] else float("nan")
            if not spot:
                continue

            self._exits(today, spot, realised)
            nav = self._nav(today, spot, realised)
            self.nav_history.append((today, nav))
            if nav <= 0:
                self.warnings.append(f"NAV reached zero on {today:%Y-%m-%d}; the replay stopped there.")
                break
            self._entries(today, spot, realised, rank, nav)

        return self._compile()

    def _nav(self, today, spot, realised) -> float:
        """Cash plus, for each open position, collateral less the cost to close."""
        nav = self.cash
        for pos in self.positions:
            if not pos.is_open or pos.symbol not in spot:
                continue
            t = max((pos.expiration - today).days / 365.0, 1e-6)
            nav += pos.collateral - pos.value(spot[pos.symbol], self.implied_vol(realised[pos.symbol]), t, self.cfg.risk_free_rate)
        return nav

    def _exits(self, today, spot, realised) -> None:
        cfg = self.cfg
        for pos in self.positions:
            if not pos.is_open or pos.symbol not in spot:
                continue

            days_left = (pos.expiration - today).days
            t = max(days_left / 365.0, 1e-6)
            debit = pos.value(spot[pos.symbol], self.implied_vol(realised[pos.symbol]), t, cfg.risk_free_rate)
            credit = pos.entry_credit * pos.contracts * OPTION_MULTIPLIER

            # Profit and stop are checked before the time exit: a position that
            # has already hit its target should be booked as a target, not
            # relabelled by the calendar.
            if credit > 0 and debit <= credit * (1 - cfg.profit_target):
                reason = "profit_target"
            elif credit > 0 and debit >= credit * (1 + cfg.stop_loss):
                reason = "stop_loss"
            elif days_left <= 0:
                reason = "expired"
            elif days_left <= cfg.dte_exit:
                reason = "time_exit"
            else:
                continue

            self._close(pos, today, debit, reason)

    def _close(self, pos: BacktestPosition, today, debit: float, reason: str) -> None:
        credit = pos.entry_credit * pos.contracts * OPTION_MULTIPLIER
        pos.exit_date = today
        pos.exit_debit = debit / (pos.contracts * OPTION_MULTIPLIER)
        pos.pnl = credit - debit
        pos.exit_reason = reason
        self.cash += pos.collateral - debit
        self.trades.append(
            {
                "symbol": pos.symbol,
                "opened_at": pos.entry_date.strftime("%Y-%m-%d"),
                "closed_at": today.strftime("%Y-%m-%d"),
                "strategy": self.cfg.strategy,
                "contracts": pos.contracts,
                "strikes": " / ".join(f"{'S' if l.qty < 0 else 'L'}{'C' if l.is_call else 'P'}{l.strike:g}" for l in pos.legs),
                "entry_iv": round(pos.entry_iv, 4),
                "entry_rank": round(pos.entry_rank, 1),
                "credit": round(pos.entry_credit, 4),
                "exit_debit": round(pos.exit_debit, 4),
                "pnl_usd": round(pos.pnl, 2),
                "collateral": round(pos.collateral, 2),
                "days_held": (today - pos.entry_date).days,
                "exit_reason": reason,
                "status": "closed",
            }
        )

    def _entries(self, today, spot, realised, rank, nav) -> None:
        cfg = self.cfg
        committed = sum(p.collateral for p in self.positions if p.is_open)

        for symbol in sorted(spot):
            symbol_rank = rank.get(symbol, float("nan"))
            if symbol_rank != symbol_rank or symbol_rank <= cfg.vol_rank_threshold:
                continue

            open_here = [p for p in self.positions if p.is_open and p.symbol == symbol]
            if cfg.one_position_per_underlying and open_here:
                continue

            deployed = sum(p.collateral for p in open_here)
            pool = max(nav * cfg.max_allocation_per_asset - deployed, 0.0)
            if pool <= 0:
                continue

            t = cfg.dte_entry / 365.0
            iv = self.implied_vol(realised[symbol])
            built = self._build_legs(spot[symbol], t, iv)
            if built is None:
                continue
            legs, collateral_per_contract = built
            if collateral_per_contract <= 0:
                continue

            credit = 0.0
            for leg in legs:
                price = engine.bs_price(spot[symbol], leg.strike, t, iv, cfg.risk_free_rate, leg.is_call)
                credit += -leg.qty * price
            if credit <= 0:
                continue

            budget = pool * cfg.trade_allocation_pct
            contracts = int(budget // collateral_per_contract)
            if contracts < 1:
                # Contracts are indivisible. Allow one if the whole pool covers
                # it; otherwise say precisely why, because a backtest that takes
                # no trades and does not explain itself is indistinguishable
                # from a broken one.
                if collateral_per_contract <= pool:
                    contracts = 1
                else:
                    self._reject(
                        symbol,
                        "collateral",
                        f"{cfg.strategy} needs ~${collateral_per_contract:,.0f} collateral per contract but "
                        f"the per-asset pool is ~${pool:,.0f} — raise capital, raise the per-asset "
                        f"allocation, or use a defined-risk strategy",
                    )
                    continue
            collateral = collateral_per_contract * contracts

            # The live margin ceiling, applied forward: never let total committed
            # collateral cross the limit.
            if (committed + collateral) / nav > cfg.max_margin_utilization:
                self._reject(symbol, "margin", f"margin ceiling {cfg.max_margin_utilization:.0%} of NAV would be breached")
                continue
            if collateral > self.cash:
                self._reject(symbol, "cash", "insufficient cash for collateral")
                continue

            position = BacktestPosition(
                symbol=symbol,
                entry_date=today,
                expiration=today + timedelta(days=cfg.dte_entry),
                legs=legs,
                entry_credit=credit,
                contracts=contracts,
                collateral=collateral,
                entry_iv=iv,
                entry_rank=symbol_rank,
            )
            self.positions.append(position)
            self.cash += credit * contracts * OPTION_MULTIPLIER - collateral
            committed += collateral

    # ------------------------------------------------------------------ results
    def _compile(self) -> BacktestResult:
        import pandas as pd

        nav = pd.Series(
            [v for _, v in self.nav_history],
            index=pd.to_datetime([d for d, _ in self.nav_history]),
            name="nav",
        )
        metrics = compute_metrics(nav, self.trades, self.cfg)

        open_count = sum(1 for p in self.positions if p.is_open)
        if open_count:
            self.warnings.append(
                f"{open_count} position(s) were still open at the end date and are excluded from trade statistics."
            )
        if not self.trades:
            if self.rejections:
                ranked = sorted(self.rejections.items(), key=lambda kv: kv[1][0], reverse=True)[:3]
                for (symbol, _code), (count, message) in ranked:
                    self.warnings.append(f"No trades — {symbol}: {message} (blocked on {count} days)")
            else:
                self.warnings.append(
                    "No trades were taken — the vol-rank filter never passed. Lower the threshold "
                    "or widen the date range."
                )
        if self.cfg.entry_window_enabled:
            self.warnings.append(
                f"Entry window {self.cfg.entry_window_start_min}–{self.cfg.entry_window_end_min} min "
                "after the open was NOT applied: this replay uses daily bars, which carry one price "
                "per day and no intraday timestamps. The window is enforced live by the bot; free "
                "intraday history only reaches back ~60 days, far short of one 45-DTE cycle."
            )
        if self.cfg.vrp_points == 0:
            self.warnings.append(
                "vrp_points = 0: options are priced at realised vol, so no variance risk premium exists "
                "by construction. This is the null hypothesis, not a forecast."
            )
        return BacktestResult(nav=nav, trades=self.trades, config=self.cfg, metrics=metrics, warnings=self.warnings)


def compute_metrics(nav: "pd.Series", trades: List[dict], cfg: BacktestConfig) -> dict:
    """Return, risk and trade statistics. Expectancy reuses the live engine."""
    if nav.empty:
        return {}

    returns = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    total_return = nav.iloc[-1] / cfg.initial_capital - 1.0
    cagr = (nav.iloc[-1] / cfg.initial_capital) ** (1 / years) - 1.0 if nav.iloc[-1] > 0 else -1.0

    volatility = returns.std() * math.sqrt(TRADING_DAYS) if len(returns) > 1 else 0.0
    sharpe = ((returns.mean() * TRADING_DAYS) - cfg.risk_free_rate) / volatility if volatility > 0 else 0.0
    downside = returns[returns < 0].std() * math.sqrt(TRADING_DAYS) if (returns < 0).any() else 0.0
    sortino = ((returns.mean() * TRADING_DAYS) - cfg.risk_free_rate) / downside if downside > 0 else 0.0

    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_dd = float(drawdown.min())

    expectancy = engine.compute_expectancy(trades)
    return {
        "start": nav.index[0].strftime("%Y-%m-%d"),
        "end": nav.index[-1].strftime("%Y-%m-%d"),
        "years": years,
        "final_nav": float(nav.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0,
        "trades": len(trades),
        "win_rate": expectancy.p_win,
        "expectancy": expectancy.expectancy,
        "profit_factor": expectancy.profit_factor,
        "avg_win": expectancy.avg_win,
        "avg_loss": expectancy.avg_loss,
        "breakeven_win_rate": engine.breakeven_win_rate(cfg.profit_target, cfg.stop_loss),
        "avg_days_held": (sum(t["days_held"] for t in trades) / len(trades)) if trades else 0.0,
        "drawdown_series": drawdown,
    }


def run_backtest(cfg: BacktestConfig, broker=None) -> BacktestResult:
    """Load history and replay the rules. The one call the UI needs."""
    prices = load_history(cfg.symbols, cfg.start_date, cfg.end_date, broker=broker)
    missing = [s for s in cfg.symbols if s not in prices]
    result = Backtester(cfg, prices).run()
    if missing:
        result.warnings.insert(0, f"No history for {', '.join(missing)} — excluded from the run.")
    return result



# ======================================================================================
# Robustness — the checks that separate an edge from a curve fit
# ======================================================================================
#: Every knob that could be tuned against the data. Counted, not hidden: the
#: ratio of trades to free parameters is the first thing an overfit strategy
#: fails, and it is arithmetic rather than opinion.
TUNABLE_PARAMETERS = (
    "short_delta", "long_delta", "dte_entry", "dte_exit", "profit_target",
    "stop_loss", "vol_rank_threshold", "vrp_points", "max_allocation_per_asset",
    "trade_allocation_pct",
)


def sample_adequacy(result: BacktestResult) -> dict:
    """How much independent evidence is actually behind the result.

    Trade count alone flatters a strategy that holds overlapping positions: two
    short puts open in the same week through the same selloff are close to one
    observation, not two. The effective count here divides by the average number
    of positions held concurrently, which is crude but directionally honest.
    """
    trades = result.trades
    if not trades:
        return {"trades": 0, "parameters": len(TUNABLE_PARAMETERS), "trades_per_parameter": 0.0,
                "effective_trades": 0.0, "verdict": "no trades"}

    import pandas as pd

    spans = [(pd.Timestamp(t["opened_at"]), pd.Timestamp(t["closed_at"])) for t in trades]
    total_days = sum(max((c - o).days, 1) for o, c in spans)
    span_days = max((max(c for _, c in spans) - min(o for o, _ in spans)).days, 1)
    concurrency = max(total_days / span_days, 1.0)
    effective = len(trades) / concurrency

    per_param = effective / len(TUNABLE_PARAMETERS)
    if per_param >= 20:
        verdict = "adequate"
    elif per_param >= 10:
        verdict = "thin"
    else:
        verdict = "insufficient"
    return {
        "trades": len(trades),
        "parameters": len(TUNABLE_PARAMETERS),
        "concurrency": concurrency,
        "effective_trades": effective,
        "trades_per_parameter": per_param,
        "verdict": verdict,
    }


def split_sample(
    cfg: BacktestConfig,
    prices: Dict[str, "pd.DataFrame"],
    train_fraction: float = 0.6,
) -> dict:
    """Run the first slice of history, then the rest, and compare.

    A strategy tuned to its history looks strong on the slice it was tuned on
    and falls apart afterwards. Splitting chronologically — never randomly, which
    would leak the future into the past — is the cheapest way to see that.
    """
    import pandas as pd

    start, end = pd.to_datetime(cfg.start_date), pd.to_datetime(cfg.end_date)
    cut = start + (end - start) * train_fraction
    cut_str = cut.strftime("%Y-%m-%d")

    from dataclasses import replace as _replace

    in_sample = Backtester(_replace(cfg, end_date=cut_str), prices).run()
    out_sample = Backtester(_replace(cfg, start_date=cut_str), prices).run()

    def summarise(result):
        m = result.metrics
        return {
            "start": m.get("start"), "end": m.get("end"),
            "cagr": m.get("cagr", 0.0), "sharpe": m.get("sharpe", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0), "trades": m.get("trades", 0),
            "win_rate": m.get("win_rate", 0.0), "expectancy": m.get("expectancy", 0.0),
        }

    a, b = summarise(in_sample), summarise(out_sample)
    decay = (b["cagr"] - a["cagr"]) if a["trades"] and b["trades"] else None
    return {"in_sample": a, "out_of_sample": b, "cagr_decay": decay, "cut": cut_str}


def walk_forward(
    cfg: BacktestConfig,
    prices: Dict[str, "pd.DataFrame"],
    folds: int = 4,
) -> List[dict]:
    """Sequential, non-overlapping slices of history, each scored on its own.

    One good year can carry a six-year total. Per-fold results show whether the
    edge recurs or whether it was one regime.
    """
    import pandas as pd

    from dataclasses import replace as _replace

    start, end = pd.to_datetime(cfg.start_date), pd.to_datetime(cfg.end_date)
    edges = [start + (end - start) * (i / folds) for i in range(folds + 1)]
    out = []
    for i in range(folds):
        fold_cfg = _replace(
            cfg,
            start_date=edges[i].strftime("%Y-%m-%d"),
            end_date=edges[i + 1].strftime("%Y-%m-%d"),
        )
        m = Backtester(fold_cfg, prices).run().metrics
        out.append({
            "fold": i + 1,
            "start": m.get("start"), "end": m.get("end"),
            "cagr": m.get("cagr", 0.0), "total_return": m.get("total_return", 0.0),
            "sharpe": m.get("sharpe", 0.0), "max_drawdown": m.get("max_drawdown", 0.0),
            "trades": m.get("trades", 0),
        })
    return out


def sensitivity(
    cfg: BacktestConfig,
    prices: Dict[str, "pd.DataFrame"],
    parameter: str,
    values: Sequence[float],
) -> List[dict]:
    """Sweep one parameter and report the curve.

    This is the most informative overfitting test available here. A real edge
    sits on a **plateau** — nudging the parameter moves the result a little. A
    curve fit sits on a **spike**: the chosen value is a peak surrounded by much
    worse neighbours, which means it was selected to fit noise.
    """
    from dataclasses import replace as _replace

    rows = []
    for value in values:
        result = Backtester(_replace(cfg, **{parameter: value}), prices).run()
        m = result.metrics
        rows.append({
            "value": value,
            "total_return": m.get("total_return", 0.0),
            "cagr": m.get("cagr", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0),
            "trades": m.get("trades", 0),
        })
    return rows


def plateau_score(rows: List[dict], chosen: float, metric: str = "total_return") -> dict:
    """Is the chosen value a plateau or a spike?

    Compares the chosen setting against the median of the whole sweep. A value
    far above its own neighbourhood is the signature of a fit; a value close to
    the median of a broadly positive sweep is the signature of an edge that does
    not depend on the exact number.
    """
    scored = [r for r in rows if r["trades"] > 0]
    if len(scored) < 3:
        return {"verdict": "insufficient sweep", "chosen": None, "median": None, "ratio": None}

    values = sorted(r[metric] for r in scored)
    median = values[len(values) // 2]
    here = min(scored, key=lambda r: abs(r["value"] - chosen))[metric]
    positive = sum(1 for v in values if v > 0) / len(values)

    # Graded on the gap and the share of the sweep that works, never on a
    # ratio: when the median is negative or near zero, here/median flips sign
    # or explodes, and a genuine spike gets graded as merely fragile.
    gap = here - median
    if positive >= 0.7:
        verdict = "plateau — result survives the neighbourhood"
    elif gap > 0 and positive <= 0.4:
        verdict = "SPIKE — the chosen value looks fitted"
    else:
        verdict = "fragile — much of the sweep loses money"
    return {"verdict": verdict, "chosen": here, "median": median, "gap": gap,
            "share_positive": positive}


#: Two-expiry structures. The replay prices one expiry per trade, so these are
#: refused rather than approximated — see :meth:`Backtester._build_legs`.
TWO_EXPIRY_STRATEGIES = ("calendar_spread", "diagonal_spread")

#: Everything this backtester can replay, in the same order as the library.
ALL_STRATEGIES = tuple(k for k in strategies.STRATEGY_KEYS if k not in TWO_EXPIRY_STRATEGIES)

#: ``short_put`` was the original name for the cash-secured put. Old configs,
#: saved comparisons and the CLI keep working.
STRATEGY_ALIASES = {"short_put": "cash_secured_put"}


def _canonical_strategy(name: str) -> str:
    return STRATEGY_ALIASES.get(name, name)


def compare_strategies(
    cfg: BacktestConfig,
    prices: Dict[str, "pd.DataFrame"],
    strategies: Sequence[str] = ALL_STRATEGIES,
) -> Dict[str, BacktestResult]:
    """Replay several structures over the same history and capital.

    One price set, one starting balance, one rule set — only the structure
    changes, so the comparison isolates the thing being compared. Running them
    separately with different downloads would not.
    """
    from dataclasses import replace as _replace

    results: Dict[str, BacktestResult] = {}
    for name in strategies:
        results[name] = Backtester(_replace(cfg, strategy=name), prices).run()
    return results


def comparison_table(results: Dict[str, BacktestResult]) -> List[dict]:
    """Flatten a comparison into rows, ranked by return on capital."""
    rows = []
    for name, result in results.items():
        m = result.metrics
        if not m:
            continue
        rows.append(
            {
                "strategy": name,
                "total_return": m["total_return"],
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "calmar": m["calmar"],
                "trades": m["trades"],
                "win_rate": m["win_rate"],
                "expectancy": m["expectancy"],
                "final_nav": m["final_nav"],
                "blocked": bool(result.warnings and m["trades"] == 0),
            }
        )
    rows.sort(key=lambda r: r["total_return"], reverse=True)
    return rows


def main() -> int:
    """CLI: python backtest.py — runs the live configuration over history."""
    import argparse

    parser = argparse.ArgumentParser(description="Brickvestcapitalterminal backtester")
    parser.add_argument("--symbols", default=None, help="comma-separated, e.g. SPY,QQQ")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy", default=None, choices=list(ALL_STRATEGIES) + ["short_put"])
    parser.add_argument("--vrp", type=float, default=None, help="vol points of premium (0 = null hypothesis)")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--capital", type=float, default=None, help="starting capital in USD")
    parser.add_argument("--compare", action="store_true", help="run every strategy over the same history")
    args = parser.parse_args()

    cfg = BacktestConfig.from_settings()
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",")]
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end
    if args.strategy:
        cfg.strategy = args.strategy
    if args.vrp is not None:
        cfg.vrp_points = args.vrp
    if args.delta is not None:
        cfg.short_delta = args.delta
    if args.capital is not None:
        cfg.initial_capital = args.capital

    if args.compare:
        prices = load_history(cfg.symbols, cfg.start_date, cfg.end_date)
        results = compare_strategies(cfg, prices)
        print(f"{cfg.start_date} → {cfg.end_date} · ${cfg.initial_capital:,.0f} · "
              f"IV = RV + {cfg.vrp_points * 100:.1f}pts\n")
        header = f"{'STRATEGY':<20}{'RETURN':>9}{'CAGR':>8}{'SHARPE':>8}{'MAXDD':>8}{'TRADES':>8}{'WIN':>7}{'E/TRADE':>10}"
        print(header)
        print("-" * len(header))
        for row in comparison_table(results):
            print(f"{row['strategy']:<20}{row['total_return'] * 100:>8.1f}%{row['cagr'] * 100:>7.1f}%"
                  f"{row['sharpe']:>8.2f}{row['max_drawdown'] * 100:>7.1f}%{row['trades']:>8d}"
                  f"{row['win_rate'] * 100:>6.0f}%{row['expectancy']:>10,.0f}")
        for name, result in results.items():
            for note in result.warnings:
                print(f"\n! {name}: {note}")
        return 0

    print(f"Replaying {cfg.strategy} on {', '.join(cfg.symbols)} — {cfg.start_date} to {cfg.end_date}")
    print(f"IV surface = RV + {cfg.vrp_points * 100:.1f} vol points\n")
    result = run_backtest(cfg)
    m = result.metrics

    print(f"{'Final NAV':<22}${m['final_nav']:>14,.2f}")
    print(f"{'Total return':<22}{m['total_return'] * 100:>14.2f}%")
    print(f"{'CAGR':<22}{m['cagr'] * 100:>14.2f}%")
    print(f"{'Volatility':<22}{m['volatility'] * 100:>14.2f}%")
    print(f"{'Sharpe':<22}{m['sharpe']:>14.2f}")
    print(f"{'Sortino':<22}{m['sortino']:>14.2f}")
    print(f"{'Max drawdown':<22}{m['max_drawdown'] * 100:>14.2f}%")
    print(f"{'Trades':<22}{m['trades']:>14d}")
    print(f"{'Win rate':<22}{m['win_rate'] * 100:>14.1f}%  (breakeven {m['breakeven_win_rate'] * 100:.0f}%)")
    print(f"{'Expectancy/trade':<22}${m['expectancy']:>14,.2f}")
    for note in result.warnings:
        print(f"\n! {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
