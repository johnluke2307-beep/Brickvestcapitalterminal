"""
Brickvestcapitalterminal — historical backtester.

Replays the terminal's own mechanical rules over history: sell at the target
DTE and delta when volatility is rich, take profit at 50% of the credit, stop at
200%, and close anything still open inside the time exit. It reuses
``engine.py`` for pricing, volatility and expectancy, so a backtest and a live
cycle are scored by the same code rather than two implementations that drift.

Three strategies, all short premium:
    short_put          cash-secured put (options level 2) — the live default
    put_credit_spread  defined-risk vertical (level 3)
    iron_condor        both wings (level 3)

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

    strategy: str = "short_put"  # short_put | put_credit_spread | iron_condor
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

    @classmethod
    def from_settings(cls, settings: Optional[config.Settings] = None) -> "BacktestConfig":
        """Seed the backtest from whatever the live terminal is configured to do."""
        s = settings or config.load_settings()
        return cls(
            symbols=list(s.universe[:4]),
            strategy=s.strategy if s.strategy in {"short_put", "put_credit_spread"} else "short_put",
            dte_entry=s.target_dte,
            dte_exit=s.time_exit_dte,
            short_delta=abs(s.target_delta),
            profit_target=s.profit_target_pct,
            stop_loss=s.stop_loss_multiple,
            vol_rank_threshold=s.min_iv_rank,
            risk_free_rate=s.risk_free_rate,
            max_margin_utilization=s.max_margin_utilization,
            one_position_per_underlying=s.one_position_per_underlying,
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
            frame = raw[symbol] if len(symbols) > 1 else raw
            frame = frame[["Open", "High", "Low", "Close"]].dropna()
            frame.columns = [c.lower() for c in frame.columns]
            if getattr(frame.index, "tz", None) is not None:
                frame.index = frame.index.tz_localize(None)
            if not frame.empty:
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
        """Strikes for the configured structure, plus the collateral per contract."""
        cfg = self.cfg
        rate = cfg.risk_free_rate

        short_put = self._round_strike(
            engine.strike_from_delta(spot, t, iv, rate, -cfg.short_delta, is_call=False)
        )
        if short_put >= spot:  # a short put must sit below spot
            return None

        if cfg.strategy == "short_put":
            # Cash-secured: the collateral is the full strike notional.
            return [Leg(short_put, False, -1)], short_put * OPTION_MULTIPLIER

        long_put = self._round_strike(
            engine.strike_from_delta(spot, t, iv, rate, -cfg.long_delta, is_call=False)
        )
        long_put = min(long_put, short_put - cfg.strike_increment)
        if long_put <= 0:
            return None
        put_width = short_put - long_put
        legs = [Leg(short_put, False, -1), Leg(long_put, False, 1)]

        if cfg.strategy == "put_credit_spread":
            return legs, put_width * OPTION_MULTIPLIER

        # Iron condor — add the call side.
        short_call = self._round_strike(
            engine.strike_from_delta(spot, t, iv, rate, cfg.short_delta, is_call=True)
        )
        long_call = self._round_strike(
            engine.strike_from_delta(spot, t, iv, rate, cfg.long_delta, is_call=True)
        )
        short_call = max(short_call, spot + cfg.strike_increment)
        long_call = max(long_call, short_call + cfg.strike_increment)
        call_width = long_call - short_call
        legs += [Leg(short_call, True, -1), Leg(long_call, True, 1)]
        # Only one side can lose at expiry, so margin is the wider wing.
        return legs, max(put_width, call_width) * OPTION_MULTIPLIER

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


def main() -> int:
    """CLI: python backtest.py — runs the live configuration over history."""
    import argparse

    parser = argparse.ArgumentParser(description="Brickvestcapitalterminal backtester")
    parser.add_argument("--symbols", default=None, help="comma-separated, e.g. SPY,QQQ")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy", default=None, choices=["short_put", "put_credit_spread", "iron_condor"])
    parser.add_argument("--vrp", type=float, default=None, help="vol points of premium (0 = null hypothesis)")
    parser.add_argument("--delta", type=float, default=None)
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
