"""
Brickvestcapitalterminal — quantitative engine.

This module owns every number the platform reasons about. It has **no broker
dependency**: give it prices and it gives you an edge estimate, which keeps the
maths unit-testable and makes swapping Alpaca for IBKR a broker-layer concern
only.

It covers four things:

1. **Volatility** — Black-Scholes pricing/greeks, implied-volatility inversion,
   and three realised-volatility estimators (close-to-close, Parkinson,
   Yang-Zhang).
2. **The VRP edge** — implied minus realised volatility, the IV/RV ratio, and an
   IV Rank computed from a locally persisted implied-volatility history.
3. **Expectancy** — the account's realised ``E = (P_win × W) − (P_loss × L)``
   from the trade log, plus the theoretical expectancy implied by the short
   strike's delta.
4. **Currency** — live USD/ZAR so every dollar of premium is reported against the
   R10,000/month baseline.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import config

TRADING_DAYS = 252
SQRT_TRADING_DAYS = math.sqrt(TRADING_DAYS)


# ======================================================================================
# Black-Scholes: pricing, greeks and implied-volatility inversion
# ======================================================================================
def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no SciPy dependency — keeps the deploy lightweight)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, t: float, vol: float, rate: float) -> tuple[float, float]:
    vol_sqrt_t = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(spot: float, strike: float, t: float, vol: float, rate: float, is_call: bool) -> float:
    """European Black-Scholes price. ``t`` is in years, ``vol`` annualised."""
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(spot, strike, t, vol, rate)
    discount = math.exp(-rate * t)
    if is_call:
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, vol: float, rate: float, is_call: bool) -> float:
    """Spot delta. Puts are returned negative, matching broker conventions."""
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1, _ = _d1_d2(spot, strike, t, vol, rate)
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def bs_vega(spot: float, strike: float, t: float, vol: float, rate: float) -> float:
    """Vega per 1.00 (100 vol points) change in volatility."""
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t, vol, rate)
    return spot * _norm_pdf(d1) * math.sqrt(t)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    is_call: bool,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """Invert Black-Scholes for implied volatility using bisection.

    Bisection rather than Newton on purpose: it cannot diverge on the wide, stale
    or crossed quotes that a free market-data feed will occasionally hand us. The
    price is bracketed first so a no-solution quote returns ``None`` instead of a
    bogus number that would silently pollute the VRP calculation.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return None

    intrinsic = max((spot - strike) if is_call else (strike - spot), 0.0)
    if price < intrinsic - 1e-8:  # below intrinsic → arbitrage/stale quote, reject
        return None

    if bs_price(spot, strike, t, hi, rate, is_call) < price:
        return None  # even 500% vol cannot reach this price

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = bs_price(spot, strike, t, mid, rate, is_call) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF, via the stdlib — no SciPy on the free tier."""
    return statistics.NormalDist().inv_cdf(min(max(p, 1e-12), 1 - 1e-12))


def strike_from_delta(
    spot: float,
    t: float,
    vol: float,
    rate: float,
    target_delta: float,
    is_call: bool,
) -> float:
    """Invert Black-Scholes delta for a strike — closed form, no root-finding.

    Calls: ``Δ = N(d1)`` so ``d1 = N⁻¹(Δ)``.
    Puts:  ``Δ = N(d1) − 1`` so ``d1 = N⁻¹(Δ + 1)`` (pass Δ negative).

    Then ``K = S · exp(−d1·σ√T + (r + σ²/2)·T)``.
    """
    if spot <= 0 or t <= 0 or vol <= 0:
        return spot
    d1 = norm_ppf(target_delta if is_call else target_delta + 1.0)
    return spot * math.exp(-d1 * vol * math.sqrt(t) + (rate + 0.5 * vol * vol) * t)


def year_fraction(expiration: date, asof: Optional[date] = None) -> float:
    """Calendar-day year fraction, floored so same-day expiries stay finite."""
    asof = asof or datetime.now(timezone.utc).date()
    return max((expiration - asof).days, 0) / 365.0


def days_to_expiry(expiration: date, asof: Optional[date] = None) -> int:
    asof = asof or datetime.now(timezone.utc).date()
    return (expiration - asof).days


# ======================================================================================
# Realised volatility
# ======================================================================================
def realized_vol_close_to_close(closes: Sequence[float], window: Optional[int] = None) -> Optional[float]:
    """Annualised close-to-close realised volatility of log returns."""
    prices = [p for p in closes if p and p > 0]
    if window:
        prices = prices[-(window + 1) :]
    if len(prices) < 3:
        return None
    returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    if len(returns) < 2:
        return None
    return statistics.stdev(returns) * SQRT_TRADING_DAYS


def realized_vol_parkinson(highs: Sequence[float], lows: Sequence[float], window: Optional[int] = None) -> Optional[float]:
    """Parkinson high/low estimator — ~5× more efficient than close-to-close.

    Blind to gaps, so it is reported alongside (never instead of) close-to-close.
    """
    pairs = [(h, l) for h, l in zip(highs, lows) if h and l and h > 0 and l > 0 and h >= l]
    if window:
        pairs = pairs[-window:]
    if len(pairs) < 3:
        return None
    factor = 1.0 / (4.0 * math.log(2.0))
    mean_sq = sum(math.log(h / l) ** 2 for h, l in pairs) / len(pairs)
    return math.sqrt(factor * mean_sq) * SQRT_TRADING_DAYS


def realized_vol_yang_zhang(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: Optional[int] = None,
) -> Optional[float]:
    """Yang-Zhang estimator: drift-independent and gap-aware.

    This is the fairest comparison against implied volatility, because implied
    vol prices overnight gap risk that close-to-close under-measures.
    """
    bars = [
        (o, h, l, c)
        for o, h, l, c in zip(opens, highs, lows, closes)
        if all(x and x > 0 for x in (o, h, l, c))
    ]
    if window:
        bars = bars[-(window + 1) :]
    n = len(bars) - 1
    if n < 3:
        return None

    overnight, open_close, rs = [], [], []
    for i in range(1, len(bars)):
        prev_close = bars[i - 1][3]
        o, h, l, c = bars[i]
        overnight.append(math.log(o / prev_close))
        open_close.append(math.log(c / o))
        rs.append(math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l / o))

    var_overnight = statistics.variance(overnight)
    var_open_close = statistics.variance(open_close)
    var_rs = sum(rs) / n
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = var_overnight + k * var_open_close + (1 - k) * var_rs
    return math.sqrt(max(variance, 0.0)) * SQRT_TRADING_DAYS


# ======================================================================================
# IV history store → IV Rank / IV Percentile
# ======================================================================================
class IVHistoryStore:
    """Append-only CSV of daily ATM implied volatility, one row per symbol/day.

    Alpaca's free tier exposes *current* implied volatility but no long IV
    history, and IV Rank is definitionally a trailing-range statistic. So the
    terminal builds its own history: every scan records today's ATM IV, and the
    rank sharpens as the file grows. Until there are enough observations the
    rank is flagged as a proxy (see :meth:`rank`) rather than silently faked.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or config.IV_HISTORY_PATH)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as handle:
                csv.writer(handle).writerow(["date", "symbol", "iv", "spot", "rv"])

    # ------------------------------------------------------------------ writes
    def record(self, symbol: str, iv: float, spot: Optional[float] = None, rv: Optional[float] = None) -> None:
        """Store one observation, overwriting an existing same-day row."""
        if iv is None or iv <= 0:
            return
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            rows = self._read_rows()
            rows = [r for r in rows if not (r["symbol"] == symbol.upper() and r["date"] == today)]
            rows.append(
                {
                    "date": today,
                    "symbol": symbol.upper(),
                    "iv": f"{iv:.6f}",
                    "spot": f"{spot:.4f}" if spot else "",
                    "rv": f"{rv:.6f}" if rv else "",
                }
            )
            rows.sort(key=lambda r: (r["date"], r["symbol"]))
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", "symbol", "iv", "spot", "rv"])
                writer.writeheader()
                writer.writerows(rows)

    def record_on(self, symbol: str, day, iv: float) -> None:
        """Store an observation for a *specific* date, for broker backfill.

        :meth:`record` always writes today. A venue that carries historical
        implied volatility can hand over a year of it at once, which is what
        turns IV Rank from a proxy into the real statistic — but only if the
        rows land on their own dates.
        """
        if iv is None or iv <= 0:
            return
        if hasattr(day, "date") and not isinstance(day, date):
            day = day.date()
        stamp = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]

        with self._lock:
            rows = self._read_rows()
            key = (symbol.upper(), stamp)
            if any((r["symbol"], r["date"]) == key for r in rows):
                return  # never overwrite an existing observation on backfill
            rows.append({"date": stamp, "symbol": symbol.upper(), "iv": f"{iv:.6f}",
                         "spot": "", "rv": ""})
            rows.sort(key=lambda r: (r["date"], r["symbol"]))
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", "symbol", "iv", "spot", "rv"])
                writer.writeheader()
                writer.writerows(rows)

    # ------------------------------------------------------------------- reads
    def _read_rows(self) -> List[dict]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as handle:
            return [row for row in csv.DictReader(handle) if row.get("symbol")]

    def series(self, symbol: str, window: Optional[int] = None) -> List[tuple[str, float]]:
        """Trailing ``(date, iv)`` observations for a symbol, oldest first."""
        rows = [r for r in self._read_rows() if r["symbol"] == symbol.upper()]
        out: List[tuple[str, float]] = []
        for row in rows:
            try:
                out.append((row["date"], float(row["iv"])))
            except (TypeError, ValueError):
                continue
        out.sort(key=lambda item: item[0])
        window = window or config.SETTINGS.iv_rank_window
        return out[-window:]

    def records(self, symbol: str) -> List[dict]:
        """Full stored rows for a symbol (IV, spot and RV), oldest first."""
        rows = [r for r in self._read_rows() if r["symbol"] == symbol.upper()]
        rows.sort(key=lambda r: r["date"])
        return rows

    def rank(self, symbol: str, current_iv: float, rv_history: Optional[Sequence[float]] = None) -> "IVRank":
        """IV Rank in [0, 100] with an explicit statement of how it was derived.

        ``rank = (IV − min) / (max − min) × 100`` over the trailing window.
        ``percentile`` is the share of stored observations below the current IV,
        which is more robust when a single spike dominates the range.
        """
        observations = [iv for _, iv in self.series(symbol)]
        settings = config.SETTINGS

        if len(observations) >= settings.iv_rank_min_samples:
            lo, hi = min(observations), max(observations)
            spread = hi - lo
            rank = 100.0 * (current_iv - lo) / spread if spread > 1e-9 else 50.0
            below = sum(1 for iv in observations if iv < current_iv)
            percentile = 100.0 * below / len(observations)
            return IVRank(
                value=_clamp(rank, 0.0, 100.0),
                percentile=_clamp(percentile, 0.0, 100.0),
                samples=len(observations),
                source="iv_history",
                low=lo,
                high=hi,
            )

        # ---- Bootstrap proxy -------------------------------------------------
        # Not enough stored IV yet. Rank the current IV against the trailing
        # *realised* vol distribution instead. Directionally correct (rich IV vs
        # its own recent regime) and flagged as a proxy everywhere it surfaces.
        proxy = [rv for rv in (rv_history or []) if rv and rv > 0]
        if len(proxy) >= 20:
            lo, hi = min(proxy), max(proxy)
            spread = hi - lo
            rank = 100.0 * (current_iv - lo) / spread if spread > 1e-9 else 50.0
            below = sum(1 for rv in proxy if rv < current_iv)
            return IVRank(
                value=_clamp(rank, 0.0, 100.0),
                percentile=_clamp(100.0 * below / len(proxy), 0.0, 100.0),
                samples=len(observations),
                source="rv_proxy",
                low=lo,
                high=hi,
            )

        return IVRank(value=None, percentile=None, samples=len(observations), source="insufficient_data")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class IVRank:
    """IV Rank result carrying its own provenance."""

    value: Optional[float]
    percentile: Optional[float] = None
    samples: int = 0
    source: str = "insufficient_data"  # iv_history | rv_proxy | insufficient_data
    low: Optional[float] = None
    high: Optional[float] = None

    @property
    def is_proxy(self) -> bool:
        return self.source != "iv_history"

    @property
    def label(self) -> str:
        return {
            "iv_history": f"IV history ({self.samples} obs)",
            "rv_proxy": "RV proxy — building IV history",
            "insufficient_data": "no data",
        }[self.source]


# ======================================================================================
# The VRP edge
# ======================================================================================
@dataclass
class VRPSnapshot:
    """Everything the scanner knows about one underlying at one moment."""

    symbol: str
    spot: Optional[float] = None
    implied_vol: Optional[float] = None
    realized_vol: Optional[float] = None
    realized_vol_yz: Optional[float] = None
    realized_vol_parkinson: Optional[float] = None
    iv_rank: IVRank = field(default_factory=lambda: IVRank(value=None))
    expiration: Optional[date] = None
    dte: Optional[int] = None
    error: Optional[str] = None
    asof: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------ derived edge
    @property
    def vrp(self) -> Optional[float]:
        """IV − RV in annualised volatility points. Positive = we are paid."""
        if self.implied_vol is None or self.reference_rv is None:
            return None
        return self.implied_vol - self.reference_rv

    @property
    def reference_rv(self) -> Optional[float]:
        """Yang-Zhang when available (gap-aware), else close-to-close."""
        return self.realized_vol_yz or self.realized_vol

    @property
    def vrp_ratio(self) -> Optional[float]:
        """IV / RV. Historically ~1.1–1.3 on index products; < 1.0 = no edge."""
        if self.implied_vol is None or not self.reference_rv:
            return None
        return self.implied_vol / self.reference_rv

    @property
    def is_tradeable(self) -> bool:
        settings = config.SETTINGS
        return (
            self.error is None
            and self.vrp is not None
            and self.vrp >= settings.min_vrp
            and self.iv_rank.value is not None
            and self.iv_rank.value >= settings.min_iv_rank
        )

    def reject_reason(self) -> Optional[str]:
        """Human-readable explanation of why this symbol is not tradeable."""
        settings = config.SETTINGS
        if self.error:
            return self.error
        if self.implied_vol is None:
            return "no implied volatility available"
        if self.reference_rv is None:
            return "insufficient price history for realised vol"
        if self.iv_rank.value is None:
            return f"IV Rank unavailable ({self.iv_rank.label})"
        if self.iv_rank.value < settings.min_iv_rank:
            # One decimal: a 49.6 shown as "50 < 50" reads like a bug.
            return f"IV Rank {self.iv_rank.value:.1f} < {settings.min_iv_rank:.0f}"
        if self.vrp is not None and self.vrp < settings.min_vrp:
            return f"VRP {self.vrp * 100:.1f}pts < {settings.min_vrp * 100:.1f}pts"
        return None


def build_vrp_snapshot(
    symbol: str,
    spot: Optional[float],
    atm_iv: Optional[float],
    bars: Optional[dict],
    iv_store: IVHistoryStore,
    *,
    expiration: Optional[date] = None,
    error: Optional[str] = None,
) -> VRPSnapshot:
    """Assemble a :class:`VRPSnapshot` from a spot price, an IV and OHLC bars.

    ``bars`` is a dict of parallel lists: ``{"open": [...], "high": [...],
    "low": [...], "close": [...]}``. Missing keys simply disable the estimators
    that need them.
    """
    settings = config.SETTINGS
    snapshot = VRPSnapshot(symbol=symbol.upper(), spot=spot, implied_vol=atm_iv, expiration=expiration, error=error)
    if expiration:
        snapshot.dte = days_to_expiry(expiration)

    if bars:
        closes = bars.get("close") or []
        highs = bars.get("high") or []
        lows = bars.get("low") or []
        opens = bars.get("open") or []
        snapshot.realized_vol = realized_vol_close_to_close(closes, settings.rv_window)
        snapshot.realized_vol_parkinson = realized_vol_parkinson(highs, lows, settings.rv_window)
        snapshot.realized_vol_yz = realized_vol_yang_zhang(opens, highs, lows, closes, settings.rv_window)

        if atm_iv:
            # Rolling RV series powers the IV Rank bootstrap proxy.
            rv_series = _rolling_rv_series(closes, settings.rv_window)
            snapshot.iv_rank = iv_store.rank(symbol, atm_iv, rv_series)
            iv_store.record(symbol, atm_iv, spot, snapshot.reference_rv)
    elif atm_iv:
        snapshot.iv_rank = iv_store.rank(symbol, atm_iv)

    return snapshot


def _rolling_rv_series(closes: Sequence[float], window: int) -> List[float]:
    """Trailing series of rolling realised volatilities (for the proxy rank)."""
    prices = [p for p in closes if p and p > 0]
    if len(prices) < window + 2:
        return []
    out: List[float] = []
    for end in range(window + 1, len(prices) + 1):
        rv = realized_vol_close_to_close(prices[end - window - 1 : end])
        if rv:
            out.append(rv)
    return out


# ======================================================================================
# Trade log and expectancy
# ======================================================================================
TRADE_FIELDS = [
    "trade_id",
    "opened_at",
    "closed_at",
    "underlying",
    "symbol",
    "strategy",
    "contracts",
    "strike",
    "expiration",
    "entry_delta",
    "entry_iv",
    "entry_rv",
    "entry_iv_rank",
    # Signed net premium per contract: positive when the structure was sold for
    # a credit, negative when it was bought for a debit. The column keeps its
    # original name so logs written before the strategy library still load.
    "credit",
    "exit_debit",      # per-contract net cost to close
    # JSON list of ``{"symbol", "action", "ratio"}``. The whole reason a condor
    # can be managed as one trade rather than four unrelated positions — without
    # it, the exit logic cannot tell a long wing from a naked short.
    "legs",
    "capital_required",
    "max_loss",
    "pnl_usd",
    "pnl_zar",
    "usd_zar",
    "status",          # open | closed | cancelled
    "exit_reason",     # profit_target | stop_loss | time_exit | manual | expired
    "note",
]


class TradeLog:
    """CSV-backed record of every position the bot opens and closes.

    Deliberately a flat file: it survives a Streamlit Cloud restart when placed on
    a mounted volume, is trivially exportable, and needs no database service on a
    free tier.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or config.TRADE_LOG_PATH)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=TRADE_FIELDS).writeheader()
        else:
            self._migrate_header()

    def _migrate_header(self) -> None:
        """Rewrite an older log in the current schema, preserving every row.

        Appending new-schema rows to a file with an old header silently shifts
        every column after the first new one, which corrupts the P&L record
        rather than failing. Migrating on open is cheap and the file is small.
        """
        try:
            with self.path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames == TRADE_FIELDS:
                    return
                rows = list(reader)
        except OSError:
            return
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in TRADE_FIELDS} for row in rows)

    # ------------------------------------------------------------------ writes
    def append(self, record: dict) -> None:
        row = {key: record.get(key, "") for key in TRADE_FIELDS}
        with self._lock, self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=TRADE_FIELDS).writerow(row)

    def update(self, trade_id: str, **changes) -> bool:
        """Patch an existing row in place. Returns ``True`` when a row matched."""
        with self._lock:
            rows = self._read()
            found = False
            for row in rows:
                if row.get("trade_id") == trade_id:
                    row.update({k: v for k, v in changes.items() if k in TRADE_FIELDS})
                    found = True
            if found:
                with self.path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=TRADE_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
            return found

    # ------------------------------------------------------------------- reads
    def _read(self) -> List[dict]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as handle:
            return [row for row in csv.DictReader(handle) if row.get("trade_id")]

    def all(self) -> List[dict]:
        return self._read()

    def open_trades(self) -> List[dict]:
        return [row for row in self._read() if row.get("status") == "open"]

    def live_trades(self) -> List[dict]:
        """Open trades plus those with an exit order submitted but not yet filled."""
        return [row for row in self._read() if row.get("status") in {"open", "closing"}]

    def closed_trades(self) -> List[dict]:
        return [row for row in self._read() if row.get("status") == "closed"]

    def find(self, trade_id: str) -> Optional[dict]:
        return next((row for row in self._read() if row.get("trade_id") == trade_id), None)

    def find_by_symbol(self, symbol: str, status: str = "open") -> Optional[dict]:
        return next(
            (row for row in self._read() if row.get("symbol") == symbol and row.get("status") == status),
            None,
        )

    def opened_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        return sum(1 for row in self._read() if str(row.get("opened_at", "")).startswith(today))


@dataclass
class Expectancy:
    """Realised expectancy per trade, in USD, plus the inputs that produced it."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    p_win: float = 0.0
    p_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    profit_factor: Optional[float] = None
    largest_win: float = 0.0
    largest_loss: float = 0.0
    #: Win rate this trade's payoff geometry must beat to break even.
    breakeven_p_win: Optional[float] = None

    @property
    def has_data(self) -> bool:
        return self.trades > 0

    @property
    def edge_vs_breakeven(self) -> Optional[float]:
        """Realised win rate minus the rate the bracket requires. Positive = edge."""
        if self.breakeven_p_win is None or not self.trades:
            return None
        return self.p_win - self.breakeven_p_win


def compute_expectancy(closed_trades: Iterable[dict]) -> Expectancy:
    """``E = (P_win × W) − (P_loss × L)`` over the realised trade record.

    ``W`` and ``L`` are *average* win and average absolute loss in USD, so ``E``
    is the expected dollar outcome of the next trade at the current sample.
    """
    pnls: List[float] = []
    for row in closed_trades:
        try:
            pnls.append(float(row.get("pnl_usd") or 0.0))
        except (TypeError, ValueError):
            continue

    result = Expectancy()
    if not pnls:
        return result

    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p < 0]  # stored as positive magnitudes

    result.trades = len(pnls)
    result.wins = len(wins)
    result.losses = len(losses)
    result.p_win = len(wins) / len(pnls)
    result.p_loss = len(losses) / len(pnls)
    result.avg_win = sum(wins) / len(wins) if wins else 0.0
    result.avg_loss = sum(losses) / len(losses) if losses else 0.0
    result.expectancy = result.p_win * result.avg_win - result.p_loss * result.avg_loss
    result.total_pnl = sum(pnls)
    result.largest_win = max(wins) if wins else 0.0
    result.largest_loss = max(losses) if losses else 0.0
    gross_loss = sum(losses)
    result.profit_factor = (sum(wins) / gross_loss) if gross_loss > 0 else None
    result.breakeven_p_win = breakeven_win_rate()
    return result


def breakeven_win_rate(
    profit_target_pct: Optional[float] = None,
    stop_loss_multiple: Optional[float] = None,
) -> float:
    """The win rate the managed bracket must beat for expectancy to be positive.

    Setting ``E = p·W − (1−p)·L = 0`` gives ``p = L / (W + L)``. With the default
    50% profit target and 200% stop that is ``2.0 / (0.5 + 2.0) = 80%``.

    This number is the honest hurdle for the whole strategy. A driftless price
    process hits either barrier in proportion to its distance, so bracket
    geometry alone can never produce an edge — the *only* thing that lifts the
    win rate above the hurdle is selling volatility that is genuinely richer than
    what subsequently realises. That is why the entry filter insists on both
    ``IV > RV`` and ``IV Rank > 50``.
    """
    settings = config.SETTINGS
    win = settings.profit_target_pct if profit_target_pct is None else profit_target_pct
    loss = settings.stop_loss_multiple if stop_loss_multiple is None else stop_loss_multiple
    total = win + loss
    return (loss / total) if total > 0 else 1.0


def theoretical_expectancy(
    credit: float,
    delta: float,
    *,
    profit_target_pct: Optional[float] = None,
    stop_loss_multiple: Optional[float] = None,
    contracts: int = 1,
) -> Expectancy:
    """Ex-ante expectancy of a managed short-premium trade, in USD.

    ``W`` and ``L`` are the *managed* outcomes: the profit taker caps the win at
    50% of the credit, the stop caps the loss at 200% of it.

    ``P(win)`` is approximated as ``1 − |delta|``, delta being the risk-neutral
    probability of finishing in the money. Read it as a **conservative lower
    bound**, not a forecast: the position is closed at 50% of max profit weeks
    before expiry, so its real win rate sits somewhere between this figure and 1.
    Compare the result against :func:`breakeven_win_rate` — the gap is the hurdle
    the variance risk premium has to cover, and only the realised trade log
    (:func:`compute_expectancy`) settles whether it did.
    """
    settings = config.SETTINGS
    profit_target_pct = settings.profit_target_pct if profit_target_pct is None else profit_target_pct
    stop_loss_multiple = settings.stop_loss_multiple if stop_loss_multiple is None else stop_loss_multiple

    multiplier = 100 * max(contracts, 1)
    p_win = _clamp(1.0 - abs(delta), 0.0, 1.0)
    win = credit * profit_target_pct * multiplier
    loss = credit * stop_loss_multiple * multiplier

    result = Expectancy()
    result.trades = 1
    result.p_win = p_win
    result.p_loss = 1.0 - p_win
    result.avg_win = win
    result.avg_loss = loss
    result.expectancy = p_win * win - (1.0 - p_win) * loss
    result.profit_factor = (p_win * win) / ((1.0 - p_win) * loss) if (1.0 - p_win) * loss > 0 else None
    result.breakeven_p_win = breakeven_win_rate(profit_target_pct, stop_loss_multiple)
    return result


def monthly_pnl(closed_trades: Iterable[dict], currency: str = "zar") -> Dict[str, float]:
    """Realised P&L bucketed by ``YYYY-MM`` of the close date."""
    key = "pnl_zar" if currency.lower() == "zar" else "pnl_usd"
    buckets: Dict[str, float] = {}
    for row in closed_trades:
        closed_at = str(row.get("closed_at") or "")
        if len(closed_at) < 7:
            continue
        try:
            buckets[closed_at[:7]] = buckets.get(closed_at[:7], 0.0) + float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return dict(sorted(buckets.items()))


def daily_pnl(closed_trades: Iterable[dict], currency: str = "usd") -> Dict[str, float]:
    """Realised P&L bucketed by ``YYYY-MM-DD`` of the close date."""
    key = "pnl_zar" if currency.lower() == "zar" else "pnl_usd"
    buckets: Dict[str, float] = {}
    for row in closed_trades:
        closed_at = str(row.get("closed_at") or "")
        if len(closed_at) < 10:
            continue
        try:
            buckets[closed_at[:10]] = buckets.get(closed_at[:10], 0.0) + float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return dict(sorted(buckets.items()))


def performance_metrics(closed_trades: Iterable[dict], currency: str = "usd") -> Dict[str, object]:
    """Risk-adjusted performance of the realised record.

    Sharpe and Sortino are computed on the series of *daily* realised P&L, not
    per trade. Per-trade ratios flatter a strategy that trades rarely, and this
    one deliberately trades rarely — two entries a day at most, often none.

    Every ratio is returned beside ``trading_days`` and ``trades`` so a reader
    can tell an estimate from a number. A Sharpe computed on eleven days is a
    rumour; the caller has to be able to see that it is.
    """
    rows = list(closed_trades)
    series = daily_pnl(rows, currency)
    values = list(series.values())
    n = len(values)

    mean = (sum(values) / n) if n else 0.0
    variance = (sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    stdev = math.sqrt(variance)
    downside = [v for v in values if v < 0]
    downside_dev = (
        math.sqrt(sum(v * v for v in downside) / len(downside)) if downside else 0.0
    )

    # Annualised on 252 trading days. The series is daily *realised* P&L, so
    # days with no closes are genuinely zero-P&L days, not gaps.
    sharpe = (mean / stdev * math.sqrt(252)) if stdev > 0 else None
    sortino = (mean / downside_dev * math.sqrt(252)) if downside_dev > 0 else None

    equity, peak, max_dd = 0.0, 0.0, 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    expectancy = compute_expectancy(rows)
    return {
        "currency": currency.lower(),
        "trades": expectancy.trades,
        "trading_days": n,
        "win_rate": expectancy.p_win,
        "breakeven_win_rate": expectancy.breakeven_p_win,
        "edge_vs_breakeven": expectancy.edge_vs_breakeven,
        "expectancy_per_trade": expectancy.expectancy,
        "profit_factor": expectancy.profit_factor,
        "total_pnl": expectancy.total_pnl,
        "avg_win": expectancy.avg_win,
        "avg_loss": expectancy.avg_loss,
        "largest_win": expectancy.largest_win,
        "largest_loss": expectancy.largest_loss,
        "daily_mean": mean,
        "daily_stdev": stdev,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "reliability": (
            "insufficient" if n < 20 else "thin" if n < 60 else "usable"
        ),
    }


# ======================================================================================
# USD → ZAR conversion
# ======================================================================================
@dataclass
class FXQuote:
    rate: float
    source: str
    asof: datetime
    stale: bool = False

    @property
    def is_live(self) -> bool:
        return self.source not in {"fallback", "manual"}


class ForexConverter:
    """Live USD/ZAR with layered fallbacks and an on-disk cache.

    Free, key-less endpoints are tried in order; the last good rate is cached to
    disk so a restart (or an outage) still produces sensible Rand figures. If
    every source fails, the configured fallback rate is used and flagged stale so
    the UI never presents a guessed rate as live.
    """

    SOURCES = (
        ("open.er-api.com", "https://open.er-api.com/v6/latest/USD", lambda d: d["rates"]["ZAR"]),
        ("frankfurter.app", "https://api.frankfurter.app/latest?from=USD&to=ZAR", lambda d: d["rates"]["ZAR"]),
        ("exchangerate.host", "https://api.exchangerate.host/latest?base=USD&symbols=ZAR", lambda d: d["rates"]["ZAR"]),
    )

    def __init__(self, cache_path: Optional[Path] = None, ttl_seconds: int = 900) -> None:
        self.cache_path = Path(cache_path or config.FX_CACHE_PATH)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._quote: Optional[FXQuote] = None
        self._fetched_at: float = 0.0

    # ------------------------------------------------------------------ public
    def get_rate(self, force_refresh: bool = False) -> FXQuote:
        """Return the current USD/ZAR quote, refreshing at most every ``ttl``."""
        with self._lock:
            fresh = self._quote and (time.time() - self._fetched_at) < self.ttl_seconds
            if fresh and not force_refresh:
                return self._quote  # type: ignore[return-value]

            quote = self._fetch_live()
            if quote:
                self._quote = quote
                self._fetched_at = time.time()
                self._write_cache(quote)
                return quote

            cached = self._read_cache()
            if cached:
                cached.stale = True
                self._quote = cached
                self._fetched_at = time.time()
                return cached

            fallback = FXQuote(
                rate=config.SETTINGS.fallback_usd_zar,
                source="fallback",
                asof=datetime.now(timezone.utc),
                stale=True,
            )
            self._quote = fallback
            self._fetched_at = time.time()
            return fallback

    def set_manual_rate(self, rate: float) -> FXQuote:
        """Pin the rate by hand (useful when a host blocks outbound HTTP)."""
        quote = FXQuote(rate=float(rate), source="manual", asof=datetime.now(timezone.utc))
        with self._lock:
            self._quote = quote
            self._fetched_at = time.time()
            self._write_cache(quote)
        return quote

    def usd_to_zar(self, usd: float, quote: Optional[FXQuote] = None) -> float:
        return float(usd) * (quote or self.get_rate()).rate

    def zar_to_usd(self, zar: float, quote: Optional[FXQuote] = None) -> float:
        rate = (quote or self.get_rate()).rate
        return float(zar) / rate if rate else 0.0

    # ----------------------------------------------------------------- private
    def _fetch_live(self) -> Optional[FXQuote]:
        import urllib.request

        for name, url, extract in self.SOURCES:
            try:
                with urllib.request.urlopen(url, timeout=6) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                rate = float(extract(payload))
                if rate > 0:
                    return FXQuote(rate=rate, source=name, asof=datetime.now(timezone.utc))
            except Exception:
                continue
        return None

    def _write_cache(self, quote: FXQuote) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps({"rate": quote.rate, "source": quote.source, "asof": quote.asof.isoformat()})
            )
        except OSError:
            pass

    def _read_cache(self) -> Optional[FXQuote]:
        try:
            payload = json.loads(self.cache_path.read_text())
            return FXQuote(
                rate=float(payload["rate"]),
                source=str(payload.get("source", "cache")),
                asof=datetime.fromisoformat(payload["asof"]),
            )
        except Exception:
            return None


# Shared singletons — cheap to construct, safe to import from anywhere.
IV_STORE = IVHistoryStore()
TRADE_LOG = TradeLog()
FX = ForexConverter()
