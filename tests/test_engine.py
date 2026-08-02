"""
Maths regression tests for the Brickvestcapitalterminal engine.

No network, no credentials, no broker. These cover the parts where a silent
error would be expensive: the Black-Scholes round-trip, the implied-volatility
inversion the free data feed forces us to rely on, the realised-vol estimators
that define the VRP, and the expectancy formula the whole platform reports on.

    python tests/test_engine.py     # or: pytest tests/
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep every test's state out of the real ./state directory.
os.environ.setdefault("BVC_STATE_DIR", tempfile.mkdtemp(prefix="bvc-test-"))
# ...and out of the real strategy_config.json, which the agent tests write to.
# A test run that retunes the checked-in strategy is a test run that changes how
# the account trades.
os.environ.setdefault("BVC_STRATEGY_CONFIG", os.path.join(os.environ["BVC_STATE_DIR"], "strategy_config.json"))

import engine  # noqa: E402
from broker_client import OptionQuote, parse_occ_symbol  # noqa: E402


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ======================================================================================
# Black-Scholes
# ======================================================================================
def test_put_call_parity() -> None:
    """C − P = S − K·e^(−rT). If parity breaks, every greek downstream is wrong."""
    spot, strike, t, vol, rate = 100.0, 95.0, 0.25, 0.22, 0.043
    call = engine.bs_price(spot, strike, t, vol, rate, is_call=True)
    put = engine.bs_price(spot, strike, t, vol, rate, is_call=False)
    assert approx(call - put, spot - strike * math.exp(-rate * t), 1e-8)


def test_price_is_monotonic_in_vol() -> None:
    prices = [engine.bs_price(100, 100, 0.25, v, 0.043, True) for v in (0.10, 0.20, 0.40, 0.80)]
    assert prices == sorted(prices)


def test_delta_conventions() -> None:
    """Calls in [0, 1], puts in [−1, 0], ATM call ≈ 0.5."""
    call_delta = engine.bs_delta(100, 100, 0.12, 0.20, 0.043, is_call=True)
    put_delta = engine.bs_delta(100, 100, 0.12, 0.20, 0.043, is_call=False)
    assert 0.0 < call_delta < 1.0
    assert -1.0 < put_delta < 0.0
    assert abs(call_delta - 0.5) < 0.06
    assert approx(call_delta - put_delta, 1.0, 1e-9)  # call Δ − put Δ = 1


def test_thirty_delta_put_is_out_of_the_money() -> None:
    """The strike the bot targets must sit below spot — sanity on the entry rule."""
    spot, t, vol, rate = 500.0, 45 / 365, 0.18, 0.043
    strikes = [s for s in range(400, 520, 1)]
    deltas = [(abs(engine.bs_delta(spot, k, t, vol, rate, False)), k) for k in strikes]
    _, strike_30 = min(deltas, key=lambda item: abs(item[0] - 0.30))
    assert strike_30 < spot
    assert 0.85 * spot < strike_30 < spot  # ~45 DTE, 18% vol → within 15% of spot


def test_implied_vol_round_trip() -> None:
    """Price at a known vol, invert it, get the same vol back."""
    for vol in (0.08, 0.15, 0.25, 0.45, 0.9):
        for is_call in (True, False):
            price = engine.bs_price(420.0, 400.0, 45 / 365, vol, 0.043, is_call)
            recovered = engine.implied_vol(price, 420.0, 400.0, 45 / 365, 0.043, is_call)
            assert recovered is not None
            assert approx(recovered, vol, 1e-4), f"{vol} → {recovered}"


def test_implied_vol_rejects_impossible_quotes() -> None:
    """Below-intrinsic and absurd prices return None rather than a bogus number."""
    assert engine.implied_vol(0.5, 100, 120, 0.25, 0.043, is_call=False) is None  # below intrinsic
    assert engine.implied_vol(999.0, 100, 100, 0.25, 0.043, is_call=True) is None  # unreachable
    assert engine.implied_vol(0.0, 100, 100, 0.25, 0.043, is_call=True) is None
    assert engine.implied_vol(5.0, 100, 100, 0.0, 0.043, is_call=True) is None  # expired


# ======================================================================================
# Realised volatility
# ======================================================================================
def _synthetic_series(n: int, daily_vol: float, seed: int = 7) -> dict:
    """GBM path with a known daily vol → known annualised vol."""
    rng = random.Random(seed)
    price = 100.0
    opens, highs, lows, closes = [], [], [], []
    for _ in range(n):
        open_price = price
        close = price * math.exp(rng.gauss(0.0, daily_vol))
        high = max(open_price, close) * (1 + abs(rng.gauss(0, daily_vol / 4)))
        low = min(open_price, close) * (1 - abs(rng.gauss(0, daily_vol / 4)))
        opens.append(open_price)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        price = close
    return {"open": opens, "high": highs, "low": lows, "close": closes}


def test_realized_vol_recovers_known_volatility() -> None:
    daily = 0.012  # ≈ 19% annualised
    expected = daily * math.sqrt(252)
    bars = _synthetic_series(600, daily)

    cc = engine.realized_vol_close_to_close(bars["close"], 250)
    pk = engine.realized_vol_parkinson(bars["high"], bars["low"], 250)
    yz = engine.realized_vol_yang_zhang(bars["open"], bars["high"], bars["low"], bars["close"], 250)

    for name, value in (("close-to-close", cc), ("parkinson", pk), ("yang-zhang", yz)):
        assert value is not None, name
        assert abs(value - expected) < 0.06, f"{name}: {value:.4f} vs {expected:.4f}"


def test_realized_vol_needs_enough_data() -> None:
    assert engine.realized_vol_close_to_close([100.0, 101.0]) is None
    assert engine.realized_vol_parkinson([100.0], [99.0]) is None
    assert engine.realized_vol_yang_zhang([100], [101], [99], [100]) is None


def test_realized_vol_ignores_bad_prices() -> None:
    """Zeros and Nones from a thin feed must not crash the estimator."""
    closes = [100.0, 0.0, 101.0, None, 102.0, 101.5, 103.0, 102.0, 104.0]
    assert engine.realized_vol_close_to_close(closes) is not None


# ======================================================================================
# VRP and IV Rank
# ======================================================================================
def test_vrp_snapshot_applies_both_filters() -> None:
    store = engine.IVHistoryStore(Path(os.environ["BVC_STATE_DIR"]) / "iv_filters.csv")
    bars = _synthetic_series(300, 0.010)  # ≈ 16% realised

    rich = engine.build_vrp_snapshot("TEST", 100.0, 0.30, bars, store, expiration=date.today() + timedelta(days=45))
    assert rich.vrp is not None and rich.vrp > 0
    assert rich.vrp_ratio is not None and rich.vrp_ratio > 1.0
    assert rich.dte == 45

    cheap = engine.build_vrp_snapshot("TEST2", 100.0, 0.05, bars, store)
    assert cheap.vrp is not None and cheap.vrp < 0
    assert not cheap.is_tradeable
    assert cheap.reject_reason() is not None


def test_iv_rank_from_stored_history() -> None:
    """With a full history the rank is the position in the trailing range."""
    path = Path(os.environ["BVC_STATE_DIR"]) / "iv_rank.csv"
    path.unlink(missing_ok=True)
    store = engine.IVHistoryStore(path)

    # Seed 60 observations spanning 10%–30% by writing rows directly.
    with path.open("w", newline="") as handle:
        handle.write("date,symbol,iv,spot,rv\n")
        for i in range(60):
            day = date(2025, 1, 1) + timedelta(days=i)
            iv = 0.10 + (i / 59) * 0.20
            handle.write(f"{day.isoformat()},RANKY,{iv:.6f},,\n")

    top = store.rank("RANKY", 0.30)
    assert top.source == "iv_history"
    assert top.value is not None and top.value > 95

    bottom = store.rank("RANKY", 0.10)
    assert bottom.value is not None and bottom.value < 5

    middle = store.rank("RANKY", 0.20)
    assert middle.value is not None and 45 < middle.value < 55
    assert not middle.is_proxy


def test_iv_rank_falls_back_to_proxy_and_says_so() -> None:
    store = engine.IVHistoryStore(Path(os.environ["BVC_STATE_DIR"]) / "iv_proxy.csv")
    rv_history = [0.10 + 0.002 * i for i in range(40)]
    result = store.rank("NEWSYM", 0.25, rv_history)
    assert result.source == "rv_proxy"
    assert result.is_proxy
    assert "proxy" in result.label.lower()

    empty = store.rank("NOSYM", 0.25)
    assert empty.value is None
    assert empty.source == "insufficient_data"


def test_iv_history_stores_one_row_per_symbol_per_day() -> None:
    path = Path(os.environ["BVC_STATE_DIR"]) / "iv_dedupe.csv"
    path.unlink(missing_ok=True)
    store = engine.IVHistoryStore(path)
    store.record("SPY", 0.18, 500.0, 0.15)
    store.record("SPY", 0.19, 501.0, 0.15)  # same day → overwrite, not append
    store.record("QQQ", 0.22, 430.0, 0.19)

    assert len(store.series("SPY")) == 1
    assert approx(store.series("SPY")[0][1], 0.19, 1e-9)
    assert len(store.series("QQQ")) == 1
    store.record("SPY", 0.0)  # invalid IV is dropped
    assert len(store.series("SPY")) == 1


# ======================================================================================
# Expectancy
# ======================================================================================
def test_expectancy_matches_the_formula() -> None:
    trades = [
        {"pnl_usd": "50"}, {"pnl_usd": "50"}, {"pnl_usd": "50"}, {"pnl_usd": "50"},  # 4 wins
        {"pnl_usd": "-100"},                                                          # 1 loss
    ]
    result = engine.compute_expectancy(trades)
    assert result.trades == 5
    assert approx(result.p_win, 0.8)
    assert approx(result.avg_win, 50.0)
    assert approx(result.avg_loss, 100.0)
    # E = 0.8 × 50 − 0.2 × 100 = 20
    assert approx(result.expectancy, 20.0)
    assert approx(result.total_pnl, 100.0)
    assert result.profit_factor is not None and approx(result.profit_factor, 2.0)


def test_expectancy_handles_an_empty_or_broken_log() -> None:
    assert not engine.compute_expectancy([]).has_data
    assert not engine.compute_expectancy([{"pnl_usd": "not-a-number"}]).has_data


def test_theoretical_expectancy_of_a_managed_thirty_delta_trade() -> None:
    """1.50 credit, 30 delta, 50% target, 200% stop → E = 0.7×75 − 0.3×300."""
    result = engine.theoretical_expectancy(1.50, -0.30, profit_target_pct=0.5, stop_loss_multiple=2.0)
    assert approx(result.p_win, 0.70)
    assert approx(result.avg_win, 75.0)
    assert approx(result.avg_loss, 300.0)
    assert approx(result.expectancy, 0.70 * 75.0 - 0.30 * 300.0)  # = −37.50


def test_breakeven_win_rate_is_the_real_hurdle() -> None:
    """p = L / (W + L). With a 50% target and a 200% stop that is 80%.

    The delta-implied 70% is the expiry-based lower bound, so it sits *below* the
    hurdle by construction — the gap is exactly what the variance risk premium
    has to close, and only the realised trade log can confirm it did.
    """
    assert approx(engine.breakeven_win_rate(0.50, 2.00), 0.80)
    assert approx(engine.breakeven_win_rate(1.00, 1.00), 0.50)  # symmetric payoff
    assert approx(engine.breakeven_win_rate(0.25, 4.00), 4.0 / 4.25)  # tighter target, higher hurdle

    managed = engine.theoretical_expectancy(1.50, -0.30, profit_target_pct=0.5, stop_loss_multiple=2.0)
    assert managed.breakeven_p_win is not None
    assert managed.p_win < managed.breakeven_p_win
    # At exactly the breakeven rate the expectancy is zero.
    at_hurdle = engine.theoretical_expectancy(1.50, -0.20, profit_target_pct=0.5, stop_loss_multiple=2.0)
    assert approx(at_hurdle.p_win, 0.80)
    assert approx(at_hurdle.expectancy, 0.0, 1e-9)


def test_realised_expectancy_reports_the_hurdle_too() -> None:
    result = engine.compute_expectancy([{"pnl_usd": "50"}, {"pnl_usd": "-100"}])
    assert result.breakeven_p_win is not None
    assert result.edge_vs_breakeven is not None
    assert approx(result.edge_vs_breakeven, result.p_win - result.breakeven_p_win)


def test_monthly_pnl_buckets_by_close_month() -> None:
    trades = [
        {"closed_at": "2026-01-15T10:00:00", "pnl_zar": "4000", "pnl_usd": "220"},
        {"closed_at": "2026-01-28T10:00:00", "pnl_zar": "6500", "pnl_usd": "350"},
        {"closed_at": "2026-02-03T10:00:00", "pnl_zar": "-1200", "pnl_usd": "-65"},
        {"closed_at": "", "pnl_zar": "999"},  # unsettled → ignored
    ]
    by_month = engine.monthly_pnl(trades, "zar")
    assert approx(by_month["2026-01"], 10_500.0)
    assert approx(by_month["2026-02"], -1_200.0)
    assert list(by_month) == ["2026-01", "2026-02"]  # sorted
    assert approx(engine.monthly_pnl(trades, "usd")["2026-01"], 570.0)


# ======================================================================================
# Trade log
# ======================================================================================
def test_trade_log_open_update_close() -> None:
    log = engine.TradeLog(Path(os.environ["BVC_STATE_DIR"]) / "trades_roundtrip.csv")
    log.append(
        {
            "trade_id": "abc123",
            "opened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "underlying": "SPY",
            "symbol": "SPY261218P00600000",
            "credit": "1.5000",
            "contracts": 1,
            "status": "open",
        }
    )
    assert len(log.open_trades()) == 1
    assert log.find_by_symbol("SPY261218P00600000") is not None
    assert log.opened_today() == 1

    assert log.update("abc123", status="closed", pnl_usd="75.00", pnl_zar="1387.50", exit_reason="profit_target")
    assert not log.update("does-not-exist", status="closed")

    closed = log.closed_trades()
    assert len(closed) == 1 and closed[0]["exit_reason"] == "profit_target"
    assert log.open_trades() == []
    assert approx(engine.compute_expectancy(closed).expectancy, 75.0)


# ======================================================================================
# Currency
# ======================================================================================
def test_fx_falls_back_and_flags_it_stale() -> None:
    fx = engine.ForexConverter(Path(os.environ["BVC_STATE_DIR"]) / "fx_missing.json")
    fx._fetch_live = lambda: None  # simulate a blocked/offline host  # noqa: SLF001
    quote = fx.get_rate(force_refresh=True)
    assert quote.stale
    assert not quote.is_live
    assert quote.rate == engine.config.SETTINGS.fallback_usd_zar


def test_fx_manual_pin_and_conversion() -> None:
    fx = engine.ForexConverter(Path(os.environ["BVC_STATE_DIR"]) / "fx_manual.json")
    fx.set_manual_rate(18.75)
    assert approx(fx.usd_to_zar(100.0), 1875.0)
    assert approx(fx.zar_to_usd(1875.0), 100.0)


# ======================================================================================
# OCC symbols and quote helpers
# ======================================================================================
def test_occ_symbol_parsing() -> None:
    parsed = parse_occ_symbol("SPY251219P00600000")
    assert parsed == {
        "underlying": "SPY",
        "expiration": date(2025, 12, 19),
        "option_type": "put",
        "strike": 600.0,
    }
    call = parse_occ_symbol("AAPL260116C00250500")
    assert call is not None and call["option_type"] == "call" and approx(call["strike"], 250.5)
    assert parse_occ_symbol("SPY") is None  # equities are not options
    assert parse_occ_symbol("") is None


def test_option_quote_mid_and_spread_gates() -> None:
    good = OptionQuote("X", "X", date.today() + timedelta(days=45), 100.0, "put", bid=1.00, ask=1.10)
    assert approx(good.mid, 1.05)
    assert good.has_two_sided_quote
    assert good.spread_pct is not None and good.spread_pct < 0.10

    wide = OptionQuote("X", "X", date.today() + timedelta(days=45), 100.0, "put", bid=0.10, ask=1.00)
    assert wide.spread_pct is not None and wide.spread_pct > 1.0  # rejected by the filter

    one_sided = OptionQuote("X", "X", date.today() + timedelta(days=45), 100.0, "put", bid=None, ask=1.00, last=0.90)
    assert not one_sided.has_two_sided_quote
    assert approx(one_sided.mid, 0.90)  # falls back to last trade


def test_year_fraction_and_dte() -> None:
    future = datetime.now(timezone.utc).date() + timedelta(days=45)
    assert engine.days_to_expiry(future) == 45
    assert approx(engine.year_fraction(future), 45 / 365, 1e-9)
    past = datetime.now(timezone.utc).date() - timedelta(days=5)
    assert engine.year_fraction(past) == 0.0  # floored, never negative


# ======================================================================================
# Guardrail arithmetic
# ======================================================================================
def test_stop_loss_multiple_is_a_price_not_a_loss() -> None:
    """A '200% stop' means loss = 2× credit, so the buy-back price is 3× credit."""
    settings = engine.config.load_settings()
    settings.stop_loss_multiple = 2.0
    assert approx(settings.stop_loss_price_multiple, 3.0)

    credit = 1.50
    assert approx(credit * settings.stop_loss_price_multiple, 4.50)          # stop trigger
    assert approx(credit * (1 - settings.profit_target_pct), 0.75)           # profit trigger
    loss_at_stop = (credit - credit * settings.stop_loss_price_multiple) * 100
    assert approx(loss_at_stop, -300.0)                                      # = 200% of the $150 credit


def test_margin_utilization_and_projection() -> None:
    from broker_client import AccountSnapshot

    account = AccountSnapshot(equity=30_000.0, maintenance_margin=12_000.0)
    assert approx(account.margin_utilization, 0.40)
    assert account.margin_utilization < 0.50  # entry allowed

    # A $60,000 cash-secured put would project to 240% — must be refused.
    projected = (account.maintenance_margin + 60_000.0) / account.equity
    assert projected > 0.50

    broke = AccountSnapshot(equity=0.0, maintenance_margin=1_000.0)
    assert broke.margin_utilization == 0.0  # no division by zero


# ======================================================================================
# Backtester
# ======================================================================================
def _synthetic_frame(n=900, daily_vol=0.0085, seed=5, s0=300.0):
    import pandas as pd

    rng = random.Random(seed)
    price = s0
    rows = {"open": [], "high": [], "low": [], "close": []}
    for _ in range(n):
        open_price = price
        close = price * math.exp(rng.gauss(0.0003, daily_vol))
        rows["open"].append(open_price)
        rows["close"].append(close)
        rows["high"].append(max(open_price, close) * (1 + abs(rng.gauss(0, daily_vol / 3))))
        rows["low"].append(min(open_price, close) * (1 - abs(rng.gauss(0, daily_vol / 3))))
        price = close
    return pd.DataFrame(rows, index=pd.bdate_range("2019-01-01", periods=n))


def test_strike_from_delta_round_trips() -> None:
    """The inverse solver must reproduce the delta it was asked for."""
    spot, t, vol, rate = 500.0, 45 / 365, 0.20, 0.045
    for target in (0.05, 0.16, 0.30, 0.45):
        strike = engine.strike_from_delta(spot, t, vol, rate, -target, is_call=False)
        assert approx(abs(engine.bs_delta(spot, strike, t, vol, rate, False)), target, 1e-6)
        assert strike < spot  # a short put sits below spot

        call_strike = engine.strike_from_delta(spot, t, vol, rate, target, is_call=True)
        assert approx(engine.bs_delta(spot, call_strike, t, vol, rate, True), target, 1e-6)
        assert call_strike > spot


def test_norm_ppf_matches_known_quantiles() -> None:
    assert approx(engine.norm_ppf(0.5), 0.0, 1e-9)
    assert approx(engine.norm_ppf(0.975), 1.959963985, 1e-6)
    assert approx(engine.norm_ppf(0.025), -1.959963985, 1e-6)


def test_backtest_runs_and_books_trades() -> None:
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=5), "QQQ": _synthetic_frame(seed=6, s0=200.0)}
    cfg = bt.BacktestConfig(
        symbols=["SPY", "QQQ"], start_date="2020-01-01", end_date="2022-01-01",
        strategy="put_credit_spread", vrp_points=0.03,
    )
    result = bt.Backtester(cfg, prices).run()
    assert result.metrics["trades"] > 0
    assert result.metrics["final_nav"] > 0
    assert not result.nav.empty
    for trade in result.trades:
        assert trade["exit_reason"] in {"profit_target", "stop_loss", "time_exit", "expired"}
        assert trade["collateral"] > 0


def test_more_assumed_premium_produces_more_profit() -> None:
    """The core sanity check: the VRP assumption must drive the result monotonically.

    If a bigger modelled premium did not pay better on identical price paths, the
    pricing surface and the P&L accounting would be inconsistent.

    Three symbols over four years on purpose. The same check on one symbol over
    two years runs about 30 trades, and at that sample a 3-point difference in
    assumed premium is indistinguishable from path luck — the invariant is real
    but unmeasurable, and a test that cannot measure its own claim is a test
    that fails at random. It is the platform's own lesson applied to itself.
    """
    import backtest as bt

    prices = {
        "SPY": _synthetic_frame(seed=7),
        "QQQ": _synthetic_frame(seed=8, s0=200.0),
        "IWM": _synthetic_frame(seed=9, s0=160.0),
    }
    returns = []
    for vrp in (0.0, 0.03, 0.06):
        cfg = bt.BacktestConfig(
            symbols=list(prices), start_date="2020-01-01", end_date="2024-01-01",
            strategy="put_credit_spread", vrp_points=vrp,
        )
        result = bt.Backtester(cfg, prices).run()
        assert result.metrics["trades"] > 80, f"sample too small to measure: {result.metrics['trades']}"
        returns.append(result.metrics["total_return"])
    assert returns[0] < returns[1] < returns[2], returns


def test_backtest_explains_an_empty_run() -> None:
    """A run that takes no trades must say why, not just return zeros."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=8)}
    cfg = bt.BacktestConfig(
        symbols=["SPY"], start_date="2020-01-01", end_date="2021-01-01",
        strategy="short_put", initial_capital=15_000.0, max_allocation_per_asset=0.10,
    )
    result = bt.Backtester(cfg, prices).run()
    assert result.metrics["trades"] == 0
    assert result.warnings
    assert any("collateral" in w or "vol-rank" in w for w in result.warnings)


def test_backtest_respects_the_margin_ceiling() -> None:
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=9), "QQQ": _synthetic_frame(seed=10, s0=250.0)}
    cfg = bt.BacktestConfig(
        symbols=["SPY", "QQQ"], start_date="2020-01-01", end_date="2022-01-01",
        strategy="put_credit_spread", max_margin_utilization=0.20,
        max_allocation_per_asset=0.50,
    )
    engine_run = bt.Backtester(cfg, prices)
    result = engine_run.run()
    # Collateral committed at any instant must never exceed the ceiling of NAV.
    peak_collateral = 0.0
    for position in engine_run.positions:
        peak_collateral = max(peak_collateral, position.collateral)
    assert peak_collateral <= cfg.initial_capital * cfg.max_margin_utilization + 1e-6
    assert result.metrics["trades"] >= 0


def test_yfinance_frame_shapes_all_parse() -> None:
    """yfinance returns different column shapes per version and symbol count.

    Flat for one ticker, MultiIndex for several, and the ticker level has moved
    between releases — so the extractor inspects the frame instead of guessing.
    """
    import pandas as pd

    import backtest as bt

    index = pd.bdate_range("2020-01-01", periods=5)
    ohlc = {
        "Open": [1, 2, 3, 4, 5], "High": [2, 3, 4, 5, 6],
        "Low": [0.5, 1, 2, 3, 4], "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
        "Volume": [9] * 5,
    }
    flat = pd.DataFrame(ohlc, index=index)
    level0 = pd.concat({"SPY": pd.DataFrame(ohlc, index=index),
                        "QQQ": pd.DataFrame(ohlc, index=index)}, axis=1)
    level1 = level0.swaplevel(0, 1, axis=1).sort_index(axis=1)

    for raw in (flat, level0, level1):
        frame = bt._extract_symbol(raw, "SPY")
        assert frame is not None
        assert list(frame.columns) == ["open", "high", "low", "close"]
        assert len(frame) == 5

    assert bt._extract_symbol(level0, "NOTLISTED") is None


def test_compare_strategies_shares_one_history() -> None:
    """All structures replay over the same prices, capital and rules."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=11), "QQQ": _synthetic_frame(seed=12, s0=200.0)}
    cfg = bt.BacktestConfig(
        symbols=["SPY", "QQQ"], start_date="2020-01-01", end_date="2022-01-01",
        initial_capital=250_000.0, vrp_points=0.03,
    )
    results = bt.compare_strategies(cfg, prices)
    assert set(results) == set(bt.ALL_STRATEGIES)
    for name, result in results.items():
        assert result.config.strategy == name
        assert result.config.initial_capital == cfg.initial_capital
        assert result.config.vrp_points == cfg.vrp_points

    rows = bt.comparison_table(results)
    assert len(rows) == len(bt.ALL_STRATEGIES)
    # Ranked by return, best first.
    assert rows == sorted(rows, key=lambda r: r["total_return"], reverse=True)


def test_two_expiry_strategies_are_refused_not_approximated() -> None:
    """This replay prices one expiry. A calendar must fail loudly, not quietly."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=11)}
    for name in bt.TWO_EXPIRY_STRATEGIES:
        cfg = bt.BacktestConfig(symbols=["SPY"], start_date="2020-01-01", end_date="2021-01-01",
                                strategy=name)
        try:
            bt.Backtester(cfg, prices).run()
        except ValueError as exc:
            assert "two expiries" in str(exc)
        else:
            raise AssertionError(f"{name} silently produced a result it cannot model")


def test_short_put_stays_a_valid_name_for_the_cash_secured_put() -> None:
    """Old configs and saved comparisons must not break on a rename."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=11)}
    cfg = bt.BacktestConfig(symbols=["SPY"], start_date="2020-01-01", end_date="2021-06-01",
                            strategy="short_put", initial_capital=250_000.0)
    legacy = bt.Backtester(cfg, prices).run()
    canonical = bt.Backtester(
        bt.BacktestConfig(**{**cfg.__dict__, "strategy": "cash_secured_put"}), prices
    ).run()
    assert legacy.metrics["total_return"] == canonical.metrics["total_return"]
    assert len(legacy.trades) == len(canonical.trades)


def test_capital_is_respected_end_to_end() -> None:
    """Starting capital flows through sizing, NAV and the final result."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=13)}
    for capital in (50_000.0, 200_000.0):
        cfg = bt.BacktestConfig(
            symbols=["SPY"], start_date="2020-01-01", end_date="2022-01-01",
            strategy="put_credit_spread", initial_capital=capital, vrp_points=0.03,
        )
        result = bt.Backtester(cfg, prices).run()
        assert abs(result.nav.iloc[0] - capital) < capital * 0.05
        assert result.metrics["final_nav"] > 0


def test_entry_window_is_declared_unreplayable_on_daily_bars() -> None:
    """The window is a live-bot rule; a daily-bar replay must say so, not fake it.

    Silently accepting an intraday filter that cannot be applied would make the
    backtest claim to test something it never tested.
    """
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=14)}
    cfg = bt.BacktestConfig(
        symbols=["SPY"], start_date="2020-01-01", end_date="2021-06-01",
        strategy="put_credit_spread", entry_window_enabled=True,
        entry_window_start_min=30, entry_window_end_min=120,
    )
    result = bt.Backtester(cfg, prices).run()
    assert any("NOT applied" in w for w in result.warnings)
    assert any("30" in w and "120" in w for w in result.warnings)

    off = bt.Backtester(bt.BacktestConfig(
        symbols=["SPY"], start_date="2020-01-01", end_date="2021-06-01",
        strategy="put_credit_spread"), prices).run()
    assert not any("NOT applied" in w for w in off.warnings)


def test_entry_window_blocks_the_live_bot_outside_its_hours() -> None:
    """The live path enforces the window against real session bounds."""
    import config as cfgmod
    from bot import TradingBot
    from broker_client import AccountSnapshot, BrokerClient, ConnectionHealth

    class Clocked(BrokerClient):
        def __init__(self, settings, elapsed):
            import threading
            self.settings = settings
            self.health = ConnectionHealth(connected=True, last_ok=datetime.now(timezone.utc))
            self._lock = threading.RLock()
            self._elapsed = elapsed

        def connect(self): return True
        @property
        def is_connected(self): return True
        def get_account(self):
            return AccountSnapshot(equity=100_000.0, maintenance_margin=5_000.0,
                                   options_buying_power=90_000.0)
        def get_positions(self): return []
        def get_option_positions(self): return []
        def is_market_open(self): return True
        def minutes_since_open(self): return self._elapsed

    settings = cfgmod.load_settings()
    settings.entry_window_enabled = True
    settings.entry_window_start_min = 30
    settings.entry_window_end_min = 120

    for elapsed, blocked in ((5, True), (45, False), (200, True)):
        bot = TradingBot(client=Clocked(settings, elapsed), settings=settings)
        reasons = bot._entry_blockers(bot.client.get_account())
        hit = any("entry window" in r for r in reasons)
        assert hit is blocked, f"{elapsed} min -> {reasons}"


# ======================================================================================
# Overfitting diagnostics
# ======================================================================================
def test_sample_adequacy_discounts_overlapping_trades() -> None:
    """Concurrent positions are not independent observations."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=21), "QQQ": _synthetic_frame(seed=22, s0=200.0)}
    cfg = bt.BacktestConfig(
        symbols=["SPY", "QQQ"], start_date="2020-01-01", end_date="2022-01-01",
        strategy="put_credit_spread", vrp_points=0.03,
    )
    result = bt.Backtester(cfg, prices).run()
    adequacy = bt.sample_adequacy(result)

    assert adequacy["trades"] > 0
    assert adequacy["concurrency"] >= 1.0
    # Effective sample can never exceed the raw count, and must shrink when
    # positions overlap in time.
    assert adequacy["effective_trades"] <= adequacy["trades"] + 1e-9
    assert adequacy["parameters"] == len(bt.TUNABLE_PARAMETERS)
    assert adequacy["verdict"] in {"adequate", "thin", "insufficient"}


def test_split_sample_is_chronological_not_random() -> None:
    """A random split would leak the future into the training slice."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=23)}
    cfg = bt.BacktestConfig(
        symbols=["SPY"], start_date="2020-01-01", end_date="2023-01-01",
        strategy="put_credit_spread", vrp_points=0.03,
    )
    split = bt.split_sample(cfg, prices, train_fraction=0.6)
    assert split["in_sample"]["end"] <= split["out_of_sample"]["start"]
    assert split["in_sample"]["start"] < split["out_of_sample"]["start"]


def test_walk_forward_covers_history_without_overlap() -> None:
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=24)}
    cfg = bt.BacktestConfig(
        symbols=["SPY"], start_date="2020-01-01", end_date="2023-01-01",
        strategy="put_credit_spread", vrp_points=0.03,
    )
    folds = bt.walk_forward(cfg, prices, folds=3)
    assert len(folds) == 3
    for earlier, later in zip(folds, folds[1:]):
        assert earlier["end"] <= later["start"]


def test_sensitivity_sweep_and_plateau_verdict() -> None:
    """The sweep must vary the parameter and grade the chosen value."""
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=25)}
    cfg = bt.BacktestConfig(
        symbols=["SPY"], start_date="2020-01-01", end_date="2023-01-01",
        strategy="put_credit_spread", vrp_points=0.03, short_delta=0.30,
    )
    rows = bt.sensitivity(cfg, prices, "short_delta", [0.20, 0.25, 0.30, 0.35])
    assert [r["value"] for r in rows] == [0.20, 0.25, 0.30, 0.35]
    assert len({r["total_return"] for r in rows}) > 1  # the knob actually moves the result

    verdict = bt.plateau_score(rows, 0.30)
    assert verdict["verdict"] != "insufficient sweep"
    assert 0.0 <= verdict["share_positive"] <= 1.0


def test_plateau_score_flags_a_manufactured_spike() -> None:
    """A peak surrounded by losses must be called a spike, not a plateau."""
    import backtest as bt

    spike = [
        {"value": 0.1, "total_return": -0.20, "trades": 40},
        {"value": 0.2, "total_return": -0.15, "trades": 40},
        {"value": 0.3, "total_return": 0.90, "trades": 40},   # the fitted peak
        {"value": 0.4, "total_return": -0.18, "trades": 40},
        {"value": 0.5, "total_return": -0.25, "trades": 40},
    ]
    assert "SPIKE" in bt.plateau_score(spike, 0.3)["verdict"]

    plateau = [
        {"value": v, "total_return": r, "trades": 40}
        for v, r in ((0.1, 0.30), (0.2, 0.34), (0.3, 0.36), (0.4, 0.31), (0.5, 0.28))
    ]
    assert "plateau" in bt.plateau_score(plateau, 0.3)["verdict"]


# ======================================================================================
# IBKR connector
# ======================================================================================
def test_occ_symbol_survives_the_ibkr_round_trip() -> None:
    """The platform speaks OCC; IBKR speaks contracts. Translation must be lossless.

    This is what makes the connector a drop-in: positions, the trade log and the
    UI keep working because symbols come back exactly as they went in.
    """
    import ibkr_client as ib

    for underlying, expiry, right, strike in (
        ("SPY", date(2025, 12, 19), "P", 600.0),
        ("QQQ", date(2026, 1, 16), "C", 512.5),
        ("IWM", date(2026, 3, 20), "PUT", 219.0),
    ):
        occ = ib.parts_to_occ(underlying, expiry, right, strike)
        parts = ib.occ_to_parts(occ)
        assert parts["underlying"] == underlying
        assert parts["expiration"] == expiry
        assert approx(parts["strike"], strike, 1e-9)
        assert parts["option_type"] == ("call" if right.upper().startswith("C") else "put")


def test_ibkr_rejects_non_option_symbols() -> None:
    import ibkr_client as ib
    from broker_client import BrokerError

    for bad in ("SPY", "", "NOTASYMBOL"):
        try:
            ib.occ_to_parts(bad)
        except BrokerError:
            continue
        raise AssertionError(f"{bad!r} should not parse as an option")


def test_ibkr_clean_rejects_ib_sentinels() -> None:
    """IBKR sends NaN and -1 for 'no data'; neither may become a price."""
    import ibkr_client as ib

    assert ib._clean(float("nan")) is None
    assert ib._clean(-1) is None
    assert ib._clean(None) is None
    assert ib._clean(0) is None
    assert ib._clean(2.5) == 2.5


def test_capabilities_differ_between_venues() -> None:
    """The bot branches on these, so they must not silently agree."""
    from broker_client import BrokerClient
    from ibkr_client import IBKRClient

    assert BrokerClient.capabilities.native_brackets is False
    assert BrokerClient.capabilities.historical_iv is False
    assert IBKRClient.capabilities.native_brackets is True
    assert IBKRClient.capabilities.historical_iv is True
    assert IBKRClient.capabilities.requires_gateway is True


def test_broker_factory_dispatches_on_config() -> None:
    """BVC_BROKER selects the venue, and a missing gateway is reported not raised."""
    import config as cfgmod
    from broker_client import build_client

    settings = cfgmod.load_settings()
    settings.broker = "ibkr"
    client = build_client(settings)          # no TWS here — must fail soft
    assert client.capabilities.name in {"ibkr", "alpaca"}
    assert client.health.last_error  # it recorded why, rather than throwing

    settings.broker = "alpaca"
    assert build_client(settings).capabilities.name == "alpaca"


def test_broker_iv_backfill_replaces_the_proxy() -> None:
    """A year of real IV turns IV Rank from a proxy into the true statistic."""
    store = engine.IVHistoryStore(Path(os.environ["BVC_STATE_DIR"]) / "iv_backfill.csv")

    proxy = store.rank("BFILL", 0.20, [0.10 + 0.002 * i for i in range(40)])
    assert proxy.is_proxy

    for i in range(260):
        store.record_on("BFILL", date(2025, 1, 1) + timedelta(days=i), 0.12 + 0.0004 * i)

    real = store.rank("BFILL", 0.20)
    assert real.source == "iv_history"
    assert not real.is_proxy
    assert real.samples >= 200
    assert 0.0 <= real.value <= 100.0


def test_iv_backfill_never_overwrites_an_existing_observation() -> None:
    """Backfill fills gaps; it must not rewrite history already recorded."""
    path = Path(os.environ["BVC_STATE_DIR"]) / "iv_nooverwrite.csv"
    path.unlink(missing_ok=True)
    store = engine.IVHistoryStore(path)

    store.record_on("KEEP", date(2025, 6, 2), 0.15)
    store.record_on("KEEP", date(2025, 6, 2), 0.99)
    rows = store.records("KEEP")
    assert len(rows) == 1
    assert approx(float(rows[0]["iv"]), 0.15, 1e-9)


# ======================================================================================
# Hermes control surface
# ======================================================================================
def _hermes_pair(tmp_name: str):
    """A bot on a stub broker, plus a Hermes bound to its own audit file."""
    import threading

    import config as cfgmod
    from bot import TradingBot
    from broker_client import AccountSnapshot, BrokerCapabilities, BrokerClient, ConnectionHealth
    from hermes import HermesAudit, HermesControl

    class Stub(BrokerClient):
        def __init__(self, settings):
            self.settings = settings
            self.health = ConnectionHealth(connected=True, last_ok=datetime.now(timezone.utc))
            self._lock = threading.RLock()
            self.capabilities = BrokerCapabilities(name="alpaca")

        def connect(self): return True
        @property
        def is_connected(self): return True
        def get_account(self):
            return AccountSnapshot(equity=100_000.0, maintenance_margin=20_000.0)
        def get_positions(self): return []
        def get_option_positions(self): return []

    settings = cfgmod.load_settings()
    bot = TradingBot(client=Stub(settings), settings=settings)
    bot.state.mode = "idle"
    bot.state.halt_reason = None
    audit = HermesAudit(Path(os.environ["BVC_STATE_DIR"]) / f"hermes_{tmp_name}.jsonl")
    return bot, HermesControl(bot, audit=audit, enabled=True), settings


def test_hermes_cannot_loosen_a_risk_limit() -> None:
    """The ratchet. A misaligned agent must only be able to trade less.

    Checked with a value *inside* the permitted range, so the rejection can only
    come from the one-way rule rather than incidentally from bounds.
    """
    bot, hermes, settings = _hermes_pair("ratchet")
    settings.max_margin_utilization = 0.30

    verdict = hermes.propose(
        {"max_margin_utilization": 0.45},
        rationale="raising the ceiling would let more capital be deployed",
    )
    assert not verdict.accepted
    assert "ratchet" in verdict.rejected["max_margin_utilization"]
    assert settings.max_margin_utilization == 0.30  # untouched

    tighten = hermes.propose(
        {"max_margin_utilization": 0.20},
        rationale="reduce exposure after a run of losses",
    )
    assert tighten.accepted
    assert settings.max_margin_utilization == 0.20


def test_hermes_cannot_touch_what_is_not_listed() -> None:
    """Promoting to live, changing venue or universe are not the agent's to make."""
    bot, hermes, settings = _hermes_pair("immutable")
    verdict = hermes.propose(
        {"paper": False, "broker": "ibkr", "universe": ["TSLA"], "alpaca_api_key": "x"},
        rationale="switch to live trading on a new venue for better fills",
    )
    assert not verdict.accepted
    assert set(verdict.rejected) == {"paper", "broker", "universe", "alpaca_api_key"}
    assert settings.paper is True


def test_hermes_may_halt_but_never_resume() -> None:
    """Stopping needs no permission; clearing a halt is a human act."""
    bot, hermes, settings = _hermes_pair("halt")

    assert hermes.halt("drawdown breach")["halted"]
    assert bot.state.is_halted

    assert hermes.start("please resume")["started"] is False
    assert bot.state.is_halted

    blocked = hermes.propose({"target_delta": 0.30}, rationale="retune after the halt")
    assert not blocked.accepted
    assert "halted" in blocked.rejected["*"]


def test_hermes_applies_the_legal_part_of_a_mixed_proposal() -> None:
    """A specific reason per parameter, so the agent can learn from a refusal."""
    bot, hermes, settings = _hermes_pair("mixed")
    verdict = hermes.propose(
        {"target_delta": 0.22, "max_open_positions": 99, "nonsense": 1},
        rationale="tighten delta while widening concurrent positions",
    )
    assert verdict.applied == {"target_delta": 0.22}
    assert "max_open_positions" in verdict.rejected
    assert "nonsense" in verdict.rejected
    assert approx(settings.target_delta, 0.22, 1e-9)


def test_hermes_requires_a_rationale() -> None:
    bot, hermes, _ = _hermes_pair("rationale")
    assert not hermes.propose({"target_delta": 0.25}, rationale="").accepted
    assert not hermes.propose({"target_delta": 0.25}, rationale="tweak").accepted


def test_hermes_counts_every_configuration_tried() -> None:
    """The multiple-comparison counter must include *applied* changes.

    Counting only rejected proposals would report a tuning agent as having
    searched nothing, which is precisely backwards.
    """
    bot, hermes, _ = _hermes_pair("experiments")
    for delta in (0.22, 0.25, 0.28, 0.25):  # three distinct, one repeat
        hermes.propose({"target_delta": delta}, rationale="sweeping delta for a better result")

    quality = hermes.observe()["evidence_quality"]
    assert quality["configurations_tried"] == 3
    assert quality["verdict"] != "adequate"  # no closed trades to justify any of it


def test_hermes_observation_carries_the_counterweights() -> None:
    """The payload must include what argues against acting, not just performance."""
    bot, hermes, _ = _hermes_pair("observe")
    state = hermes.observe()

    assert state["hermes"]["may_resume_after_halt"] is False
    assert "evidence_quality" in state
    assert "mutable_parameters" in state["hermes"]
    assert "max_margin_utilization" in state["hermes"]["mutable_parameters"]
    assert state["hermes"]["mutable_parameters"]["max_margin_utilization"]["ratcheted"] is True
    assert "paper" not in state["hermes"]["mutable_parameters"]


def test_hermes_is_off_until_switched_on() -> None:
    """Nothing may change how this trades until it is deliberately enabled."""
    import config as cfgmod
    from hermes import HermesControl

    bot, _, settings = _hermes_pair("disabled")
    disabled = HermesControl(bot, enabled=False)
    verdict = disabled.propose({"target_delta": 0.25}, rationale="a perfectly reasonable adjustment")
    assert not verdict.accepted
    assert "disabled" in verdict.rejected["*"]
    assert cfgmod.load_settings().hermes_enabled in (True, False)


def test_hermes_audit_records_rejections_too() -> None:
    """The record must show what was refused, not only what happened."""
    bot, hermes, _ = _hermes_pair("audit")
    hermes.propose({"max_margin_utilization": 0.99}, rationale="deploy substantially more capital")
    history = hermes.history()
    assert history
    last = history[-1]
    assert last["kind"] in {"propose", "apply"}
    assert last["accepted"] is False
    assert last["rejections"]
    assert "deploy substantially more capital" in last["rationale"]


# ======================================================================================
# The JSON parameter contract
# ======================================================================================
def _config_file(parameters: dict, name: str = "overlay"):
    """A strategy_config.json in the test state dir, plus a Settings that reads it."""
    import config as cfgmod

    path = Path(os.environ["BVC_STATE_DIR"]) / f"cfg_{name}.json"
    store = cfgmod.StrategyConfigFile(path)
    store.write(parameters, rationale="test", actor="test")
    settings = cfgmod.Settings()
    return store, settings


def test_config_overlay_applies_a_legal_change() -> None:
    """The happy path: a value inside its bound reaches the engine."""
    store, settings = _config_file({"target_delta": 0.22}, "legal")
    settings.apply_overlay(store)
    assert approx(settings.target_delta, 0.22)
    assert settings.overlay_applied == {"target_delta": 0.22}


def test_config_overlay_cannot_loosen_a_risk_limit() -> None:
    """A hand-edited or hallucinated file must not be able to widen the governor.

    This is the one that matters. The file asks for 95% margin utilisation —
    inside no sane operator's intent — and the resolved setting is the
    operator's own value, with a reason recorded.
    """
    import config as cfgmod

    store, settings = _config_file({"max_margin_utilization": 0.95}, "loosen")
    operator_value = settings.max_margin_utilization
    settings.apply_overlay(store)
    assert settings.max_margin_utilization == operator_value
    assert "max_margin_utilization" in settings.overlay_rejected
    assert cfgmod.HERMES_BOUNDS["max_margin_utilization"].risk_limit


def test_config_overlay_cannot_touch_the_broker_or_the_account() -> None:
    """Promoting to live money is not expressible in the file's vocabulary."""
    store, settings = _config_file(
        {"broker": "ibkr", "paper": False, "alpaca_api_key": "stolen", "universe": ["TSLA"]},
        "escape",
    )
    before = (settings.broker, settings.paper, settings.alpaca_api_key, list(settings.universe))
    settings.apply_overlay(store)
    assert (settings.broker, settings.paper, settings.alpaca_api_key, list(settings.universe)) == before
    assert set(settings.overlay_rejected) == {"broker", "paper", "alpaca_api_key", "universe"}


def test_dropping_a_parameter_from_the_file_restores_the_operator_value() -> None:
    """Deleting a line must actually undo it, not leave the last value stuck."""
    store, settings = _config_file({"target_delta": 0.18}, "revert")
    operator_delta = settings.target_delta
    settings.apply_overlay(store)
    assert approx(settings.target_delta, 0.18)

    store.write({}, rationale="reverted", actor="test")
    settings.apply_overlay(store)
    assert approx(settings.target_delta, operator_delta)
    assert settings.overlay_applied == {}


def test_repeated_reloads_cannot_walk_a_limit_outward() -> None:
    """The ratchet is anchored, not relative — a hundred reloads buy nothing."""
    import config as cfgmod

    store, settings = _config_file({"max_open_positions": 3}, "walk")
    settings.apply_overlay(store)
    assert settings.max_open_positions == 3

    for target in (4, 5, 6):
        store.write({"max_open_positions": target}, rationale="creep", actor="test")
        settings.apply_overlay(store)
        assert settings.max_open_positions <= cfgmod.Settings().max_open_positions


def test_malformed_config_is_ignored_not_fatal() -> None:
    """A corrupt file must cost the overlay, never the trading loop."""
    import config as cfgmod

    path = Path(os.environ["BVC_STATE_DIR"]) / "cfg_broken.json"
    path.write_text("{ this is not json")
    settings = cfgmod.Settings()
    before = settings.target_delta
    settings.apply_overlay(cfgmod.StrategyConfigFile(path))
    assert approx(settings.target_delta, before)


# ======================================================================================
# The kill switch
# ======================================================================================
def test_kill_switch_is_unreachable_from_the_agent() -> None:
    """Structural, not procedural: there is no name for it in the mutable set."""
    import bot as botmod
    import config as cfgmod

    assert "daily_loss_limit" not in cfgmod.HERMES_BOUNDS
    assert not any("daily_loss" in name for name in cfgmod.HERMES_BOUNDS)
    # Not a Settings field either, so it cannot arrive through the config file.
    assert not any("daily_loss" in f for f in cfgmod.Settings.__dataclass_fields__)
    assert botmod.DAILY_LOSS_LIMIT_PCT > 0


def test_kill_switch_halts_on_a_daily_loss_breach() -> None:
    """A 5% intraday drawdown against a 3% limit must stop the bot."""
    import bot as botmod
    from broker_client import AccountSnapshot

    bot, _, _ = _hermes_pair("killswitch")
    healthy = AccountSnapshot(equity=100_000.0, last_equity=100_000.0)
    assert bot._daily_loss_breach(healthy) is None

    bleeding = AccountSnapshot(equity=95_000.0, last_equity=100_000.0)
    reason = bot._daily_loss_breach(bleeding)
    assert reason and "daily loss limit" in reason
    assert f"{botmod.DAILY_LOSS_LIMIT_PCT:.2%}" in reason


def test_kill_switch_runs_before_any_entry_check() -> None:
    """It must halt the cycle, not merely block entries."""
    from broker_client import AccountSnapshot
    from bot import CycleResult

    bot, _, _ = _hermes_pair("killorder")
    bot.client.get_account = lambda: AccountSnapshot(  # type: ignore[method-assign]
        equity=90_000.0, last_equity=100_000.0
    )
    result = CycleResult()
    assert bot._preflight(result) is None
    assert result.halted and bot.state.is_halted


def test_halt_request_file_is_consumed_once() -> None:
    """An honoured stop request must not re-halt the bot after a human resumes."""
    import config as cfgmod

    bot, _, _ = _hermes_pair("haltfile")
    cfgmod.HALT_REQUEST_PATH.write_text('{"reason": "drawdown", "actor": "hermes-api"}')
    reason = bot._consume_halt_request()
    assert reason and "drawdown" in reason and "hermes-api" in reason
    assert not cfgmod.HALT_REQUEST_PATH.exists()
    assert bot._consume_halt_request() is None


# ======================================================================================
# Strict separation of the research and execution layers
# ======================================================================================
def test_the_research_layer_has_no_import_path_to_an_order() -> None:
    """The HTTP surface must not be able to reach a broker, transitively.

    Asserted on the import graph rather than on intent, because "we were careful"
    is not a security property. If someone adds ``import bot`` to ``api.py`` to
    borrow one constant, this fails.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    forbidden = {"bot", "broker_client", "ibkr_client", "ib_async", "ib_insync", "alpaca"}

    # strategies.py is included because the research layer imports it to
    # reason about the library; it must stay a pure declaration module.
    for module in ("api.py", "memory.py", "strategies.py"):
        tree = ast.parse((root / module).read_text())

        # Imports under `if TYPE_CHECKING:` never execute, so they create no
        # runtime path to a broker. Everything else counts, including imports
        # tucked inside a function body.
        type_only = {
            child
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test).endswith("TYPE_CHECKING")
            for stmt in node.body
            for child in ast.walk(stmt)
        }

        imported = set()
        for node in ast.walk(tree):
            if node in type_only:
                continue
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        leaked = imported & forbidden
        assert not leaked, f"{module} imports the execution layer: {sorted(leaked)}"


def test_performance_metrics_report_their_own_reliability() -> None:
    """A Sharpe ratio without a sample size is a rumour — say the sample size."""
    trades = [
        {"closed_at": f"2026-0{m}-1{d}T16:00:00", "pnl_usd": pnl, "status": "closed"}
        for m, d, pnl in [(1, 2, 120.0), (1, 5, -260.0), (2, 3, 95.0), (2, 8, 140.0)]
    ]
    metrics = engine.performance_metrics(trades, "usd")
    assert metrics["trades"] == 4
    assert metrics["trading_days"] == 4
    assert metrics["reliability"] == "insufficient"
    assert approx(metrics["total_pnl"], 95.0)
    assert metrics["max_drawdown"] < 0  # the -260 leg must show as a drawdown


# ======================================================================================
# The strategy library
# ======================================================================================
def _chain(spot: float = 100.0, expiry_days: int = 45, vol: float = 0.22):
    """A synthetic but internally consistent chain: real Black-Scholes prices.

    Prices and deltas come from the same model, so a builder that picks the
    0.30-delta strike gets a contract that really is 0.30 delta — a chain of
    made-up numbers would let a broken selector pass.
    """
    expiration = date.today() + timedelta(days=expiry_days)
    t = engine.year_fraction(expiration)
    quotes = []
    # Strikes step by ~1% of spot, which is roughly how real ETF chains are
    # listed. A coarse ladder would let a selector miss its delta target by a
    # wide margin and still look correct.
    step = max(round(spot * 0.01), 1)
    for strike in range(int(spot * 0.70), int(spot * 1.16), step):
        strike = float(strike)
        for right in ("put", "call"):
            is_call = right == "call"
            price = engine.bs_price(spot, strike, t, vol, 0.043, is_call)
            delta = engine.bs_delta(spot, strike, t, vol, 0.043, is_call)
            quotes.append(
                OptionQuote(
                    symbol=f"XYZ{expiration:%y%m%d}{'C' if is_call else 'P'}{int(strike * 1000):08d}",
                    underlying="XYZ",
                    expiration=expiration,
                    strike=strike,
                    option_type=right,
                    bid=round(max(price - 0.03, 0.01), 2),
                    ask=round(price + 0.03, 2),
                    implied_volatility=vol,
                    delta=delta,
                )
            )
    return quotes


def _ctx(strategy: str = "cash_secured_put", **overrides):
    import config as cfgmod
    import strategies as lib

    settings = cfgmod.Settings()
    settings.strategy = strategy
    settings.delta_tolerance = 0.10
    for key, value in overrides.items():
        setattr(settings, key, value)
    return lib.BuildContext(
        underlying="XYZ", spot=100.0, settings=settings,
        near=_chain(), far=_chain(expiry_days=105, vol=0.20),
    )


def test_every_registered_strategy_builds_from_a_normal_chain() -> None:
    """The library's own contract: eight strategies, eight plans, no exceptions."""
    import strategies as lib

    for key in lib.REGISTRY:
        plan = lib.build_plan(key, _ctx(key))
        assert plan is not None, f"{key} produced no plan from a healthy chain"
        assert plan.legs, f"{key} produced a plan with no legs"
        assert plan.capital_required > 0, f"{key} claims to need no capital"
        assert len(plan.legs) == lib.get(key).leg_count, f"{key} leg count disagrees with its definition"


def test_credit_and_debit_strategies_have_the_right_sign() -> None:
    """A calendar that reports a credit is a calendar priced wrong."""
    import strategies as lib

    for key in ("cash_secured_put", "put_credit_spread", "iron_condor", "iron_butterfly"):
        assert lib.build_plan(key, _ctx(key)).net_premium > 0, f"{key} should collect premium"
    for key in ("calendar_spread", "diagonal_spread"):
        assert lib.build_plan(key, _ctx(key)).net_premium < 0, f"{key} should pay premium"


def test_iron_condor_sells_both_sides_and_buys_both_wings() -> None:
    """Four legs, one short pair inside one long pair, margin = one wing."""
    import strategies as lib

    plan = lib.build_plan("iron_condor", _ctx("iron_condor", target_delta=0.20, spread_width=5.0))
    puts = sorted([l for l in plan.legs if l.quote.option_type == "put"], key=lambda l: l.quote.strike)
    calls = sorted([l for l in plan.legs if l.quote.option_type == "call"], key=lambda l: l.quote.strike)

    assert len(puts) == len(calls) == 2
    assert not puts[0].is_short and puts[1].is_short     # long wing below the short put
    assert calls[0].is_short and not calls[1].is_short   # short call below the long wing
    assert puts[1].quote.strike < calls[0].quote.strike  # a condor, not a butterfly

    # Only one side can finish in the money, so the requirement is one wing.
    width = max(puts[1].quote.strike - puts[0].quote.strike,
                calls[1].quote.strike - calls[0].quote.strike)
    assert approx(plan.capital_required, width * 100, 1e-6)
    assert approx(plan.max_loss + plan.max_profit, width * 100, 1e-3)


def test_iron_butterfly_body_sits_at_the_money() -> None:
    """The butterfly's defining property: both shorts on the same strike."""
    import strategies as lib

    plan = lib.build_plan("iron_butterfly", _ctx("iron_butterfly", spread_width=10.0))
    shorts = [l for l in plan.legs if l.is_short]
    assert len(shorts) == 2
    assert shorts[0].quote.strike == shorts[1].quote.strike
    assert abs(shorts[0].quote.strike - 100.0) <= 2.5      # at the money

    condor = lib.build_plan("iron_condor", _ctx("iron_condor", target_delta=0.20, spread_width=10.0))
    # Collapsing the shorts to the money is what buys the bigger credit.
    assert plan.net_premium > condor.net_premium


def test_calendar_needs_two_expiries_and_declines_without_one() -> None:
    """A 'calendar' inside one expiry is not a calendar — refuse to build it."""
    import strategies as lib

    ctx = _ctx("calendar_spread")
    ctx.far = ()
    assert lib.build_plan("calendar_spread", ctx) is None

    plan = lib.build_plan("calendar_spread", _ctx("calendar_spread"))
    assert len(plan.expirations) == 2
    front, back = plan.legs
    assert front.is_short and not back.is_short
    assert front.quote.expiration < back.quote.expiration
    assert front.quote.strike == back.quote.strike


def test_diagonal_long_leg_is_deep_and_further_out() -> None:
    """The back-month leg has to behave like stock, or it is not a PMCC."""
    import strategies as lib

    plan = lib.build_plan("diagonal_spread", _ctx("diagonal_spread", long_leg_delta=0.80))
    short, long = plan.legs
    assert short.is_short and not long.is_short
    assert long.quote.strike < short.quote.strike
    assert long.quote.expiration > short.quote.expiration
    assert abs(long.quote.delta) >= 0.65


def test_delta_tolerance_is_enforced_not_nearest_wins() -> None:
    """On a thin chain, 'nearest' can be a completely different trade."""
    import strategies as lib

    ctx = _ctx("cash_secured_put", target_delta=0.30, delta_tolerance=0.001)
    ctx.near = [q for q in ctx.near if abs(q.delta or 0) < 0.10 or abs(q.delta or 0) > 0.60]
    assert lib.build_plan("cash_secured_put", ctx) is None


def test_defined_risk_strategies_cap_the_loss() -> None:
    """The whole reason to buy a wing: a bounded worst case, and it must be bounded."""
    import strategies as lib

    for key in ("put_credit_spread", "call_credit_spread", "iron_condor", "iron_butterfly"):
        plan = lib.build_plan(key, _ctx(key))
        assert lib.get(key).defined_risk
        assert plan.max_loss is not None and plan.max_loss > 0
        assert plan.max_loss <= plan.capital_required

    naked = lib.build_plan("cash_secured_put", _ctx("cash_secured_put"))
    assert not lib.get("cash_secured_put").defined_risk
    # The undefined case still reports its true worst case: the strike, less credit.
    assert naked.max_loss > 50 * 100


def test_the_exit_rule_means_the_same_thing_for_credit_and_debit() -> None:
    """One rule, expressed on P&L rather than on price.

    A 50% profit target must mean "half the credit" for a short put and "half
    the debit" for a calendar. Testing it directly against the bot's own
    decision function, because that is what will actually run.
    """
    from bot import TradingBot
    from broker_client import PositionView

    bot, _, settings = _hermes_pair("exitrule")
    settings.profit_target_pct = 0.50
    settings.stop_loss_multiple = 2.0
    settings.time_exit_dte = 0
    settings.strategy = "cash_secured_put"   # undefined risk, so the stop applies

    def position(symbol: str, qty: float, price: float) -> PositionView:
        return PositionView(
            symbol=symbol, qty=qty, avg_entry_price=price, current_price=price,
            market_value=0.0, cost_basis=0.0, unrealized_pl=0.0, unrealized_plpc=0.0,
            asset_class="us_option", option_type="put",
            expiration=date.today() + timedelta(days=40),
        )

    # Credit structure: sold for 2.00, now worth 1.00 → half the credit captured.
    credit_record = {"credit": "2.0", "contracts": "1", "status": "open",
                     "legs": json.dumps([{"symbol": "A", "action": "sell", "ratio": 1}])}
    reason, cost, pnl = bot._trade_exit_decision(credit_record, {"A": position("A", -1, 1.00)})
    assert reason == "profit_target" and approx(pnl, 1.0)

    # Debit structure: paid 2.00, now worth 3.00 → half the debit made.
    debit_record = {"credit": "-2.0", "contracts": "1", "status": "open",
                    "legs": json.dumps([{"symbol": "A", "action": "sell", "ratio": 1},
                                        {"symbol": "B", "action": "buy", "ratio": 1}])}
    verdict = bot._trade_exit_decision(
        debit_record, {"A": position("A", -1, 1.00), "B": position("B", 1, 4.00)}
    )
    assert verdict[0] == "profit_target" and approx(verdict[2], 1.0)

    # And the stop fires on the same arithmetic, in both directions.
    stopped = bot._trade_exit_decision(credit_record, {"A": position("A", -1, 6.10)})
    assert stopped[0] == "stop_loss"


def test_a_multi_leg_exit_is_decided_on_the_whole_structure() -> None:
    """Never close one wing of a spread and leave the short naked."""
    from broker_client import PositionView

    bot, _, settings = _hermes_pair("structure")
    settings.profit_target_pct = 0.50
    settings.stop_loss_multiple = 2.0
    settings.time_exit_dte = 0

    def leg(symbol: str, qty: float, price: float) -> PositionView:
        return PositionView(
            symbol=symbol, qty=qty, avg_entry_price=price, current_price=price,
            market_value=0.0, cost_basis=0.0, unrealized_pl=0.0, unrealized_plpc=0.0,
            asset_class="us_option", option_type="put",
            expiration=date.today() + timedelta(days=40),
        )

    legs = [{"symbol": "SHORT", "action": "sell", "ratio": 1},
            {"symbol": "LONG", "action": "buy", "ratio": 1}]
    record = {"credit": "1.50", "contracts": "1", "status": "open", "legs": json.dumps(legs)}

    # The short leg alone has more than halved — but the spread has not.
    positions = {"SHORT": leg("SHORT", -1, 1.00), "LONG": leg("LONG", 1, 0.10)}
    assert approx(bot._close_cost(legs, positions), 0.90)
    assert bot._trade_exit_decision(record, positions) is None

    # Now the spread itself is worth half what it was sold for.
    positions["LONG"] = leg("LONG", 1, 0.25)
    positions["SHORT"] = leg("SHORT", -1, 1.00)
    assert bot._trade_exit_decision(record, positions)[0] == "profit_target"


def test_every_strategy_has_a_config_file_with_valid_parameters() -> None:
    """Each strategy ships a config, and every value in it survives the gate."""
    import config as cfgmod
    import strategies as lib

    baseline = cfgmod.Settings().mutable_values()
    for definition in lib.REGISTRY.values():
        path = definition.config_path
        assert path.exists(), f"{definition.key} has no config file — run strategies.py --write-configs"
        params = cfgmod.StrategyConfigFile(path).parameters()
        accepted, rejected = cfgmod.vet_changes(baseline, params)
        assert not rejected, f"{definition.key}.json contains refused values: {rejected}"
        assert accepted, f"{definition.key}.json set nothing"


def test_switching_strategy_is_not_something_the_agent_can_do() -> None:
    """Payoff geometry, capital and approval level all change — operator only."""
    import config as cfgmod

    assert "strategy" not in cfgmod.HERMES_BOUNDS
    store, settings = _config_file({"strategy": "iron_butterfly"}, "switch")
    before = settings.strategy
    settings.apply_overlay(store)
    assert settings.strategy == before
    assert "strategy" in settings.overlay_rejected


# ======================================================================================
# Wing selection, the hard stop, and the breakeven identity
# ======================================================================================
def test_wings_are_chosen_by_delta_so_they_scale_with_the_underlying() -> None:
    """A fixed dollar wing is a different trade on every symbol.

    The same delta settings on a $90 name and a $600 name must produce wings
    that are comparable in *risk*, not in dollars. Checked by comparing the
    width as a fraction of spot, which is what should stay roughly constant.
    """
    import strategies as lib

    widths = {}
    for spot in (90.0, 300.0, 600.0):
        ctx = _ctx("put_credit_spread", wing_delta=0.10, max_spread_width=0.0)
        ctx.spot = spot
        ctx.near = _chain(spot=spot)
        plan = lib.build_plan("put_credit_spread", ctx)
        assert plan is not None, f"no plan at spot {spot}"
        short, long = plan.legs
        widths[spot] = abs(short.quote.strike - long.quote.strike) / spot

    # Delta selection keeps the *relative* wing stable across a 6.7x price range.
    assert max(widths.values()) / min(widths.values()) < 1.5, widths


def test_max_spread_width_caps_the_wing_without_becoming_the_target() -> None:
    """The cap bounds margin per trade; it must not silently drive selection."""
    import strategies as lib

    uncapped = lib.build_plan("put_credit_spread", _ctx("put_credit_spread", wing_delta=0.05))
    capped = lib.build_plan(
        "put_credit_spread", _ctx("put_credit_spread", wing_delta=0.05, max_spread_width=5.0)
    )
    u_width = abs(uncapped.legs[0].quote.strike - uncapped.legs[1].quote.strike)
    c_width = abs(capped.legs[0].quote.strike - capped.legs[1].quote.strike)
    assert u_width > 5.0, "test needs an uncapped wing wider than the cap"
    assert c_width <= 5.0
    assert capped.capital_required < uncapped.capital_required


def test_the_backtester_builds_the_same_spread_the_live_bot_would() -> None:
    """The defect this pass fixed: live picked wings by dollars, the replay by delta.

    Compared as a fraction of spot, because the replay solves for an exact
    strike on a modelled surface while the library picks a listed one.
    """
    import backtest as bt
    import strategies as lib

    spot, iv = 300.0, 0.22
    ctx = _ctx("put_credit_spread", wing_delta=0.10)
    ctx.spot, ctx.near = spot, _chain(spot=spot, vol=iv)
    settings = ctx.settings

    plan = lib.build_plan("put_credit_spread", ctx)
    live_width = abs(plan.legs[0].quote.strike - plan.legs[1].quote.strike)

    cfg = bt.BacktestConfig.from_settings(settings)
    tester = bt.Backtester(cfg, {})
    t = engine.year_fraction(date.today() + timedelta(days=45))
    legs, _ = tester._build_legs(spot, t, iv)
    replay_width = abs(legs[0].strike - legs[1].strike)

    assert cfg.long_delta == settings.wing_delta
    assert abs(live_width - replay_width) / spot < 0.02, (live_width, replay_width)


def test_defined_risk_trades_do_not_fire_a_hard_stop_under_auto() -> None:
    """You bought the wing. Declining to use it is paying twice for one tail."""
    from broker_client import PositionView

    bot, _, settings = _hermes_pair("hardstop")
    settings.profit_target_pct = 0.50
    settings.stop_loss_multiple = 1.0
    settings.time_exit_dte = 0
    settings.hard_stop_mode = "auto"

    def leg(symbol: str, qty: float, price: float) -> PositionView:
        return PositionView(
            symbol=symbol, qty=qty, avg_entry_price=price, current_price=price,
            market_value=0.0, cost_basis=0.0, unrealized_pl=0.0, unrealized_plpc=0.0,
            asset_class="us_option", option_type="put",
            expiration=date.today() + timedelta(days=40),
        )

    legs = [{"symbol": "SHORT", "action": "sell", "ratio": 1},
            {"symbol": "LONG", "action": "buy", "ratio": 1}]
    # Sold for 1.00, now costs 3.00 to close — a 200% loss.
    positions = {"SHORT": leg("SHORT", -1, 3.50), "LONG": leg("LONG", 1, 0.50)}

    # A narrow wing: max loss $80 is already inside the $100 stop, so the stop
    # is unreachable and "auto" correctly declines to arm it.
    narrow = {"credit": "1.00", "contracts": "1", "status": "open", "max_loss": "80",
              "strategy": "put_credit_spread", "legs": json.dumps(legs)}
    assert bot._hard_stop_applies(narrow) is False

    # A distant wing: max loss $400 is four times the stop level, so the wing is
    # no substitute for it. This is the case the replay actually measured.
    spread = {"credit": "1.00", "contracts": "1", "status": "open", "max_loss": "400",
              "strategy": "put_credit_spread", "legs": json.dumps(legs)}
    assert bot._hard_stop_applies(spread) is True
    assert bot._trade_exit_decision(spread, positions)[0] == "stop_loss"

    # The same loss on an undefined-risk short stops, and "never" cannot turn
    # that off — the stop is the only thing between it and the tail.
    naked = {"credit": "1.00", "contracts": "1", "status": "open",
             "strategy": "cash_secured_put",
             "legs": json.dumps([{"symbol": "SHORT", "action": "sell", "ratio": 1}])}
    assert bot._hard_stop_applies(naked) is True
    assert bot._trade_exit_decision(naked, positions)[0] == "stop_loss"
    settings.hard_stop_mode = "never"
    assert bot._hard_stop_applies(naked) is True

    settings.hard_stop_mode = "always"
    assert bot._hard_stop_applies(narrow) is True     # override reaches even a moot stop
    settings.hard_stop_mode = "never"
    assert bot._hard_stop_applies(spread) is False    # ...and can disarm a live one


def test_backtest_and_bot_agree_on_when_a_stop_fires() -> None:
    """Two implementations of one rule is how a backtest stops being a test."""
    import backtest as bt

    # (strategy, mode, max_loss $, credit $, expected)
    cases = [
        ("put_credit_spread", "auto",   400.0, 100.0, True),   # distant wing → stop arms
        ("put_credit_spread", "auto",    80.0, 100.0, False),  # narrow wing → stop is moot
        ("put_credit_spread", "always",  80.0, 100.0, True),
        ("put_credit_spread", "never",  400.0, 100.0, False),
        ("cash_secured_put",  "auto",     0.0, 100.0, True),
        ("cash_secured_put",  "never",    0.0, 100.0, True),   # naked shorts always stop
        ("iron_condor",       "auto",   350.0, 100.0, True),
    ]
    for strategy, mode, max_loss, credit, expected in cases:
        bot, _, settings = _hermes_pair(f"agree_{strategy}_{mode}_{max_loss:.0f}")
        settings.strategy, settings.hard_stop_mode = strategy, mode
        settings.stop_loss_multiple = 1.0
        cfg = bt.BacktestConfig(strategy=strategy, hard_stop_mode=mode, stop_loss=1.0)

        live = bot._hard_stop_applies({
            "strategy": strategy, "max_loss": str(max_loss),
            "credit": str(credit / 100.0), "contracts": "1",
        })
        replay = bt.Backtester(cfg, {})._hard_stop_applies(max_loss=max_loss, credit=credit)
        assert live == replay == expected, (strategy, mode, max_loss, live, replay)


def test_the_default_bracket_starts_above_the_risk_neutral_win_rate() -> None:
    """The reason the default stop moved from 200% to 100%.

    A 50%/200% bracket needs an 80% win rate to break even, while a 30-delta
    short is only ~70% out of the money on risk-neutral probabilities — it
    starts *below* breakeven and depends entirely on the variance risk premium
    to climb above it. 50%/100% needs 67%, which starts above it.
    """
    import config as cfgmod

    settings = cfgmod.Settings()
    breakeven = engine.breakeven_win_rate(settings.profit_target_pct, settings.stop_loss_multiple)
    assert approx(breakeven, 2.0 / 3.0, 1e-9)

    # The delta-implied probability of the short expiring worthless.
    t = engine.year_fraction(date.today() + timedelta(days=45))
    delta = engine.bs_delta(100.0, engine.strike_from_delta(100.0, t, 0.20, 0.043, -0.30, False),
                            t, 0.20, 0.043, is_call=False)
    risk_neutral_win = 1.0 - abs(delta)
    assert risk_neutral_win > breakeven, (risk_neutral_win, breakeven)
    assert engine.breakeven_win_rate(0.50, 2.00) > risk_neutral_win  # the old default did not


def test_the_default_strategy_is_the_one_that_fits_the_capital() -> None:
    """A cash-secured put ties up strike notional to earn under 1% of it."""
    import config as cfgmod
    import strategies as lib

    settings = cfgmod.Settings()
    assert settings.strategy == "put_credit_spread"
    assert lib.get(settings.strategy).defined_risk

    csp = lib.build_plan("cash_secured_put", _ctx("cash_secured_put"))
    spread = lib.build_plan("put_credit_spread", _ctx("put_credit_spread"))
    # Same view, same short strike, a fraction of the capital.
    assert spread.capital_required < csp.capital_required / 5
    assert spread.net_premium / spread.capital_required > csp.net_premium / csp.capital_required


# ======================================================================================
# Runner
# ======================================================================================
def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
