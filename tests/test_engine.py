"""
Maths regression tests for the Brickvestcapitalterminal engine.

No network, no credentials, no broker. These cover the parts where a silent
error would be expensive: the Black-Scholes round-trip, the implied-volatility
inversion the free data feed forces us to rely on, the realised-vol estimators
that define the VRP, and the expectancy formula the whole platform reports on.

    python tests/test_engine.py     # or: pytest tests/
"""

from __future__ import annotations

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
    """
    import backtest as bt

    prices = {"SPY": _synthetic_frame(seed=7)}
    returns = []
    for vrp in (0.0, 0.03, 0.06):
        cfg = bt.BacktestConfig(
            symbols=["SPY"], start_date="2020-01-01", end_date="2022-01-01",
            strategy="put_credit_spread", vrp_points=vrp,
        )
        returns.append(bt.Backtester(cfg, prices).run().metrics["total_return"])
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
