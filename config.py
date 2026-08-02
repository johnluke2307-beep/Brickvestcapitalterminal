"""
Brickvestcapitalterminal — central configuration.

Every tunable lives here so the strategy can be audited in one place. Values are
resolved with the following precedence:

    1. Environment variables            (best for Hugging Face Spaces / Docker)
    2. ``st.secrets``                   (best for Streamlit Community Cloud)
    3. The defaults declared below      (safe, paper-trading oriented)

Nothing in this module imports Streamlit at module scope, so ``bot.py`` can be
run head-less (cron, a worker dyno, a terminal) without pulling in the UI stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List

# --------------------------------------------------------------------------------------
# Paths — everything the platform persists lives under ./state (git-ignored).
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("BVC_STATE_DIR", ROOT_DIR / "state"))
IV_HISTORY_PATH = STATE_DIR / "iv_history.csv"
TRADE_LOG_PATH = STATE_DIR / "trade_log.csv"
BOT_STATE_PATH = STATE_DIR / "bot_state.json"
FX_CACHE_PATH = STATE_DIR / "fx_cache.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Secret / setting resolution
# --------------------------------------------------------------------------------------
def _from_streamlit_secrets(key: str) -> str | None:
    """Read a key from ``st.secrets`` without hard-depending on Streamlit."""
    try:  # pragma: no cover - depends on runtime host
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
        # Allow a nested [alpaca] table as well, e.g. st.secrets["alpaca"]["API_KEY"].
        for section in ("alpaca", "brickvest"):
            if section in st.secrets and key in st.secrets[section]:
                return str(st.secrets[section][key])
    except Exception:
        return None
    return None


def setting(key: str, default: Any = None) -> Any:
    """Resolve a single setting from env → Streamlit secrets → default."""
    value = os.getenv(key)
    if value is None:
        value = _from_streamlit_secrets(key)
    return default if value is None or value == "" else value


def _float(key: str, default: float) -> float:
    try:
        return float(setting(key, default))
    except (TypeError, ValueError):
        return default


def _int(key: str, default: int) -> int:
    try:
        return int(float(setting(key, default)))
    except (TypeError, ValueError):
        return default


def _bool(key: str, default: bool) -> bool:
    raw = setting(key, None)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _list(key: str, default: List[str]) -> List[str]:
    raw = setting(key, None)
    if not raw:
        return list(default)
    return [item.strip().upper() for item in str(raw).replace(";", ",").split(",") if item.strip()]


# Liquid, optionable, high-volume underlyings. ETFs are preferred for mechanical
# premium selling: no earnings gaps, tighter spreads, index-like kurtosis.
DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "XLF", "EEM", "GLD", "TLT"]


@dataclass
class Settings:
    """Immutable snapshot of every runtime parameter."""

    # ---------------------------------------------------------------- credentials
    alpaca_api_key: str = field(default_factory=lambda: str(setting("ALPACA_API_KEY", "")))
    alpaca_secret_key: str = field(default_factory=lambda: str(setting("ALPACA_SECRET_KEY", "")))
    #: Paper trading is the default. Flip ALPACA_PAPER=false only with intent.
    paper: bool = field(default_factory=lambda: _bool("ALPACA_PAPER", True))

    # ---------------------------------------------------------------- market data
    #: Free plan → "indicative" options feed and "iex" equity feed. Paid → "opra"/"sip".
    options_feed: str = field(default_factory=lambda: str(setting("ALPACA_OPTIONS_FEED", "indicative")).lower())
    stock_feed: str = field(default_factory=lambda: str(setting("ALPACA_STOCK_FEED", "iex")).lower())

    # ---------------------------------------------------------------- strategy core
    universe: List[str] = field(default_factory=lambda: _list("BVC_UNIVERSE", DEFAULT_UNIVERSE))
    #: Mechanical entry tenor. The VRP is richest and gamma still tame around 45 DTE.
    target_dte: int = field(default_factory=lambda: _int("BVC_TARGET_DTE", 45))
    dte_min: int = field(default_factory=lambda: _int("BVC_DTE_MIN", 35))
    dte_max: int = field(default_factory=lambda: _int("BVC_DTE_MAX", 56))
    #: Short-strike delta target (0.30 ≈ 70% theoretical probability of profit).
    target_delta: float = field(default_factory=lambda: _float("BVC_TARGET_DELTA", 0.30))
    delta_tolerance: float = field(default_factory=lambda: _float("BVC_DELTA_TOLERANCE", 0.08))
    #: Only sell when volatility is objectively rich versus its own trailing range.
    min_iv_rank: float = field(default_factory=lambda: _float("BVC_MIN_IV_RANK", 50.0))
    #: Minimum IV − RV spread (annualised vol points) required to call it an edge.
    min_vrp: float = field(default_factory=lambda: _float("BVC_MIN_VRP", 0.02))
    #: "short_put" (cash-secured put, options level 2) or "put_credit_spread" (level 3).
    strategy: str = field(default_factory=lambda: str(setting("BVC_STRATEGY", "short_put")).lower())
    #: Width in strikes-dollars for the long wing of a credit spread.
    spread_width: float = field(default_factory=lambda: _float("BVC_SPREAD_WIDTH", 5.0))

    # ---------------------------------------------------------------- exit management
    #: Buy back at 50% of the credit received — the classic VRP profit taker.
    profit_target_pct: float = field(default_factory=lambda: _float("BVC_PROFIT_TARGET_PCT", 0.50))
    #: Stop when the open loss reaches 200% of the credit (i.e. price = 3× credit).
    stop_loss_multiple: float = field(default_factory=lambda: _float("BVC_STOP_LOSS_MULTIPLE", 2.00))
    #: Mechanical time stop: close anything still open inside 21 DTE (gamma risk).
    time_exit_dte: int = field(default_factory=lambda: _int("BVC_TIME_EXIT_DTE", 21))

    # ---------------------------------------------------------------- risk guardrails
    #: Hard ceiling on maintenance-margin utilisation. Fat-tail isolation.
    max_margin_utilization: float = field(default_factory=lambda: _float("BVC_MAX_MARGIN_UTIL", 0.50))
    #: Max simultaneous short-premium positions.
    max_open_positions: int = field(default_factory=lambda: _int("BVC_MAX_OPEN_POSITIONS", 6))
    #: Max new entries per calendar day — throttles correlated same-day risk.
    max_new_positions_per_day: int = field(default_factory=lambda: _int("BVC_MAX_NEW_PER_DAY", 2))
    #: One position per underlying at a time.
    one_position_per_underlying: bool = field(default_factory=lambda: _bool("BVC_ONE_PER_UNDERLYING", True))
    contracts_per_trade: int = field(default_factory=lambda: _int("BVC_CONTRACTS_PER_TRADE", 1))
    #: Reject illiquid contracts: minimum credit and maximum bid/ask spread ratio.
    min_credit_usd: float = field(default_factory=lambda: _float("BVC_MIN_CREDIT_USD", 0.35))
    max_spread_pct: float = field(default_factory=lambda: _float("BVC_MAX_SPREAD_PCT", 0.20))
    #: Refuse to open anything if account equity drops below this floor.
    min_equity_usd: float = field(default_factory=lambda: _float("BVC_MIN_EQUITY_USD", 2000.0))
    #: Restrict entries to a window measured in minutes after the opening bell.
    #: The first half hour is the widest-spread, least-representative part of the
    #: day; by two hours the morning volatility premium has largely decayed.
    entry_window_enabled: bool = field(default_factory=lambda: _bool("BVC_ENTRY_WINDOW", False))
    entry_window_start_min: int = field(default_factory=lambda: _int("BVC_ENTRY_WINDOW_START", 30))
    entry_window_end_min: int = field(default_factory=lambda: _int("BVC_ENTRY_WINDOW_END", 120))

    # ---------------------------------------------------------------- income target
    #: The whole point of the terminal: R10,000 of realised premium per month.
    monthly_target_zar: float = field(default_factory=lambda: _float("BVC_MONTHLY_TARGET_ZAR", 10_000.0))
    #: Fallback USD/ZAR used only when every live FX source is unreachable.
    fallback_usd_zar: float = field(default_factory=lambda: _float("BVC_FALLBACK_USD_ZAR", 18.50))

    # ---------------------------------------------------------------- engine plumbing
    #: Risk-free rate used for Black-Scholes greeks / IV inversion.
    risk_free_rate: float = field(default_factory=lambda: _float("BVC_RISK_FREE_RATE", 0.043))
    #: Realised-volatility lookback (trading days).
    rv_window: int = field(default_factory=lambda: _int("BVC_RV_WINDOW", 20))
    #: Trailing window for the IV Rank calculation (trading days).
    iv_rank_window: int = field(default_factory=lambda: _int("BVC_IV_RANK_WINDOW", 252))
    #: Below this many stored IV observations the rank falls back to an RV proxy.
    iv_rank_min_samples: int = field(default_factory=lambda: _int("BVC_IV_RANK_MIN_SAMPLES", 40))
    #: Seconds between automated bot cycles.
    loop_interval_seconds: int = field(default_factory=lambda: _int("BVC_LOOP_INTERVAL", 300))
    #: When true the bot scans, scores and logs but never sends an order.
    dry_run: bool = field(default_factory=lambda: _bool("BVC_DRY_RUN", False))

    # ---------------------------------------------------------------- derived helpers
    @property
    def credentials_present(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def stop_loss_price_multiple(self) -> float:
        """Contract price at which the stop fires, as a multiple of the credit.

        A "200% stop" means the *loss* equals 200% of the credit received, so the
        option must be bought back at ``credit × (1 + 2.0) = 3× credit``.
        """
        return 1.0 + self.stop_loss_multiple

    def as_dict(self) -> dict:
        """Redacted view, safe to render in the UI or dump to logs."""
        data = asdict(self)
        for secret in ("alpaca_api_key", "alpaca_secret_key"):
            value = data.get(secret) or ""
            data[secret] = f"{value[:4]}…{value[-2:]}" if len(value) > 8 else ("<set>" if value else "<missing>")
        return data


def load_settings() -> Settings:
    """Build a fresh :class:`Settings` snapshot (re-reads env and secrets)."""
    return Settings()


#: Import-time singleton for convenience; call :func:`load_settings` for a fresh read.
SETTINGS = load_settings()
