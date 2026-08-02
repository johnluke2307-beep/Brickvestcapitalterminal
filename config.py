"""
Brickvestcapitalterminal — central configuration.

Every tunable lives here so the strategy can be audited in one place. Values are
resolved with the following precedence:

    1. ``strategies/<strategy>.json``   (the agent-writable overlay — see below)
    2. Environment variables            (best for Hugging Face Spaces / Docker)
    3. ``st.secrets``                   (best for Streamlit Community Cloud)
    4. The defaults declared below      (safe, paper-trading oriented)

Nothing in this module imports Streamlit at module scope, so ``bot.py`` can be
run head-less (cron, a worker dyno, a terminal) without pulling in the UI stack.

The agent-writable overlay
--------------------------
The research layer (Hermes) never calls into the execution layer. It writes a
JSON file; the execution loop reads it. That file is the *entire* interface, and
this module is where it is policed:

There is one such file per strategy — ``strategies/iron_condor.json``,
``strategies/cash_secured_put.json`` and so on — and the active one is chosen by
``BVC_STRATEGY``, which is itself *not* agent-writable.

* Only keys in :data:`HERMES_BOUNDS` are read. A ``broker``, ``paper``,
  ``strategy`` or ``alpaca_api_key`` entry in the file is ignored, not applied —
  neither promoting to live money nor switching strategy is expressible in the
  agent's vocabulary.
* Every value must sit inside its declared bound.
* **Risk limits ratchet.** The overlay may only move a ``risk_limit`` parameter
  in the safer direction *relative to the operator's own env/default baseline*.
  A hand-edited file claiming ``max_margin_utilization: 0.95`` resolves to the
  operator's value, and says so in :attr:`Settings.overlay_notes`.

So the worst case of a corrupted, hostile or hallucinated config file is a
terminal that trades less than the operator allowed — never more.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("brickvest.config")

# --------------------------------------------------------------------------------------
# Paths — everything the platform persists lives under ./state (git-ignored).
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
STATE_DIR = Path(os.getenv("BVC_STATE_DIR", ROOT_DIR / "state"))
IV_HISTORY_PATH = STATE_DIR / "iv_history.csv"
TRADE_LOG_PATH = STATE_DIR / "trade_log.csv"
BOT_STATE_PATH = STATE_DIR / "bot_state.json"
FX_CACHE_PATH = STATE_DIR / "fx_cache.json"
#: Drop-box for an out-of-process stop request. Declared here, not in ``bot``,
#: so a process that must never import the broker layer can still write it.
HALT_REQUEST_PATH = STATE_DIR / "halt_request.json"

#: The Hermes ↔ engine contract: one config file per strategy, versioned in the
#: repo rather than hidden under ``state/``, because it is a reviewable artifact.
#: You should be able to read the diff of what the agent changed about your
#: strategy. Switching strategies switches a whole coherent parameter set — a
#: 20-delta condor and a 30-delta cash-secured put are not the same trade with a
#: different leg count, and must not share tuning.
STRATEGIES_DIR = Path(os.getenv("BVC_STRATEGIES_DIR", ROOT_DIR / "strategies"))
#: Fallback for a strategy with no file of its own, and for single-strategy
#: deployments that would rather keep one file at the root.
STRATEGY_CONFIG_PATH = Path(ROOT_DIR / "strategy_config.json")


def strategy_config_path(strategy: str) -> Path:
    """Where the active strategy's parameters live.

    ``BVC_STRATEGY_CONFIG`` wins when set (tests and single-file deployments).
    Otherwise the per-strategy file, falling back to a shared root file so an
    install that predates the strategy library keeps working untouched.
    """
    explicit = os.getenv("BVC_STRATEGY_CONFIG")
    if explicit:
        return Path(explicit)
    candidate = STRATEGIES_DIR / f"{strategy}.json"
    return candidate if candidate.exists() else STRATEGY_CONFIG_PATH


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


# --------------------------------------------------------------------------------------
# What an agent is allowed to touch
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Bound:
    """One tunable, its permitted range, and which direction counts as safer.

    ``safer`` says which way tightens risk. ``risk_limit=True`` means the value
    may only move in that direction — the ratchet that makes a misaligned agent
    trade less rather than more.
    """

    low: float
    high: float
    safer: str = "either"     # "lower" | "higher" | "either"
    risk_limit: bool = False
    note: str = ""

    def clamp_ok(self, value: float) -> bool:
        return self.low <= value <= self.high

    def loosens(self, proposed: float, baseline: float) -> bool:
        """True when moving ``baseline`` → ``proposed`` takes on more risk."""
        if not self.risk_limit or float(proposed) == float(baseline):
            return False
        if self.safer == "lower":
            return float(proposed) > float(baseline)
        if self.safer == "higher":
            return float(proposed) < float(baseline)
        return False


#: The complete set of parameters an agent can move. Anything absent is
#: immutable from the agent's side — including the universe, the broker,
#: credentials and the paper/live flag. Promoting to live is a human decision by
#: construction, because the agent has no word for it.
HERMES_BOUNDS: Dict[str, Bound] = {
    # ---- strategy shape: the agent may explore freely inside sane limits ----
    "target_delta": Bound(0.10, 0.45, "lower", note="lower delta = further OTM = safer"),
    "delta_tolerance": Bound(0.02, 0.15),
    "target_dte": Bound(21, 60),
    "min_iv_rank": Bound(0.0, 90.0, "higher", note="higher bar = fewer, richer entries"),
    "min_vrp": Bound(0.0, 0.15, "higher"),
    "profit_target_pct": Bound(0.20, 0.90),
    "stop_loss_multiple": Bound(1.0, 4.0, "lower", note="a tighter stop caps the tail"),
    "time_exit_dte": Bound(0, 30, "higher", note="exiting earlier reduces gamma risk"),
    "min_credit_usd": Bound(0.10, 5.00, "higher"),
    "max_spread_pct": Bound(0.02, 0.50, "lower"),
    # ---- multi-leg shape, used only by the strategies that have those legs --
    "wing_delta": Bound(0.03, 0.25, "higher", note="a closer wing caps the defined loss"),
    "max_spread_width": Bound(0.0, 100.0),
    "long_leg_delta": Bound(0.60, 0.95, "higher", note="deeper long leg tracks the stock more closely"),
    "back_month_dte": Bound(60, 240),

    # ---- risk limits: ratcheted. Tighten only. -----------------------------
    "max_margin_utilization": Bound(0.05, 0.50, "lower", risk_limit=True,
                                    note="the fat-tail governor — may only be reduced"),
    "max_open_positions": Bound(1, 6, "lower", risk_limit=True),
    "max_new_positions_per_day": Bound(1, 2, "lower", risk_limit=True),
    "contracts_per_trade": Bound(1, 10, "lower", risk_limit=True),
    "min_equity_usd": Bound(2000.0, 1_000_000.0, "higher", risk_limit=True),
}


def vet_changes(
    baseline: Dict[str, Any], changes: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Filter a proposed parameter dict down to what is actually permitted.

    The single gate every mutation path goes through — the JSON overlay, the
    in-process control surface and the HTTP API all call this, so there is one
    place to read to know what an agent can do to this account.

    Returns ``(accepted, rejected)`` where ``rejected`` maps each refused key to
    a specific reason. Keys are judged independently on purpose: a proposal that
    mixes one legal and one illegal change applies the legal half and explains
    the rest, which is something an agent can learn from. A blanket refusal is
    not.
    """
    accepted: Dict[str, Any] = {}
    rejected: Dict[str, str] = {}

    for name, raw in changes.items():
        bound = HERMES_BOUNDS.get(name)
        if bound is None:
            rejected[name] = "not an agent-mutable parameter"
            continue
        if name not in baseline:
            rejected[name] = "no baseline value to compare against"
            continue
        current = baseline[name]
        try:
            value = type(current)(raw) if not isinstance(current, bool) else bool(raw)
            numeric = float(value)
        except (TypeError, ValueError):
            rejected[name] = f"{raw!r} is not a valid value"
            continue
        if not bound.clamp_ok(numeric):
            rejected[name] = f"outside the permitted range [{bound.low}, {bound.high}]"
            continue
        if bound.loosens(numeric, float(current)):
            rejected[name] = (
                f"risk limits ratchet one way: {name} may only move "
                f"{bound.safer} (baseline {current})"
            )
            continue
        accepted[name] = value
    return accepted, rejected


# --------------------------------------------------------------------------------------
# The strategy config file — the whole of the agent's write surface
# --------------------------------------------------------------------------------------
class StrategyConfigFile:
    """Read/write access to ``strategy_config.json``.

    Nothing here touches a broker, an account or an order. That is the point:
    the research layer's only reachable verb is "write a number into a file",
    and the execution layer decides for itself when and whether to read it.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or STRATEGY_CONFIG_PATH)

    def mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def read(self) -> dict:
        """The file as written, or an empty document when absent/corrupt."""
        try:
            payload = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            # A malformed config must not stop the bot; it must stop the
            # *overlay*. The engine falls back to the operator's own settings.
            logger.warning("strategy config unreadable, ignoring overlay: %s", exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def parameters(self) -> Dict[str, Any]:
        params = self.read().get("parameters")
        return dict(params) if isinstance(params, dict) else {}

    def write(
        self,
        parameters: Dict[str, Any],
        *,
        rationale: str = "",
        evidence: Dict[str, Any] | None = None,
        actor: str = "hermes",
    ) -> dict:
        """Persist a full parameter set with the provenance that produced it.

        The rationale and evidence are stored beside the numbers deliberately.
        A parameter file without the argument for its values is a set of magic
        constants, and six weeks later nobody can tell a validated change from
        a lucky one.
        """
        document = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "updated_by": actor,
            "rationale": rationale,
            "evidence": evidence or {},
            "parameters": parameters,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
        tmp.replace(self.path)  # atomic: the bot never reads a half-written file
        return document


STRATEGY_CONFIG = StrategyConfigFile()


@dataclass
class Settings:
    """Immutable snapshot of every runtime parameter."""

    # ---------------------------------------------------------------- venue
    #: "alpaca" (REST, free, no gateway) or "ibkr" (needs TWS/IB Gateway, but
    #: brings resting brackets on option legs and real historical IV).
    broker: str = field(default_factory=lambda: str(setting("BVC_BROKER", "alpaca")).lower())

    # ---------------------------------------------------------------- IBKR
    ibkr_host: str = field(default_factory=lambda: str(setting("IBKR_HOST", "127.0.0.1")))
    #: 7497 paper TWS · 7496 live TWS · 4002 paper Gateway · 4001 live Gateway.
    ibkr_port: int = field(default_factory=lambda: _int("IBKR_PORT", 7497))
    ibkr_client_id: int = field(default_factory=lambda: _int("IBKR_CLIENT_ID", 17))
    ibkr_account: str = field(default_factory=lambda: str(setting("IBKR_ACCOUNT", "")))
    ibkr_readonly: bool = field(default_factory=lambda: _bool("IBKR_READONLY", False))
    ibkr_timeout: float = field(default_factory=lambda: _float("IBKR_TIMEOUT", 15.0))
    #: Seed IV Rank from the broker's own implied-vol history when it has one.
    use_broker_iv_history: bool = field(default_factory=lambda: _bool("BVC_USE_BROKER_IV", True))

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
    #: Which strategy from the library to run — see ``strategies.REGISTRY``.
    #: Deliberately *not* agent-mutable. Switching strategy changes the payoff
    #: geometry, the capital requirement and the options approval level needed;
    #: it is an operator decision. Hermes can recommend a switch in its report,
    #: and can compare strategies in the backtester, but cannot make one.
    #: Defaults to the defined-risk vertical rather than the cash-secured put.
    #: A CSP ties up the full strike notional to earn under 1% of it per cycle,
    #: which needs several hundred thousand dollars behind it to produce a
    #: meaningful income. The spread expresses the same view on the width.
    strategy: str = field(default_factory=lambda: str(setting("BVC_STRATEGY", "put_credit_spread")).lower())
    #: Delta of the long wing of a vertical, condor or butterfly.
    #:
    #: Selected by **delta, not by dollars**. A fixed $5 wing means something
    #: completely different on TLT at $90 and SPY at $600, and it made the
    #: backtester construct a different spread from the live bot — so the
    #: backtest was not testing the strategy that would actually run.
    wing_delta: float = field(default_factory=lambda: _float("BVC_WING_DELTA", 0.10))
    #: Optional ceiling on wing width in strike dollars, applied after the delta
    #: selection so margin per trade stays bounded on high-priced underlyings.
    #: ``0`` (the default) leaves the delta choice untouched.
    max_spread_width: float = field(default_factory=lambda: _float("BVC_MAX_SPREAD_WIDTH", 0.0))
    #: Delta of the long back-month leg in a diagonal (PMCC). Deep enough that
    #: the leg behaves like stock; 0.80 is the usual floor.
    long_leg_delta: float = field(default_factory=lambda: _float("BVC_LONG_LEG_DELTA", 0.80))
    #: Target DTE for the back month of a calendar or diagonal.
    back_month_dte: int = field(default_factory=lambda: _int("BVC_BACK_MONTH_DTE", 105))

    # ---------------------------------------------------------------- exit management
    #: Buy back at 50% of the credit received — the classic VRP profit taker.
    profit_target_pct: float = field(default_factory=lambda: _float("BVC_PROFIT_TARGET_PCT", 0.50))
    #: Stop when the open loss reaches this multiple of the premium at risk.
    #:
    #: The breakeven win rate of a ``profit_target/stop`` bracket is
    #: ``stop / (profit_target + stop)``. The traditional 50%/200% pair needs an
    #: **80%** win rate to break even, while a 30-delta short is only ~70%
    #: out-of-the-money on risk-neutral probabilities — so it starts below
    #: breakeven and depends entirely on the variance risk premium to climb
    #: above it. 100% needs 67%, which starts above the risk-neutral rate.
    stop_loss_multiple: float = field(default_factory=lambda: _float("BVC_STOP_LOSS_MULTIPLE", 1.00))
    #: When a hard stop applies at all.
    #:
    #: ``auto`` (default) fires the stop only on undefined-risk strategies. On a
    #: spread or a condor the tail has *already been bought*: the loss is capped
    #: at width minus credit, and a stop on top of that mostly converts
    #: drawdowns that would have recovered into realised losses. ``always`` and
    #: ``never`` override. Not agent-mutable — this is structural, like the
    #: choice of strategy.
    hard_stop_mode: str = field(default_factory=lambda: str(setting("BVC_HARD_STOP", "auto")).lower())
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
    # ---------------------------------------------------------------- hermes
    #: Master switch for the external self-improvement agent. Off by default:
    #: nothing may change how this trades until it is turned on deliberately.
    hermes_enabled: bool = field(default_factory=lambda: _bool("BVC_HERMES_ENABLED", False))

    #: When true the bot scans, scores and logs but never sends an order.
    dry_run: bool = field(default_factory=lambda: _bool("BVC_DRY_RUN", False))

    # ---------------------------------------------------------------- overlay provenance
    #: Which parameters came from ``strategy_config.json`` rather than from the
    #: operator, and what the file asked for that was refused. Rendered in the
    #: UI and returned by the API so an agent's footprint is never invisible.
    overlay_applied: dict = field(default_factory=dict)
    overlay_rejected: dict = field(default_factory=dict)
    #: What each overlaid parameter was *before* the file touched it. This is
    #: the ratchet's fixed reference — without it, re-reading the file would
    #: let a risk limit be walked outward one legal step per reload.
    overlay_baseline: dict = field(default_factory=dict)
    overlay_updated_at: str = ""
    overlay_updated_by: str = ""

    # ---------------------------------------------------------------- derived helpers
    def mutable_values(self) -> Dict[str, Any]:
        """Current value of every agent-mutable parameter."""
        return {name: getattr(self, name) for name in HERMES_BOUNDS if hasattr(self, name)}

    def baseline_values(self) -> Dict[str, Any]:
        """Agent-mutable parameters as the *operator* set them, overlay undone."""
        values = self.mutable_values()
        values.update(self.overlay_baseline)
        return values

    def config_store(self) -> StrategyConfigFile:
        """The parameter file for whichever strategy is active."""
        return StrategyConfigFile(strategy_config_path(self.strategy))

    def apply_overlay(self, source: StrategyConfigFile | None = None) -> "Settings":
        """Fold ``strategy_config.json`` in on top of the operator's baseline.

        Called against a ``Settings`` that has *not* yet been overlaid, so the
        ratchet always compares against what the human configured — repeated
        reloads can never walk a risk limit outward one small step at a time.
        """
        source = source or self.config_store()
        document = source.read()
        requested = document.get("parameters")
        baseline = self.baseline_values()

        # A parameter the file no longer mentions reverts to the operator's
        # value. Dropping a line from the config must actually undo it.
        for name, value in self.overlay_baseline.items():
            setattr(self, name, value)
        self.overlay_applied, self.overlay_rejected, self.overlay_baseline = {}, {}, {}

        if not isinstance(requested, dict) or not requested:
            return self

        accepted, rejected = vet_changes(baseline, requested)
        for name, value in accepted.items():
            setattr(self, name, value)
        self.overlay_baseline = {name: baseline[name] for name in accepted}
        self.overlay_applied = accepted
        self.overlay_rejected = rejected
        self.overlay_updated_at = str(document.get("updated_at") or "")
        self.overlay_updated_by = str(document.get("updated_by") or "")
        for name, why in rejected.items():
            logger.warning("strategy config: %s refused — %s", name, why)
        return self

    @property
    def uses_ibkr(self) -> bool:
        return self.broker in {"ibkr", "ib", "interactive_brokers"}

    @property
    def credentials_present(self) -> bool:
        """IBKR authenticates at the gateway, so there are no keys to check."""
        if self.uses_ibkr:
            return True
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


def load_settings(*, overlay: bool = True) -> Settings:
    """Build a fresh :class:`Settings` snapshot (re-reads env, secrets and file).

    Pass ``overlay=False`` for the operator baseline with no agent influence at
    all — that is what the ratchet is measured against, and what the UI shows as
    "your settings" beside "what the agent changed".
    """
    settings = Settings()
    return settings.apply_overlay() if overlay else settings


#: Import-time singleton for convenience; call :func:`load_settings` for a fresh read.
SETTINGS = load_settings()
