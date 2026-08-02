"""
Brickvestcapitalterminal — the strategy library.

One execution engine, one risk engine, one dashboard, many strategies. A
strategy here is a *declaration*, not a program: which legs to select from a
chain, what capital they consume, and what the payoff geometry is. Everything
after that — the VRP filter, the entry window, the margin ceiling, the daily
loss kill switch, position management, the trade log, the Rand conversion — is
shared and does not know which strategy it is running.

That split is what makes the strategies comparable. If each one carried its own
risk handling you would never be able to tell whether the iron condor beat the
cash-secured put or just had a looser stop.

Adding a strategy
-----------------
Write a builder that turns a chain into legs, register a
:class:`StrategyDefinition`, and drop a JSON file in ``strategies/``. Nothing in
``bot.py`` changes.

Credit and debit in one vocabulary
----------------------------------
Six of these collect premium; calendars and diagonals pay it. Rather than
special-case them, every plan reports a **signed net premium** — positive when
you are paid, negative when you pay — and exits are decided on P&L as a
fraction of the premium at risk:

    profit when   pnl >= profit_target_pct × |net premium|
    stop when     pnl <= −stop_loss_multiple × |net premium|

For a short put that is exactly the classic "close at 50% of the credit, stop at
200%". For a calendar it means "take half the debit as profit, stop at twice
it". One rule, and the comparison between strategies stays honest.

What the numbers are not
------------------------
``capital_required`` is the platform's own conservative estimate, used to size
trades and to project margin utilisation *before* sending an order. It is not
the broker's margin calculation. IBKR's real requirement for an undefined-risk
short is portfolio-margin dependent and can exceed the naive figure; the
estimates here are deliberately no lower than the standard Reg-T requirement,
and the live guardrail still reads maintenance margin from the account.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence

import config

if TYPE_CHECKING:  # pragma: no cover - avoids importing the broker layer
    from broker_client import OptionQuote

logger = logging.getLogger("brickvest.strategies")

OPTION_MULTIPLIER = 100

STRATEGIES_DIR = config.ROOT_DIR / "strategies"


# ======================================================================================
# Selected legs and the resulting plan
# ======================================================================================
@dataclass
class Leg:
    """One contract in a plan, with the direction it will be traded."""

    quote: "OptionQuote"
    action: str          # "sell" | "buy"
    ratio: int = 1

    @property
    def symbol(self) -> str:
        return self.quote.symbol

    @property
    def is_short(self) -> bool:
        return self.action == "sell"

    @property
    def premium(self) -> float:
        """Signed premium per share: positive when this leg pays us."""
        mid = self.quote.mid or 0.0
        return (mid if self.is_short else -mid) * self.ratio

    def describe(self) -> str:
        return (
            f"{'-' if self.is_short else '+'}{self.ratio} {self.quote.expiration:%d%b%y} "
            f"{self.quote.strike:g}{self.quote.option_type[0].upper()}"
        )


@dataclass
class StrategyPlan:
    """A complete, priced, checkable trade — before anything is sent."""

    strategy: str
    underlying: str
    legs: List[Leg]
    capital_required: float
    max_loss: Optional[float] = None        # per contract, in dollars; None = undefined
    max_profit: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def net_premium(self) -> float:
        """Signed net premium per share. Positive = credit, negative = debit."""
        return round(sum(leg.premium for leg in self.legs), 4)

    @property
    def is_credit(self) -> bool:
        return self.net_premium > 0

    @property
    def premium_at_risk(self) -> float:
        """The magnitude the exit rules are measured against."""
        return abs(self.net_premium)

    @property
    def short_leg(self) -> Optional[Leg]:
        """The primary short — what the delta target and IV Rank filter refer to."""
        shorts = [leg for leg in self.legs if leg.is_short]
        if not shorts:
            return None
        return max(shorts, key=lambda leg: abs(leg.quote.delta or 0.0))

    @property
    def expirations(self) -> List[date]:
        return sorted({leg.quote.expiration for leg in self.legs})

    @property
    def nearest_expiration(self) -> date:
        return self.expirations[0]

    @property
    def net_delta(self) -> Optional[float]:
        deltas = [leg.quote.delta for leg in self.legs]
        if any(d is None for d in deltas):
            return None
        return round(
            sum((-d if leg.is_short else d) * leg.ratio
                for leg, d in zip(self.legs, deltas)),
            4,
        )

    def describe(self) -> str:
        return f"{self.underlying} {' '.join(leg.describe() for leg in self.legs)}"

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "underlying": self.underlying,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "action": leg.action,
                    "ratio": leg.ratio,
                    "strike": leg.quote.strike,
                    "right": leg.quote.option_type,
                    "expiration": leg.quote.expiration.isoformat(),
                    "mid": leg.quote.mid,
                    "delta": leg.quote.delta,
                }
                for leg in self.legs
            ],
            "net_premium": self.net_premium,
            "capital_required": self.capital_required,
            "max_loss": self.max_loss,
            "max_profit": self.max_profit,
        }


@dataclass
class BuildContext:
    """Everything a builder is allowed to look at."""

    underlying: str
    spot: float
    settings: config.Settings
    near: Sequence["OptionQuote"]           # the ~45 DTE chain, calls and puts
    far: Sequence["OptionQuote"] = ()       # the back-month chain, when needed


# ======================================================================================
# Chain selection helpers
# ======================================================================================
def _side(chain: Sequence["OptionQuote"], right: str) -> List["OptionQuote"]:
    return [q for q in chain if q.option_type == right and q.mid]


def by_delta(
    chain: Sequence["OptionQuote"], right: str, target: float, tolerance: float
) -> Optional["OptionQuote"]:
    """The contract whose |delta| is nearest ``target``, inside ``tolerance``.

    Tolerance is enforced rather than "nearest wins": on a thin chain the
    nearest strike to 30 delta can be 12 delta, and selling that is a different
    strategy from the one that was backtested.
    """
    scored = [
        (abs(abs(q.delta) - abs(target)), q)
        for q in _side(chain, right)
        if q.delta is not None and abs(abs(q.delta) - abs(target)) <= tolerance
    ]
    if not scored:
        return None
    return min(scored, key=lambda item: item[0])[1]


def at_strike(chain: Sequence["OptionQuote"], right: str, strike: float) -> Optional["OptionQuote"]:
    """The listed strike nearest ``strike``."""
    side = _side(chain, right)
    return min(side, key=lambda q: abs(q.strike - strike)) if side else None


def at_the_money(chain: Sequence["OptionQuote"], right: str, spot: float) -> Optional["OptionQuote"]:
    return at_strike(chain, right, spot)


def _fail(reason: str) -> None:
    logger.debug("strategy build rejected: %s", reason)


# ======================================================================================
# Builders — one per strategy
# ======================================================================================
def build_cash_secured_put(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Sell the 30-delta put. Undefined risk down to zero, fully cash-secured."""
    s = ctx.settings
    short = by_delta(ctx.near, "put", s.target_delta, s.delta_tolerance)
    if short is None:
        return None
    legs = [Leg(short, "sell")]
    credit = short.mid or 0.0
    return StrategyPlan(
        strategy="cash_secured_put",
        underlying=ctx.underlying,
        legs=legs,
        capital_required=short.strike * OPTION_MULTIPLIER,
        max_loss=(short.strike - credit) * OPTION_MULTIPLIER,
        max_profit=credit * OPTION_MULTIPLIER,
        notes=["assignment leaves you long 100 shares at the strike"],
    )


def build_covered_call(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Sell the 30-delta call against stock you already own."""
    s = ctx.settings
    short = by_delta(ctx.near, "call", s.target_delta, s.delta_tolerance)
    if short is None:
        return None
    credit = short.mid or 0.0
    return StrategyPlan(
        strategy="covered_call",
        underlying=ctx.underlying,
        legs=[Leg(short, "sell")],
        # The capital is the stock, not the option. Reported honestly so the
        # margin projection does not flatter this strategy against the others.
        capital_required=ctx.spot * OPTION_MULTIPLIER,
        max_loss=(ctx.spot - credit) * OPTION_MULTIPLIER,
        max_profit=(short.strike - ctx.spot + credit) * OPTION_MULTIPLIER,
        notes=["requires 100 shares per contract already held — the bot will not buy them"],
    )


def _vertical(
    ctx: BuildContext, right: str, key: str, *, wing_below: bool
) -> Optional[StrategyPlan]:
    """Shared body for the two credit verticals."""
    s = ctx.settings
    short = by_delta(ctx.near, right, s.target_delta, s.delta_tolerance)
    if short is None:
        return None
    target = short.strike - s.spread_width if wing_below else short.strike + s.spread_width
    long = at_strike(ctx.near, right, target)
    if long is None or long.strike == short.strike:
        return None

    width = abs(short.strike - long.strike)
    credit = (short.mid or 0.0) - (long.mid or 0.0)
    return StrategyPlan(
        strategy=key,
        underlying=ctx.underlying,
        legs=[Leg(short, "sell"), Leg(long, "buy")],
        capital_required=width * OPTION_MULTIPLIER,
        max_loss=(width - credit) * OPTION_MULTIPLIER,
        max_profit=credit * OPTION_MULTIPLIER,
        notes=[f"defined risk, {width:g}-wide wing"],
    )


def build_put_credit_spread(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Bull put spread: the cash-secured put with the tail bought back."""
    return _vertical(ctx, "put", "put_credit_spread", wing_below=True)


def build_call_credit_spread(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Bear call spread — the same trade expressed on the upside."""
    return _vertical(ctx, "call", "call_credit_spread", wing_below=False)


def build_iron_condor(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Short strangle with both tails bought back. The canonical VRP harvest.

    Only one side can finish in the money, so the margin is one wing's width —
    which is why a condor collects roughly twice the premium of a single
    vertical for the same capital, and why it is the workhorse of premium
    selling when there is no directional view.
    """
    s = ctx.settings
    short_put = by_delta(ctx.near, "put", s.target_delta, s.delta_tolerance)
    short_call = by_delta(ctx.near, "call", s.target_delta, s.delta_tolerance)
    if short_put is None or short_call is None:
        return None
    long_put = at_strike(ctx.near, "put", short_put.strike - s.spread_width)
    long_call = at_strike(ctx.near, "call", short_call.strike + s.spread_width)
    if long_put is None or long_call is None:
        return None
    if long_put.strike >= short_put.strike or long_call.strike <= short_call.strike:
        return None

    put_width = short_put.strike - long_put.strike
    call_width = long_call.strike - short_call.strike
    width = max(put_width, call_width)
    credit = (
        (short_put.mid or 0.0) + (short_call.mid or 0.0)
        - (long_put.mid or 0.0) - (long_call.mid or 0.0)
    )
    return StrategyPlan(
        strategy="iron_condor",
        underlying=ctx.underlying,
        legs=[
            Leg(long_put, "buy"), Leg(short_put, "sell"),
            Leg(short_call, "sell"), Leg(long_call, "buy"),
        ],
        # Only one side can lose, so the requirement is the wider wing, not both.
        capital_required=width * OPTION_MULTIPLIER,
        max_loss=(width - credit) * OPTION_MULTIPLIER,
        max_profit=credit * OPTION_MULTIPLIER,
        notes=[f"profit zone {long_put.strike:g}–{long_call.strike:g}, "
               f"breakevens {short_put.strike - credit:.2f} / {short_call.strike + credit:.2f}"],
    )


def build_iron_butterfly(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Condor with the shorts collapsed to the money.

    Much larger credit, much narrower profit zone. It is the highest-conviction
    expression of "realised volatility will come in below implied" — and the
    least forgiving if it does not.
    """
    s = ctx.settings
    body_put = at_the_money(ctx.near, "put", ctx.spot)
    body_call = at_strike(ctx.near, "call", body_put.strike) if body_put else None
    if body_put is None or body_call is None:
        return None
    long_put = at_strike(ctx.near, "put", body_put.strike - s.spread_width)
    long_call = at_strike(ctx.near, "call", body_call.strike + s.spread_width)
    if long_put is None or long_call is None:
        return None
    if long_put.strike >= body_put.strike or long_call.strike <= body_call.strike:
        return None

    width = max(body_put.strike - long_put.strike, long_call.strike - body_call.strike)
    credit = (
        (body_put.mid or 0.0) + (body_call.mid or 0.0)
        - (long_put.mid or 0.0) - (long_call.mid or 0.0)
    )
    return StrategyPlan(
        strategy="iron_butterfly",
        underlying=ctx.underlying,
        legs=[
            Leg(long_put, "buy"), Leg(body_put, "sell"),
            Leg(body_call, "sell"), Leg(long_call, "buy"),
        ],
        capital_required=width * OPTION_MULTIPLIER,
        max_loss=(width - credit) * OPTION_MULTIPLIER,
        max_profit=credit * OPTION_MULTIPLIER,
        notes=[f"body at {body_put.strike:g}, breakevens "
               f"{body_put.strike - credit:.2f} / {body_call.strike + credit:.2f}"],
    )


def build_calendar_spread(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Sell the near-dated at-the-money, buy the same strike further out.

    A **debit** trade, and the one strategy here that is long volatility. It
    profits from the front month decaying faster than the back month, so it
    wants a *low* IV Rank at entry — the opposite of everything else in this
    library. The VRP filter has to be inverted for it, which the definition
    declares via ``prefers_high_iv=False``.
    """
    s = ctx.settings
    if not ctx.far:
        return None
    front = at_the_money(ctx.near, "put", ctx.spot)
    if front is None:
        return None
    back = at_strike(ctx.far, "put", front.strike)
    if back is None or back.expiration <= front.expiration:
        return None

    debit = (back.mid or 0.0) - (front.mid or 0.0)
    if debit <= 0:
        return None  # a "calendar" collected for a credit is mispriced data
    return StrategyPlan(
        strategy="calendar_spread",
        underlying=ctx.underlying,
        legs=[Leg(front, "sell"), Leg(back, "buy")],
        capital_required=debit * OPTION_MULTIPLIER,
        max_loss=debit * OPTION_MULTIPLIER,   # the debit is the whole risk
        max_profit=None,                      # depends on the back month's IV at the front expiry
        notes=[f"debit {debit:.2f}; wants low IV Rank and a quiet tape",
               f"front {front.expiration:%d%b%y}, back {back.expiration:%d%b%y}"],
    )


def build_diagonal_spread(ctx: BuildContext) -> Optional[StrategyPlan]:
    """Poor man's covered call: long a deep back-month call, short a near OTM call.

    A stock replacement — the long call stands in for 100 shares at a fraction
    of the capital, and the short call is the covered call written against it.
    Also a debit trade.
    """
    s = ctx.settings
    if not ctx.far:
        return None
    short = by_delta(ctx.near, "call", s.target_delta, s.delta_tolerance)
    long = by_delta(ctx.far, "call", s.long_leg_delta, max(s.delta_tolerance, 0.10))
    if short is None or long is None:
        return None
    if long.strike >= short.strike or long.expiration <= short.expiration:
        return None

    debit = (long.mid or 0.0) - (short.mid or 0.0)
    if debit <= 0:
        return None
    return StrategyPlan(
        strategy="diagonal_spread",
        underlying=ctx.underlying,
        legs=[Leg(short, "sell"), Leg(long, "buy")],
        capital_required=debit * OPTION_MULTIPLIER,
        max_loss=debit * OPTION_MULTIPLIER,
        # Approximate: ignores whatever extrinsic value the back month still
        # carries at the front expiry, so it understates rather than flatters.
        max_profit=(short.strike - long.strike - debit) * OPTION_MULTIPLIER,
        notes=[f"debit {debit:.2f}; {abs(long.delta or 0):.2f}-delta back-month call "
               f"replaces the stock"],
    )


# ======================================================================================
# The registry
# ======================================================================================
@dataclass(frozen=True)
class StrategyDefinition:
    """What the shared engine needs to know to run a strategy it has never seen."""

    key: str
    label: str
    thesis: str
    build: Callable[[BuildContext], Optional[StrategyPlan]]
    #: True when the strategy collects premium and wants rich implied vol.
    #: False inverts the IV Rank filter — a calendar wants cheap front-month vol.
    prefers_high_iv: bool = True
    defined_risk: bool = True
    leg_count: int = 1
    #: Needs a second, later expiry fetched and passed to the builder.
    needs_back_month: bool = False
    #: Needs 100 shares of the underlying already held per contract.
    needs_shares: bool = False
    #: Alpaca/IBKR options approval level the account must hold.
    options_level: int = 3
    #: Per-strategy parameter defaults, layered under the JSON config file.
    defaults: Dict[str, float] = field(default_factory=dict)

    @property
    def is_multi_leg(self) -> bool:
        return self.leg_count > 1

    @property
    def config_path(self) -> Path:
        return STRATEGIES_DIR / f"{self.key}.json"


REGISTRY: Dict[str, StrategyDefinition] = {
    d.key: d
    for d in [
        StrategyDefinition(
            key="cash_secured_put",
            label="Cash-secured put",
            thesis="Sell the 30-delta put outright. The simplest VRP harvest; "
                   "assignment leaves you long stock at a discount.",
            build=build_cash_secured_put,
            defined_risk=False,
            leg_count=1,
            options_level=2,
            defaults={"target_delta": 0.30, "profit_target_pct": 0.50, "stop_loss_multiple": 2.0},
        ),
        StrategyDefinition(
            key="covered_call",
            label="Covered call",
            thesis="Sell the 30-delta call against stock already held. Converts "
                   "an equity holding into a premium stream, capped on the upside.",
            build=build_covered_call,
            defined_risk=False,
            leg_count=1,
            needs_shares=True,
            options_level=1,
            defaults={"target_delta": 0.30, "profit_target_pct": 0.50, "stop_loss_multiple": 2.0},
        ),
        StrategyDefinition(
            key="put_credit_spread",
            label="Put credit spread",
            thesis="The cash-secured put with the tail bought back. Same "
                   "direction, a fraction of the capital, a bounded worst case.",
            build=build_put_credit_spread,
            leg_count=2,
            defaults={"target_delta": 0.30, "spread_width": 5.0},
        ),
        StrategyDefinition(
            key="call_credit_spread",
            label="Call credit spread",
            thesis="The same structure on the upside. Useful when IV is rich but "
                   "the skew makes puts the expensive side to be short.",
            build=build_call_credit_spread,
            leg_count=2,
            defaults={"target_delta": 0.30, "spread_width": 5.0},
        ),
        StrategyDefinition(
            key="iron_condor",
            label="Iron condor",
            thesis="Both tails sold, both bought back. Roughly double the credit "
                   "of one vertical for the same margin, with no directional view.",
            build=build_iron_condor,
            leg_count=4,
            defaults={"target_delta": 0.20, "spread_width": 5.0, "profit_target_pct": 0.50},
        ),
        StrategyDefinition(
            key="iron_butterfly",
            label="Iron butterfly",
            thesis="Condor with the shorts at the money. Much larger credit, "
                   "much narrower profit zone — the sharpest bet that realised "
                   "volatility comes in under implied.",
            build=build_iron_butterfly,
            leg_count=4,
            defaults={"spread_width": 10.0, "profit_target_pct": 0.25},
        ),
        StrategyDefinition(
            key="calendar_spread",
            label="Calendar spread",
            thesis="Sell the front month, own the back month at the same strike. "
                   "Long volatility and long time decay — it wants a cheap, quiet "
                   "tape, which is exactly when the rest of this library stands down.",
            build=build_calendar_spread,
            prefers_high_iv=False,
            leg_count=2,
            needs_back_month=True,
            defaults={"profit_target_pct": 0.35, "stop_loss_multiple": 1.0, "min_iv_rank": 0.0},
        ),
        StrategyDefinition(
            key="diagonal_spread",
            label="Diagonal spread (PMCC)",
            thesis="A deep back-month call stands in for the stock; a near-month "
                   "OTM call is written against it. A covered call at a fraction "
                   "of the capital.",
            build=build_diagonal_spread,
            leg_count=2,
            needs_back_month=True,
            defaults={"target_delta": 0.30, "long_leg_delta": 0.80,
                      "profit_target_pct": 0.35, "stop_loss_multiple": 1.0},
        ),
    ]
}

#: Ordered for the UI and for ``compare_strategies`` — simplest first.
STRATEGY_KEYS: List[str] = list(REGISTRY)

#: The premium-selling subset, which is what the VRP filter is actually about.
CREDIT_STRATEGIES: List[str] = [k for k, d in REGISTRY.items() if d.prefers_high_iv]


def get(key: str) -> StrategyDefinition:
    """Look up a strategy, failing loudly on a typo rather than silently."""
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown strategy {key!r}; available: {', '.join(REGISTRY)}"
        ) from None


def build_plan(key: str, ctx: BuildContext) -> Optional[StrategyPlan]:
    """Construct the trade for ``key``, or ``None`` if the chain cannot support it."""
    return get(key).build(ctx)


def catalogue() -> List[dict]:
    """The whole library as plain data — for the UI, the API and the optimizer."""
    return [
        {
            "key": d.key,
            "label": d.label,
            "thesis": d.thesis,
            "legs": d.leg_count,
            "defined_risk": d.defined_risk,
            "prefers_high_iv": d.prefers_high_iv,
            "needs_back_month": d.needs_back_month,
            "needs_shares": d.needs_shares,
            "options_level": d.options_level,
            "config": str(d.config_path),
            "config_exists": d.config_path.exists(),
        }
        for d in REGISTRY.values()
    ]


# ======================================================================================
# Per-strategy configuration files
# ======================================================================================
def write_default_configs(overwrite: bool = False) -> List[Path]:
    """Materialise ``strategies/<key>.json`` for every registered strategy.

    Each file carries the strategy's own parameter defaults merged over the
    platform-wide ones, so switching strategies switches a coherent parameter
    set rather than inheriting the previous strategy's tuning — a 20-delta
    condor and a 30-delta cash-secured put are not the same trade with a
    different leg count.
    """
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    baseline = config.Settings()
    written: List[Path] = []

    for definition in REGISTRY.values():
        path = definition.config_path
        if path.exists() and not overwrite:
            continue
        parameters = {
            name: definition.defaults.get(name, getattr(baseline, name))
            for name in config.HERMES_BOUNDS
            if hasattr(baseline, name)
        }
        parameters.update(
            {k: v for k, v in definition.defaults.items() if k not in parameters}
        )
        document = {
            "version": 1,
            "strategy": definition.key,
            "label": definition.label,
            "thesis": definition.thesis,
            "updated_at": "",
            "updated_by": "operator",
            "rationale": f"Starting parameters for {definition.label}. Not a fitted result.",
            "evidence": {"out_of_sample_validated": False},
            "parameters": parameters,
        }
        path.write_text(json.dumps(document, indent=2) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import argparse

    parser = argparse.ArgumentParser(description="Brickvestcapitalterminal strategy library")
    parser.add_argument("--write-configs", action="store_true", help="create strategies/*.json")
    parser.add_argument("--overwrite", action="store_true", help="replace existing config files")
    args = parser.parse_args()

    if args.write_configs:
        for path in write_default_configs(overwrite=args.overwrite):
            print(f"wrote {path}")
    else:
        for entry in catalogue():
            print(f"{entry['key']:<20} {entry['legs']} legs  "
                  f"{'defined' if entry['defined_risk'] else 'UNDEFINED':<9} risk  "
                  f"{'high' if entry['prefers_high_iv'] else 'LOW':<4} IV  — {entry['thesis'][:60]}")
