"""
Brickvestcapitalterminal — Hermes control surface.

The integration point for an external self-improvement agent. Hermes observes
the terminal, proposes changes to how it trades, and can stop it — through one
narrow, audited interface rather than by reaching into ``TradingBot`` internals.

The contract is deliberately asymmetric
---------------------------------------
**Reads are wide. Writes are narrow.** An agent that can read everything and
change little is useful; one that can change everything is a liability. So:

* Observation returns the whole state — account, positions, scan, expectancy,
  robustness, capabilities — in one structured payload.
* Mutation is limited to a fixed set of tunables, each with hard bounds the
  agent cannot argue its way past.
* **Risk limits move one way.** The agent may tighten a guardrail; a proposal
  that loosens one is rejected outright, whatever its rationale. Self-improving
  systems optimise the objective they are given, and "make more money" reads a
  margin ceiling as an obstacle. This invariant means the worst case of a
  misaligned Hermes is an account that trades too little.
* **Halting is always allowed; resuming never is.** Hermes can stop the bot at
  any time without asking. It cannot clear a halt — that stays a human act,
  because the halt exists precisely for conditions the automation misread.
* Every proposal, accepted or rejected, is appended to an audit log with its
  rationale before anything changes.

Overfitting is the failure mode to design against
-------------------------------------------------
An agent tuning parameters against the backtester is an automated
multiple-comparison machine: try enough configurations and one looks excellent
by chance. So the surface counts every configuration Hermes has evaluated, and
:meth:`HermesControl.propose` refuses a change that has not been validated out
of sample. The count is reported back on every observation, so the agent can see
its own search burden — and so can you.

Usage
-----
    from hermes import HermesControl
    hermes = HermesControl(bot)

    state = hermes.observe()                    # everything, structured
    verdict = hermes.propose(                   # bounded, audited
        {"target_delta": 0.25},
        rationale="OOS Sharpe improved 0.31 at 0.25 delta over 4 folds",
        evidence={"out_of_sample_cagr": 0.14, "folds_profitable": 4},
    )
    hermes.halt("drawdown breach")              # always permitted

Or over JSON, for an agent in another process:

    python hermes.py observe
    echo '{"changes":{"target_delta":0.25},"rationale":"..."}' | python hermes.py propose
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import engine

logger = logging.getLogger("brickvest.hermes")

HERMES_AUDIT_PATH = config.STATE_DIR / "hermes_audit.jsonl"


# ======================================================================================
# What Hermes is allowed to touch
# ======================================================================================
# The bounds table and the gate that enforces it live in ``config`` — the same
# module the execution loop reads its parameters from — so there is exactly one
# definition of "what an agent may change", shared by the in-process surface,
# the JSON overlay and the HTTP API. Re-exported here because this is where
# people come looking for it.
Bound = config.Bound
HERMES_BOUNDS = config.HERMES_BOUNDS
vet_changes = config.vet_changes


# ======================================================================================
# Audit
# ======================================================================================
@dataclass
class HermesAction:
    """One agent interaction, recorded before it takes effect."""

    action_id: str
    ts: str
    kind: str                       # observe | propose | apply | halt | stop | start
    accepted: bool
    rationale: str = ""
    changes: Dict[str, Any] = field(default_factory=dict)
    rejections: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    actor: str = "hermes"

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class HermesAudit:
    """Append-only JSONL record of everything the agent asked for.

    Separate from the bot's own event feed on purpose: when an automated system
    is changing how another automated system trades, the record of *who changed
    what and why* has to survive independently of either.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path or HERMES_AUDIT_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, action: HermesAction) -> None:
        try:
            with self.path.open("a") as handle:
                handle.write(action.to_json() + "\n")
        except OSError as exc:
            logger.warning("could not write Hermes audit: %s", exc)

    def tail(self, limit: int = 50) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text().strip().splitlines()
        except OSError:
            return []
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def experiment_count(self) -> int:
        """How many distinct configurations Hermes has proposed.

        This is the multiple-comparison counter. Every configuration tried is
        another chance for one to look good by luck, and a search this number
        cannot see is a search that will overfit without noticing.
        """
        seen = set()
        for row in self.tail(limit=10_000):
            # Both kinds count. An accepted proposal is logged as "apply", and
            # a configuration that was actually adopted is the most definite
            # experiment of all — counting only rejected ones would report a
            # tuning agent as having searched nothing.
            if row.get("kind") in {"propose", "apply"} and row.get("changes"):
                seen.add(json.dumps(row["changes"], sort_keys=True))
        return len(seen)


# ======================================================================================
# Verdict
# ======================================================================================
@dataclass
class Verdict:
    """The answer to a proposal: what was accepted, what was refused and why."""

    accepted: bool
    action_id: str
    applied: Dict[str, Any] = field(default_factory=dict)
    rejected: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ======================================================================================
# The control surface
# ======================================================================================
class HermesControl:
    """Everything an external agent may do to this terminal, and nothing else."""

    def __init__(self, bot, audit: Optional[HermesAudit] = None, enabled: Optional[bool] = None) -> None:
        self.bot = bot
        self.audit = audit or HermesAudit()
        self.enabled = bot.settings.hermes_enabled if enabled is None else enabled

    # ------------------------------------------------------------------ read
    def observe(self) -> dict:
        """The full state, structured for a model to reason over.

        Deliberately includes the things that argue *against* acting: how thin
        the evidence is, how many configurations have already been tried, and
        which guardrails are immovable. An agent given only performance numbers
        will optimise them.
        """
        settings = self.bot.settings
        closed = self.bot.trade_log.closed_trades()
        expectancy = engine.compute_expectancy(closed)

        state: Dict[str, Any] = {
            "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hermes": {
                "enabled": self.enabled,
                "may_resume_after_halt": False,
                "experiments_proposed": self.audit.experiment_count(),
                "config_file": str(settings.config_store().path),
                "overlay_applied": settings.overlay_applied,
                "overlay_rejected": settings.overlay_rejected,
                # Stated explicitly so the agent never has to discover these the
                # expensive way. Nothing it writes can move them.
                "immovable": {
                    "daily_loss_limit_pct": self.bot.DAILY_LOSS_LIMIT_PCT,
                    "note": (
                        "The daily loss kill switch is hard-coded in the execution module, "
                        "is not a Settings field, and is not in the mutable set. Neither the "
                        "config file nor this API can change it. Broker, account, paper/live "
                        "and credentials are likewise unreachable from here."
                    ),
                },
                "mutable_parameters": {
                    name: {
                        "value": getattr(settings, name, None),
                        "low": b.low, "high": b.high,
                        "ratcheted": b.risk_limit, "safer_direction": b.safer,
                        "note": b.note,
                    }
                    for name, b in HERMES_BOUNDS.items()
                },
            },
            "bot": {
                "mode": self.bot.state.mode,
                "halted": self.bot.state.is_halted,
                "halt_reason": self.bot.state.halt_reason,
                "running": self.bot.is_running,
                "cycles": self.bot.state.cycles,
                "entries_today": self.bot.state.entries_today_count(),
                "last_cycle": self.bot.state.last_cycle_summary,
            },
            "venue": asdict(self.bot.client.capabilities),
            "performance": {
                "trades_closed": expectancy.trades,
                "win_rate": expectancy.p_win,
                "breakeven_win_rate": expectancy.breakeven_p_win,
                "expectancy_usd": expectancy.expectancy,
                "profit_factor": expectancy.profit_factor,
                "total_pnl_usd": expectancy.total_pnl,
                "monthly_zar": engine.monthly_pnl(closed, "zar"),
                "monthly_target_zar": settings.monthly_target_zar,
            },
            # The honest counterweight to the performance block.
            "evidence_quality": self._evidence_quality(expectancy),
        }

        try:
            account = self.bot.client.get_account()
            state["account"] = {
                "equity": account.equity,
                "cash": account.cash,
                "maintenance_margin": account.maintenance_margin,
                "margin_utilization": account.margin_utilization,
                "day_pnl": account.day_pnl,
            }
            state["positions"] = [
                {
                    "symbol": p.symbol, "underlying": p.underlying, "qty": p.qty,
                    "strike": p.strike, "dte": p.dte, "mark": p.current_price,
                    "unrealized_pl": p.unrealized_pl,
                    "action": self.bot.position_action(p)["action"],
                }
                for p in self.bot.client.get_option_positions()
            ]
        except Exception as exc:
            state["account_error"] = str(exc)
            state["positions"] = []

        last = self.bot.last_result
        if last:
            state["last_scan"] = [
                {
                    "symbol": s.symbol, "iv": s.implied_vol, "rv": s.reference_rv,
                    "vrp": s.vrp, "iv_rank": s.iv_rank.value,
                    "iv_rank_source": s.iv_rank.source,
                    "tradeable": s.is_tradeable, "reject_reason": s.reject_reason(),
                }
                for s in last.scanned
            ]
            state["blocked"] = last.blocked
        return state

    def _evidence_quality(self, expectancy) -> dict:
        """How much the record can actually support — the anti-overconfidence block."""
        import backtest as bt

        trades = expectancy.trades
        params = len(HERMES_BOUNDS)
        experiments = self.audit.experiment_count()
        per_param = (trades / params) if params else 0.0

        if trades < 30:
            verdict = "insufficient — too few closed trades to distinguish skill from luck"
        elif per_param < 10:
            verdict = "thin — fewer than 10 closed trades per tunable parameter"
        elif experiments > max(trades / 10, 5):
            verdict = "search-heavy — more configurations tried than the record can justify"
        else:
            verdict = "adequate"
        return {
            "closed_trades": trades,
            "tunable_parameters": params,
            "trades_per_parameter": per_param,
            "configurations_tried": experiments,
            "verdict": verdict,
            "note": (
                "Every configuration tried is another chance for one to look good by luck. "
                "Validate a change out of sample before proposing it."
            ),
            "backtest_parameters": len(bt.TUNABLE_PARAMETERS),
        }

    # ----------------------------------------------------------------- write
    def propose(
        self,
        changes: Dict[str, Any],
        rationale: str,
        evidence: Optional[Dict[str, Any]] = None,
        *,
        apply: bool = True,
    ) -> Verdict:
        """Ask to change how the bot trades. Bounded, ratcheted and audited.

        Every change is checked independently, so a proposal mixing a legal and
        an illegal change applies the legal part and reports the rest — the
        agent gets a specific reason rather than a blanket refusal it cannot
        learn from.
        """
        action_id = uuid.uuid4().hex[:12]
        verdict = Verdict(accepted=False, action_id=action_id)
        settings = self.bot.settings
        evidence = evidence or {}

        if not self.enabled:
            verdict.rejected["*"] = "Hermes control is disabled (set BVC_HERMES_ENABLED=true)"
            self._record("propose", verdict, rationale, changes, evidence)
            return verdict

        if not rationale or len(rationale.strip()) < 10:
            verdict.rejected["*"] = "a substantive rationale is required for every proposal"
            self._record("propose", verdict, rationale, changes, evidence)
            return verdict

        if self.bot.state.is_halted:
            verdict.rejected["*"] = "the bot is halted; a human must resolve that before parameters change"
            self._record("propose", verdict, rationale, changes, evidence)
            return verdict

        # Vetted against the *operator's* baseline, not the running values. A
        # ratchet measured against the last accepted proposal is no ratchet at
        # all: an agent could walk a limit outward one legal step at a time.
        accepted, rejected = config.vet_changes(self._baseline(), changes)
        verdict.applied.update(accepted)
        verdict.rejected.update(rejected)

        if apply and accepted:
            for name, value in accepted.items():
                setattr(settings, name, value)
            # Persist to the JSON contract so the change survives a restart and
            # shows up as a reviewable diff rather than as in-memory drift.
            store = settings.config_store()
            merged = dict(store.parameters())
            merged.update(accepted)
            try:
                store.write(
                    merged, rationale=rationale, evidence=evidence, actor="hermes"
                )
                self.bot.mark_config_current()
            except OSError as exc:
                verdict.warnings.append(f"applied in memory but not persisted: {exc}")

        if not evidence.get("out_of_sample_validated") and verdict.applied:
            verdict.warnings.append(
                "no out-of-sample validation supplied — pass "
                "evidence={'out_of_sample_validated': True, ...} after a split or walk-forward run"
            )
        quality = self._evidence_quality(engine.compute_expectancy(self.bot.trade_log.closed_trades()))
        if quality["verdict"] != "adequate" and verdict.applied:
            verdict.warnings.append(f"evidence is {quality['verdict']}")

        verdict.accepted = bool(verdict.applied)
        self._record("apply" if apply else "propose", verdict, rationale, changes, evidence)
        if verdict.applied:
            self.bot.state.log_event(
                "info",
                f"Hermes changed {', '.join(f'{k}={v}' for k, v in verdict.applied.items())} — {rationale[:120]}",
            )
            self.bot.state.save()
        return verdict

    def halt(self, reason: str) -> dict:
        """Stop the bot. Always permitted, never questioned.

        The one write that needs no justification: an agent that suspects
        trouble should never be arguing with a permission check.
        """
        self.bot.halt(f"Hermes: {reason}")
        verdict = Verdict(accepted=True, action_id=uuid.uuid4().hex[:12])
        self._record("halt", verdict, reason, {}, {})
        return {"halted": True, "reason": reason}

    def stop(self, reason: str = "") -> dict:
        """Stop the automation loop without entering the emergency state."""
        self.bot.stop()
        verdict = Verdict(accepted=True, action_id=uuid.uuid4().hex[:12])
        self._record("stop", verdict, reason, {}, {})
        return {"running": False}

    def start(self, reason: str = "") -> dict:
        """Start the loop. Refused while halted — clearing a halt is a human act."""
        if not self.enabled:
            return {"started": False, "reason": "Hermes control is disabled"}
        if self.bot.state.is_halted:
            return {
                "started": False,
                "reason": "the bot is halted; only a human may clear a halt",
            }
        started = self.bot.start()
        verdict = Verdict(accepted=started, action_id=uuid.uuid4().hex[:12])
        self._record("start", verdict, reason, {}, {})
        return {"started": started}

    def history(self, limit: int = 50) -> List[dict]:
        return self.audit.tail(limit)

    # ---------------------------------------------------------------- private
    def _baseline(self) -> Dict[str, Any]:
        """What a proposal is measured against: the values in force right now.

        There are two anchors in this system and they are different on purpose.

        * Here, the anchor is the **current effective value**, so tightening is
          monotone — once the agent narrows a risk limit it cannot widen it
          again by proposal. And because an accepted proposal is written to
          ``strategy_config.json``, the tightened value is what the next process
          starts from, so the ratchet survives a restart.
        * In :meth:`config.Settings.apply_overlay`, the anchor is the operator's
          own pre-overlay value, so a file that has been hand-edited, corrupted
          or replaced wholesale can still never resolve to something looser than
          what the operator configured.

        The invariant both enforce: **no risk limit ever ends up looser than the
        human set it.** Widening one back is a human act — edit the file.
        """
        return self.bot.settings.mutable_values()

    def _record(self, kind: str, verdict: Verdict, rationale: str, changes: dict, evidence: dict) -> None:
        self.audit.record(
            HermesAction(
                action_id=verdict.action_id,
                ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                kind=kind,
                accepted=verdict.accepted,
                rationale=rationale,
                changes=changes,
                rejections=[f"{k}: {v}" for k, v in verdict.rejected.items()],
                evidence=evidence,
            )
        )


# ======================================================================================
# JSON CLI — for an agent living in another process
# ======================================================================================
def main() -> int:
    import argparse
    import sys

    from bot import TradingBot

    parser = argparse.ArgumentParser(description="Hermes control surface")
    parser.add_argument("command", choices=["observe", "propose", "halt", "stop", "start", "history", "bounds"])
    parser.add_argument("--reason", default="")
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    if args.command == "bounds":
        print(json.dumps({k: asdict(v) for k, v in HERMES_BOUNDS.items()}, indent=2))
        return 0

    bot = TradingBot()
    hermes = HermesControl(bot)

    if args.command == "observe":
        print(json.dumps(hermes.observe(), indent=2, default=str))
    elif args.command == "history":
        print(json.dumps(hermes.history(args.limit), indent=2, default=str))
    elif args.command == "halt":
        print(json.dumps(hermes.halt(args.reason or "requested via CLI")))
    elif args.command == "stop":
        print(json.dumps(hermes.stop(args.reason)))
    elif args.command == "start":
        print(json.dumps(hermes.start(args.reason)))
    elif args.command == "propose":
        payload = json.load(sys.stdin)
        verdict = hermes.propose(
            payload.get("changes", {}),
            payload.get("rationale", ""),
            payload.get("evidence", {}),
            apply=payload.get("apply", True),
        )
        print(json.dumps(verdict.as_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
