"""
Brickvestcapitalterminal — read-mostly HTTP surface for the research layer.

What this process is
--------------------
A **separate process from the trading loop, with no broker connection at all.**
It never imports ``ibkr_client``, never constructs a ``TradingBot``, never opens
a socket to TWS and has no way to place, modify or cancel an order. Everything
it serves it reads from the artifacts the execution loop leaves on disk — the
trade log, the bot state file, the audit log, the strategy config.

That constraint is the architecture, not an implementation detail. The failure
mode being designed against is an LLM with a tool that eventually reaches
``placeOrder``. The defence is not a careful prompt; it is that no such path
exists in this module's import graph. There is a test that asserts it.

What it can change
------------------
Exactly two things, both by writing a file the execution loop chooses when to
read:

* ``POST /config`` — parameters, filtered through :func:`config.vet_changes`.
  Out-of-bounds values are refused; risk limits may only be tightened.
* ``POST /halt`` — writes a stop request. There is no matching resume. Clearing
  a halt is a human act performed at the machine that holds the account.

Neither call reaches into the trading process. Both leave a file. The loop picks
it up between cycles, so heavy inference on this side can never stall the
ib_async event loop or delay a fill.

Running it
----------
    pip install fastapi uvicorn
    BVC_API_TOKEN=$(openssl rand -hex 24) uvicorn api:app --port 8787

Set ``BVC_API_TOKEN`` and every endpoint requires ``Authorization: Bearer …``.
Leave it unset and the service refuses to start unless ``BVC_API_ALLOW_OPEN=true``
is also set — an unauthenticated endpoint that can retune a trading strategy is
not a default anyone should be able to reach by accident.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import config
import engine

logger = logging.getLogger("brickvest.api")

API_TOKEN = str(config.setting("BVC_API_TOKEN", "") or "")
ALLOW_OPEN = str(config.setting("BVC_API_ALLOW_OPEN", "false")).lower() in {"1", "true", "yes", "on"}

if not API_TOKEN and not ALLOW_OPEN:
    raise RuntimeError(
        "BVC_API_TOKEN is not set. This service can retune a live trading strategy; "
        "it will not start without a token. Set BVC_API_ALLOW_OPEN=true to override "
        "on a trusted loopback-only host."
    )

app = FastAPI(
    title="Brickvestcapitalterminal",
    description="Read-mostly telemetry and bounded parameter control. No order path.",
    version="1.0.0",
)
_bearer = HTTPBearer(auto_error=False)


def require_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    if not API_TOKEN:
        return
    if not credentials or credentials.credentials != API_TOKEN:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")


# ======================================================================================
# Reading what the execution loop left behind
# ======================================================================================
def _trade_log() -> engine.TradeLog:
    return engine.TradeLog()


def _bot_state() -> dict:
    try:
        return json.loads(config.BOT_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _audit_tail(limit: int) -> List[dict]:
    path = config.STATE_DIR / "hermes_audit.jsonl"
    try:
        lines = path.read_text().strip().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _settings() -> config.Settings:
    """Fresh settings including the current overlay — what the loop would use."""
    return config.load_settings()


# ======================================================================================
# Schemas
# ======================================================================================
class ConfigProposal(BaseModel):
    """A parameter change, its argument, and the evidence behind it."""

    changes: Dict[str, float] = Field(..., description="parameter → new value")
    rationale: str = Field(..., min_length=10, description="why, in a sentence a human can audit")
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "backtest support. Include out_of_sample_validated=true only when the "
            "change was measured on data the search never saw."
        ),
    )


class HaltRequest(BaseModel):
    reason: str = Field(..., min_length=3)
    actor: str = "hermes"


# ======================================================================================
# Telemetry
# ======================================================================================
@app.get("/health")
def health() -> dict:
    """Liveness plus the execution loop's own last word about itself."""
    state = _bot_state()
    return {
        "service": "ok",
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authenticated": bool(API_TOKEN),
        "bot": {
            "mode": state.get("mode", "unknown"),
            "halted": state.get("mode") == "halted",
            "halt_reason": state.get("halt_reason"),
            "cycles": state.get("cycles", 0),
            "last_cycle_at": state.get("last_cycle_at"),
            "last_cycle_summary": state.get("last_cycle_summary", ""),
        },
    }


@app.get("/trades", dependencies=[Depends(require_token)])
def trades(
    status_filter: Optional[str] = Query(None, alias="status", description="open | closing | closed"),
    limit: int = Query(200, ge=1, le=5000),
) -> dict:
    """The daily trade log, newest last — the primary record for the optimizer."""
    rows = _trade_log().all()
    if status_filter:
        rows = [r for r in rows if r.get("status") == status_filter]
    return {"count": len(rows), "trades": rows[-limit:]}


@app.get("/pnl", dependencies=[Depends(require_token)])
def pnl(
    by: str = Query("day", pattern="^(day|month)$"),
    currency: str = Query("usd", pattern="^(usd|zar)$"),
) -> dict:
    """Realised P&L by close date, in USD or Rand, against the monthly target."""
    closed = _trade_log().closed_trades()
    settings = _settings()
    buckets = (
        engine.daily_pnl(closed, currency) if by == "day" else engine.monthly_pnl(closed, currency)
    )
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    month_zar = engine.monthly_pnl(closed, "zar").get(month_key, 0.0)
    return {
        "by": by,
        "currency": currency,
        "buckets": buckets,
        "month_to_date_zar": month_zar,
        "monthly_target_zar": settings.monthly_target_zar,
        "target_progress": (
            month_zar / settings.monthly_target_zar if settings.monthly_target_zar else 0.0
        ),
    }


@app.get("/metrics", dependencies=[Depends(require_token)])
def metrics(currency: str = Query("usd", pattern="^(usd|zar)$")) -> dict:
    """Sharpe, Sortino, win rate, expectancy, drawdown — with their own caveats.

    ``reliability`` and ``evidence`` are part of the payload rather than a
    footnote. An optimizer handed only performance numbers will optimise them,
    including the ones computed from eleven days of data.
    """
    closed = _trade_log().closed_trades()
    payload = engine.performance_metrics(closed, currency)
    payload["evidence"] = _evidence_quality(len(closed))
    return payload


def _evidence_quality(trades: int) -> dict:
    """How far the record can actually be pushed — the anti-overconfidence block."""
    params = len(config.HERMES_BOUNDS)
    experiments = len({
        json.dumps(row.get("changes"), sort_keys=True)
        for row in _audit_tail(10_000)
        if row.get("kind") in {"propose", "apply"} and row.get("changes")
    })
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
            "Validate out of sample before proposing."
        ),
    }


@app.get("/events", dependencies=[Depends(require_token)])
def events(limit: int = Query(100, ge=1, le=200)) -> dict:
    """The execution loop's event feed — entries, exits, blocks, halts."""
    return {"events": (_bot_state().get("events") or [])[-limit:]}


@app.get("/audit", dependencies=[Depends(require_token)])
def audit(limit: int = Query(50, ge=1, le=1000)) -> dict:
    """Every parameter change ever asked for, accepted or refused."""
    return {"actions": _audit_tail(limit)}


# ======================================================================================
# Configuration — the only write path, and it writes a file
# ======================================================================================
@app.get("/config", dependencies=[Depends(require_token)])
def get_config() -> dict:
    """Current parameters, their bounds, and what cannot be changed from here."""
    settings = _settings()
    document = config.STRATEGY_CONFIG.read()
    return {
        "path": str(config.STRATEGY_CONFIG.path),
        "updated_at": document.get("updated_at"),
        "updated_by": document.get("updated_by"),
        "rationale": document.get("rationale"),
        "parameters": settings.mutable_values(),
        "operator_baseline": settings.baseline_values(),
        "overlay_applied": settings.overlay_applied,
        "overlay_rejected": settings.overlay_rejected,
        "bounds": {
            name: {
                "low": b.low,
                "high": b.high,
                "ratcheted": b.risk_limit,
                "safer_direction": b.safer,
                "note": b.note,
            }
            for name, b in config.HERMES_BOUNDS.items()
        },
        "immovable": {
            "note": (
                "Not settable from this API by construction, not by policy: the daily "
                "loss kill switch, the broker, the account, paper/live, credentials and "
                "the universe are not Hermes-mutable parameters and are not read from "
                "the config file."
            ),
            "daily_loss_limit_pct": _daily_loss_limit(),
        },
    }


def _daily_loss_limit() -> float:
    """Report the kill switch without importing the execution module.

    ``bot`` pulls in the broker layer, and this process must not have that in
    its import graph. The value is read from the same environment variable the
    execution module reads it from.
    """
    try:
        return float(config.setting("BVC_DAILY_LOSS_LIMIT_PCT", 0.03))
    except (TypeError, ValueError):
        return 0.03


@app.post("/config", dependencies=[Depends(require_token)])
def post_config(proposal: ConfigProposal) -> dict:
    """Propose parameter changes. Bounded, ratcheted, audited, then written.

    Returns the same shape whether it accepted everything, some of it or none:
    ``applied`` and ``rejected`` with a specific reason per key, so an agent
    that got something wrong is told which thing and why.
    """
    settings = _settings()
    accepted, rejected = config.vet_changes(settings.baseline_values(), proposal.changes)

    warnings: List[str] = []
    if accepted and not proposal.evidence.get("out_of_sample_validated"):
        warnings.append(
            "no out-of-sample validation supplied — pass "
            "evidence={'out_of_sample_validated': true, ...} after a split or walk-forward run"
        )
    quality = _evidence_quality(len(_trade_log().closed_trades()))
    if accepted and quality["verdict"] != "adequate":
        warnings.append(f"evidence is {quality['verdict']}")

    if accepted:
        merged = dict(config.STRATEGY_CONFIG.parameters())
        merged.update(accepted)
        try:
            config.STRATEGY_CONFIG.write(
                merged,
                rationale=proposal.rationale,
                evidence=proposal.evidence,
                actor="hermes-api",
            )
        except OSError as exc:
            raise HTTPException(status.HTTP_507_INSUFFICIENT_STORAGE, f"could not write config: {exc}")

    _record_audit(
        kind="apply" if accepted else "propose",
        accepted=bool(accepted),
        rationale=proposal.rationale,
        changes=proposal.changes,
        rejections=[f"{k}: {v}" for k, v in rejected.items()],
        evidence=proposal.evidence,
    )
    return {
        "accepted": bool(accepted),
        "applied": accepted,
        "rejected": rejected,
        "warnings": warnings,
        "effective": "at the start of the execution loop's next cycle",
    }


@app.post("/halt", dependencies=[Depends(require_token)])
def halt(request: HaltRequest) -> dict:
    """Stop the bot. Always permitted, never questioned, and not reversible here.

    The one write that needs no justification: an agent that suspects trouble
    should never be arguing with a permission check. There is deliberately no
    ``/resume`` — the halt exists for conditions the automation misread, so
    clearing it is a human act performed where the account is.
    """
    payload = {
        "reason": request.reason,
        "actor": request.actor,
        "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    config.HALT_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.HALT_REQUEST_PATH.write_text(json.dumps(payload, indent=2))
    _record_audit("halt", True, request.reason, {}, [], {})
    return {
        "halt_requested": True,
        "honoured": "at the start of the execution loop's next cycle",
        "resume": "not available from this API — a human must clear a halt",
    }


def _record_audit(
    kind: str,
    accepted: bool,
    rationale: str,
    changes: dict,
    rejections: List[str],
    evidence: dict,
) -> None:
    """Append to the same audit log the in-process control surface writes.

    Written before the caller is told anything, so a request that crashes this
    process still left a record that it was made.
    """
    import uuid

    path = config.STATE_DIR / "hermes_audit.jsonl"
    row = {
        "action_id": uuid.uuid4().hex[:12],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "accepted": accepted,
        "rationale": rationale,
        "changes": changes,
        "rejections": rejections,
        "evidence": evidence,
        "actor": "hermes-api",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
    except OSError as exc:
        logger.warning("could not write audit row: %s", exc)


if __name__ == "__main__":  # pragma: no cover - convenience runner
    import uvicorn

    uvicorn.run(app, host=str(config.setting("BVC_API_HOST", "127.0.0.1")),
                port=int(config.setting("BVC_API_PORT", 8787)))
