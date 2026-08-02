"""
Brickvestcapitalterminal — Honcho memory adapter for the research layer.

Why the research layer needs memory at all
------------------------------------------
The optimizer runs once a day, cold. Without memory it cannot tell that it
already tried 0.25 delta three weeks ago and the out-of-sample folds said no —
so it tries again, and again, and every retry is another draw in a
multiple-comparison lottery it cannot see itself playing.

So what gets written here is not a transcript. It is the two things that
actually constrain tomorrow's search:

* **What was tried, and what happened next.** Every proposal, its rationale,
  its evidence, and — crucially — the realised outcome once enough trades have
  closed to judge it.
* **What was refused, and why.** A ratchet rejection is the most useful memory
  in the system: it is the boundary of the permitted space, learned once.

Degradation is deliberate
-------------------------
Honcho is optional. Without ``honcho-ai`` installed or ``HONCHO_API_KEY`` set,
this falls back to an append-only JSONL file in ``state/`` with the same API.
The research layer keeps working; it just remembers locally. A trading system
whose parameter history depends on a third-party service being up is a trading
system with an extra way to fail, and none of what is stored here needs to be
anywhere but on the operator's own disk.

Nothing in this module can trade. It has no broker import and no order path.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger("brickvest.memory")

MEMORY_PATH = config.STATE_DIR / "research_memory.jsonl"

#: Identifiers used in Honcho. The peer/session split maps naturally: the
#: terminal and the optimizer are two peers holding one long conversation about
#: one strategy.
DEFAULT_WORKSPACE = "brickvestcapitalterminal"
AGENT_PEER = "hermes"
TERMINAL_PEER = "terminal"


class ResearchMemory:
    """Durable notes for the optimizer. Honcho when available, local file otherwise."""

    def __init__(
        self,
        *,
        session_id: str = "strategy-optimization",
        path: Optional[Path] = None,
        api_key: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.path = Path(path or MEMORY_PATH)
        self.session_id = session_id
        self.backend = "local"
        self._client = None
        self._session = None
        self._peer = None

        wanted = (
            enabled
            if enabled is not None
            else str(config.setting("BVC_HONCHO_ENABLED", "false")).lower() in {"1", "true", "yes", "on"}
        )
        if wanted:
            self._connect(api_key or str(config.setting("HONCHO_API_KEY", "") or ""))

    # ------------------------------------------------------------------ setup
    def _connect(self, api_key: str) -> None:
        """Attach to Honcho, or log why not and carry on locally."""
        if not api_key:
            logger.info("Honcho enabled but HONCHO_API_KEY is unset — using local memory")
            return
        try:
            from honcho import Honcho
        except ImportError:
            logger.info("honcho-ai is not installed — using local memory (pip install honcho-ai)")
            return
        try:
            self._client = Honcho(
                api_key=api_key,
                workspace_id=str(config.setting("HONCHO_WORKSPACE", DEFAULT_WORKSPACE)),
                base_url=str(config.setting("HONCHO_URL", "")) or None,
            )
            self._peer = self._client.peer(AGENT_PEER)
            self._session = self._client.session(self.session_id)
            self._session.add_peers([self._peer, self._client.peer(TERMINAL_PEER)])
            self.backend = "honcho"
        except Exception as exc:
            # Any Honcho failure is a research-quality problem, never a trading
            # one — this module is not in the execution path.
            logger.warning("Honcho unavailable (%s) — using local memory", exc)
            self._client = self._session = self._peer = None

    # ------------------------------------------------------------------ write
    def remember(self, kind: str, summary: str, **detail: Any) -> dict:
        """Record one research event. Always writes locally; mirrors to Honcho.

        The local file is the source of truth even when Honcho is connected. A
        remote store that is authoritative is a remote store that can lose your
        parameter history during an outage you find out about later.
        """
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "summary": summary,
            **detail,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("could not write research memory: %s", exc)

        if self._session and self._peer:
            try:
                self._session.add_messages([
                    self._peer.message(json.dumps(entry, default=str))
                ])
            except Exception as exc:
                logger.warning("Honcho write failed, kept locally: %s", exc)
        return entry

    def remember_proposal(self, verdict: dict, rationale: str, evidence: Dict[str, Any]) -> dict:
        """Record a parameter proposal and its verdict, accepted or not."""
        applied = verdict.get("applied") or {}
        rejected = verdict.get("rejected") or {}
        summary = (
            f"proposed {json.dumps(applied or rejected, sort_keys=True)} — "
            f"{'accepted' if applied else 'refused'}: {rationale[:160]}"
        )
        return self.remember(
            "proposal",
            summary,
            applied=applied,
            rejected=rejected,
            rationale=rationale,
            evidence=evidence,
        )

    def remember_outcome(self, parameters: Dict[str, Any], metrics: Dict[str, Any]) -> dict:
        """Record how a configuration actually performed once trades closed.

        This is the entry that makes the memory worth keeping. A log of
        proposals tells the optimizer what it *believed*; only this tells it
        what was true.
        """
        return self.remember(
            "outcome",
            f"config {json.dumps(parameters, sort_keys=True)} → "
            f"{metrics.get('trades', 0)} trades, Sharpe {metrics.get('sharpe')}, "
            f"win rate {metrics.get('win_rate')}",
            parameters=parameters,
            metrics=metrics,
        )

    # ------------------------------------------------------------------- read
    def recall(self, limit: int = 50, kind: Optional[str] = None) -> List[dict]:
        """The most recent research events, newest last."""
        try:
            lines = self.path.read_text().strip().splitlines()
        except OSError:
            return []
        rows: List[dict] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None or row.get("kind") == kind:
                rows.append(row)
        return rows[-limit:]

    def ask(self, question: str) -> Optional[str]:
        """Natural-language query over the stored history, when Honcho is attached.

        Returns ``None`` on the local backend rather than faking an answer —
        the caller should fall back to :meth:`recall` and read the rows itself.
        """
        if not (self._peer and self._session):
            return None
        try:
            return self._peer.chat(question, session=self._session)
        except Exception as exc:
            logger.warning("Honcho query failed: %s", exc)
            return None

    def tried_configurations(self) -> List[Dict[str, Any]]:
        """Every distinct parameter set already evaluated — the search burden.

        The optimizer should read this before proposing anything. Re-proposing a
        configuration that was already tried and rejected is not persistence,
        it is a second draw from the same lottery.
        """
        seen: List[Dict[str, Any]] = []
        keys = set()
        for row in self.recall(limit=10_000, kind="proposal"):
            payload = row.get("applied") or row.get("rejected") or {}
            key = json.dumps(payload, sort_keys=True)
            if payload and key not in keys:
                keys.add(key)
                seen.append(payload)
        return seen


#: Import-time singleton for the research layer.
MEMORY = ResearchMemory()
