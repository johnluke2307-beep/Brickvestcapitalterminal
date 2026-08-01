# Brickvestcapitalterminal

Options intelligent insights — an automated options strategy dashboard that
harvests the **variance risk premium (VRP)**: the persistent gap between what
options *imply* volatility will be and what volatility *turns out* to be. It runs
a mechanical 45 DTE / 30 delta premium-selling programme benchmarked against a
**R10,000 per month** income target.

Broker and market data: **Alpaca** (free options-enabled paper account, free
market data). Deploys on Streamlit Community Cloud or Hugging Face Spaces with no
paid dependency and no local gateway process.

---

## The edge, stated plainly

Implied volatility trades above subsequent realised volatility most of the time
because option buyers pay for insurance. Selling that insurance systematically is
positive-expectancy *only* when three things hold together:

1. **The premium is actually there** — `VRP = IV − RV > 0`, measured against a
   gap-aware realised-vol estimator (Yang-Zhang), not a naive close-to-close one.
2. **Volatility is rich relative to its own history** — `IV Rank > 50`, so we are
   not selling cheap vol just because it is positive.
3. **The tail is bounded** — a hard stop, a margin ceiling, and position limits,
   because the return distribution of short premium is negatively skewed and one
   unmanaged loss undoes many wins.

The account's realised edge is tracked with the expectancy formula

```
E = (P_win × W) − (P_loss × L)
```

where `W` is the average winning trade and `L` the average absolute loss, both in
USD. The dashboard shows this alongside the *ex-ante* expectancy of each new
trade, computed from the short strike's delta and the managed exits.

---

## Architecture

| File | Responsibility |
|---|---|
| `app.py` | Streamlit dashboard — six tabs, no business logic of its own |
| `broker_client.py` | Alpaca connection, account, positions, option chains, order routing, connection health |
| `engine.py` | Black-Scholes, realised-volatility estimators, VRP, IV Rank, expectancy, USD→ZAR |
| `bot.py` | The execution loop: preflight → manage → scan → enter |
| `config.py` | Every tunable, resolved from env vars → `st.secrets` → defaults |
| `requirements.txt` | Four dependencies, all free-tier friendly |
| `tests/test_engine.py` | Maths regression tests (no network, no credentials) |

`engine.py` has **no broker dependency** — give it prices, it gives you an edge
estimate. `bot.py` and `app.py` talk to the broker only through `BrokerClient`,
whose surface is small enough that an IBKR implementation can replace it without
touching anything else.

### Cycle

```
preflight ─┬─ broker reachable?       ── no ──► HALT
           ├─ account unblocked?      ── no ──► HALT
           ├─ data feed healthy?      ── no ──► HALT
           └─ margin < 50% of equity? ── no ──► no new entries (existing risk still managed)
manage    ─── every short position: 50% profit target · 200% stop · 21 DTE time exit
scan      ─── universe → IV, RV, VRP, IV Rank
enter     ─── best qualifying candidate, subject to every guardrail
```

---

## Three deviations from the original IBKR specification

These are deliberate, and each one is visible in the UI rather than hidden.

**1. Alpaca replaces IBKR.** IBKR's API requires a running TWS or IB Gateway
process (via IBC) alongside the app. No free Streamlit or Hugging Face host will
do that — the container has no desktop session and is recycled on idle. Alpaca is
pure REST with an options-enabled paper account, so the whole system deploys free.
`ib_insync`'s asyncio integration is therefore not used; the bot runs on a daemon
thread and the client is lock-guarded, which keeps the Streamlit script thread
free for exactly the same reason.

**2. The 50%/200% bracket is synthetic.** Alpaca does not accept bracket, OCO or
OTO orders on option legs. The bot enforces the pair itself: trigger prices are
computed at entry from the actual fill, written to the trade log, shown on the
Positions tab, and evaluated every cycle. The consequence is real and stated on
that tab — **if the bot is stopped, open positions are not being managed.** The
cycle interval (default 300s) is a risk parameter, not a cosmetic one.

**3. IV Rank builds its own history.** IV Rank is by definition a trailing-range
statistic, and no free tier serves a year of historical implied volatility. Every
scan appends today's ATM IV to `state/iv_history.csv`; once
`BVC_IV_RANK_MIN_SAMPLES` (default 40) observations exist, the rank is a true IV
Rank. Before that the terminal ranks current IV against the trailing *realised*
vol distribution and labels it **"RV proxy — building IV history"** everywhere it
appears. It is never presented as something it is not.

---

## Risk guardrails

| Guardrail | Default | Where it is enforced |
|---|---|---|
| Maintenance margin ceiling | 50% of equity | `bot._entry_blockers` and `bot._buying_power_ok` — the latter checks *projected* post-trade utilisation, so the limit is forward-looking |
| Equity floor | $2,000 | no entries below it |
| Max open positions | 6 | counted from live broker positions, not the log |
| Max new entries per day | 2 | throttles correlated same-day risk |
| One position per underlying | on | prevents stacking the same tail |
| Liquidity filter | credit ≥ $0.35, spread ≤ 20% of mid | slippage is the tax on a small edge |
| Stop loss | 200% of credit | i.e. buy back at 3× credit; sent as a marketable close so it actually gets out |
| Time exit | 21 DTE | gamma risk rises faster than remaining theta |
| Fail-safe | any broker or data failure | halts the bot, persists the reason, flags the UI red; only a human clears it |

The fail-safe is the important one. `BrokerClient._guard` is the single choke
point for every outbound call: it retries transient failures with backoff, records
the outcome on a `ConnectionHealth` object, and re-raises as `BrokerError`. The
bot catches that and calls `halt()`, which writes `state/bot_state.json`. An
unattended premium seller running on a broken data feed is the most expensive
failure mode in this design, so it stops instead of guessing.

---

## Setup

### 1. Alpaca credentials

Create a free account at [alpaca.markets](https://alpaca.markets), then in the
**Paper Trading** dashboard generate an API key pair and enable options trading.

- **Level 2** is enough for the default `short_put` strategy (cash-secured puts).
- **Level 3** is required for `put_credit_spread` (multi-leg, defined risk).

### 2. Local run

```bash
pip install -r requirements.txt

export ALPACA_API_KEY=PK…
export ALPACA_SECRET_KEY=…
export ALPACA_PAPER=true

streamlit run app.py
```

Head-less execution without the UI:

```bash
python bot.py --dry-run          # one cycle, scores everything, sends nothing
python bot.py                    # one live cycle
python bot.py --loop             # continuous, 5-minute cadence
python bot.py --resume           # clear a persisted halt
```

Start with `--dry-run` for a few sessions. It exercises the whole path — chain
pricing, greeks, filters, sizing — and logs what it *would* have done.

### 3. Streamlit Community Cloud

1. Push this repository to GitHub.
2. New app → point it at `app.py`.
3. **Settings → Secrets** → paste the contents of
   `.streamlit/secrets.toml.example` with your real keys.

Note the free tier sleeps on inactivity. A sleeping app is a stopped bot, and a
stopped bot is unmanaged positions — for continuous management run
`python bot.py --loop` somewhere always-on and use the dashboard purely as a view.

### 4. Hugging Face Spaces

Create a Streamlit Space, push these files, and add the same keys under
**Settings → Variables and secrets** as environment variables.

---

## Configuration

Every parameter is an environment variable or a `st.secrets` key, listed with its
default in `config.py` and rendered live on the **Settings** tab. The ones worth
knowing:

| Variable | Default | Meaning |
|---|---|---|
| `BVC_UNIVERSE` | 8 liquid ETFs | ETFs are preferred — no earnings gaps, tight spreads |
| `BVC_TARGET_DTE` | 45 | richest premium decay per unit of gamma risk |
| `BVC_TARGET_DELTA` | 0.30 | ≈ 70% theoretical probability of profit |
| `BVC_MIN_IV_RANK` | 50 | sell rich vol only |
| `BVC_MIN_VRP` | 0.02 | require 2 vol points of edge |
| `BVC_MAX_MARGIN_UTIL` | 0.50 | the fat-tail governor |
| `BVC_MONTHLY_TARGET_ZAR` | 10000 | the income baseline |
| `BVC_DRY_RUN` | false | scan and log, never trade |

---

## State

Everything the platform persists lives in `state/` (git-ignored):

- `trade_log.csv` — every position opened and closed, with entry IV/RV/IV-Rank,
  credit, exit debit and P&L in both USD and ZAR. This is what expectancy is
  computed from; it is exportable from the Trade log tab.
- `iv_history.csv` — one ATM IV observation per symbol per day.
- `bot_state.json` — mode, halt reason, daily entry count, event feed.
- `fx_cache.json` — last good USD/ZAR.

On an ephemeral host this directory does not survive a restart. Point
`BVC_STATE_DIR` at a mounted volume if you want the history to persist.

---

## Currency

USD/ZAR is fetched from free key-less endpoints (`open.er-api.com`, then
`frankfurter.app`, then `exchangerate.host`), cached for 15 minutes and written to
disk. If every source fails, the configured fallback is used and flagged **stale**
in the sidebar — a guessed rate is never presented as a live one. The rate can
also be pinned by hand, which is useful on hosts that block outbound HTTP.

---

## Tests

```bash
python tests/test_engine.py       # or: pytest tests/
```

Covers Black-Scholes round-trips, the implied-volatility inversion, all three
realised-vol estimators, expectancy, IV Rank and OCC symbol parsing. No network
and no credentials required.

---

## What this is not

Paper trading, by construction. Options are leveraged instruments with asymmetric
downside; a 30-delta short put loses far more than its credit in a gap. The margin
ceiling, the stop and the position limits bound the damage — they do not eliminate
it. Run it in paper until the trade log itself, not the theory, shows a positive
expectancy over a sample large enough to mean something.
