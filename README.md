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
| `terminal.py` | The same deck in a console, via Rich — for an always-on host |
| `backtest.py` | Historical replay of the same rules, with the premium assumption exposed |
| `broker_client.py` | The broker surface + Alpaca implementation, plus the venue factory |
| `ibkr_client.py` | Interactive Brokers via ib_async — resting brackets, real IV history |
| `engine.py` | Black-Scholes, realised-volatility estimators, VRP, IV Rank, expectancy, USD→ZAR |
| `bot.py` | The execution loop: preflight → manage → scan → enter |
| `strategies.py` | The strategy library — eight structures as declarations, no broker imports |
| `strategies/*.json` | One parameter file per strategy; the agent's entire write surface |
| `hermes.py` | In-process control surface for an external self-improvement agent |
| `api.py` | Read-mostly HTTP telemetry + bounded parameter control. No order path |
| `memory.py` | Durable research notes via Honcho, falling back to a local JSONL |
| `skills/options_optimizer.md` | The daily post-close review procedure Hermes follows |
| `config.py` | Every tunable, resolved from `strategies/<strategy>.json` → env → `st.secrets` → defaults |
| `.streamlit/config.toml` | Terminal chrome — dark deck, monospace figures |
| `requirements.txt` | Six dependencies, all free-tier friendly |
| `tests/test_engine.py` | Maths regression tests (no network, no credentials) |

`engine.py` has **no broker dependency** — give it prices, it gives you an edge
estimate. `bot.py` and `app.py` talk to the broker only through `BrokerClient`,
whose surface is small enough that an IBKR implementation can replace it without
touching anything else.

`api.py`, `memory.py` and `strategies.py` have no broker dependency either, and
that is enforced rather than intended: a test parses their import graphs and
fails if any of them can reach `bot`, `broker_client`, `ibkr_client`, `ib_async`
or `alpaca` at runtime. The research layer is separated from the execution layer
by the module graph, not by good behaviour.

---

## The strategy library

One execution engine, one risk engine, one dashboard, eight strategies. A
strategy is a *declaration* — which legs to select from a chain, what capital
they consume, what the payoff geometry is. Everything after that is shared and
does not know which strategy it is running.

| Strategy | Legs | Risk | Wants | Notes |
|---|---|---|---|---|
| `cash_secured_put` | 1 | undefined | high IV | The live default. Assignment leaves you long stock |
| `covered_call` | 1 | undefined | high IV | Needs 100 shares per contract already held |
| `put_credit_spread` | 2 | defined | high IV | The CSP with the tail bought back |
| `call_credit_spread` | 2 | defined | high IV | The same trade on the upside |
| `iron_condor` | 4 | defined | high IV | ~2× the credit of one vertical for the same margin |
| `iron_butterfly` | 4 | defined | high IV | Shorts at the money — bigger credit, narrower zone |
| `calendar_spread` | 2 | defined | **low IV** | A debit trade, and the only long-volatility one here |
| `diagonal_spread` | 2 | defined | high IV | Poor man's covered call |

Select with `BVC_STRATEGY`; parameters come from `strategies/<key>.json`.
Regenerate the files with `python strategies.py --write-configs`.

**Credit and debit share one exit rule.** Six of these collect premium; the
calendar and diagonal pay it. Rather than special-case them, every plan reports
a *signed net premium* and exits are decided on P&L as a fraction of the premium
at risk:

```
profit when   pnl >=  profit_target_pct × |net premium|
stop   when   pnl <= −stop_loss_multiple × |net premium|
```

For a short put that is exactly "close at 50% of the credit, stop at 100%". For
a calendar it reads "take half the debit as profit, stop at twice it". One rule,
so a comparison between two strategies is a comparison of the strategies rather
than of two different exit regimes.

**Multi-leg trades are managed as trades, not as positions.** A condor's exit is
one decision about four legs at one net price. Evaluating each leg on its own
would close the tested wing of a spread and leave the short naked — the single
worst thing the loop could do — so the trade log carries the legs and the exit
logic reads them.

Adding a strategy: write a builder, register a `StrategyDefinition`, drop a JSON
file in `strategies/`. Nothing in `bot.py` changes.

### Cycle

```
preflight ─┬─ broker reachable?       ── no ──► HALT
           ├─ account unblocked?      ── no ──► HALT
           ├─ data feed healthy?      ── no ──► HALT
           └─ margin < 50% of equity? ── no ──► no new entries (existing risk still managed)
manage    ─── every open structure: 50% profit target · 100% stop · 21 DTE time exit
scan      ─── universe → IV, RV, VRP, IV Rank
enter     ─── best qualifying candidate, subject to every guardrail
```

---

## Three deviations from the original IBKR specification

These are deliberate, and each one is visible in the UI rather than hidden.

**Either venue, chosen by `BVC_BROKER`.** Both implement one interface, and the
bot reads a `BrokerCapabilities` object rather than assuming what a venue can do.

| | Alpaca | IBKR |
|---|---|---|
| Gateway process | none | TWS or IB Gateway must be running |
| Free hosting | yes | no — needs an always-on host |
| Brackets on options | bot-managed each cycle | **resting at the exchange (OCA)** |
| Historical implied vol | none — IV Rank uses an RV proxy | **yes, backfilled into IV Rank** |
| Credentials | API key pair | authenticated at the gateway |

```bash
BVC_BROKER=ibkr IBKR_PORT=7497 python bot.py --loop     # paper TWS
BVC_BROKER=ibkr IBKR_PORT=4002 streamlit run app.py     # paper Gateway
```

Ports: 7497 paper TWS · 7496 live TWS · 4002 paper Gateway · 4001 live Gateway.
In TWS enable **API → Settings → Enable ActiveX and Socket Clients**.

Two things change materially on IBKR. The 50%/200% pair becomes a real OCA
bracket held by IBKR, so **a stopped bot no longer means unmanaged positions** —
the largest operational risk in the Alpaca build. And the bot backfills a year of
`OPTION_IMPLIED_VOLATILITY` into the IV store on its first cycle, so IV Rank
becomes the true trailing-range statistic instead of the realised-vol proxy. The
time exit stays with the bot either way: no exchange order can express "close at
21 DTE".

**1. Alpaca was the original substitution for IBKR.** IBKR's API requires a running TWS or IB Gateway
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

## Backtesting

The **Backtest** tab (and `python backtest.py`) replays the same mechanical rules
over history, reusing `engine.py` for pricing, volatility and expectancy so a
backtest and a live cycle are scored by identical code.

**The volatility assumption is the whole ballgame.** No free source carries years
of historical *implied* volatility, so option prices are modelled. Pricing every
option at trailing realised vol — the obvious approach — quietly destroys the
thing being measured: if IV equals RV there is no variance risk premium, the
seller collects exactly fair value, and the run measures nothing but path luck.

So entries and marks price on a surface of `IV(t) = RV(t) + vrp_points`, where
`vrp_points` is the premium the market has historically paid over realised vol
(2–4 points on index products; 3 by default). Profit then comes from the actual
forward path being calmer than that surface implied, which is the real mechanism.

Set the slider to **0 points** to run the null hypothesis — no edge exists — and
the tab plots both curves together. If they track each other, the result is path
luck rather than edge. Always look at both.

```bash
python backtest.py --strategy put_credit_spread --vrp 0.03
python backtest.py --strategy put_credit_spread --vrp 0     # the null
python backtest.py --compare --capital 250000               # every structure, side by side
```

**Comparison mode** (`ALL — compare` in the tab, `--compare` on the CLI) replays
every structure over one price set, one starting balance and one rule set, so
only the structure differs. Return on capital is the honest yardstick: a
cash-secured put posts the full strike as collateral, so it can look safe and
still be the worst use of the money — on a $100k account it typically cannot size
a single SPY contract at all, and says so instead of returning zeros.

**Capital** is adjustable from $1,000 to $10m. It changes what is reachable, not
just the scale of the result.

**Intraday entry window.** The live bot can be restricted to a window measured in
minutes after the opening bell (`BVC_ENTRY_WINDOW=true`,
`BVC_ENTRY_WINDOW_START=30`, `BVC_ENTRY_WINDOW_END=120`), enforced against the
exchange calendar so half-days and DST are handled. **The backtest cannot replay
it** — daily bars carry one price per day and no intraday timestamps, and free
intraday history reaches back about 60 days, well short of a single 45-DTE cycle.
Enabling it prints that warning rather than silently ignoring the setting.

It runs on a hosted Streamlit deployment with **no Alpaca keys** — history comes
from yfinance, so the Backtest tab is usable before any secrets are set. A
six-year run over four symbols takes about 5 seconds and a couple of megabytes;
downloads are cached for an hour and shared with the null run, which halves the
requests. If Yahoo rate-limits the host (shared cloud IP ranges do get 429s), the
loader falls back to the broker feed when the Alpaca client is connected.

### Is it an edge or a curve fit?

The **Robustness** panel (and `sample_adequacy` / `split_sample` / `walk_forward` /
`sensitivity` / `plateau_score` in `backtest.py`) runs the checks that separate
the two:

- **Sample adequacy.** Trades, free parameters, and the ratio between them.
  Overlapping positions are not independent observations, so the effective count
  divides the trade count by average concurrency. Below ~10 effective trades per
  free parameter the result cannot distinguish edge from noise however good it
  looks.
- **In-sample vs out-of-sample.** A chronological 60/40 split — never random,
  which would leak the future into the training slice. A large CAGR drop on the
  unseen slice is the classic overfitting signature.
- **Walk-forward folds.** Four sequential, non-overlapping slices. One good
  regime can carry a multi-year total; an edge should recur.
- **Parameter sensitivity.** The most informative test available here. A real
  edge sits on a **plateau** — nudging delta from 0.30 to 0.28 moves the result a
  little. A curve fit sits on a **spike**: the chosen value is a peak surrounded
  by much worse neighbours, meaning it was picked to fit noise. A flat or
  downward-sloping sweep says the filter is not earning its place.

Two things worth stating plainly. First, the shipped defaults — 45 DTE, 30 delta,
50% profit, 200% stop, 21-DTE exit — were **not** fitted to this data; they are
long-standing conventions chosen before any backtest was run, which is the
strongest anti-overfitting property the platform has. Second, that property is
destroyed the moment the sliders are used to hunt for the best combination. Every
configuration tried is a silent multiple-comparison, and the tab does not know how
many you have tried. Decide the rules first, then test them.

What it does **not** model: bid/ask spread, early assignment, dividend and pin
risk, volatility skew across strikes, or whether a strike was actually listed and
liquid. Real fills are worse than these. Treat the output as a sanity check on the
rules, never as a forecast.

A note on capital: a cash-secured put ties up `strike × 100` — about $28,000 on a
$280 ETF — so a $100k account cannot hold many. Defined-risk spreads post only the
wing width and show dramatically better return on capital for the same rules. When
a run takes no trades it names the binding constraint rather than returning zeros.

---

## Hermes — the agent control surface

`hermes.py` is the integration point for an external self-improvement agent. It
observes, proposes changes, and can stop the bot — through one narrow, audited
interface rather than by reaching into `TradingBot`.

```python
from hermes import HermesControl
hermes = HermesControl(bot)

state   = hermes.observe()                     # everything, structured
verdict = hermes.propose({"target_delta": 0.25},
                         rationale="OOS Sharpe +0.31 across 4 folds",
                         evidence={"out_of_sample_validated": True})
hermes.halt("drawdown breach")                 # always permitted
```

Or over JSON from another process: `python hermes.py observe`,
`python hermes.py bounds`, `echo '{...}' | python hermes.py propose`.

### The separated path (this is the one to use)

The agent does not call into the trading process at all. It writes a file; the
execution loop reads it between cycles.

```
research process                    execution process
────────────────                    ─────────────────
api.py  ──writes──►  strategies/iron_condor.json  ──read at cycle start──►  bot.py
        ──writes──►  state/halt_request.json      ──consumed once──────────►
        ◄──reads───  state/trade_log.csv, bot_state.json, hermes_audit.jsonl
```

`api.py` runs standalone and has **no broker in its import graph**, so heavy LLM
inference on that side can never stall the ib_async event loop or delay a fill.

```bash
pip install fastapi uvicorn
BVC_API_TOKEN=$(openssl rand -hex 24) uvicorn api:app --port 8787
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is the loop alive, is it halted |
| `GET /metrics` | Sharpe, Sortino, win rate, expectancy, drawdown — **plus its own reliability verdict** |
| `GET /metrics/by?field=` | The same, cut by any trade-log column: strategy, exit reason, underlying |
| `GET /trades`, `/pnl`, `/events` | The realised record and the Rand target |
| `GET /strategies` | The library, which is active, each one's parameters |
| `GET /config` | Parameters, bounds, ratchet directions, what is immovable |
| `POST /config` | One bounded proposal — vetted, audited, written to the strategy file |
| `POST /halt` | Stop the bot. No matching resume |

The daily procedure the agent follows is `skills/options_optimizer.md`. Its
standing default is **propose nothing**, and most days that is the correct
output: this edge is structural and small, and it survives on being executed the
same way for a long time.

### The kill switch Hermes cannot reach

`bot.DAILY_LOSS_LIMIT_PCT` (default 3% of start-of-day equity) halts the loop
before any other account check. It is **not** a field on `config.Settings`, so
it is not in `HERMES_BOUNDS` and cannot be expressed in a strategy config file
at all. An agent tuning for return has every incentive to widen a daily loss
limit; the design answer is not to trust it not to, but to put the limit
somewhere the agent has no word for. A test asserts this, and would fail if
anyone added one.

Set `BVC_KILL_SWITCH_FLATTEN=true` on a venue with no resting brackets, where a
halted bot means an unmanaged short.

**The contract is deliberately asymmetric — reads wide, writes narrow:**

| Invariant | Why |
|---|---|
| **Risk limits ratchet one way** | The agent may tighten a guardrail, never loosen it — whatever the rationale. A system optimising "make more money" reads a margin ceiling as an obstacle. The worst case of a misaligned Hermes is an account that trades too little. |
| **Halt always, resume never** | Stopping needs no permission. Clearing a halt stays a human act, because the halt exists for exactly the conditions the automation misread. |
| **Fixed mutable set** | Venue, universe, credentials, the paper/live flag and the daily loss kill switch are not the agent's to change. Promoting to live is a human decision by construction. |
| **Strategy selection is the operator's** | Switching from a cash-secured put to an iron condor changes the payoff geometry, the capital per trade and the options approval level the account needs. Hermes can compare structures and recommend one; it cannot switch. |
| **Rationale required** | Every proposal carries one, and every proposal — accepted or rejected — is appended to `state/hermes_audit.jsonl` before anything changes. |
| **Off by default** | `BVC_HERMES_ENABLED=true` is required before anything can change how this trades. |

**Overfitting is the failure mode this is designed against.** An agent tuning
parameters against the backtester is an automated multiple-comparison machine.
So the surface counts every distinct configuration proposed — including the ones
adopted — and returns that count, plus a blunt `evidence_quality` verdict, on
every observation. An agent given only performance numbers will optimise them;
this one is also handed the reasons not to act.

---

## Risk guardrails

| Guardrail | Default | Where it is enforced |
|---|---|---|
| Maintenance margin ceiling | 50% of equity | `bot._entry_blockers` and `bot._capital_reject_reason` — the latter checks *projected* post-trade utilisation, so the limit is forward-looking |
| Equity floor | $2,000 | no entries below it |
| Max open positions | 6 | counted from live broker positions, not the log |
| Max new entries per day | 2 | throttles correlated same-day risk |
| One position per underlying | on | prevents stacking the same tail |
| Liquidity filter | credit ≥ $0.35, spread ≤ 20% of mid | slippage is the tax on a small edge |
| Stop loss | 100% of premium at risk | breakeven 67%, vs 80% for the traditional 200% pair. Skipped when the structure's own wing already caps the loss tighter |
| Time exit | 21 DTE | gamma risk rises faster than remaining theta |
| Fail-safe | any broker or data failure | halts the bot, persists the reason, flags the UI red; only a human clears it |

Both front ends compute the ACTION column through `TradingBot.position_action`,
which routes to the same `_exit_decision` the trading loop uses. A risk panel
that can disagree with the engine is worse than no risk panel, so they cannot
drift apart.

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

Two front ends over one engine:

```bash
streamlit run app.py         # browser deck
python terminal.py           # console deck (Rich), monitor only
python terminal.py --live    # console deck that also runs the trading loop
```

The console deck exists because Streamlit Community Cloud sleeps on inactivity,
and a sleeping app is a stopped bot, which is unmanaged positions. On a VPS or a
Pi, run `terminal.py --live` (or `bot.py --loop`) and keep the browser dashboard
for analysis. Run `streamlit run app.py` from the repository root so
`.streamlit/config.toml` is picked up — Streamlit resolves it against the working
directory, and without it the deck falls back to light chrome.

Head-less execution without any UI:

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
